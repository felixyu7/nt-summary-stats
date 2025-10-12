#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kNumStats = 9;

std::array<double, kNumStats> empty_stats() {
    std::array<double, kNumStats> stats{};
    stats.fill(0.0);
    return stats;
}

bool is_sorted(const std::vector<double>& values) {
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i - 1] > values[i]) {
            return false;
        }
    }
    return true;
}

void sort_by_time(std::vector<double>& times, std::vector<double>& charges) {
    std::vector<std::size_t> order(times.size());
    std::iota(order.begin(), order.end(), 0);
    // Stable ordering is not required here; equal-time elements have identical
    // timestamps so the chosen representative for a bin is unchanged.
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return times[a] < times[b];
    });
    std::vector<double> sorted_times(times.size());
    std::vector<double> sorted_charges(charges.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        sorted_times[i] = times[order[i]];
        sorted_charges[i] = charges[order[i]];
    }
    times.swap(sorted_times);
    charges.swap(sorted_charges);
}

std::pair<std::vector<double>, std::vector<double>> group_hits_by_window(
    const std::vector<double>& times,
    const std::vector<double>& charges,
    double window_ns) {
    if (times.empty()) {
        return {std::vector<double>{}, std::vector<double>{}};
    }
    std::vector<double> grouped_times;
    std::vector<double> grouped_charges;
    grouped_times.reserve(times.size());
    grouped_charges.reserve(times.size());

    const double base_time = times.front();
    double bin_time = times.front();
    double bin_charge = charges.front();
    auto current_bin = static_cast<long long>(0);
    double current_bin_end = base_time + window_ns;  // end of current bin (exclusive)

    for (std::size_t i = 1; i < times.size(); ++i) {
        const double time = times[i];
        if (time < current_bin_end) {
            bin_charge += charges[i];
        } else {
            // Finish the current non-empty bin.
            grouped_times.push_back(bin_time);
            grouped_charges.push_back(bin_charge);
            // Jump directly to the bin containing this time without iterating per empty bin.
            const auto new_bin = static_cast<long long>(std::floor((time - base_time) / window_ns));
            current_bin = new_bin;
            current_bin_end = base_time + (static_cast<double>(current_bin) + 1.0) * window_ns;
            bin_time = time;
            bin_charge = charges[i];
        }
    }

    // Flush the final bin
    grouped_times.push_back(bin_time);
    grouped_charges.push_back(bin_charge);
    return {std::move(grouped_times), std::move(grouped_charges)};
}

std::array<double, kNumStats> compute_stats_from_sorted(
    const std::vector<double>& times,
    const std::vector<double>& charges) {
    const std::size_t n = times.size();
    if (n == 0) {
        return empty_stats();
    }
    if (n == 1) {
        const double time = times.front();
        const double charge = charges.front();
        return {charge, charge, charge, time, time, time, time, time, 0.0};
    }

    const double first_time = times.front();
    const double last_time = times.back();

    // First pass: totals, weighted moments, and fixed-window charges.
    const double cutoff_100 = first_time + 100.0;
    const double cutoff_500 = first_time + 500.0;

    double total_charge = 0.0;
    double charge_100 = 0.0;
    double charge_500 = 0.0;
    double sum_qt = 0.0;
    double sum_qt2 = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double t = times[i];
        const double q = charges[i];
        total_charge += q;
        sum_qt += q * t;
        sum_qt2 += q * t * t;
        if (t <= cutoff_100) charge_100 += q;
        if (t <= cutoff_500) charge_500 += q;
    }

    // Second pass: times at which 20% and 50% charge have been collected.
    const double threshold_20 = total_charge * 0.2;
    const double threshold_50 = total_charge * 0.5;
    double running = 0.0;
    double charge_20_time = first_time;
    double charge_50_time = first_time;
    bool have_20 = false;
    bool have_50 = false;
    for (std::size_t i = 0; i < n; ++i) {
        running += charges[i];
        // Match NumPy's searchsorted(..., side='right') semantics: first index with cumulative > threshold.
        if (!have_20 && running > threshold_20) {
            charge_20_time = times[i];
            have_20 = true;
        }
        if (!have_50 && running > threshold_50) {
            charge_50_time = times[i];
            have_50 = true;
            if (have_20) break; // both found
        }
    }

    double weighted_mean = 0.0;
    double weighted_std = 0.0;
    if (total_charge > 0.0) {
        weighted_mean = sum_qt / total_charge;
        const double variance = (sum_qt2 / total_charge) - (weighted_mean * weighted_mean);
        weighted_std = variance > 0.0 ? std::sqrt(variance) : 0.0;
    }

    return {total_charge,
            charge_100,
            charge_500,
            first_time,
            last_time,
            charge_20_time,
            charge_50_time,
            weighted_mean,
            weighted_std};
}

std::array<double, kNumStats> compute_stats_single_sensor_impl(
    std::vector<double> times,
    std::vector<double> charges,
    const std::optional<double>& grouping_window_ns) {
    if (times.empty()) {
        return empty_stats();
    }

    if (!is_sorted(times)) {
        sort_by_time(times, charges);
    }

    if (grouping_window_ns.has_value() && grouping_window_ns.value() > 0.0) {
        auto grouped = group_hits_by_window(times, charges, grouping_window_ns.value());
        return compute_stats_from_sorted(grouped.first, grouped.second);
    }

    return compute_stats_from_sorted(times, charges);
}

py::array_t<double> to_array(const std::array<double, kNumStats>& stats) {
    py::array_t<double> result({kNumStats});
    auto buf = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < kNumStats; ++i) {
        buf(i) = stats[i];
    }
    return result;
}

py::array_t<double> compute_summary_stats_py(py::array_t<double, py::array::c_style | py::array::forcecast> times,
                                             py::array_t<double, py::array::c_style | py::array::forcecast> charges) {
    if (times.ndim() != 1 || charges.ndim() != 1) {
        throw std::invalid_argument("times and charges must be 1D arrays");
    }
    if (times.shape(0) != charges.shape(0)) {
        throw std::invalid_argument("times and charges must have the same length");
    }

    std::vector<double> times_vec(times.shape(0));
    std::vector<double> charges_vec(charges.shape(0));
    std::memcpy(times_vec.data(), times.data(), times.shape(0) * sizeof(double));
    std::memcpy(charges_vec.data(), charges.data(), charges.shape(0) * sizeof(double));

    std::array<double, kNumStats> stats;
    {
        // Heavy compute section; allow other Python threads to run.
        py::gil_scoped_release release;
        stats = compute_stats_single_sensor_impl(std::move(times_vec), std::move(charges_vec), std::nullopt);
    }
    return to_array(stats);
}

py::array_t<double> process_sensor_data_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> times,
    py::object charges_obj,
    std::optional<double> grouping_window_ns) {
    if (times.ndim() != 1) {
        throw std::invalid_argument("sensor_times must be a 1D array");
    }

    std::vector<double> times_vec(times.shape(0));
    std::memcpy(times_vec.data(), times.data(), times.shape(0) * sizeof(double));

    std::vector<double> charges_vec;
    if (charges_obj.is_none()) {
        charges_vec.assign(times_vec.size(), 1.0);
    } else {
        py::array_t<double, py::array::c_style | py::array::forcecast> charges = charges_obj.cast<py::array>();
        if (charges.ndim() != 1 || charges.shape(0) != times.shape(0)) {
            throw std::invalid_argument("sensor_charges must be 1D and match sensor_times length");
        }
        charges_vec.resize(charges.shape(0));
        std::memcpy(charges_vec.data(), charges.data(), charges.shape(0) * sizeof(double));
    }

    std::array<double, kNumStats> stats;
    {
        py::gil_scoped_release release;
        stats = compute_stats_single_sensor_impl(std::move(times_vec), std::move(charges_vec), grouping_window_ns);
    }
    return to_array(stats);
}

py::tuple process_event_arrays_py(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> string_ids,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> sensor_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> times,
    py::array_t<double, py::array::c_style | py::array::forcecast> pos_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> pos_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> pos_z,
    py::object charges_obj,
    std::optional<double> grouping_window_ns,
    std::optional<int> /*n_threads*/) {
    if (string_ids.ndim() != 1 || sensor_ids.ndim() != 1 || times.ndim() != 1 ||
        pos_x.ndim() != 1 || pos_y.ndim() != 1 || pos_z.ndim() != 1) {
        throw std::invalid_argument("All event arrays must be 1D");
    }

    const std::size_t n_hits = times.shape(0);
    const auto n_hits_ssize = static_cast<py::ssize_t>(n_hits);
    if (string_ids.shape(0) != n_hits_ssize || sensor_ids.shape(0) != n_hits_ssize ||
        pos_x.shape(0) != n_hits_ssize || pos_y.shape(0) != n_hits_ssize || pos_z.shape(0) != n_hits_ssize) {
        throw std::invalid_argument("All event arrays must have identical lengths");
    }

    // Snapshot input pointers before releasing the GIL.
    auto* string_ptr = string_ids.data();
    auto* sensor_ptr = sensor_ids.data();
    auto* times_ptr = times.data();
    auto* pos_x_ptr = pos_x.data();
    auto* pos_y_ptr = pos_y.data();
    auto* pos_z_ptr = pos_z.data();

    std::vector<double> charges_vec;
    if (charges_obj.is_none()) {
        charges_vec.assign(n_hits, 1.0);
    } else {
        py::array_t<double, py::array::c_style | py::array::forcecast> charges = charges_obj.cast<py::array>();
        if (charges.ndim() != 1 || charges.shape(0) != n_hits_ssize) {
            throw std::invalid_argument("charges must be 1D and match times length");
        }
        charges_vec.resize(n_hits);
        std::memcpy(charges_vec.data(), charges.data(), n_hits * sizeof(double));
    }

    // Data structures computed without holding the GIL
    std::vector<std::size_t> order(n_hits);
    std::vector<std::size_t> sensor_offsets;
    std::vector<std::array<double, 3>> sensor_positions_local;

    std::vector<std::array<double, kNumStats>> sensor_stats_local; // filled later

    {
        py::gil_scoped_release release;

        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            if (string_ptr[a] != string_ptr[b]) {
                return string_ptr[a] < string_ptr[b];
            }
            if (sensor_ptr[a] != sensor_ptr[b]) {
                return sensor_ptr[a] < sensor_ptr[b];
            }
            return times_ptr[a] < times_ptr[b];
        });

        if (n_hits == 0) {
            // Nothing to do, will return empty arrays below with GIL held
        } else {
            sensor_offsets.reserve(n_hits + 1);
            sensor_positions_local.reserve(n_hits);

            sensor_offsets.push_back(0);
            for (std::size_t i = 0; i < n_hits; ++i) {
                const auto idx = order[i];
                if (i == 0 || string_ptr[idx] != string_ptr[order[i - 1]] || sensor_ptr[idx] != sensor_ptr[order[i - 1]]) {
                    if (i != 0) sensor_offsets.push_back(i);
                    sensor_positions_local.push_back({pos_x_ptr[idx], pos_y_ptr[idx], pos_z_ptr[idx]});
                }
            }
            sensor_offsets.push_back(n_hits);

            const std::size_t n_sensors = sensor_positions_local.size();
            sensor_stats_local.resize(n_sensors);

            for (std::size_t s = 0; s < n_sensors; ++s) {
                const std::size_t start = sensor_offsets[s];
                const std::size_t end = sensor_offsets[s + 1];
                std::vector<double> times_slice;
                std::vector<double> charges_slice;
                times_slice.reserve(end - start);
                charges_slice.reserve(end - start);
                for (std::size_t i = start; i < end; ++i) {
                    const auto idx = order[i];
                    times_slice.push_back(times_ptr[idx]);
                    charges_slice.push_back(charges_vec[idx]);
                }
                sensor_stats_local[s] = compute_stats_single_sensor_impl(
                    std::move(times_slice),
                    std::move(charges_slice),
                    grouping_window_ns
                );
            }
        }
    }

    // With the GIL held, materialise the Python arrays for output
    if (n_hits == 0) {
        py::array_t<double> empty_positions(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(3)});
        py::array_t<double> empty_stats(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(kNumStats)});
        return py::make_tuple(empty_positions, empty_stats);
    }

    const std::size_t n_sensors = sensor_positions_local.size();

    py::array_t<double> positions(py::array::ShapeContainer{
        static_cast<py::ssize_t>(n_sensors),
        py::ssize_t(3)});
    py::array_t<double> stats(py::array::ShapeContainer{
        static_cast<py::ssize_t>(n_sensors),
        static_cast<py::ssize_t>(kNumStats)});

    auto positions_buf = positions.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n_sensors; ++i) {
        positions_buf(i, 0) = sensor_positions_local[i][0];
        positions_buf(i, 1) = sensor_positions_local[i][1];
        positions_buf(i, 2) = sensor_positions_local[i][2];
    }

    auto stats_buf = stats.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n_sensors; ++i) {
        for (std::size_t j = 0; j < kNumStats; ++j) {
            stats_buf(i, j) = sensor_stats_local[i][j];
        }
    }

    return py::make_tuple(positions, stats);
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "C++ backend for nt_summary_stats";

    m.def("compute_summary_stats",
          &compute_summary_stats_py,
          py::arg("times"),
          py::arg("charges"),
          "Compute summary statistics for a single sensor.");

    m.def("process_sensor_data",
          &process_sensor_data_py,
          py::arg("times"),
          py::arg("charges") = py::none(),
          py::arg("grouping_window_ns") = py::none(),
          "Process sensor data with an optional grouping window.");

    m.def("process_event_arrays",
          &process_event_arrays_py,
          py::arg("string_ids"),
          py::arg("sensor_ids"),
          py::arg("times"),
          py::arg("pos_x"),
          py::arg("pos_y"),
          py::arg("pos_z"),
          py::arg("charges") = py::none(),
          py::arg("grouping_window_ns") = py::none(),
          py::arg("n_threads") = py::none(),
          "Process full event arrays into positions and summary statistics.");
}
