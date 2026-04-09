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
constexpr std::size_t kNumStatsExtended = 25;

template<std::size_t N>
std::array<double, N> empty_stats() {
    std::array<double, N> stats{};
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

template<bool Extended>
auto compute_stats_from_sorted(
    const std::vector<double>& times,
    const std::vector<double>& charges) {
    constexpr std::size_t NumStats = Extended ? kNumStatsExtended : kNumStats;
    const std::size_t n = times.size();

    if (n == 0) {
        return empty_stats<NumStats>();
    }

    const double time = times.front();
    const double charge = charges.front();

    if (n == 1) {
        if constexpr (Extended) {
            return std::array<double, kNumStatsExtended>{
                charge, charge, charge, time, time, time, time, time, 0.0,
                time, time, time, time, time, time,
                charge, charge, charge, charge, charge, charge,
                1.0,    // n_pulses
                1.0,    // q_max_frac (single pulse: max == total)
                0.0,    // n_string_neighbors (filled by process_event)
                0.0     // t_skewness (undefined for n < 3)
            };
        } else {
            return std::array<double, kNumStats>{
                charge, charge, charge, time, time, time, time, time, 0.0
            };
        }
    }

    const double first_time = times.front();
    const double last_time = times.back();

    // First pass: totals, weighted moments, and fixed-window charges.
    double total_charge = 0.0;
    double sum_qt = 0.0;
    double sum_qt2 = 0.0;

    // Standard time windows
    const double cutoff_100 = first_time + 100.0;
    const double cutoff_500 = first_time + 500.0;
    double charge_100 = 0.0;
    double charge_500 = 0.0;

    // Extended time windows
    double charge_10 = 0.0, charge_20 = 0.0, charge_50 = 0.0;
    double charge_200 = 0.0, charge_1000 = 0.0, charge_2000 = 0.0;

    // Extended accumulators: max charge and third raw moment
    double max_charge = 0.0;
    double sum_qt3 = 0.0;

    if constexpr (Extended) {
        const double cutoff_10 = first_time + 10.0;
        const double cutoff_20 = first_time + 20.0;
        const double cutoff_50 = first_time + 50.0;
        const double cutoff_200 = first_time + 200.0;
        const double cutoff_1000 = first_time + 1000.0;
        const double cutoff_2000 = first_time + 2000.0;

        for (std::size_t i = 0; i < n; ++i) {
            const double t = times[i];
            const double q = charges[i];
            total_charge += q;
            sum_qt += q * t;
            sum_qt2 += q * t * t;
            sum_qt3 += q * t * t * t;
            if (q > max_charge) max_charge = q;
            if (t <= cutoff_10) charge_10 += q;
            if (t <= cutoff_20) charge_20 += q;
            if (t <= cutoff_50) charge_50 += q;
            if (t <= cutoff_100) charge_100 += q;
            if (t <= cutoff_200) charge_200 += q;
            if (t <= cutoff_500) charge_500 += q;
            if (t <= cutoff_1000) charge_1000 += q;
            if (t <= cutoff_2000) charge_2000 += q;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const double t = times[i];
            const double q = charges[i];
            total_charge += q;
            sum_qt += q * t;
            sum_qt2 += q * t * t;
            if (t <= cutoff_100) charge_100 += q;
            if (t <= cutoff_500) charge_500 += q;
        }
    }

    // Second pass: times at which charge percentiles are reached
    double charge_20_time = first_time;
    double charge_50_time = first_time;
    double charge_5_time = first_time, charge_10_time = first_time;
    double charge_25_time = first_time, charge_75_time = first_time;
    double charge_90_time = first_time, charge_95_time = first_time;

    if constexpr (Extended) {
        const double threshold_5 = total_charge * 0.05;
        const double threshold_10 = total_charge * 0.1;
        const double threshold_20 = total_charge * 0.2;
        const double threshold_25 = total_charge * 0.25;
        const double threshold_50 = total_charge * 0.5;
        const double threshold_75 = total_charge * 0.75;
        const double threshold_90 = total_charge * 0.9;
        const double threshold_95 = total_charge * 0.95;

        double running = 0.0;
        bool have_5 = false, have_10 = false, have_20 = false, have_25 = false;
        bool have_50 = false, have_75 = false, have_90 = false, have_95 = false;

        for (std::size_t i = 0; i < n; ++i) {
            running += charges[i];
            if (!have_5 && running > threshold_5) { charge_5_time = times[i]; have_5 = true; }
            if (!have_10 && running > threshold_10) { charge_10_time = times[i]; have_10 = true; }
            if (!have_20 && running > threshold_20) { charge_20_time = times[i]; have_20 = true; }
            if (!have_25 && running > threshold_25) { charge_25_time = times[i]; have_25 = true; }
            if (!have_50 && running > threshold_50) { charge_50_time = times[i]; have_50 = true; }
            if (!have_75 && running > threshold_75) { charge_75_time = times[i]; have_75 = true; }
            if (!have_90 && running > threshold_90) { charge_90_time = times[i]; have_90 = true; }
            if (!have_95 && running > threshold_95) { charge_95_time = times[i]; have_95 = true; break; }
        }
    } else {
        const double threshold_20 = total_charge * 0.2;
        const double threshold_50 = total_charge * 0.5;
        double running = 0.0;
        bool have_20 = false;
        bool have_50 = false;

        for (std::size_t i = 0; i < n; ++i) {
            running += charges[i];
            if (!have_20 && running > threshold_20) {
                charge_20_time = times[i];
                have_20 = true;
            }
            if (!have_50 && running > threshold_50) {
                charge_50_time = times[i];
                have_50 = true;
                if (have_20) break;
            }
        }
    }

    double weighted_mean = 0.0;
    double weighted_std = 0.0;
    if (total_charge > 0.0) {
        weighted_mean = sum_qt / total_charge;
        const double variance = (sum_qt2 / total_charge) - (weighted_mean * weighted_mean);
        weighted_std = variance > 0.0 ? std::sqrt(variance) : 0.0;
    }

    // Skewness from raw moments (extended only)
    double t_skewness = 0.0;
    if constexpr (Extended) {
        if (n >= 3 && weighted_std > 0.0 && total_charge > 0.0) {
            const double mu = weighted_mean;
            const double e_x2 = sum_qt2 / total_charge;
            const double e_x3 = sum_qt3 / total_charge;
            const double sigma3 = weighted_std * weighted_std * weighted_std;
            t_skewness = (e_x3 - 3.0 * mu * e_x2 + 2.0 * mu * mu * mu) / sigma3;
        }
    }

    if constexpr (Extended) {
        return std::array<double, kNumStatsExtended>{
            total_charge, charge_100, charge_500,
            first_time, last_time,
            charge_20_time, charge_50_time,
            weighted_mean, weighted_std,
            charge_5_time, charge_10_time, charge_25_time,
            charge_75_time, charge_90_time, charge_95_time,
            charge_10, charge_20, charge_50,
            charge_200, charge_1000, charge_2000,
            static_cast<double>(n),                                    // n_pulses
            (total_charge > 0.0 ? max_charge / total_charge : 0.0),   // q_max_frac
            0.0,                                                       // n_string_neighbors
            t_skewness                                                 // t_skewness
        };
    } else {
        return std::array<double, kNumStats>{
            total_charge, charge_100, charge_500,
            first_time, last_time,
            charge_20_time, charge_50_time,
            weighted_mean, weighted_std
        };
    }
}

template<bool Extended>
auto compute_stats_single_sensor_impl(
    std::vector<double> times,
    std::vector<double> charges,
    const std::optional<double>& grouping_window_ns) {
    constexpr std::size_t NumStats = Extended ? kNumStatsExtended : kNumStats;

    if (times.empty()) {
        return empty_stats<NumStats>();
    }

    if (!is_sorted(times)) {
        sort_by_time(times, charges);
    }

    if (grouping_window_ns.has_value() && grouping_window_ns.value() > 0.0) {
        auto grouped = group_hits_by_window(times, charges, grouping_window_ns.value());
        return compute_stats_from_sorted<Extended>(grouped.first, grouped.second);
    }

    return compute_stats_from_sorted<Extended>(times, charges);
}

template<std::size_t N>
py::array_t<double> to_array(const std::array<double, N>& stats) {
    py::array_t<double> result({static_cast<py::ssize_t>(N)});
    auto buf = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < N; ++i) {
        buf(i) = stats[i];
    }
    return result;
}

py::array_t<double> compute_summary_stats_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> times,
    py::array_t<double, py::array::c_style | py::array::forcecast> charges,
    bool extended) {
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

    py::array_t<double> result;
    {
        // Heavy compute section; allow other Python threads to run.
        py::gil_scoped_release release;
        if (extended) {
            auto stats = compute_stats_single_sensor_impl<true>(std::move(times_vec), std::move(charges_vec), std::nullopt);
            py::gil_scoped_acquire acquire;
            result = to_array(stats);
        } else {
            auto stats = compute_stats_single_sensor_impl<false>(std::move(times_vec), std::move(charges_vec), std::nullopt);
            py::gil_scoped_acquire acquire;
            result = to_array(stats);
        }
    }
    return result;
}

py::array_t<double> process_sensor_data_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> times,
    py::object charges_obj,
    std::optional<double> grouping_window_ns,
    bool extended) {
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

    const auto pre_grouping_n = static_cast<double>(times_vec.size());

    py::array_t<double> result;
    {
        py::gil_scoped_release release;
        if (extended) {
            auto stats = compute_stats_single_sensor_impl<true>(std::move(times_vec), std::move(charges_vec), grouping_window_ns);
            py::gil_scoped_acquire acquire;
            result = to_array(stats);
        } else {
            auto stats = compute_stats_single_sensor_impl<false>(std::move(times_vec), std::move(charges_vec), grouping_window_ns);
            py::gil_scoped_acquire acquire;
            result = to_array(stats);
        }
    }

    // Override n_pulses with pre-grouping count when grouping is applied
    if (extended && grouping_window_ns.has_value() && grouping_window_ns.value() > 0.0) {
        auto buf = result.mutable_unchecked<1>();
        buf(21) = pre_grouping_n;
    }

    return result;
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
    std::optional<int> /*n_threads*/,
    bool extended) {
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
    const std::size_t num_stats = extended ? kNumStatsExtended : kNumStats;
    std::vector<std::size_t> order(n_hits);
    std::vector<std::size_t> sensor_offsets;
    std::vector<std::array<double, 3>> sensor_positions_local;

    std::vector<double> sensor_stats_flat; // flattened stats array

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

            std::vector<int32_t> sensor_string_ids;
            std::vector<int32_t> sensor_sensor_ids;

            sensor_offsets.push_back(0);
            for (std::size_t i = 0; i < n_hits; ++i) {
                const auto idx = order[i];
                if (i == 0 || string_ptr[idx] != string_ptr[order[i - 1]] || sensor_ptr[idx] != sensor_ptr[order[i - 1]]) {
                    if (i != 0) sensor_offsets.push_back(i);
                    sensor_positions_local.push_back({pos_x_ptr[idx], pos_y_ptr[idx], pos_z_ptr[idx]});
                    sensor_string_ids.push_back(string_ptr[idx]);
                    sensor_sensor_ids.push_back(sensor_ptr[idx]);
                }
            }
            sensor_offsets.push_back(n_hits);

            const std::size_t n_sensors = sensor_positions_local.size();
            sensor_stats_flat.reserve(n_sensors * num_stats);

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

                if (extended) {
                    auto stats = compute_stats_single_sensor_impl<true>(
                        std::move(times_slice),
                        std::move(charges_slice),
                        grouping_window_ns
                    );
                    sensor_stats_flat.insert(sensor_stats_flat.end(), stats.begin(), stats.end());
                } else {
                    auto stats = compute_stats_single_sensor_impl<false>(
                        std::move(times_slice),
                        std::move(charges_slice),
                        grouping_window_ns
                    );
                    sensor_stats_flat.insert(sensor_stats_flat.end(), stats.begin(), stats.end());
                }
            }

            // Extended-only post-processing: n_string_neighbors and n_pulses override
            if (extended) {
                // HLC-style neighbor count: same string, +-2 sensor_id, +-1000ns coincidence
                for (std::size_t s = 0; s < n_sensors; ++s) {
                    int count = 0;
                    const int32_t my_str = sensor_string_ids[s];
                    const int32_t my_sid = sensor_sensor_ids[s];
                    const std::size_t scan_start = (s >= 4) ? s - 4 : 0;
                    const std::size_t scan_end = std::min(n_sensors, s + 5);
                    for (std::size_t other = scan_start; other < scan_end; ++other) {
                        if (other == s) continue;
                        if (sensor_string_ids[other] != my_str) continue;
                        const int32_t sid_diff = std::abs(sensor_sensor_ids[other] - my_sid);
                        if (sid_diff < 1 || sid_diff > 2) continue;
                        // Merge-scan for +-1000ns coincidence on pre-grouping times
                        // times_ptr[order[i]] gives original times, time-sorted within each sensor
                        std::size_t ai = sensor_offsets[s], bi = sensor_offsets[other];
                        const std::size_t a_end = sensor_offsets[s + 1];
                        const std::size_t b_end = sensor_offsets[other + 1];
                        bool coincident = false;
                        while (ai < a_end && bi < b_end) {
                            const double ta = times_ptr[order[ai]];
                            const double tb = times_ptr[order[bi]];
                            if (std::abs(ta - tb) <= 1000.0) { coincident = true; break; }
                            if (ta < tb) ++ai; else ++bi;
                        }
                        if (coincident) ++count;
                    }
                    sensor_stats_flat[s * num_stats + 23] = static_cast<double>(count);
                }

                // Override n_pulses with pre-grouping count when grouping is applied
                if (grouping_window_ns.has_value() && grouping_window_ns.value() > 0.0) {
                    for (std::size_t s = 0; s < n_sensors; ++s) {
                        sensor_stats_flat[s * num_stats + 21] =
                            static_cast<double>(sensor_offsets[s + 1] - sensor_offsets[s]);
                    }
                }
            }
        }
    }

    // With the GIL held, materialise the Python arrays for output
    if (n_hits == 0) {
        py::array_t<double> empty_positions(py::array::ShapeContainer{py::ssize_t(0), py::ssize_t(3)});
        py::array_t<double> empty_stats(py::array::ShapeContainer{py::ssize_t(0), static_cast<py::ssize_t>(num_stats)});
        return py::make_tuple(empty_positions, empty_stats);
    }

    const std::size_t n_sensors = sensor_positions_local.size();

    py::array_t<double> positions(py::array::ShapeContainer{
        static_cast<py::ssize_t>(n_sensors),
        py::ssize_t(3)});
    py::array_t<double> stats(py::array::ShapeContainer{
        static_cast<py::ssize_t>(n_sensors),
        static_cast<py::ssize_t>(num_stats)});

    auto positions_buf = positions.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n_sensors; ++i) {
        positions_buf(i, 0) = sensor_positions_local[i][0];
        positions_buf(i, 1) = sensor_positions_local[i][1];
        positions_buf(i, 2) = sensor_positions_local[i][2];
    }

    auto stats_buf = stats.mutable_unchecked<2>();
    for (std::size_t i = 0; i < n_sensors; ++i) {
        for (std::size_t j = 0; j < num_stats; ++j) {
            stats_buf(i, j) = sensor_stats_flat[i * num_stats + j];
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
          py::arg("extended") = false,
          "Compute summary statistics for a single sensor.");

    m.def("process_sensor_data",
          &process_sensor_data_py,
          py::arg("times"),
          py::arg("charges") = py::none(),
          py::arg("grouping_window_ns") = py::none(),
          py::arg("extended") = false,
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
          py::arg("extended") = false,
          "Process full event arrays into positions and summary statistics.");
}
