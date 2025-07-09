#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "core.hpp"

namespace py = pybind11;

namespace nt_summary_stats {

// Helper function to convert numpy array to std::vector
std::vector<double> numpy_to_vector(py::array_t<double> input) {
    auto buf = input.request();
    double* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

// Helper function to convert std::array to numpy array
py::array_t<double> array_to_numpy(const std::array<double, 9>& arr) {
    return py::array_t<double>(
        arr.size(),
        arr.data(),
        py::cast(arr) // This ensures the array lifetime is managed
    );
}

// Python wrapper for compute_summary_stats
py::array_t<double> py_compute_summary_stats(py::array_t<double> times, py::array_t<double> charges) {
    auto times_vec = numpy_to_vector(times);
    auto charges_vec = numpy_to_vector(charges);
    
    auto result = compute_summary_stats(times_vec, charges_vec);
    
    // Create numpy array from result
    auto result_array = py::array_t<double>(9);
    auto buf = result_array.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < 9; ++i) {
        ptr[i] = result[i];
    }
    
    return result_array;
}

// Python wrapper for compute_summary_stats_batch
py::array_t<double> py_compute_summary_stats_batch(py::list times_list, py::list charges_list) {
    std::vector<std::vector<double>> cpp_times_list;
    std::vector<std::vector<double>> cpp_charges_list;
    
    // Convert Python lists to C++ vectors
    for (auto item : times_list) {
        py::array_t<double> arr = py::cast<py::array_t<double>>(item);
        cpp_times_list.push_back(numpy_to_vector(arr));
    }
    
    for (auto item : charges_list) {
        py::array_t<double> arr = py::cast<py::array_t<double>>(item);
        cpp_charges_list.push_back(numpy_to_vector(arr));
    }
    
    auto results = compute_summary_stats_batch(cpp_times_list, cpp_charges_list);
    
    // Convert results back to numpy array
    size_t n_sensors = results.size();
    
    // Create 2D numpy array with shape (n_sensors, 9)
    std::vector<size_t> shape = {n_sensors, 9};
    auto result_array = py::array_t<double>(shape);
    auto buf = result_array.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    for (size_t i = 0; i < n_sensors; ++i) {
        for (size_t j = 0; j < 9; ++j) {
            ptr[i * 9 + j] = results[i][j];
        }
    }
    
    return result_array;
}

} // namespace nt_summary_stats

PYBIND11_MODULE(_cpp_core, m) {
    m.doc() = "C++ implementation of neutrino telescope summary statistics";
    
    m.def("compute_summary_stats", &nt_summary_stats::py_compute_summary_stats,
          "Compute the 9 traditional summary statistics for neutrino telescope sensors",
          py::arg("times"), py::arg("charges"));
    
    m.def("compute_summary_stats_batch", &nt_summary_stats::py_compute_summary_stats_batch,
          "Compute summary statistics for multiple sensors in batch",
          py::arg("times_list"), py::arg("charges_list"));
    
    // Add version info
    m.attr("__version__") = "0.1.0";
}