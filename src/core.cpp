#include "core.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace nt_summary_stats {

namespace {
    // Helper function to get empty statistics array
    std::array<double, 9> empty_stats() {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
}

std::array<double, 9> compute_summary_stats(const std::vector<double>& times, 
                                           const std::vector<double>& charges) {
    // Handle empty input
    if (times.empty()) {
        return empty_stats();
    }
    
    // Validate input lengths match
    if (times.size() != charges.size()) {
        throw std::invalid_argument("times and charges must have the same length, got " + 
                                  std::to_string(times.size()) + " and " + 
                                  std::to_string(charges.size()));
    }
    
    // Create indices for sorting
    std::vector<size_t> indices(times.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Check if already sorted to avoid unnecessary sorting
    bool is_sorted = std::is_sorted(times.begin(), times.end());
    
    std::vector<double> times_sorted, charges_sorted;
    
    if (is_sorted) {
        times_sorted = times;
        charges_sorted = charges;
    } else {
        // Sort indices by time
        std::sort(indices.begin(), indices.end(), 
                  [&times](size_t i1, size_t i2) { return times[i1] < times[i2]; });
        
        // Create sorted arrays
        times_sorted.reserve(times.size());
        charges_sorted.reserve(charges.size());
        
        for (size_t idx : indices) {
            times_sorted.push_back(times[idx]);
            charges_sorted.push_back(charges[idx]);
        }
    }
    
    // Precompute values used multiple times
    const double total_charge = std::accumulate(charges_sorted.begin(), charges_sorted.end(), 0.0);
    const double first_pulse_time = times_sorted[0];
    const double last_pulse_time = times_sorted.back();
    
    // Optimized time window calculations using binary search
    const double time_100ns_cutoff = first_pulse_time + 100.0;
    const double time_500ns_cutoff = first_pulse_time + 500.0;
    
    // Use upper_bound for efficient binary search (equivalent to numpy searchsorted with side='right')
    auto it_100ns = std::upper_bound(times_sorted.begin(), times_sorted.end(), time_100ns_cutoff);
    auto it_500ns = std::upper_bound(times_sorted.begin(), times_sorted.end(), time_500ns_cutoff);
    
    size_t idx_100ns = std::distance(times_sorted.begin(), it_100ns);
    size_t idx_500ns = std::distance(times_sorted.begin(), it_500ns);
    
    // Calculate charge sums within time windows
    double charge_100ns = 0.0;
    double charge_500ns = 0.0;
    
    for (size_t i = 0; i < idx_100ns; ++i) {
        charge_100ns += charges_sorted[i];
    }
    
    for (size_t i = 0; i < idx_500ns; ++i) {
        charge_500ns += charges_sorted[i];
    }
    
    // Efficient percentile calculations using cumulative sum
    std::vector<double> cumulative_charge(charges_sorted.size());
    std::partial_sum(charges_sorted.begin(), charges_sorted.end(), cumulative_charge.begin());
    
    const double charge_20_percent = 0.2 * total_charge;
    const double charge_50_percent = 0.5 * total_charge;
    
    // Use upper_bound for percentile searches
    auto it_20 = std::upper_bound(cumulative_charge.begin(), cumulative_charge.end(), charge_20_percent);
    auto it_50 = std::upper_bound(cumulative_charge.begin(), cumulative_charge.end(), charge_50_percent);
    
    size_t idx_20 = std::distance(cumulative_charge.begin(), it_20);
    size_t idx_50 = std::distance(cumulative_charge.begin(), it_50);
    
    // Ensure indices are within bounds
    size_t n_times = times_sorted.size();
    double charge_20_percent_time = times_sorted[std::min(idx_20, n_times - 1)];
    double charge_50_percent_time = times_sorted[std::min(idx_50, n_times - 1)];
    
    // Vectorized weighted statistics
    double charge_weighted_mean_time = 0.0;
    double charge_weighted_std_time = 0.0;
    
    if (total_charge > 0) {
        // Calculate weighted mean
        double weighted_sum = 0.0;
        for (size_t i = 0; i < times_sorted.size(); ++i) {
            weighted_sum += times_sorted[i] * charges_sorted[i];
        }
        charge_weighted_mean_time = weighted_sum / total_charge;
        
        // Calculate weighted variance
        double weighted_variance = 0.0;
        for (size_t i = 0; i < times_sorted.size(); ++i) {
            double time_diff = times_sorted[i] - charge_weighted_mean_time;
            weighted_variance += charges_sorted[i] * time_diff * time_diff;
        }
        weighted_variance /= total_charge;
        charge_weighted_std_time = std::sqrt(weighted_variance);
    }
    
    return {
        total_charge,
        charge_100ns,
        charge_500ns,
        first_pulse_time,
        last_pulse_time,
        charge_20_percent_time,
        charge_50_percent_time,
        charge_weighted_mean_time,
        charge_weighted_std_time
    };
}

std::vector<std::array<double, 9>> compute_summary_stats_batch(
    const std::vector<std::vector<double>>& times_list,
    const std::vector<std::vector<double>>& charges_list) {
    
    if (times_list.size() != charges_list.size()) {
        throw std::invalid_argument("times_list and charges_list must have the same length");
    }
    
    std::vector<std::array<double, 9>> results;
    results.reserve(times_list.size());
    
    for (size_t i = 0; i < times_list.size(); ++i) {
        results.push_back(compute_summary_stats(times_list[i], charges_list[i]));
    }
    
    return results;
}

} // namespace nt_summary_stats