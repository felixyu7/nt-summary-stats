#pragma once

#include <vector>
#include <array>

namespace nt_summary_stats {

/**
 * @brief Compute the 9 traditional summary statistics for neutrino telescope sensors
 * 
 * This function computes timing and charge statistics for a sequence of pulses
 * on a single sensor (optical module). The implementation is optimized for speed
 * using efficient C++ algorithms and data structures.
 * 
 * @param times Vector of pulse arrival times (in ns)
 * @param charges Vector of pulse charges (in arbitrary units)
 * @return Array containing the 9 summary statistics:
 *         [0] total_charge: Total charge collected by the sensor
 *         [1] charge_100ns: Charge within 100ns of first pulse
 *         [2] charge_500ns: Charge within 500ns of first pulse
 *         [3] first_pulse_time: Time of the first pulse
 *         [4] last_pulse_time: Time of the last pulse
 *         [5] charge_20_percent_time: Time at which 20% of charge is collected
 *         [6] charge_50_percent_time: Time at which 50% of charge is collected
 *         [7] charge_weighted_mean_time: Charge-weighted mean time
 *         [8] charge_weighted_std_time: Charge-weighted standard deviation time
 * 
 * @throws std::invalid_argument if times and charges have different lengths
 */
std::array<double, 9> compute_summary_stats(const std::vector<double>& times, 
                                           const std::vector<double>& charges);

/**
 * @brief Compute summary statistics for multiple sensors in batch
 * 
 * @param times_list Vector of time vectors, one per sensor
 * @param charges_list Vector of charge vectors, one per sensor
 * @return Vector of arrays containing summary statistics for each sensor
 * 
 * @throws std::invalid_argument if times_list and charges_list have different lengths
 */
std::vector<std::array<double, 9>> compute_summary_stats_batch(
    const std::vector<std::vector<double>>& times_list,
    const std::vector<std::vector<double>>& charges_list);

} // namespace nt_summary_stats