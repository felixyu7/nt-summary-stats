"""
Core summary statistics computation functions.

This module provides the core functionality for computing the 9 traditional
summary statistics for neutrino telescope sensors.
"""

import numpy as np
from typing import Dict, Union


def compute_summary_stats(times: Union[np.ndarray, list],
                         charges: Union[np.ndarray, list],
                         extended: bool = False) -> np.ndarray:
    """
    Compute summary statistics for neutrino telescope sensors.

    This function computes timing and charge statistics for a sequence of pulses
    on a single sensor (optical module). The implementation is optimized for speed
    using vectorized operations.

    Args:
        times: Array of pulse arrival times (in ns)
        charges: Array of pulse charges (in arbitrary units)
        extended: If True, compute 25 statistics (9 standard + 16 extended).
                  If False (default), compute 9 traditional statistics.

    Returns:
        np.ndarray containing summary statistics:

        When extended=False (9 stats):
        [0] total_charge: Total charge collected by the sensor
        [1] charge_100ns: Charge within 100ns of first pulse
        [2] charge_500ns: Charge within 500ns of first pulse
        [3] first_pulse_time: Time of the first pulse
        [4] last_pulse_time: Time of the last pulse
        [5] charge_20_percent_time: Time at which 20% of charge is collected
        [6] charge_50_percent_time: Time at which 50% of charge is collected
        [7] charge_weighted_mean_time: Charge-weighted mean of pulse times
        [8] charge_weighted_std_time: Charge-weighted standard deviation of pulse times

        When extended=True (25 stats, includes above plus):
        [9] charge_5_percent_time: Time at which 5% of charge is collected
        [10] charge_10_percent_time: Time at which 10% of charge is collected
        [11] charge_25_percent_time: Time at which 25% of charge is collected
        [12] charge_75_percent_time: Time at which 75% of charge is collected
        [13] charge_90_percent_time: Time at which 90% of charge is collected
        [14] charge_95_percent_time: Time at which 95% of charge is collected
        [15] charge_10ns: Charge within 10ns of first pulse
        [16] charge_20ns: Charge within 20ns of first pulse
        [17] charge_50ns: Charge within 50ns of first pulse
        [18] charge_200ns: Charge within 200ns of first pulse
        [19] charge_1000ns: Charge within 1000ns of first pulse
        [20] charge_2000ns: Charge within 2000ns of first pulse
        [21] n_pulses: Number of input pulses (pre-grouping count)
        [22] q_max_frac: Peak charge fraction (max pulse charge / total charge)
        [23] n_string_neighbors: HLC-style neighbor count (0 unless via process_event)
        [24] t_skewness: Charge-weighted time skewness (0 for < 3 pulses)

    Example:
        >>> import numpy as np
        >>> from nt_summary_stats import compute_summary_stats
        >>> times = np.array([10.0, 15.0, 25.0, 100.0])
        >>> charges = np.array([1.0, 2.0, 1.5, 0.5])
        >>> stats = compute_summary_stats(times, charges)
        >>> print(stats[0])  # total_charge
        5.0
        >>> extended_stats = compute_summary_stats(times, charges, extended=True)
        >>> print(extended_stats.shape)  # (25,)
        (25,)
    """
    # Convert to numpy arrays for consistent handling
    times = np.asarray(times, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    
    # Handle empty input
    if len(times) == 0:
        return _empty_stats(extended)

    # Validate input lengths match
    if len(times) != len(charges):
        raise ValueError(f"times and charges must have the same length, got {len(times)} and {len(charges)}")

    n_times = len(times)

    # Fast path for single pulse
    if n_times == 1:
        return _single_pulse_stats(times[0], charges[0], extended)
    
    # Optimized sorting check using early termination
    is_sorted = True
    for i in range(1, min(n_times, 100)):  # Check first 100 elements for early termination
        if times[i-1] > times[i]:
            is_sorted = False
            break
    
    if is_sorted and n_times > 100:
        # Check remaining elements if first 100 were sorted
        is_sorted = np.all(times[99:-1] <= times[100:])
    
    if is_sorted:
        times_sorted = times
        charges_sorted = charges
    else:
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        charges_sorted = charges[sort_idx]
    
    # Precompute values used multiple times
    total_charge = np.sum(charges_sorted)
    first_pulse_time = times_sorted[0]
    last_pulse_time = times_sorted[-1]
    
    # Combined time window calculations using single searchsorted call
    if extended:
        time_cutoffs = np.array([first_pulse_time + 10.0, first_pulse_time + 20.0,
                                 first_pulse_time + 50.0, first_pulse_time + 100.0,
                                 first_pulse_time + 200.0, first_pulse_time + 500.0,
                                 first_pulse_time + 1000.0, first_pulse_time + 2000.0])
    else:
        time_cutoffs = np.array([first_pulse_time + 100.0, first_pulse_time + 500.0])

    time_indices = np.searchsorted(times_sorted, time_cutoffs, side='right')

    # Optimized charge calculations - reuse cumulative sum
    cumulative_charge = np.cumsum(charges_sorted)

    if extended:
        idx_10ns, idx_20ns, idx_50ns, idx_100ns, idx_200ns, idx_500ns, idx_1000ns, idx_2000ns = time_indices
        charge_10ns = cumulative_charge[idx_10ns - 1] if idx_10ns > 0 else 0.0
        charge_20ns = cumulative_charge[idx_20ns - 1] if idx_20ns > 0 else 0.0
        charge_50ns = cumulative_charge[idx_50ns - 1] if idx_50ns > 0 else 0.0
        charge_100ns = cumulative_charge[idx_100ns - 1] if idx_100ns > 0 else 0.0
        charge_200ns = cumulative_charge[idx_200ns - 1] if idx_200ns > 0 else 0.0
        charge_500ns = cumulative_charge[idx_500ns - 1] if idx_500ns > 0 else 0.0
        charge_1000ns = cumulative_charge[idx_1000ns - 1] if idx_1000ns > 0 else 0.0
        charge_2000ns = cumulative_charge[idx_2000ns - 1] if idx_2000ns > 0 else 0.0
    else:
        idx_100ns, idx_500ns = time_indices[0], time_indices[1]
        charge_100ns = cumulative_charge[idx_100ns - 1] if idx_100ns > 0 else 0.0
        charge_500ns = cumulative_charge[idx_500ns - 1] if idx_500ns > 0 else 0.0
    
    # Efficient percentile calculations using existing cumulative sum
    if extended:
        charge_thresholds = np.array([0.05 * total_charge, 0.1 * total_charge,
                                      0.2 * total_charge, 0.25 * total_charge,
                                      0.5 * total_charge, 0.75 * total_charge,
                                      0.9 * total_charge, 0.95 * total_charge])
    else:
        charge_thresholds = np.array([0.2 * total_charge, 0.5 * total_charge])

    percentile_indices = np.searchsorted(cumulative_charge, charge_thresholds, side='right')

    # Ensure indices are within bounds and get times
    if extended:
        idx_5, idx_10, idx_20, idx_25, idx_50, idx_75, idx_90, idx_95 = [min(i, n_times - 1) for i in percentile_indices]
        charge_5_percent_time = times_sorted[idx_5]
        charge_10_percent_time = times_sorted[idx_10]
        charge_20_percent_time = times_sorted[idx_20]
        charge_25_percent_time = times_sorted[idx_25]
        charge_50_percent_time = times_sorted[idx_50]
        charge_75_percent_time = times_sorted[idx_75]
        charge_90_percent_time = times_sorted[idx_90]
        charge_95_percent_time = times_sorted[idx_95]
    else:
        idx_20 = min(percentile_indices[0], n_times - 1)
        idx_50 = min(percentile_indices[1], n_times - 1)
        charge_20_percent_time = times_sorted[idx_20]
        charge_50_percent_time = times_sorted[idx_50]
    
    # Optimized weighted statistics
    if total_charge > 0:
        charge_weighted_mean_time = np.dot(times_sorted, charges_sorted) / total_charge
        # More efficient variance calculation avoiding intermediate array
        charge_weighted_var = (np.dot(charges_sorted, times_sorted * times_sorted) / total_charge - 
                             charge_weighted_mean_time * charge_weighted_mean_time)
        charge_weighted_std_time = np.sqrt(max(0.0, charge_weighted_var))  # Ensure non-negative
    else:
        charge_weighted_mean_time = 0.0
        charge_weighted_std_time = 0.0
    
    # Extended-only: n_pulses, q_max_frac, t_skewness
    if extended:
        n_pulses = float(n_times)
        q_max_frac = float(np.max(charges_sorted) / total_charge) if total_charge > 0 else 0.0
        t_skewness = 0.0
        if n_times >= 3 and charge_weighted_std_time > 0.0 and total_charge > 0.0:
            deviations = times_sorted - charge_weighted_mean_time
            t_skewness = float(np.dot(charges_sorted, deviations ** 3) /
                               (total_charge * charge_weighted_std_time ** 3))

    if extended:
        return np.array([
            # Standard 9 stats
            total_charge,
            charge_100ns,
            charge_500ns,
            first_pulse_time,
            last_pulse_time,
            charge_20_percent_time,
            charge_50_percent_time,
            charge_weighted_mean_time,
            charge_weighted_std_time,
            # Extended charge percentile times
            charge_5_percent_time,
            charge_10_percent_time,
            charge_25_percent_time,
            charge_75_percent_time,
            charge_90_percent_time,
            charge_95_percent_time,
            # Extended time window charges
            charge_10ns,
            charge_20ns,
            charge_50ns,
            charge_200ns,
            charge_1000ns,
            charge_2000ns,
            # New extended stats
            n_pulses,
            q_max_frac,
            0.0,              # n_string_neighbors (filled by process_event)
            t_skewness,
        ], dtype=np.float64)
    else:
        return np.array([
            total_charge,
            charge_100ns,
            charge_500ns,
            first_pulse_time,
            last_pulse_time,
            charge_20_percent_time,
            charge_50_percent_time,
            charge_weighted_mean_time,
            charge_weighted_std_time
        ], dtype=np.float64)


def _empty_stats(extended: bool = False) -> np.ndarray:
    """Return empty statistics array for zero-length inputs."""
    return np.zeros(25 if extended else 9, dtype=np.float64)


def _single_pulse_stats(time: float, charge: float, extended: bool = False) -> np.ndarray:
    """Fast path for single pulse statistics computation."""
    if extended:
        return np.array([
            # Standard 9 stats
            charge,           # total_charge
            charge,           # charge_100ns (all charge is within 100ns)
            charge,           # charge_500ns (all charge is within 500ns)
            time,             # first_pulse_time
            time,             # last_pulse_time
            time,             # charge_20_percent_time (single pulse)
            time,             # charge_50_percent_time (single pulse)
            time,             # charge_weighted_mean_time (single pulse)
            0.0,              # charge_weighted_std_time (no variance with single pulse)
            # Extended charge percentile times (all at same time for single pulse)
            time,             # charge_5_percent_time
            time,             # charge_10_percent_time
            time,             # charge_25_percent_time
            time,             # charge_75_percent_time
            time,             # charge_90_percent_time
            time,             # charge_95_percent_time
            # Extended time window charges (all charge in all windows for single pulse)
            charge,           # charge_10ns
            charge,           # charge_20ns
            charge,           # charge_50ns
            charge,           # charge_200ns
            charge,           # charge_1000ns
            charge,           # charge_2000ns
            1.0,              # n_pulses
            1.0,              # q_max_frac (single pulse: max == total)
            0.0,              # n_string_neighbors (filled by process_event)
            0.0,              # t_skewness (undefined for n < 3)
        ], dtype=np.float64)
    else:
        return np.array([
            charge,           # total_charge
            charge,           # charge_100ns (all charge is within 100ns)
            charge,           # charge_500ns (all charge is within 500ns)
            time,             # first_pulse_time
            time,             # last_pulse_time
            time,             # charge_20_percent_time (single pulse)
            time,             # charge_50_percent_time (single pulse)
            time,             # charge_weighted_mean_time (single pulse)
            0.0               # charge_weighted_std_time (no variance with single pulse)
        ], dtype=np.float64)


def compute_summary_stats_batch(times_list: list, charges_list: list, extended: bool = False) -> np.ndarray:
    """
    Compute summary statistics for multiple sensors in batch.

    Args:
        times_list: List of time arrays, one per sensor
        charges_list: List of charge arrays, one per sensor
        extended: If True, compute 25 statistics per sensor. If False, compute 9.

    Returns:
        np.ndarray of shape (N_sensors, 9) or (N_sensors, 25) containing summary statistics for each sensor
    """
    if len(times_list) != len(charges_list):
        raise ValueError("times_list and charges_list must have the same length")

    n_sensors = len(times_list)
    n_stats = 25 if extended else 9
    # Pre-allocate results array for better performance
    results = np.empty((n_sensors, n_stats), dtype=np.float64)

    # Efficient batch processing
    for i, (times, charges) in enumerate(zip(times_list, charges_list)):
        results[i] = compute_summary_stats(times, charges, extended=extended)

    return results