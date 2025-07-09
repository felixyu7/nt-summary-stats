"""
Core summary statistics computation functions.

This module provides the core functionality for computing the 9 traditional
summary statistics for neutrino telescope sensors.
"""

import numpy as np
from typing import Dict, Union


def compute_summary_stats(times: Union[np.ndarray, list], 
                         charges: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the 9 traditional summary statistics for neutrino telescope sensors.
    
    This function computes timing and charge statistics for a sequence of pulses
    on a single sensor (optical module). The implementation is optimized for speed
    using vectorized operations.
    
    Args:
        times: Array of pulse arrival times (in ns)
        charges: Array of pulse charges (in arbitrary units)
        
    Returns:
        np.ndarray containing the 9 summary statistics in the following order:
        [0] total_charge: Total charge collected by the sensor
        [1] charge_100ns: Charge within 100ns of first pulse
        [2] charge_500ns: Charge within 500ns of first pulse
        [3] first_pulse_time: Time of the first pulse
        [4] last_pulse_time: Time of the last pulse
        [5] charge_20_percent_time: Time at which 20% of charge is collected
        [6] charge_50_percent_time: Time at which 50% of charge is collected
        [7] charge_weighted_mean_time: Charge-weighted mean of pulse times
        [8] charge_weighted_std_time: Charge-weighted standard deviation of pulse times
        
    Example:
        >>> import numpy as np
        >>> from nt_summary_stats import compute_summary_stats
        >>> times = np.array([10.0, 15.0, 25.0, 100.0])
        >>> charges = np.array([1.0, 2.0, 1.5, 0.5])
        >>> stats = compute_summary_stats(times, charges)
        >>> print(stats[0])  # total_charge
        5.0
    """
    # Convert to numpy arrays for consistent handling
    times = np.asarray(times, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    
    # Handle empty input
    if len(times) == 0:
        return _empty_stats()
    
    # Validate input lengths match
    if len(times) != len(charges):
        raise ValueError(f"times and charges must have the same length, got {len(times)} and {len(charges)}")
    
    # Check if already sorted to avoid unnecessary sorting
    if len(times) > 1 and np.all(times[:-1] <= times[1:]):
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
    
    # Optimized time window calculations using searchsorted
    time_100ns_cutoff = first_pulse_time + 100.0
    time_500ns_cutoff = first_pulse_time + 500.0
    
    idx_100ns = np.searchsorted(times_sorted, time_100ns_cutoff, side='right')
    idx_500ns = np.searchsorted(times_sorted, time_500ns_cutoff, side='right')
    
    charge_100ns = np.sum(charges_sorted[:idx_100ns])
    charge_500ns = np.sum(charges_sorted[:idx_500ns])
    
    # Efficient percentile calculations using cumulative sum
    cumulative_charge = np.cumsum(charges_sorted)
    charge_20_percent = 0.2 * total_charge
    charge_50_percent = 0.5 * total_charge
    
    idx_20 = np.searchsorted(cumulative_charge, charge_20_percent, side='right')
    idx_50 = np.searchsorted(cumulative_charge, charge_50_percent, side='right')
    
    # Ensure indices are within bounds
    n_times = len(times_sorted)
    charge_20_percent_time = times_sorted[min(idx_20, n_times - 1)]
    charge_50_percent_time = times_sorted[min(idx_50, n_times - 1)]
    
    # Vectorized weighted statistics
    if total_charge > 0:
        charge_weighted_mean_time = np.dot(times_sorted, charges_sorted) / total_charge
        # Use more efficient variance calculation
        time_diff = times_sorted - charge_weighted_mean_time
        charge_weighted_var = np.dot(charges_sorted, time_diff * time_diff) / total_charge
        charge_weighted_std_time = np.sqrt(charge_weighted_var)
    else:
        charge_weighted_mean_time = 0.0
        charge_weighted_std_time = 0.0
    
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


def _empty_stats() -> np.ndarray:
    """Return empty statistics array for zero-length inputs."""
    return np.zeros(9, dtype=np.float64)


def compute_summary_stats_batch(times_list: list, charges_list: list) -> np.ndarray:
    """
    Compute summary statistics for multiple sensors in batch.
    
    Args:
        times_list: List of time arrays, one per sensor
        charges_list: List of charge arrays, one per sensor
        
    Returns:
        np.ndarray of shape (N_sensors, 9) containing summary statistics for each sensor
    """
    if len(times_list) != len(charges_list):
        raise ValueError("times_list and charges_list must have the same length")
    
    n_sensors = len(times_list)
    # Pre-allocate results array for better performance
    results = np.empty((n_sensors, 9), dtype=np.float64)
    
    # Efficient batch processing
    for i, (times, charges) in enumerate(zip(times_list, charges_list)):
        results[i] = compute_summary_stats(times, charges)
    
    return results