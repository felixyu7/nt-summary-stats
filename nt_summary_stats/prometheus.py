"""
Prometheus event data processing functions.

This module provides functions for working with Prometheus event data,
including processing full events and extracting sensor data.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from .core import compute_summary_stats


def process_sensor_data(sensor_times: Union[np.ndarray, list], 
                       sensor_charges: Optional[Union[np.ndarray, list]] = None,
                       grouping_window_ns: Optional[float] = None) -> Dict[str, float]:
    """
    Process sensor data with optional time-based grouping.
    
    This function processes timing data from a single sensor, optionally grouping
    hits within a time window before computing summary statistics.
    
    Args:
        sensor_times: Array of hit times for the sensor (in ns)
        sensor_charges: Array of hit charges. If None, assumes charge=1 for each hit
        grouping_window_ns: Time window for grouping hits (in ns). If None, no grouping is performed.
        
    Returns:
        Dictionary containing the 9 summary statistics for the sensor
        
    Example:
        >>> from nt_summary_stats import process_sensor_data
        >>> times = [10.0, 10.5, 15.0, 100.0]
        >>> charges = [1.0, 0.5, 2.0, 1.0]
        >>> # Default: no grouping
        >>> stats = process_sensor_data(times, charges)
        >>> # With grouping: group hits within 2ns windows
        >>> stats = process_sensor_data(times, charges, grouping_window_ns=2.0)
    """
    # Convert to numpy arrays
    sensor_times = np.asarray(sensor_times, dtype=np.float64)
    
    if sensor_charges is None:
        sensor_charges = np.ones_like(sensor_times, dtype=np.float64)
    else:
        sensor_charges = np.asarray(sensor_charges, dtype=np.float64)
    
    # Handle empty input
    if len(sensor_times) == 0:
        return compute_summary_stats([], [])
    
    # Group hits by time window if specified
    if grouping_window_ns is not None and grouping_window_ns > 0:
        grouped_times, grouped_charges = _group_hits_by_window(
            sensor_times, sensor_charges, grouping_window_ns
        )
    else:
        grouped_times = sensor_times
        grouped_charges = sensor_charges
    
    # Compute and return summary statistics
    return compute_summary_stats(grouped_times, grouped_charges)


def process_prometheus_event(event_data: Dict, 
                           grouping_window_ns: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Process a full Prometheus event to extract sensor positions and summary statistics.
    
    This function processes a Prometheus event dictionary, groups photon hits by sensor,
    and computes summary statistics for each sensor that has hits.
    
    Args:
        event_data: Prometheus event dictionary containing 'photons' key
        grouping_window_ns: Time window for grouping hits within each sensor (in ns). If None, no grouping is performed.
        
    Returns:
        Tuple containing:
        - sensor_positions: Array of sensor positions (N, 3) for sensors with hits
        - sensor_stats: List of summary statistics dictionaries, one per sensor
        
    Example:
        >>> event = {
        ...     'photons': {
        ...         'sensor_pos_x': [0.0, 0.0, 100.0],
        ...         'sensor_pos_y': [0.0, 0.0, 0.0], 
        ...         'sensor_pos_z': [0.0, 0.0, 50.0],
        ...         'string_id': [1, 1, 2],
        ...         'sensor_id': [1, 1, 1],
        ...         't': [10.0, 15.0, 20.0]
        ...     }
        ... }
        >>> # Default: no grouping
        >>> positions, stats = process_prometheus_event(event)
        >>> # With grouping: group hits within 2ns windows
        >>> positions, stats = process_prometheus_event(event, grouping_window_ns=2.0)
        >>> print(len(positions))  # Number of unique sensors with hits
        2
    """
    photons = event_data['photons']
    
    # Extract photon data as contiguous arrays
    sensor_pos_x = np.asarray(photons['sensor_pos_x'], dtype=np.float64)
    sensor_pos_y = np.asarray(photons['sensor_pos_y'], dtype=np.float64)
    sensor_pos_z = np.asarray(photons['sensor_pos_z'], dtype=np.float64)
    string_ids = np.asarray(photons['string_id'], dtype=np.int32)
    sensor_ids = np.asarray(photons['sensor_id'], dtype=np.int32)
    times = np.asarray(photons['t'], dtype=np.float64)
    
    # Handle charges if present, otherwise assume unit charge
    if 'charge' in photons:
        charges = np.asarray(photons['charge'], dtype=np.float64)
    else:
        charges = np.ones_like(times, dtype=np.float64)
    
    # Handle empty event
    if len(times) == 0:
        return np.empty((0, 3), dtype=np.float64), []
    
    # Group photons by sensor (string_id, sensor_id)
    sensor_keys = np.column_stack((string_ids, sensor_ids))
    unique_sensors, inverse_indices = np.unique(sensor_keys, axis=0, return_inverse=True)
    
    n_sensors = len(unique_sensors)
    
    # Pre-allocate results arrays
    sensor_positions = np.empty((n_sensors, 3), dtype=np.float64)
    sensor_stats = [None] * n_sensors
    
    # Process each unique sensor
    for sensor_idx in range(n_sensors):
        # Get mask for this sensor
        mask = inverse_indices == sensor_idx
        
        # Extract data for this sensor
        sensor_times = times[mask]
        sensor_charges = charges[mask]
        
        # Get sensor position (use first occurrence) - optimized
        first_hit_idx = np.argmax(mask)  # Faster than np.where for first occurrence
        sensor_positions[sensor_idx] = [
            sensor_pos_x[first_hit_idx],
            sensor_pos_y[first_hit_idx], 
            sensor_pos_z[first_hit_idx]
        ]
        
        # Compute summary statistics for this sensor
        sensor_stats[sensor_idx] = process_sensor_data(sensor_times, sensor_charges, grouping_window_ns)
    
    return sensor_positions, sensor_stats


def _group_hits_by_window(times: np.ndarray, charges: np.ndarray, 
                         window_ns: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group hits within a time window, summing charges for grouped hits.
    
    This is a simplified version of the grouping logic used in the Mercury model.
    Hits within the specified time window are grouped together, with their charges
    summed and the earliest time used as the group time.
    
    Args:
        times: Array of hit times (must be sorted)
        charges: Array of hit charges
        window_ns: Time window for grouping (in ns)
        
    Returns:
        Tuple of (grouped_times, grouped_charges)
    """
    if len(times) == 0:
        return np.array([]), np.array([])
    
    # Check if already sorted to avoid unnecessary sorting
    if len(times) > 1 and np.all(times[:-1] <= times[1:]):
        times_sorted = times
        charges_sorted = charges
    else:
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        charges_sorted = charges[sort_idx]
    
    if len(times_sorted) == 1:
        return times_sorted.copy(), charges_sorted.copy()
    
    # Vectorized approach for better performance
    time_diffs = np.diff(times_sorted)
    group_boundaries = np.concatenate(([True], time_diffs > window_ns))
    group_ids = np.cumsum(group_boundaries)
    
    # Use bincount for efficient grouping
    n_groups = group_ids[-1]
    grouped_charges = np.bincount(group_ids - 1, weights=charges_sorted)
    
    # Get the first time for each group
    grouped_times = np.empty(n_groups, dtype=np.float64)
    for i in range(n_groups):
        grouped_times[i] = times_sorted[group_ids == i + 1][0]
    
    return grouped_times, grouped_charges