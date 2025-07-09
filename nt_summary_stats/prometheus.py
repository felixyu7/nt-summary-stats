"""
Prometheus event data processing functions.

This module provides functions for working with Prometheus event data,
including processing full events and extracting sensor data.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from .core import compute_summary_stats

try:
    import awkward as ak
    HAS_AWKWARD = True
except ImportError:
    HAS_AWKWARD = False


def process_sensor_data(sensor_times: Union[np.ndarray, list], 
                       sensor_charges: Optional[Union[np.ndarray, list]] = None,
                       grouping_window_ns: Optional[float] = None) -> np.ndarray:
    """
    Process sensor data with optional time-based grouping.
    
    This function processes timing data from a single sensor, optionally grouping
    hits within a time window before computing summary statistics.
    
    Args:
        sensor_times: Array of hit times for the sensor (in ns)
        sensor_charges: Array of hit charges. If None, assumes charge=1 for each hit
        grouping_window_ns: Time window for grouping hits (in ns). If None, no grouping is performed.
        
    Returns:
        np.ndarray containing the 9 summary statistics for the sensor (same order as compute_summary_stats)
        
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


def process_prometheus_event(event_data: Union[Dict, Any], 
                           grouping_window_ns: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a full Prometheus event to extract sensor positions and summary statistics.
    
    This function processes a Prometheus event (dictionary or Awkward array), groups photon hits by sensor,
    and computes summary statistics for each sensor that has hits.
    
    Args:
        event_data: Prometheus event data containing photon information. Can be:
                   - Dictionary with 'photons' key containing sensor data
                   - Awkward array with 'photons' field containing sensor data
                   - Direct photon data structure (dict or awkward array)
        grouping_window_ns: Time window for grouping hits within each sensor (in ns). If None, no grouping is performed.
        
    Returns:
        Tuple containing:
        - sensor_positions: np.ndarray of shape (N_sensors, 3) containing sensor positions
        - sensor_stats: np.ndarray of shape (N_sensors, 9) containing summary statistics
        
        The arrays are aligned: sensor_positions[i] corresponds to sensor_stats[i]
        
    Example:
        >>> # Dictionary format
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
        >>> positions, stats = process_prometheus_event(event)
        >>> 
        >>> # Awkward array format (if awkward is available)
        >>> import awkward as ak
        >>> event_ak = ak.from_iter([event])[0]  # Convert to awkward array
        >>> positions, stats = process_prometheus_event(event_ak)
        >>> print(len(positions))  # Number of unique sensors with hits
        2
    """
    # Handle different input formats
    photons = _extract_photons_data(event_data)
    
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
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 9), dtype=np.float64)
    
    # Group photons by sensor (string_id, sensor_id)
    sensor_keys = np.column_stack((string_ids, sensor_ids))
    unique_sensors, inverse_indices = np.unique(sensor_keys, axis=0, return_inverse=True)
    
    n_sensors = len(unique_sensors)
    
    # Pre-allocate results arrays - ensuring perfect alignment
    sensor_positions = np.empty((n_sensors, 3), dtype=np.float64)
    sensor_stats = np.empty((n_sensors, 9), dtype=np.float64)
    
    # Process each unique sensor to ensure alignment
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
        
        # Compute summary statistics for this sensor - same index ensures alignment
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


def _extract_photons_data(event_data: Union[Dict, Any]) -> Dict:
    """
    Extract photons data from different input formats.
    
    Args:
        event_data: Event data in various formats (dict, awkward array, etc.)
        
    Returns:
        Dictionary with photon data fields
        
    Raises:
        ValueError: If the input format is not supported or photon data cannot be extracted
    """
    # Case 1: Dictionary with 'photons' key
    if isinstance(event_data, dict) and 'photons' in event_data:
        return event_data['photons']
    
    # Case 2: Awkward array with 'photons' field
    if HAS_AWKWARD and hasattr(event_data, 'photons'):
        photons = event_data.photons
        # Convert awkward array fields to dict
        result = {}
        for field in ['sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 'string_id', 'sensor_id', 't']:
            if hasattr(photons, field):
                result[field] = ak.to_numpy(getattr(photons, field))
            else:
                raise ValueError(f"Required field '{field}' not found in photons data")
        
        # Handle optional charge field
        if hasattr(photons, 'charge'):
            result['charge'] = ak.to_numpy(photons.charge)
            
        return result
    
    # Case 3: Direct photons data (dict or awkward array with photon fields)
    if isinstance(event_data, dict):
        # Check if this is direct photon data
        required_fields = ['sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 'string_id', 'sensor_id', 't']
        if all(field in event_data for field in required_fields):
            return event_data
    
    # Case 4: Awkward array with direct photon fields
    if HAS_AWKWARD and hasattr(event_data, 'sensor_pos_x'):
        result = {}
        for field in ['sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 'string_id', 'sensor_id', 't']:
            if hasattr(event_data, field):
                result[field] = ak.to_numpy(getattr(event_data, field))
            else:
                raise ValueError(f"Required field '{field}' not found in event data")
        
        # Handle optional charge field
        if hasattr(event_data, 'charge'):
            result['charge'] = ak.to_numpy(event_data.charge)
            
        return result
    
    # If we get here, the format is not supported
    raise ValueError(
        "Unsupported event_data format. Expected:\n"
        "- Dictionary with 'photons' key\n"
        "- Awkward array with 'photons' field\n"
        "- Direct photon data (dict or awkward array) with required fields:\n"
        "  ['sensor_pos_x', 'sensor_pos_y', 'sensor_pos_z', 'string_id', 'sensor_id', 't']"
    )