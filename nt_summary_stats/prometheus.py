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
    
    # Optimized processing using np.split for better performance
    # Sort all data by sensor index for efficient splitting
    sort_order = np.argsort(inverse_indices)
    sorted_times = times[sort_order]
    sorted_charges = charges[sort_order]
    sorted_positions_x = sensor_pos_x[sort_order]
    sorted_positions_y = sensor_pos_y[sort_order]
    sorted_positions_z = sensor_pos_z[sort_order]
    sorted_indices = inverse_indices[sort_order]
    
    # Find split points for each sensor
    split_points = np.where(np.diff(sorted_indices) != 0)[0] + 1
    
    # Split the data into per-sensor arrays
    sensor_times_list = np.split(sorted_times, split_points)
    sensor_charges_list = np.split(sorted_charges, split_points)
    sensor_pos_x_list = np.split(sorted_positions_x, split_points)
    sensor_pos_y_list = np.split(sorted_positions_y, split_points)
    sensor_pos_z_list = np.split(sorted_positions_z, split_points)
    
    # Process each sensor efficiently
    for i in range(n_sensors):
        # Extract sensor position (use first element of each split)
        sensor_positions[i] = [
            sensor_pos_x_list[i][0],
            sensor_pos_y_list[i][0],
            sensor_pos_z_list[i][0]
        ]
        
        # Compute summary statistics for this sensor
        sensor_stats[i] = process_sensor_data(sensor_times_list[i], sensor_charges_list[i], grouping_window_ns)
    
    return sensor_positions, sensor_stats


def _group_hits_by_window(times: np.ndarray, charges: np.ndarray, 
                         window_ns: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress arrays by grouping values that lie within `window_ns` of the first element of each group.
    
    This matches the optimized grouping logic used in Mercury and the training dataloader.
    
    Parameters
    ----------
    times : array_like
        Input hit times (any order, any numeric dtype).
    charges : array_like  
        Input hit charges corresponding to times.
    window_ns : float
        Width of the inclusion window (inclusive on the right).
        
    Returns
    -------
    grouped_times : ndarray
        Representative values (the first element in each group).
    grouped_charges : ndarray
        Sum of charges in each group.
    """
    if len(times) == 0:
        return np.array([]), np.array([])
    
    # Sort both arrays by time
    sort_idx = np.argsort(times)
    sorted_times = times[sort_idx]
    sorted_charges = charges[sort_idx]
    
    # endpoints[i] == first index where value > sorted_times[i] + window_ns
    endpoints = np.searchsorted(sorted_times,
                               sorted_times + window_ns,
                               side='right')

    reps, charge_sums = [], []
    i = 0
    n = sorted_times.size
    while i < n:
        reps.append(sorted_times[i])
        j = endpoints[i]       # jump to the start of the next group
        charge_sums.append(np.sum(sorted_charges[i:j]))
        i = j

    return np.asarray(reps), np.asarray(charge_sums)


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