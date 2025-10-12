"""
Event data processing utilities.

Provides helpers for collapsing per-hit detector event data into per-sensor
summary statistics using either the native extension or the NumPy fallback.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from . import _backend
from .core import compute_summary_stats as _compute_summary_stats_numpy


def process_sensor_data(
    sensor_times: Union[np.ndarray, list],
    sensor_charges: Optional[Union[np.ndarray, list]] = None,
    grouping_window_ns: Optional[float] = None,
) -> np.ndarray:
    """
    Process sensor data with optional time-based grouping.
    
    This function processes timing data from a single sensor, optionally grouping
    hits within a time window before computing summary statistics. When the
    compiled extension is available it is used automatically; otherwise, the
    NumPy implementation is used.
    """
    native = _backend.get_native_module()
    if native is not None:
        times_arr = np.ascontiguousarray(sensor_times, dtype=np.float64)
        charges_arr = (None if sensor_charges is None 
                       else np.ascontiguousarray(sensor_charges, dtype=np.float64))
        return native.process_sensor_data(times_arr, charges_arr, grouping_window_ns)
    
    return _process_sensor_data_numpy(sensor_times, sensor_charges, grouping_window_ns)


def _process_sensor_data_numpy(
    sensor_times: Union[np.ndarray, list],
    sensor_charges: Optional[Union[np.ndarray, list]] = None,
    grouping_window_ns: Optional[float] = None,
) -> np.ndarray:
    sensor_times = np.asarray(sensor_times, dtype=np.float64)
    
    if sensor_charges is None:
        sensor_charges = np.ones_like(sensor_times, dtype=np.float64)
    else:
        sensor_charges = np.asarray(sensor_charges, dtype=np.float64)
    
    if len(sensor_times) == 0:
        return _compute_summary_stats_numpy([], [])
    
    if grouping_window_ns is not None and grouping_window_ns > 0:
        grouped_times, grouped_charges = _group_hits_by_window(
            sensor_times, sensor_charges, grouping_window_ns
        )
    else:
        grouped_times = sensor_times
        grouped_charges = sensor_charges
    
    return _compute_summary_stats_numpy(grouped_times, grouped_charges)


def process_event(
    event_data: Dict[str, Any],
    grouping_window_ns: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a detector event to extract sensor positions and summary statistics.

    The input must be a mapping that either contains a ``"photons"`` dictionary
    with the required fields or directly exposes the photon-level fields
    ``sensor_pos_x``, ``sensor_pos_y``, ``sensor_pos_z``, ``string_id``,
    ``sensor_id``, ``t``, and optionally ``charge``. When the native extension is
    available it is used automatically; otherwise the NumPy implementation is invoked.
    """
    photons = _extract_photons_data(event_data)
    
    sensor_pos_x = np.ascontiguousarray(photons['sensor_pos_x'], dtype=np.float64)
    sensor_pos_y = np.ascontiguousarray(photons['sensor_pos_y'], dtype=np.float64)
    sensor_pos_z = np.ascontiguousarray(photons['sensor_pos_z'], dtype=np.float64)
    string_ids = np.ascontiguousarray(photons['string_id'], dtype=np.int32)
    sensor_ids = np.ascontiguousarray(photons['sensor_id'], dtype=np.int32)
    times = np.ascontiguousarray(photons['t'], dtype=np.float64)
    charges = photons.get('charge')
    charges_arr = None if charges is None else np.ascontiguousarray(charges, dtype=np.float64)
    
    if len(times) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 9), dtype=np.float64)
    
    native = _backend.get_native_module()
    if native is not None:
        return native.process_event_arrays(
            string_ids,
            sensor_ids,
            times,
            sensor_pos_x,
            sensor_pos_y,
            sensor_pos_z,
            charges=charges_arr,
            grouping_window_ns=grouping_window_ns,
            n_threads=None,
        )
    
    return _process_event_arrays_numpy(
        sensor_pos_x,
        sensor_pos_y,
        sensor_pos_z,
        string_ids,
        sensor_ids,
        times,
        charges_arr,
        grouping_window_ns,
    )


def _process_event_arrays_numpy(sensor_pos_x: np.ndarray,
                                sensor_pos_y: np.ndarray,
                                sensor_pos_z: np.ndarray,
                                string_ids: np.ndarray,
                                sensor_ids: np.ndarray,
                                times: np.ndarray,
                                charges: Optional[np.ndarray],
                                grouping_window_ns: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if charges is None:
        charges = np.ones_like(times, dtype=np.float64)
    else:
        charges = np.asarray(charges, dtype=np.float64)
    
    sensor_keys = np.column_stack((string_ids, sensor_ids))
    unique_sensors, inverse_indices = np.unique(sensor_keys, axis=0, return_inverse=True)
    
    n_sensors = len(unique_sensors)
    
    sensor_positions = np.empty((n_sensors, 3), dtype=np.float64)
    sensor_stats = np.empty((n_sensors, 9), dtype=np.float64)
    
    sort_order = np.argsort(inverse_indices)
    sorted_times = times[sort_order]
    sorted_charges = charges[sort_order]
    sorted_positions_x = sensor_pos_x[sort_order]
    sorted_positions_y = sensor_pos_y[sort_order]
    sorted_positions_z = sensor_pos_z[sort_order]
    sorted_indices = inverse_indices[sort_order]
    
    split_points = np.where(np.diff(sorted_indices) != 0)[0] + 1
    
    sensor_times_list = np.split(sorted_times, split_points)
    sensor_charges_list = np.split(sorted_charges, split_points)
    sensor_pos_x_list = np.split(sorted_positions_x, split_points)
    sensor_pos_y_list = np.split(sorted_positions_y, split_points)
    sensor_pos_z_list = np.split(sorted_positions_z, split_points)
    
    for i in range(n_sensors):
        times_slice = sensor_times_list[i]
        min_idx = np.argmin(times_slice)
        sensor_positions[i] = [
            sensor_pos_x_list[i][min_idx],
            sensor_pos_y_list[i][min_idx],
            sensor_pos_z_list[i][min_idx]
        ]
        sensor_stats[i] = _process_sensor_data_numpy(
            sensor_times_list[i],
            sensor_charges_list[i],
            grouping_window_ns,
        )
    
    return sensor_positions, sensor_stats


def _group_hits_by_window(hit_times, hit_charges, time_window, return_counts=False):
    """
    Group hits into fixed time windows, returning the first actual hit time
    in each non-empty window and the sum of charges in that window.

    Parameters
    ----------
    hit_times : array-like, shape (N,)
        Hit times in nanoseconds.
    hit_charges : array-like, shape (N,)
        Charge per hit (e.g., photoelectrons). Must align with hit_times.
    time_window : float
        Window size in nanoseconds (> 0).
    return_counts : bool, optional (default: False)
        If True, also return the number of hits per window.

    Returns
    -------
    grouped_times : np.ndarray, shape (M,)
        First actual hit time in each non-empty window (ascending by window).
    window_charges : np.ndarray, shape (M,)
        Sum of hit_charges within each window.
    hit_counts : np.ndarray, shape (M,), optional
        Number of hits in each window (only if return_counts=True).
    """
    ht = np.asarray(hit_times)
    hc = np.asarray(hit_charges)

    if ht.size == 0:
        if return_counts:
            return ht[:0], ht[:0].astype(float), ht[:0]
        else:
            return ht[:0], ht[:0].astype(float)

    if ht.shape != hc.shape:
        raise ValueError("hit_times and hit_charges must have the same shape.")
    if time_window <= 0:
        raise ValueError("time_window must be positive.")

    # Stable sort by time (stable ensures the first time in each bin is preserved if equal times occur).
    order = np.argsort(ht, kind="mergesort")
    st = ht[order]
    sc = hc[order]

    # Compute monotone bin labels with numerically robust arithmetic.
    if np.issubdtype(st.dtype, np.integer) and float(time_window).is_integer():
        tw = np.int64(time_window)
        bins = (st - st[0]) // tw
    else:
        # Cast to float64 and shift by st[0] for better precision at boundaries.
        bins = np.floor((st - st[0]).astype(np.float64) / float(time_window)).astype(np.int64)

    # Run-length encode the (sorted, hence monotone) bin labels.
    changes = np.empty(bins.size, dtype=bool)
    changes[0] = True
    np.not_equal(bins[1:], bins[:-1], out=changes[1:])
    starts = np.flatnonzero(changes)  # start index of each bin-run

    # First hit time per non-empty bin:
    grouped_times = st[starts]

    # Aggregate charges per bin efficiently:
    window_charges = np.add.reduceat(sc, starts)

    if return_counts:
        hit_counts = np.diff(np.r_[starts, st.size])
        return grouped_times, window_charges, hit_counts
    else:
        return grouped_times, window_charges


def _extract_photons_data(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve the photon-level dictionary from an event mapping.

    The function accepts either an outer dictionary containing a ``"photons"``
    key or a dictionary that already holds the required photon fields.
    """
    if not isinstance(event_data, dict):
        raise TypeError("event_data must be a dictionary")

    if "photons" in event_data:
        photons = event_data["photons"]
        if not isinstance(photons, dict):
            raise TypeError("event_data['photons'] must be a dictionary")
    else:
        photons = event_data

    required_fields = [
        "sensor_pos_x",
        "sensor_pos_y",
        "sensor_pos_z",
        "string_id",
        "sensor_id",
        "t",
    ]
    missing = [field for field in required_fields if field not in photons]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required photon fields: {missing_str}")

    return photons
