"""
NT Summary Stats - Fast neutrino telescope summary statistics computation.

This package provides efficient computation of the 9 traditional summary statistics
for neutrino telescope sensors (optical modules), as described in the IceCube paper
(https://arxiv.org/abs/2101.11589).

The 9 summary statistics are:
1. Total DOM charge
2. Charge within 100ns of first pulse
3. Charge within 500ns of first pulse
4. Time of first pulse
5. Time of last pulse
6. Time at which 20% of charge is collected
7. Time at which 50% of charge is collected
8. Charge weighted mean time
9. Charge weighted standard deviation time
"""

from __future__ import annotations

import numpy as np

from . import _backend
from .core import compute_summary_stats as _compute_summary_stats_numpy
from .event import process_event, process_sensor_data

native_available = _backend.native_available
using_native_backend = _backend.using_native_backend

__version__ = "1.0"

def compute_summary_stats(times, charges):
    """Compute summary statistics, preferring the native backend when available."""
    native = _backend.get_native_module()
    if native is not None:
        times_arr = np.ascontiguousarray(times, dtype=np.float64)
        charges_arr = np.ascontiguousarray(charges, dtype=np.float64)
        return native.compute_summary_stats(times_arr, charges_arr)
    return _compute_summary_stats_numpy(times, charges)


def compute_summary_stats_numpy(times, charges):
    """Explicitly use the NumPy implementation (useful for testing)."""
    return _compute_summary_stats_numpy(times, charges)


__all__ = [
    "__version__",
    "compute_summary_stats",
    "compute_summary_stats_numpy",
    "process_event",
    "process_sensor_data",
    "native_available",
    "using_native_backend",
]
