"""
NT Summary Stats - Fast neutrino telescope summary statistics computation.

This package provides efficient computation of summary statistics for neutrino
telescope sensors (optical modules), as described in the IceCube paper
(https://arxiv.org/abs/2101.11589).

The 9 traditional summary statistics are:
1. Total DOM charge
2. Charge within 100ns of first pulse
3. Charge within 500ns of first pulse
4. Time of first pulse
5. Time of last pulse
6. Time at which 20% of charge is collected
7. Time at which 50% of charge is collected
8. Charge weighted mean time
9. Charge weighted standard deviation time

When extended=True, 16 additional statistics are computed (25 total):
10. Time at which 5% of charge is collected
11. Time at which 10% of charge is collected
12. Time at which 25% of charge is collected
13. Time at which 75% of charge is collected
14. Time at which 90% of charge is collected
15. Time at which 95% of charge is collected
16. Charge within 10ns of first pulse
17. Charge within 20ns of first pulse
18. Charge within 50ns of first pulse
19. Charge within 200ns of first pulse
20. Charge within 1000ns of first pulse
21. Charge within 2000ns of first pulse
22. Number of pulses (pre-grouping count)
23. Peak charge fraction (max pulse charge / total charge)
24. Same-string neighbor hit count (HLC-style, +-2 sensor IDs, +-1000ns; 0 outside process_event)
25. Charge-weighted time skewness (3rd standardized moment; 0 for < 3 pulses)
"""

from __future__ import annotations

import numpy as np

from . import _backend
from .core import compute_summary_stats as _compute_summary_stats_numpy
from .event import process_event, process_sensor_data

native_available = _backend.native_available
using_native_backend = _backend.using_native_backend

__version__ = "1.0"

def compute_summary_stats(times, charges, extended=False):
    """
    Compute summary statistics, preferring the native backend when available.

    Args:
        times: Array of pulse arrival times (in ns)
        charges: Array of pulse charges
        extended: If True, compute 25 statistics. If False (default), compute 9.

    Returns:
        np.ndarray of shape (9,) or (25,) containing summary statistics
    """
    native = _backend.get_native_module()
    if native is not None:
        times_arr = np.ascontiguousarray(times, dtype=np.float64)
        charges_arr = np.ascontiguousarray(charges, dtype=np.float64)
        return native.compute_summary_stats(times_arr, charges_arr, extended)
    return _compute_summary_stats_numpy(times, charges, extended)


def compute_summary_stats_numpy(times, charges, extended=False):
    """
    Explicitly use the NumPy implementation (useful for testing).

    Args:
        times: Array of pulse arrival times (in ns)
        charges: Array of pulse charges
        extended: If True, compute 25 statistics. If False (default), compute 9.

    Returns:
        np.ndarray of shape (9,) or (25,) containing summary statistics
    """
    return _compute_summary_stats_numpy(times, charges, extended)


__all__ = [
    "__version__",
    "compute_summary_stats",
    "compute_summary_stats_numpy",
    "process_event",
    "process_sensor_data",
    "native_available",
    "using_native_backend",
]
