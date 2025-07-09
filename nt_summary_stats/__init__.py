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

# Try to import the optimized C++ implementation first
try:
    from ._cpp_core import compute_summary_stats as _cpp_compute_summary_stats
    from ._cpp_core import compute_summary_stats_batch as _cpp_compute_summary_stats_batch
    _HAS_CPP_EXTENSION = True
except ImportError:
    _HAS_CPP_EXTENSION = False

# Always import the Python fallback
from .core import compute_summary_stats as _py_compute_summary_stats
from .core import compute_summary_stats_batch as _py_compute_summary_stats_batch
from .prometheus import process_prometheus_event, process_sensor_data

# Choose the best available implementation
if _HAS_CPP_EXTENSION:
    # Use C++ implementation for better performance
    compute_summary_stats = _cpp_compute_summary_stats
    compute_summary_stats_batch = _cpp_compute_summary_stats_batch
else:
    # Fallback to Python implementation
    compute_summary_stats = _py_compute_summary_stats
    compute_summary_stats_batch = _py_compute_summary_stats_batch

__version__ = "0.1.0"
__all__ = ["compute_summary_stats", "compute_summary_stats_batch", "process_prometheus_event", "process_sensor_data"]

# Expose information about which implementation is being used
def get_implementation_info():
    """
    Get information about which implementation is currently being used.
    
    Returns:
        dict: Information about the current implementation
    """
    return {
        "has_cpp_extension": _HAS_CPP_EXTENSION,
        "using_cpp": _HAS_CPP_EXTENSION,
        "version": __version__,
    }