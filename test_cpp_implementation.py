#!/usr/bin/env python3
"""
Test script for C++ implementation of nt_summary_stats.

This script tests that the C++ implementation produces identical results
to the Python implementation for various test cases.
"""

import numpy as np
import sys
import time

# Import both implementations
import nt_summary_stats
from nt_summary_stats.core import compute_summary_stats as py_compute_summary_stats
from nt_summary_stats.core import compute_summary_stats_batch as py_compute_summary_stats_batch

def test_basic_functionality():
    """Test basic functionality with known results."""
    print("Testing basic functionality...")
    
    times = np.array([10.0, 15.0, 25.0, 100.0])
    charges = np.array([1.0, 2.0, 1.5, 0.5])
    
    cpp_result = nt_summary_stats.compute_summary_stats(times, charges)
    py_result = py_compute_summary_stats(times, charges)
    
    assert np.allclose(cpp_result, py_result), f"Results don't match: C++ {cpp_result}, Python {py_result}"
    print("✓ Basic functionality test passed")

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    test_cases = [
        # Empty arrays
        ([], []),
        # Single element
        ([10.0], [1.0]),
        # Two elements
        ([10.0, 20.0], [1.0, 2.0]),
        # Already sorted
        ([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]),
        # Unsorted
        ([30.0, 10.0, 20.0], [3.0, 1.0, 2.0]),
        # Zero charges
        ([10.0, 20.0, 30.0], [0.0, 0.0, 0.0]),
        # Mixed charges
        ([10.0, 20.0, 30.0], [1.0, 0.0, 2.0]),
        # Large values
        ([1e6, 1e6+100, 1e6+200], [1.0, 2.0, 3.0]),
    ]
    
    for i, (times, charges) in enumerate(test_cases):
        times_arr = np.array(times, dtype=np.float64)
        charges_arr = np.array(charges, dtype=np.float64)
        
        cpp_result = nt_summary_stats.compute_summary_stats(times_arr, charges_arr)
        py_result = py_compute_summary_stats(times_arr, charges_arr)
        
        assert np.allclose(cpp_result, py_result), f"Test case {i+1} failed: C++ {cpp_result}, Python {py_result}"
    
    print("✓ All edge case tests passed")

def test_batch_processing():
    """Test batch processing functionality."""
    print("Testing batch processing...")
    
    # Create test data for multiple sensors
    times_list = [
        [10.0, 15.0, 25.0],
        [5.0, 10.0, 15.0, 20.0],
        [100.0, 200.0],
        []  # Empty sensor
    ]
    
    charges_list = [
        [1.0, 2.0, 1.5],
        [0.5, 1.0, 1.5, 2.0],
        [1.0, 1.0],
        []  # Empty sensor
    ]
    
    cpp_result = nt_summary_stats.compute_summary_stats_batch(times_list, charges_list)
    py_result = py_compute_summary_stats_batch(times_list, charges_list)
    
    assert np.allclose(cpp_result, py_result), f"Batch results don't match: C++ {cpp_result}, Python {py_result}"
    print("✓ Batch processing test passed")

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    # Test mismatched lengths
    try:
        nt_summary_stats.compute_summary_stats([1.0, 2.0], [1.0])
        assert False, "Should have raised exception for mismatched lengths"
    except ValueError:
        pass  # Expected
    
    print("✓ Error handling test passed")

def performance_benchmark():
    """Compare performance between C++ and Python implementations."""
    print("Running performance benchmark...")
    
    # Generate larger test data
    np.random.seed(42)
    n_pulses = 10000
    times = np.sort(np.random.uniform(0, 1000, n_pulses))
    charges = np.random.exponential(1.0, n_pulses)
    
    # Benchmark Python implementation
    start_time = time.time()
    for _ in range(100):
        py_result = py_compute_summary_stats(times, charges)
    python_time = time.time() - start_time
    
    # Benchmark C++ implementation
    start_time = time.time()
    for _ in range(100):
        cpp_result = nt_summary_stats.compute_summary_stats(times, charges)
    cpp_time = time.time() - start_time
    
    # Verify results are the same
    assert np.allclose(cpp_result, py_result), "Performance test results don't match"
    
    speedup = python_time / cpp_time
    print(f"✓ Performance benchmark completed")
    print(f"  Python time: {python_time:.4f}s")
    print(f"  C++ time: {cpp_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")

def main():
    """Run all tests."""
    print("NT Summary Stats C++ Implementation Test Suite")
    print("=" * 50)
    
    # Check implementation info
    info = nt_summary_stats.get_implementation_info()
    print(f"Implementation info: {info}")
    
    if not info['has_cpp_extension']:
        print("❌ C++ extension not available, tests will fail")
        return 1
    
    try:
        test_basic_functionality()
        test_edge_cases()
        test_batch_processing()
        test_error_handling()
        performance_benchmark()
        
        print("\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())