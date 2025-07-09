from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import numpy

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "nt_summary_stats._cpp_core",
        sources=[
            "src/core.cpp",
            "src/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "src",
        ],
        language="c++",
        cxx_std=11,
        # Optimization flags
        extra_compile_args=[
            "-O3",
            "-ffast-math",
            "-march=native",
            "-mtune=native",
        ] if "linux" in __import__("sys").platform else ["-O3"],
        extra_link_args=[],
    ),
]

setup(
    name="nt_summary_stats",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)