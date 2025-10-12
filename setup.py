from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

ext_modules = [
    Pybind11Extension(
        "nt_summary_stats._native",
        ["src/nt_summary_stats.cpp"],
        extra_compile_args=["-O3"],
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
