"""
Lightweight loader for the optional native extension.

The environment variables below control behaviour:

NTSS_DISABLE_NATIVE=1  -> always use the NumPy implementation.
NTSS_FORCE_NATIVE=1    -> raise at import time if the native module is unavailable.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Optional

_DISABLE_NATIVE = os.environ.get("NTSS_DISABLE_NATIVE", "").lower() in {"1", "true", "yes", "on"}
_FORCE_NATIVE = os.environ.get("NTSS_FORCE_NATIVE", "").lower() in {"1", "true", "yes", "on"}

_NATIVE_INITIALISED = False
_NATIVE_MODULE: Optional[ModuleType] = None


def get_native_module() -> Optional[ModuleType]:
    """Attempt to load and cache the native extension module."""
    global _NATIVE_INITIALISED, _NATIVE_MODULE
    if not _NATIVE_INITIALISED:
        _NATIVE_INITIALISED = True
        if _DISABLE_NATIVE:
            _NATIVE_MODULE = None
        else:
            module = None
            errors: list[Exception] = []
            for name in ("nt_summary_stats._native", "nt_summary_stats_native", "nt_summary_stats_cpp_native"):
                try:
                    module = importlib.import_module(name)
                    if name != "nt_summary_stats._native":
                        sys.modules.setdefault("nt_summary_stats._native", module)
                    break
                except Exception as exc:
                    errors.append(exc)
            if module is None:
                if _FORCE_NATIVE and errors:
                    raise errors[-1]
                _NATIVE_MODULE = None
            else:
                _NATIVE_MODULE = module
    return _NATIVE_MODULE


def native_available() -> bool:
    """Return True if the native extension loaded successfully."""
    return get_native_module() is not None


def using_native_backend() -> bool:
    """Alias for native_available for backwards compatibility."""
    return native_available()


def disable_native() -> bool:
    """Return True if native execution has been explicitly disabled."""
    return _DISABLE_NATIVE


def force_native() -> bool:
    """Return True if native execution is mandatory."""
    return _FORCE_NATIVE


__all__ = [
    "get_native_module",
    "native_available",
    "using_native_backend",
    "disable_native",
    "force_native",
]
