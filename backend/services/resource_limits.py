"""Apply the persisted CPU and PyTorch VRAM inference guard."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import platform

logger = logging.getLogger(__name__)


def allowed_cpu_count(total: int, percent: int) -> int:
    return max(1, min(total, math.ceil(total * percent / 100.0)))


def _apply_cpu_affinity(enabled: bool, percent: int) -> None:
    total = os.cpu_count() or 1
    allowed = allowed_cpu_count(total, percent) if enabled else total
    if platform.system() == "Windows":
        mask = (1 << allowed) - 1
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        set_affinity = kernel32.SetProcessAffinityMask
        set_affinity.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        set_affinity.restype = ctypes.c_int
        if not set_affinity(get_current_process(), ctypes.c_size_t(mask)):
            raise ctypes.WinError(ctypes.get_last_error())
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(range(allowed)))


def _apply_vram_fraction(enabled: bool, percent: int) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    fraction = percent / 100.0 if enabled else 1.0
    torch.cuda.set_per_process_memory_fraction(fraction)


def apply_resource_limits(enabled: bool, cpu_percent: int = 80, vram_percent: int = 80) -> None:
    """Apply best-effort inference limits without changing process priority."""
    try:
        _apply_cpu_affinity(enabled, cpu_percent)
    except Exception:
        logger.exception("Could not apply the CPU inference guard")
    try:
        _apply_vram_fraction(enabled, vram_percent)
    except Exception:
        logger.exception("Could not apply the PyTorch VRAM inference guard")


def apply_persisted_resource_limits() -> None:
    from ..database import session as database_session
    from .settings import get_resource_settings

    if database_session.SessionLocal is None:
        return
    with database_session.SessionLocal() as db:
        settings = get_resource_settings(db)
        apply_resource_limits(
            settings.limits_enabled,
            settings.cpu_percent,
            settings.vram_percent,
        )
