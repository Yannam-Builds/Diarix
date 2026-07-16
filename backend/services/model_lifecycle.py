"""Warm-model lifecycle for low-latency local dictation.

The selected STT model starts loading as soon as recording begins, remains
resident for nearby dictations, and is released after the configured idle
period.  A single operation lock prevents model switches or idle cleanup from
racing an active transcription.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from ..backends import (
    ModelConfig,
    get_stt_backend_for_engine,
    unload_all_stt_backends,
    unload_other_stt_backends,
)

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_SECONDS = 300
NEVER_UNLOAD = -1
ALLOWED_IDLE_TIMEOUTS = {-1, 0, 15, 60, 300, 600, 900, 3600}


def normalize_idle_timeout(value: int | None) -> int:
    if value in ALLOWED_IDLE_TIMEOUTS:
        return int(value)
    return DEFAULT_IDLE_TIMEOUT_SECONDS


def _persisted_idle_timeout() -> int:
    """Read the timeout without requiring callers to carry a DB session."""
    try:
        from ..database import session as database_session
        from .settings import get_capture_settings

        session_factory = database_session.SessionLocal
        if session_factory is None:
            return DEFAULT_IDLE_TIMEOUT_SECONDS
        db = session_factory()
        try:
            return normalize_idle_timeout(
                get_capture_settings(db).model_unload_timeout_seconds
            )
        finally:
            db.close()
    except Exception:
        logger.exception("Could not read the dictation model idle timeout")
        return DEFAULT_IDLE_TIMEOUT_SECONDS


class STTModelLifecycle:
    def __init__(self) -> None:
        self._operation_lock: asyncio.Lock | None = None
        self._operation_loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = threading.RLock()
        self._active_operations = 0
        self._generation = 0
        self._unload_timer: threading.Timer | None = None

    def _get_operation_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._operation_lock is None or self._operation_loop is not loop:
            self._operation_lock = asyncio.Lock()
            self._operation_loop = loop
        return self._operation_lock

    def _cancel_timer_locked(self) -> None:
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

    def _begin_operation(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._cancel_timer_locked()
            self._active_operations += 1

    def _finish_operation(self) -> None:
        unload_now = False
        with self._state_lock:
            self._active_operations = max(0, self._active_operations - 1)
            if self._active_operations:
                return

            timeout = _persisted_idle_timeout()
            self._generation += 1
            generation = self._generation
            self._cancel_timer_locked()

            if timeout == NEVER_UNLOAD:
                return
            if timeout == 0:
                unload_now = True
            else:
                timer = threading.Timer(
                    timeout,
                    self._unload_if_still_idle,
                    args=(generation, timeout),
                )
                timer.daemon = True
                self._unload_timer = timer
                timer.start()

        if unload_now:
            self.unload_idle_models("immediate idle policy")

    def _unload_if_still_idle(self, generation: int, timeout: int) -> None:
        with self._state_lock:
            if generation != self._generation or self._active_operations:
                return
            self._unload_timer = None
        self.unload_idle_models(f"{timeout}s of inactivity")

    def unload_idle_models(self, reason: str) -> None:
        try:
            unload_all_stt_backends()
            logger.info("Unloaded dictation STT models after %s", reason)
        except Exception:
            logger.exception("Failed to unload dictation STT models after %s", reason)

    def reschedule_for_setting_change(self) -> None:
        """Apply a changed timeout to any model that is currently idle."""
        with self._state_lock:
            if self._active_operations:
                return
            self._generation += 1
            self._cancel_timer_locked()
        # Reuse the normal completion path without changing the active count.
        with self._state_lock:
            self._active_operations = 1
        self._finish_operation()

    @asynccontextmanager
    async def use_model(self, config: ModelConfig):
        operation_lock = self._get_operation_lock()
        async with operation_lock:
            self._begin_operation()
            try:
                # Do not retain a second engine's weights when the user changes
                # the selected model family between dictations.
                unload_other_stt_backends(config.engine)
                yield get_stt_backend_for_engine(config.engine)
            finally:
                self._finish_operation()

    async def warm_model(self, config: ModelConfig) -> None:
        async with self.use_model(config) as backend:
            if backend.is_loaded() and getattr(backend, "model_size", None) == config.model_size:
                return
            await backend.load_model(config.model_size)


stt_model_lifecycle = STTModelLifecycle()

