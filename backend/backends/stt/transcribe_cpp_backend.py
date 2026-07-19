"""Native GGUF speech recognition through Handy's transcribe.cpp runtime."""

from __future__ import annotations

import asyncio
import os
import threading
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import ProgressCallback, is_model_config_cached
from ..base import materialize_windows_snapshot_links, model_load_progress, native_windows_path
from .common import (
    MissingRuntimeError,
    config_for_engine,
    merge_overlapping_text,
    report_partial_text,
    report_progress,
    report_segments,
    stop_requested,
)

STREAM_CHUNK_SAMPLES = 1_600  # 100 ms at the shared 16 kHz input rate.
OFFLINE_FALLBACK_CHUNK_MS = 30_000
OFFLINE_OVERLAP_MS = 500


def _read_pcm16_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read the normalized media boundary as mono float32 PCM."""
    with wave.open(str(path), "rb") as source:
        if source.getframerate() != 16_000:
            raise ValueError(
                f"transcribe.cpp requires 16 kHz audio, got {source.getframerate()} Hz"
            )
        if source.getnchannels() != 1:
            raise ValueError(
                f"transcribe.cpp requires mono audio, got {source.getnchannels()} channels"
            )
        if source.getsampwidth() != 2:
            raise ValueError(
                f"transcribe.cpp requires 16-bit PCM, got {source.getsampwidth() * 8}-bit"
            )
        frames = source.readframes(source.getnframes())

    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    pcm /= 32768.0
    return pcm, 16_000


def _timestamp_mode(max_timestamp_kind: str) -> str:
    if max_timestamp_kind in {"segment", "word", "token"}:
        return "segment"
    return "none"


class TranscribeCppSTTBackend:
    """Load one GGUF model in the existing Diarix server process."""

    def __init__(self):
        self.model = None
        self.model_size = ""
        self._runtime = None
        self.backend = "auto"

    def is_loaded(self) -> bool:
        return self.model is not None

    def supports_live_streaming(self) -> bool:
        """Whether the loaded GGUF model exposes a true incremental stream."""
        return bool(
            self.model is not None
            and getattr(self.model.capabilities, "supports_streaming", False)
        )

    def open_live_stream(self, language: str | None = None):
        """Open a microphone-time stream on the already-loaded model.

        The returned object is intentionally synchronous. The WebSocket route
        owns a single-thread executor for its lifetime so every native session
        call stays serialized on one worker thread.
        """
        if self.model is None:
            raise RuntimeError("The native GGUF model is not loaded")
        if not self.supports_live_streaming():
            raise ValueError(f"{self.model_size} does not support live streaming")
        return TranscribeCppLiveStream(self.model, language)

    def _is_model_cached(self, model_size: str) -> bool:
        config = config_for_engine(model_size, "transcribe_cpp")
        return is_model_config_cached(config)

    async def load_model(self, model_size: str) -> None:
        if self.is_loaded() and self.model_size == model_size:
            return
        if self.is_loaded():
            self.unload_model()
        await asyncio.to_thread(self._load_sync, model_size)

    load_model_async = load_model

    def _load_sync(self, model_size: str) -> None:
        config = config_for_engine(model_size, "transcribe_cpp")
        if not config.artifact_filename:
            raise ValueError(f"{config.display_name} has no GGUF artifact configured")

        try:
            import transcribe_cpp
        except (ImportError, OSError, RuntimeError) as exc:
            raise MissingRuntimeError(
                "The native GGUF ASR runtime is not installed in this Diarix server build."
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise MissingRuntimeError("Hugging Face Hub is not installed.") from exc

        cached = is_model_config_cached(config)
        with model_load_progress(config.model_name, cached):
            artifact_path = Path(
                hf_hub_download(
                    repo_id=config.hf_repo_id,
                    filename=config.artifact_filename,
                    local_files_only=cached,
                )
            )
            materialize_windows_snapshot_links(artifact_path.parent)
            artifact_path = artifact_path.parent / config.artifact_filename
            requested_backend = os.environ.get("DIARIX_TRANSCRIBE_CPP_BACKEND", "auto")
            self.model = transcribe_cpp.Model(
                native_windows_path(artifact_path),
                backend=requested_backend,
            )

        self._runtime = transcribe_cpp
        self.model_size = model_size
        self.backend = str(getattr(self.model, "backend", requested_backend))

    def _cancel_watcher(
        self,
        session,
        should_stop: Optional[Callable[[], bool]],
        completed: threading.Event,
    ) -> None:
        if should_stop is None:
            return
        while not completed.wait(0.05):
            if stop_requested(should_stop):
                session.cancel()
                return

    def _run_streaming(
        self,
        session,
        pcm: np.ndarray,
        language: str | None,
        progress_callback: ProgressCallback | None,
        should_stop: Optional[Callable[[], bool]],
        partial_callback: Optional[Callable[[str], None]],
    ) -> str:
        report_progress(progress_callback, 0.0)
        latest_text = ""
        with session.stream(language=language, timestamps="none") as stream:
            for offset in range(0, len(pcm), STREAM_CHUNK_SAMPLES):
                if stop_requested(should_stop):
                    session.cancel()
                    return latest_text.strip()
                update = stream.feed(pcm[offset : offset + STREAM_CHUNK_SAMPLES])
                if update.committed_changed or update.tentative_changed:
                    view = stream.text()
                    latest_text = view.full
                    report_partial_text(partial_callback, latest_text)
                report_progress(
                    progress_callback,
                    min(1.0, (offset + STREAM_CHUNK_SAMPLES) / max(1, len(pcm))),
                )

            stream.finalize()
            latest_text = stream.text().full

        report_partial_text(partial_callback, latest_text)
        report_progress(progress_callback, 1.0)
        return latest_text.strip()

    def _run_offline(
        self,
        session,
        pcm: np.ndarray,
        sample_rate: int,
        language: str | None,
        progress_callback: ProgressCallback | None,
        should_stop: Optional[Callable[[], bool]],
        partial_callback: Optional[Callable[[str], None]],
        segments_callback: Optional[Callable[[list], None]],
    ) -> str:
        limits = session.limits
        model_limit_ms = int(getattr(limits, "effective_max_audio_ms", 0) or 0)
        chunk_ms = OFFLINE_FALLBACK_CHUNK_MS
        if model_limit_ms > 0:
            chunk_ms = max(5_000, min(chunk_ms, model_limit_ms - 250))

        chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
        overlap_samples = max(0, int(sample_rate * OFFLINE_OVERLAP_MS / 1000))
        step_samples = max(1, chunk_samples - overlap_samples)
        timestamp_mode = _timestamp_mode(self.model.capabilities.max_timestamp_kind)
        stitched = ""
        timed_segments: list[dict] = []
        report_progress(progress_callback, 0.0)

        start = 0
        while start < len(pcm):
            if stop_requested(should_stop):
                break
            end = min(len(pcm), start + chunk_samples)
            completed = threading.Event()
            watcher = threading.Thread(
                target=self._cancel_watcher,
                args=(session, should_stop, completed),
                daemon=True,
            )
            watcher.start()
            try:
                result = session.run(
                    pcm[start:end],
                    language=language,
                    timestamps=timestamp_mode,
                )
            except self._runtime.errors.Aborted as exc:
                partial = getattr(exc, "partial_result", None)
                if partial is not None:
                    stitched = merge_overlapping_text(stitched, partial.text)
                break
            finally:
                completed.set()
                watcher.join(timeout=0.2)

            stitched = merge_overlapping_text(stitched, result.text)
            report_partial_text(partial_callback, stitched)
            for segment in result.segments:
                timed_segments.append(
                    {
                        "start": (start / sample_rate) + (segment.t0_ms / 1000.0),
                        "end": (start / sample_rate) + (segment.t1_ms / 1000.0),
                        "text": segment.text.strip(),
                    }
                )
            report_progress(progress_callback, end / max(1, len(pcm)))
            if end >= len(pcm):
                break
            start += step_samples

        report_segments(segments_callback, timed_segments)
        if not stop_requested(should_stop):
            report_progress(progress_callback, 1.0)
        return stitched.strip()

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
        progress_callback: ProgressCallback | None = None,
        should_stop: Optional[Callable[[], bool]] = None,
        partial_callback: Optional[Callable[[str], None]] = None,
        segments_callback: Optional[Callable[[list], None]] = None,
    ) -> str:
        resolved = model_size or self.model_size
        await self.load_model(resolved)

        def _transcribe_sync() -> str:
            pcm, sample_rate = _read_pcm16_wav(audio_path)
            with self.model.session() as session:
                if self.model.capabilities.supports_streaming:
                    return self._run_streaming(
                        session,
                        pcm,
                        language,
                        progress_callback,
                        should_stop,
                        partial_callback,
                    )
                return self._run_offline(
                    session,
                    pcm,
                    sample_rate,
                    language,
                    progress_callback,
                    should_stop,
                    partial_callback,
                    segments_callback,
                )

        return await asyncio.to_thread(_transcribe_sync)

    def unload_model(self) -> None:
        model = self.model
        self.model = None
        self.model_size = ""
        self.backend = "auto"
        self._runtime = None
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                close()


class TranscribeCppLiveStream:
    """One native microphone-time transcription session."""

    def __init__(self, model, language: str | None):
        self._session = model.session()
        self._stream_context = self._session.stream(
            language=language,
            timestamps="none",
        )
        self._stream = self._stream_context.__enter__()
        self._closed = False

    def feed(self, pcm: np.ndarray) -> dict:
        if self._closed:
            raise RuntimeError("Live transcription stream is closed")
        update = self._stream.feed(np.asarray(pcm, dtype=np.float32))
        view = self._stream.text()
        return {
            "changed": bool(update.committed_changed or update.tentative_changed),
            "full": view.full,
            "committed": view.committed,
            "tentative": view.tentative,
            "input_received_ms": int(update.input_received_ms),
            "audio_committed_ms": int(update.audio_committed_ms),
            "revision": int(update.revision),
            "is_final": bool(update.is_final),
        }

    def finalize(self) -> dict:
        if self._closed:
            raise RuntimeError("Live transcription stream is closed")
        update = self._stream.finalize()
        view = self._stream.text()
        return {
            "changed": True,
            "full": view.full,
            "committed": view.committed,
            "tentative": view.tentative,
            "input_received_ms": int(update.input_received_ms),
            "audio_committed_ms": int(update.audio_committed_ms),
            "revision": int(update.revision),
            "is_final": True,
        }

    def cancel(self) -> None:
        if not self._closed:
            self._session.cancel()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream_context.__exit__(None, None, None)
        finally:
            self._session.close()
