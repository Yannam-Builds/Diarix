"""Native CTranslate2 adapter for Faster-Whisper model repositories."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from .. import ProgressCallback
from ..base import (
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    materialize_windows_snapshot_links,
    model_load_progress,
    native_windows_path,
)
from .common import (
    config_for_engine,
    report_partial_text,
    report_progress,
    report_segments,
    require_import,
    stop_requested,
)


class FasterWhisperSTTBackend:
    """Run Systran Faster-Whisper checkpoints without a second worker process."""

    def __init__(self):
        self.model = None
        self.model_size = ""
        self.device = get_torch_device()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        config = config_for_engine(model_size, "faster_whisper")
        return is_model_cached(config.hf_repo_id)

    async def load_model(self, model_size: str) -> None:
        if self.is_loaded() and self.model_size == model_size:
            return
        if self.is_loaded():
            self.unload_model()
        await asyncio.to_thread(self._load_sync, model_size)

    load_model_async = load_model

    def _load_sync(self, model_size: str) -> None:
        config = config_for_engine(model_size, "faster_whisper")
        faster_whisper = require_import("faster_whisper", "Faster-Whisper")
        huggingface_hub = require_import("huggingface_hub", "Hugging Face Hub")
        runtime_device = "cuda" if self.device == "cuda" else "cpu"
        compute_type = "float16" if runtime_device == "cuda" else "int8"
        cached = is_model_cached(config.hf_repo_id)
        with model_load_progress(config.model_name, cached):
            snapshot_path = huggingface_hub.snapshot_download(
                repo_id=config.hf_repo_id,
                local_files_only=cached,
            )
            snapshot_path = materialize_windows_snapshot_links(snapshot_path)
            self.model = faster_whisper.WhisperModel(
                native_windows_path(snapshot_path),
                device=runtime_device,
                compute_type=compute_type,
            )
        self.model_size = model_size

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
            # faster-whisper's transcribe() returns a lazy generator that
            # decodes one segment at a time, so this loop is already a real
            # interruption point — no separate chunking needed here.
            segments, info = self.model.transcribe(
                audio_path,
                language=language or None,
                vad_filter=True,
                condition_on_previous_text=True,
            )
            duration = max(0.0, float(getattr(info, "duration", 0.0) or 0.0))
            lines: list[str] = []
            timed_segments: list[dict] = []
            report_progress(progress_callback, 0.0)
            for segment in segments:
                if stop_requested(should_stop):
                    break
                text = segment.text.strip()
                if text:
                    lines.append(text)
                    timed_segments.append(
                        {
                            "start": float(getattr(segment, "start", 0.0) or 0.0),
                            "end": float(getattr(segment, "end", 0.0) or 0.0),
                            "text": text,
                        }
                    )
                    report_partial_text(partial_callback, "\n".join(lines))
                if duration > 0:
                    report_progress(
                        progress_callback,
                        float(getattr(segment, "end", 0.0) or 0.0) / duration,
                    )
            report_progress(progress_callback, 1.0)
            report_segments(segments_callback, timed_segments)
            return "\n".join(lines)

        return await asyncio.to_thread(_transcribe_sync)

    def unload_model(self) -> None:
        self.model = None
        self.model_size = ""
        empty_device_cache(self.device)
