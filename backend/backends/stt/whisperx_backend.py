"""WhisperX adapter with alignment and word timestamps."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from .. import ProgressCallback
from ..base import empty_device_cache, is_model_cached, model_load_progress
from .common import config_for_engine, report_progress, report_segments, require_import


class WhisperXSTTBackend:
    def __init__(self):
        self.model = None
        self.model_size = ""
        self.device = "cuda"
        self.compute_type = "float16"

    def is_loaded(self) -> bool:
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(config_for_engine(model_size, "whisperx").hf_repo_id)

    async def load_model(self, model_size: str) -> None:
        if self.is_loaded() and self.model_size == model_size:
            return
        await asyncio.to_thread(self._load_sync, model_size)

    load_model_async = load_model

    def _load_sync(self, model_size: str) -> None:
        config = config_for_engine(model_size, "whisperx")
        whisperx = require_import("whisperx", "WhisperX")
        torch = require_import("torch", "PyTorch")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        with model_load_progress(config.model_name, is_model_cached(config.hf_repo_id)):
            self.model = whisperx.load_model(
                config.model_size,
                self.device,
                compute_type=self.compute_type,
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
        # partial_callback: also not honored — same single-call limitation
        # as should_stop above, no per-segment boundary to report from.
        # Not yet honored here: whisperx's transcribe()/align() are single
        # vendored blocking calls with no chunk or callback boundary to poll
        # (unlike the adapters that process bounded PCM chunks themselves).
        # A cancelled job still becomes eligible for unload as soon as this
        # call returns — it just can't be cut short mid-call the way the
        # chunked adapters can.
        resolved = model_size or self.model_size
        await self.load_model(resolved)
        whisperx = require_import("whisperx", "WhisperX")

        def run() -> str:
            audio = whisperx.load_audio(audio_path)
            hint = None if not language or language == "auto" else language
            report_progress(progress_callback, 0.0)

            def transcription_progress(percentage: float) -> None:
                report_progress(progress_callback, float(percentage) / 100.0 * 0.8)

            result = self.model.transcribe(
                audio,
                batch_size=16,
                language=hint,
                progress_callback=transcription_progress,
            )
            segments = result.get("segments", [])

            detected = result.get("language") or hint
            if detected and segments:
                align_model, metadata = whisperx.load_align_model(detected, self.device)
                def alignment_progress(percentage: float) -> None:
                    report_progress(
                        progress_callback,
                        0.8 + float(percentage) / 100.0 * 0.2,
                    )

                result = whisperx.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    self.device,
                    progress_callback=alignment_progress,
                )
                segments = result.get("segments", segments)

            lines = []
            timed_segments: list[dict] = []
            for segment in segments:
                text = str(segment.get("text", "")).strip()
                if not text:
                    continue
                lines.append(text)
                timed_segments.append(
                    {
                        "start": float(segment.get("start", 0.0) or 0.0),
                        "end": float(segment.get("end", 0.0) or 0.0),
                        "text": text,
                    }
                )
            report_progress(progress_callback, 1.0)
            report_segments(segments_callback, timed_segments)
            return "\n".join(lines).strip()

        return await asyncio.to_thread(run)

    def unload_model(self) -> None:
        self.model = None
        empty_device_cache(self.device)
