"""Transformers ASR adapter for Parakeet and Granite Speech."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .. import ProgressCallback
from ..base import empty_device_cache, get_torch_device, is_model_cached, model_load_progress
from .common import (
    config_for_engine,
    merge_overlapping_text,
    pcm_wav_chunks,
    report_partial_text,
    report_progress,
    require_import,
    stop_requested,
    text_from_result,
)

logger = logging.getLogger(__name__)


class TransformersASRBackend:
    def __init__(self):
        self.pipeline = None
        self.model_size = ""
        self.device = get_torch_device(allow_xpu=True, allow_directml=False)

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(config_for_engine(model_size, "transformers_asr").hf_repo_id)

    async def load_model(self, model_size: str) -> None:
        if self.is_loaded() and self.model_size == model_size:
            return
        await asyncio.to_thread(self._load_sync, model_size)

    load_model_async = load_model

    def _load_sync(self, model_size: str) -> None:
        config = config_for_engine(model_size, "transformers_asr")
        transformers = require_import("transformers", "Transformers ASR")
        torch = require_import("torch", "PyTorch")
        if self.pipeline is not None:
            self.unload_model()

        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if self.device == "cuda" else torch.float32)
        )
        progress_name = config.model_name
        with model_load_progress(progress_name, is_model_cached(config.hf_repo_id)):
            kwargs = {"model": config.hf_repo_id, "torch_dtype": dtype}
            if self.device == "cuda":
                kwargs["device"] = 0
            elif self.device == "cpu":
                kwargs["device"] = -1
            self.pipeline = transformers.pipeline("automatic-speech-recognition", **kwargs)
        self.model_size = model_size
        logger.info("Loaded %s on %s", config.display_name, self.device)

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
        # segments_callback unused — chunk boundaries only, no honest
        # segment timestamps (see nemo_backend for rationale).
        resolved = model_size or self.model_size
        await self.load_model(resolved)

        def run_chunk(chunk_path: str) -> str:
            kwargs = {}
            if language and language != "auto":
                kwargs["generate_kwargs"] = {"language": language}
            try:
                result = self.pipeline(chunk_path, **kwargs)
            except (TypeError, ValueError):
                result = self.pipeline(chunk_path)
            return text_from_result(result)

        # SpeechSeq2Seq generation exposes token streaming, but the final token
        # count is unknowable and therefore cannot produce an honest percent.
        # Complete bounded audio slices instead and report their source coverage.
        source = Path(audio_path)
        with tempfile.TemporaryDirectory(
            prefix="diarix-transformers-asr-", dir=str(source.parent)
        ) as temp_dir_name:
            chunks = await asyncio.to_thread(
                pcm_wav_chunks,
                source,
                Path(temp_dir_name),
                chunk_seconds=30.0,
                overlap_seconds=1.0,
            )
            stitched = ""
            report_progress(progress_callback, 0.0)
            for chunk in chunks:
                if stop_requested(should_stop):
                    break
                chunk_text = await asyncio.to_thread(run_chunk, str(chunk.path))
                stitched = merge_overlapping_text(stitched, chunk_text)
                report_progress(progress_callback, chunk.completed_fraction)
                report_partial_text(partial_callback, stitched)
            return stitched.strip()

    def unload_model(self) -> None:
        self.pipeline = None
        empty_device_cache(self.device)
