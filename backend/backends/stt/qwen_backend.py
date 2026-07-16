"""Official Qwen3-ASR package adapter."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .. import ProgressCallback
from ..base import empty_device_cache, is_model_cached, model_load_progress
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


QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}


class QwenASRBackend:
    def __init__(self):
        self.model = None
        self.model_size = ""
        self.device = "cuda"

    def is_loaded(self) -> bool:
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(config_for_engine(model_size, "qwen_asr").hf_repo_id)

    async def load_model(self, model_size: str) -> None:
        if self.is_loaded() and self.model_size == model_size:
            return
        await asyncio.to_thread(self._load_sync, model_size)

    load_model_async = load_model

    def _load_sync(self, model_size: str) -> None:
        config = config_for_engine(model_size, "qwen_asr")
        qwen_asr = require_import("qwen_asr", "Qwen3-ASR")
        torch = require_import("torch", "PyTorch")
        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        if self.model is not None:
            self.unload_model()
        with model_load_progress(config.model_name, is_model_cached(config.hf_repo_id)):
            self.model = qwen_asr.Qwen3ASRModel.from_pretrained(
                config.hf_repo_id,
                dtype=torch.bfloat16 if use_cuda else torch.float32,
                device_map="cuda:0" if use_cuda else "cpu",
                max_inference_batch_size=1,
                max_new_tokens=4096,
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
        # segments_callback unused — chunk boundaries only, no honest
        # segment timestamps (see nemo_backend for rationale).
        resolved = model_size or self.model_size
        await self.load_model(resolved)
        hint = None if not language or language == "auto" else QWEN_LANGUAGE_NAMES[language]
        # The Transformers backend of Qwen3-ASR has no progress callback (its
        # streaming API is vLLM-only). Keep Diarix's single-server architecture
        # and expose real completed-audio progress through bounded PCM chunks.
        source = Path(audio_path)
        with tempfile.TemporaryDirectory(
            prefix="diarix-qwen-asr-", dir=str(source.parent)
        ) as temp_dir_name:
            chunks = await asyncio.to_thread(
                pcm_wav_chunks,
                source,
                Path(temp_dir_name),
                chunk_seconds=60.0,
                overlap_seconds=1.0,
            )
            stitched = ""
            report_progress(progress_callback, 0.0)
            for chunk in chunks:
                if stop_requested(should_stop):
                    break
                result = await asyncio.to_thread(
                    self.model.transcribe,
                    audio=str(chunk.path),
                    language=hint,
                )
                stitched = merge_overlapping_text(stitched, text_from_result(result))
                report_progress(progress_callback, chunk.completed_fraction)
                report_partial_text(partial_callback, stitched)
            return stitched.strip()

    def unload_model(self) -> None:
        self.model = None
        empty_device_cache(self.device)
