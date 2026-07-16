"""NVIDIA NeMo adapter for Canary and Parakeet speech models."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import wave
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


class NeMoASRBackend:
    def __init__(self):
        self.model = None
        self.model_size = ""
        self.device = "cpu"

    def is_loaded(self) -> bool:
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(config_for_engine(model_size, "nemo_asr").hf_repo_id)

    async def load_model(self, model_size: str) -> None:
        if self.is_loaded() and self.model_size == model_size:
            return
        await asyncio.to_thread(self._load_sync, model_size)

    load_model_async = load_model

    def _load_sync(self, model_size: str) -> None:
        config = config_for_engine(model_size, "nemo_asr")
        if self.model is not None:
            self.unload_model()
        # NeMo's from_pretrained() places weights on CPU regardless of what
        # the caller sets self.device to beforehand — it does not consult
        # torch.cuda.is_available() itself. Every model here (ASRModel and
        # SALM) is a LightningModule/nn.Module subclass, so .cuda() moves it
        # the same way QwenASRBackend and TransformersASRBackend already do
        # for their own runtimes.
        resolved_device = get_torch_device()
        with model_load_progress(config.model_name, is_model_cached(config.hf_repo_id)):
            if model_size == "nvidia-canary-qwen-2.5b":
                speechlm_models = require_import("nemo.collections.speechlm2.models", "NVIDIA NeMo SpeechLM2")
                self.model = speechlm_models.SALM.from_pretrained(config.hf_repo_id)
            else:
                nemo_asr = require_import("nemo.collections.asr", "NVIDIA NeMo")
                self.model = nemo_asr.models.ASRModel.from_pretrained(config.hf_repo_id)
            if resolved_device == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.to(resolved_device)
            self.model.eval()
            if model_size == "nvidia-parakeet-tdt-0.6b-v3":
                # NVIDIA's documented long-form mode replaces global
                # attention with bounded local attention before inference.
                self.model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=[256, 256],
                )
        self.device = resolved_device
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
        # segments_callback is intentionally unused: this adapter only knows
        # 30-60s chunk boundaries, which are too coarse to be honest
        # subtitle timestamps. Timestamped export stays with the engines
        # that produce real segments (Faster-Whisper, WhisperX).
        resolved = model_size or self.model_size
        await self.load_model(resolved)
        source_language = language or "en"

        # NeMo exposes a console progress bar but no application callback for
        # a single long file. Process normalized PCM in bounded chunks so each
        # completed slice provides real source-duration progress. Canary 180M
        # remains below its documented 40-second utterance limit; the other
        # long-form models keep a wider window to preserve context.
        source = Path(audio_path)
        with tempfile.TemporaryDirectory(prefix="diarix-nemo-", dir=str(source.parent)) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            if resolved == "nvidia-canary-180m-flash":
                chunk_seconds = 30.0
                overlap_seconds = 1.0
            elif resolved == "nvidia-canary-qwen-2.5b":
                # NVIDIA trained and evaluates Canary-Qwen with utterances up
                # to 40 seconds. Non-overlapping chunks avoid duplicating text
                # because this SALM path does not emit timestamps.
                chunk_seconds = 40.0
                overlap_seconds = 0.0
            else:
                chunk_seconds = 60.0
                overlap_seconds = 1.0
            chunks = await asyncio.to_thread(
                pcm_wav_chunks,
                source,
                temp_dir,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
            )
            stitched = ""
            report_progress(progress_callback, 0.0)
            for index, chunk in enumerate(chunks):
                if stop_requested(should_stop):
                    break
                if resolved == "nvidia-parakeet-tdt-0.6b-v3":
                    result = await asyncio.to_thread(
                        self.model.transcribe,
                        [str(chunk.path)],
                        batch_size=1,
                    )
                    chunk_text = text_from_result(result)
                elif resolved == "nvidia-canary-qwen-2.5b":
                    prompts = [
                        [
                            {
                                "role": "user",
                                "content": (f"Transcribe the following: {self.model.audio_locator_tag}"),
                                "audio": [str(chunk.path)],
                            }
                        ]
                    ]
                    answer_ids = await asyncio.to_thread(
                        self.model.generate,
                        prompts=prompts,
                        max_new_tokens=512,
                    )
                    token_ids = answer_ids[0]
                    if hasattr(token_ids, "cpu"):
                        token_ids = token_ids.cpu()
                    chunk_text = _clean_canary_text(self.model.tokenizer.ids_to_text(token_ids))
                elif resolved == "nvidia-canary-1b-v2":
                    result = await asyncio.to_thread(
                        self.model.transcribe,
                        [str(chunk.path)],
                        source_lang=source_language,
                        target_lang=source_language,
                        batch_size=1,
                    )
                    chunk_text = _clean_canary_text(text_from_result(result))
                elif resolved == "nvidia-canary-180m-flash" and source_language == "en":
                    result = await asyncio.to_thread(
                        self.model.transcribe,
                        [str(chunk.path)],
                        batch_size=1,
                        pnc="yes",
                    )
                    chunk_text = _clean_canary_text(text_from_result(result))
                else:
                    manifest_path = temp_dir / f"chunk-{index:04d}.jsonl"
                    await asyncio.to_thread(
                        _write_canary_manifest,
                        manifest_path,
                        chunk.path,
                        source_language,
                    )
                    result = await asyncio.to_thread(
                        self.model.transcribe,
                        str(manifest_path),
                        batch_size=1,
                    )
                    chunk_text = _clean_canary_text(text_from_result(result))
                stitched = merge_overlapping_text(stitched, chunk_text)
                report_progress(progress_callback, chunk.completed_fraction)
                report_partial_text(partial_callback, stitched)
            return stitched.strip()

    def unload_model(self) -> None:
        self.model = None
        empty_device_cache(self.device)


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / float(source.getframerate())


def _canary_audio_chunks(
    source_path: Path,
    output_dir: Path,
    *,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 1.0,
) -> list[Path]:
    """Return <40s PCM WAV chunks, retaining overlap around word boundaries."""
    if _wav_duration_seconds(source_path) < 40.0:
        return [source_path]

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    with wave.open(str(source_path), "rb") as source:
        params = source.getparams()
        sample_rate = source.getframerate()
        total_frames = source.getnframes()
        chunk_frames = max(1, int(chunk_seconds * sample_rate))
        overlap_frames = max(0, int(overlap_seconds * sample_rate))
        step_frames = max(1, chunk_frames - overlap_frames)
        start_frame = 0
        index = 0
        while start_frame < total_frames:
            source.setpos(start_frame)
            data = source.readframes(min(chunk_frames, total_frames - start_frame))
            chunk_path = output_dir / f"audio-{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as target:
                target.setparams(params)
                target.writeframes(data)
            chunks.append(chunk_path)
            if start_frame + chunk_frames >= total_frames:
                break
            start_frame += step_frames
            index += 1
    return chunks


def _write_canary_manifest(
    manifest_path: Path,
    audio_path: Path,
    language: str,
) -> None:
    record = {
        "audio_filepath": str(audio_path.resolve()),
        "duration": _wav_duration_seconds(audio_path),
        "source_lang": language,
        "target_lang": language,
        "taskname": "asr",
        "pnc": "yes",
        "timestamp": "no",
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _normalized_word(word: str) -> str:
    return re.sub(r"(^\W+|\W+$)", "", word, flags=re.UNICODE).casefold()


def _clean_canary_text(text: str) -> str:
    """Remove decoder control tokens that are not part of a transcript."""
    cleaned = re.sub(r"<\|[^|]+\|>", "", text).strip()
    return re.sub(r"^[\s.,;:]+", "", cleaned).strip()


def _merge_overlapping_text(existing: str, incoming: str) -> str:
    """Drop an exact word overlap introduced by adjacent Canary chunks."""
    existing = existing.strip()
    incoming = incoming.strip()
    if not existing:
        return incoming
    if not incoming:
        return existing

    left = existing.split()
    right = incoming.split()
    max_overlap = min(24, len(left), len(right))
    for overlap in range(max_overlap, 1, -1):
        left_tail = [_normalized_word(word) for word in left[-overlap:]]
        right_head = [_normalized_word(word) for word in right[:overlap]]
        if left_tail == right_head and all(left_tail):
            return " ".join(left + right[overlap:]).strip()
    return f"{existing} {incoming}".strip()
