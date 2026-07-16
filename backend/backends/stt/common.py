"""Shared helpers for optional STT engines."""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .. import ProgressCallback, resolve_stt_config


class MissingRuntimeError(RuntimeError):
    """Raised when an optional engine's runtime pack is not installed."""


StopCheck = Optional[Callable[[], bool]]
PartialTextCallback = Optional[Callable[[str], None]]
SegmentsCallback = Optional[Callable[[list], None]]


def stop_requested(should_stop: StopCheck) -> bool:
    """True if the caller asked the in-flight job to stop before the next chunk."""
    return bool(should_stop is not None and should_stop())


def report_partial_text(callback: PartialTextCallback, stitched_so_far: str) -> None:
    """Report the accumulated transcript text after each completed chunk."""
    if callback is None:
        return
    try:
        callback(stitched_so_far)
    except Exception:
        pass


def report_segments(callback: SegmentsCallback, segments: list) -> None:
    """Report the final {start, end, text} segment list, when the engine has one.

    Adapters without a real segment/timestamp boundary simply never call
    this — callers treat "never called" the same as "no timestamps
    available" and fall back to plain-text-only export.
    """
    if callback is None or not segments:
        return
    try:
        callback(segments)
    except Exception:
        pass


@dataclass(frozen=True)
class AudioChunk:
    """A model-ready PCM WAV slice and its covered source-audio fraction."""

    path: Path
    completed_fraction: float


def report_progress(callback: ProgressCallback | None, fraction: float) -> None:
    """Report normalized, finite progress without coupling adapters to tasks."""
    if callback is None:
        return
    try:
        normalized = float(fraction)
    except (TypeError, ValueError):
        return
    if normalized != normalized:  # NaN
        return
    callback(min(1.0, max(0.0, normalized)))


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / float(source.getframerate())


def pcm_wav_chunks(
    source_path: Path,
    output_dir: Path,
    *,
    chunk_seconds: float,
    overlap_seconds: float = 1.0,
) -> list[AudioChunk]:
    """Split normalized PCM WAV audio and retain true source coverage metadata."""
    total_seconds = wav_duration_seconds(source_path)
    if total_seconds <= chunk_seconds:
        return [AudioChunk(source_path, 1.0)]

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
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
            frame_count = min(chunk_frames, total_frames - start_frame)
            source.setpos(start_frame)
            data = source.readframes(frame_count)
            chunk_path = output_dir / f"audio-{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as target:
                target.setparams(params)
                target.writeframes(data)
            covered_frames = min(total_frames, start_frame + frame_count)
            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    completed_fraction=covered_frames / float(total_frames),
                )
            )
            if covered_frames >= total_frames:
                break
            start_frame += step_frames
            index += 1
    return chunks


def _normalized_word(word: str) -> str:
    return re.sub(r"(^\W+|\W+$)", "", word, flags=re.UNICODE).casefold()


def merge_overlapping_text(existing: str, incoming: str) -> str:
    """Drop exact word overlap introduced by adjacent PCM chunks."""
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


def require_import(module_name: str, package_name: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise MissingRuntimeError(
            f"{package_name} is not installed. Install the Advanced ASR runtime from Settings."
        ) from exc


def config_for_engine(model_name: str, engine: str):
    config = resolve_stt_config(model_name)
    if config.engine != engine:
        raise ValueError(f"Model {model_name} does not use the {engine} engine")
    return config


def text_from_result(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, (list, tuple)) and result:
        return text_from_result(result[0])
    return str(result).strip()
