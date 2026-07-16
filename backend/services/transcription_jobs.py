"""Batch transcription orchestration built on Voicebox task infrastructure."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import config
from ..backends import ModelConfig, resolve_stt_config, unload_model_by_config
from ..utils.progress import get_progress_manager
from ..utils.tasks import TranscriptionResult, TranscriptionTask, get_task_manager
from . import captures, task_queue, transcribe
from .media_ingestion import (
    MediaIngestionError,
    cleanup_media_job_dir,
    ingest_media,
)
from .transcript_formats import (
    SUPPORTED_EXPORT_FORMATS,
    paragraphs_from_segments,
    segments_to_json,
    segments_to_srt,
    segments_to_vtt,
)


logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(frozen=True)
class PendingMedia:
    source_path: Path
    filename: str


def resolve_job_options(model: str | None, precision: str | None) -> tuple[ModelConfig, str]:
    """Validate a model selection while preserving honest precision metadata."""
    model_config = resolve_stt_config(model)
    requested_precision = (precision or "default").strip().lower() or "default"
    if requested_precision in {"auto", "default"}:
        # Adapters currently choose their own runtime dtype. Store this sentinel
        # instead of claiming a precision was applied when it was not.
        return model_config, "default"
    if requested_precision not in model_config.precision_options:
        options = ", ".join(model_config.precision_options) or "default"
        raise ValueError(
            f"Unsupported precision '{requested_precision}' for {model_config.model_name}. "
            f"Available: {options}."
        )
    return model_config, requested_precision


def resolve_job_language(model_config: ModelConfig, language: str | None) -> str:
    """Validate language hints against the selected model's real capabilities."""
    requested = (language or "auto").strip().lower() or "auto"
    supported = {code.lower() for code in model_config.languages}
    if requested == "auto":
        if "language_detection" in model_config.capabilities:
            return "auto"
        if len(supported) == 1:
            return next(iter(supported))
        raise ValueError(
            f"{model_config.display_name} requires an explicit language: "
            f"{', '.join(sorted(supported))}."
        )
    if requested not in supported:
        raise ValueError(
            f"Language '{requested}' is not supported by {model_config.display_name}. "
            f"Choose one of: {', '.join(sorted(supported))}."
        )
    return requested


def resolve_output_directory(output_dir: str | None) -> Path:
    """Resolve an explicit destination or the app-owned transcript directory."""
    if output_dir and output_dir.strip():
        destination = Path(output_dir.strip()).expanduser().resolve()
    else:
        destination = config.get_transcriptions_dir().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ValueError(f"Transcript output is not a directory: {destination}")
    return destination


def resolve_export_formats(export_formats: str | None) -> list[str]:
    """Validate a comma-separated format list against the supported set.

    Plain text is always written; srt/vtt/json are additional and only take
    effect for models whose engine reports real segment timestamps.
    """
    if not export_formats or not export_formats.strip():
        return ["txt"]
    requested = [item.strip().lower() for item in export_formats.split(",") if item.strip()]
    unknown = [item for item in requested if item not in SUPPORTED_EXPORT_FORMATS]
    if unknown:
        raise ValueError(
            f"Unsupported export format(s): {', '.join(unknown)}. "
            f"Available: {', '.join(SUPPORTED_EXPORT_FORMATS)}."
        )
    if "txt" not in requested:
        requested.insert(0, "txt")
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    return [item for item in requested if not (item in seen or seen.add(item))]


def validate_output_suffix(output_suffix: str | None) -> str:
    suffix = "_transcript" if output_suffix is None else output_suffix.strip()
    if len(suffix) > 80:
        raise ValueError("Output suffix must be 80 characters or fewer.")
    if re.search(r"[\x00-\x1f<>:\"/\\|?*]", suffix):
        raise ValueError("Output suffix contains characters that are not valid in filenames.")
    if suffix.lower().endswith(".txt"):
        suffix = suffix[:-4]
    return suffix


def task_to_public_dict(task: TranscriptionTask) -> dict:
    payload = asdict(task)
    payload.pop("work_dir", None)
    payload.pop("started_at", None)
    return payload


def publish_task(task: TranscriptionTask) -> None:
    payload = task_to_public_dict(task)
    task_id = payload.pop("task_id")
    status = payload.pop("status")
    progress = payload.pop("progress")
    stage = payload.pop("stage")
    get_progress_manager().update_task_progress(
        task_id,
        status=status,
        progress=progress,
        stage=stage,
        **payload,
    )


def _update_and_publish(task_id: str, **changes) -> TranscriptionTask | None:
    task = get_task_manager().update_transcription(task_id, **changes)
    if task is not None:
        publish_task(task)
    return task


def _unique_output_path(
    output_dir: Path, stem: str, suffix: str, extension: str = "txt"
) -> Path:
    safe_stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", "_", stem).strip(" .") or "transcript"
    candidate = output_dir / f"{safe_stem}{suffix}.{extension}"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{safe_stem}{suffix}-{counter}.{extension}"
        counter += 1
    return candidate


def _write_transcript_atomic(output_path: Path, text: str) -> None:
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


async def run_transcription_job(
    *,
    task_id: str,
    files: list[PendingMedia],
    model_config: ModelConfig,
    language: str | None,
    precision: str,
    output_suffix: str,
    output_dir: Path,
    work_dir: Path,
    export_formats: list[str] | None = None,
    silence_paragraphs: bool = False,
) -> None:
    """Process a batch sequentially and leave a terminal task snapshot."""
    manager = get_task_manager()
    total_files = len(files)
    inference_language = None if not language or language == "auto" else language
    current_progress = 0.0
    resolved_formats = export_formats or ["txt"]

    try:
        _update_and_publish(task_id, status="running", stage="loading_model", error=None)
        backend, _ = transcribe.get_stt_model(model_config.model_name)
        await transcribe.await_stt_operation(backend.load_model(model_config.model_size))

        for index, pending in enumerate(files):
            file_base = index / total_files * 100.0
            file_share = 100.0 / total_files
            current_progress = file_base
            _update_and_publish(
                task_id,
                status="running",
                current_file=pending.filename,
                stage="probing",
                progress=current_progress,
            )

            async def ingestion_progress(stage: str, percentage: float) -> None:
                nonlocal current_progress
                if stage == "probing":
                    local_progress = percentage * 0.10
                else:
                    local_progress = 10.0 + percentage * 0.25
                current_progress = file_base + file_share * local_progress / 100.0
                _update_and_publish(
                    task_id,
                    status="running",
                    current_file=pending.filename,
                    stage=stage,
                    progress=current_progress,
                )

            async with ingest_media(
                pending.source_path,
                model_config.audio_input,
                progress_callback=ingestion_progress,
            ) as media:
                current_progress = file_base + file_share * 0.35
                _update_and_publish(
                    task_id,
                    status="running",
                    current_file=pending.filename,
                    stage="transcribing",
                    progress=current_progress,
                    partial_text="",
                )

                inference_fraction = 0.0

                def inference_progress(fraction: float) -> None:
                    """Map model-native audio progress into this file's 35-90% band."""
                    nonlocal current_progress, inference_fraction
                    try:
                        normalized = min(1.0, max(0.0, float(fraction)))
                    except (TypeError, ValueError):
                        return
                    if normalized != normalized or normalized <= inference_fraction:
                        return
                    inference_fraction = normalized
                    current_progress = file_base + file_share * (
                        35.0 + inference_fraction * 55.0
                    ) / 100.0
                    _update_and_publish(
                        task_id,
                        status="running",
                        current_file=pending.filename,
                        stage="transcribing",
                        progress=current_progress,
                    )

                def partial_text_progress(chunk_text: str) -> None:
                    _update_and_publish(
                        task_id,
                        status="running",
                        current_file=pending.filename,
                        stage="transcribing",
                        progress=current_progress,
                        partial_text=chunk_text,
                    )

                file_segments: list[dict] = []

                def collect_segments(segments: list) -> None:
                    file_segments.extend(segments)

                text, actual_model_name = await transcribe.transcribe_audio(
                    str(media.audio_path),
                    model_config.model_name,
                    inference_language,
                    progress_callback=inference_progress,
                    should_stop=lambda: task_queue.is_transcription_cancel_requested(task_id),
                    partial_callback=partial_text_progress,
                    segments_callback=collect_segments,
                )
                duration = float(media.duration or 0.0)

            current_progress = file_base + file_share * 0.90
            _update_and_publish(
                task_id,
                status="running",
                current_file=pending.filename,
                stage="writing",
                progress=current_progress,
            )
            if silence_paragraphs and file_segments:
                # Same text, reflowed: break paragraphs where the segment
                # timestamps show a silence gap. Engines without segments
                # keep their original line layout.
                text = paragraphs_from_segments(file_segments) or text

            output_path = _unique_output_path(
                output_dir, Path(pending.filename).stem, output_suffix
            )
            await asyncio.to_thread(_write_transcript_atomic, output_path, text)

            extra_outputs: list[str] = []
            if file_segments:
                stem = Path(pending.filename).stem
                format_writers = {
                    "srt": segments_to_srt,
                    "vtt": segments_to_vtt,
                    "json": segments_to_json,
                }
                for format_name in resolved_formats:
                    writer = format_writers.get(format_name)
                    if writer is None:
                        continue
                    extra_path = _unique_output_path(
                        output_dir, stem, output_suffix, extension=format_name
                    )
                    await asyncio.to_thread(
                        _write_transcript_atomic, extra_path, writer(file_segments)
                    )
                    extra_outputs.append(str(extra_path))

            await asyncio.to_thread(
                captures.persist_completed_transcription,
                source_path=pending.source_path,
                filename=pending.filename,
                language=language,
                duration=duration,
                transcript=text,
                stt_model=actual_model_name,
            )
            snapshot = manager.append_transcription_result(
                task_id,
                TranscriptionResult(
                    filename=pending.filename,
                    output_path=str(output_path),
                    text=text,
                    duration=duration,
                    model_name=actual_model_name,
                    extra_outputs=extra_outputs,
                ),
            )
            current_progress = file_base + file_share
            if snapshot is not None:
                snapshot = manager.update_transcription(task_id, progress=current_progress)
                if snapshot is not None:
                    publish_task(snapshot)

        _update_and_publish(
            task_id,
            status="completed",
            current_file=None,
            stage="completed",
            progress=100.0,
            error=None,
        )
    except asyncio.CancelledError:
        _update_and_publish(
            task_id,
            status="cancelled",
            stage="cancelled",
            progress=current_progress,
            error=None,
        )
        # Free the model immediately rather than leaving it resident until
        # something else happens to load or unload it. should_stop already
        # cut the in-flight transcribe call short at the next chunk
        # boundary (see transcribe.await_stt_operation / stop_requested in
        # the STT adapters), so the model is idle and safe to release here.
        try:
            unload_model_by_config(model_config)
        except Exception:
            logger.exception("Failed to unload %s after cancelling task %s", model_config.model_name, task_id)
        raise
    except Exception as exc:
        logger.exception("Transcription job %s failed", task_id)
        if isinstance(exc, MediaIngestionError):
            message = f"[{exc.code}] {exc}"
            if exc.detail:
                message = f"{message}: {exc.detail}"
        else:
            message = str(exc) or exc.__class__.__name__
        _update_and_publish(
            task_id,
            status="failed",
            stage="failed",
            progress=current_progress,
            error=message,
        )
    finally:
        cleanup_media_job_dir(work_dir)
