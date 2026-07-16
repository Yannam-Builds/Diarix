"""Transcription endpoints backed by centralized media ingestion."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .. import models
from ..backends import resolve_stt_config
from ..backends.base import is_model_cached
from ..services import transcribe
from ..services.media_ingestion import (
    MediaIngestionError,
    MediaUploadTooLargeError,
    cleanup_media_job_dir,
    create_media_job_dir,
    ingest_media,
    save_upload_to_job,
    sanitize_upload_filename,
)
from ..services.task_queue import (
    cancel_transcription,
    create_background_task,
    enqueue_transcription,
)
from ..services.transcription_jobs import (
    PendingMedia,
    publish_task,
    resolve_export_formats,
    resolve_job_language,
    resolve_job_options,
    resolve_output_directory,
    run_transcription_job,
    task_to_public_dict,
    validate_output_suffix,
)
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

router = APIRouter()

MAX_BATCH_FILES = 100


def _job_response(task) -> models.TranscriptionJobResponse:
    return models.TranscriptionJobResponse(**task_to_public_dict(task))


def _media_http_error(exc: MediaIngestionError) -> HTTPException:
    status_code = 413 if isinstance(exc, MediaUploadTooLargeError) else 400
    detail = {"code": exc.code, "message": str(exc)}
    if exc.detail:
        detail["detail"] = exc.detail
    return HTTPException(status_code=status_code, detail=detail)


@router.post(
    "/transcription/jobs",
    response_model=models.TranscriptionJobCreateResponse,
    status_code=202,
)
async def create_transcription_job(
    files: list[UploadFile] | None = File(None),
    files_bracketed: list[UploadFile] | None = File(None, alias="files[]"),
    model: str | None = Form(None),
    language: str | None = Form(None),
    precision: str | None = Form("default"),
    output_suffix: str | None = Form("_transcript"),
    output_dir: str | None = Form(None),
    export_formats: str | None = Form(None),
    silence_paragraphs: bool = Form(False),
):
    """Stage one media batch and return immediately with a cancellable task ID."""
    uploads = [*(files or []), *(files_bracketed or [])]
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one media file is required.")
    if len(uploads) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"A transcription batch can contain at most {MAX_BATCH_FILES} files.",
        )

    try:
        model_config, resolved_precision = resolve_job_options(model, precision)
        resolved_language = resolve_job_language(model_config, language)
        destination = resolve_output_directory(output_dir)
        resolved_suffix = validate_output_suffix(output_suffix)
        resolved_formats = resolve_export_formats(export_formats)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = uuid4().hex
    work_dir = create_media_job_dir(task_id)
    pending_files: list[PendingMedia] = []
    try:
        for index, upload in enumerate(uploads):
            original_name = sanitize_upload_filename(upload.filename, f"media-{index + 1}")
            source_path, _ = await save_upload_to_job(upload, work_dir, index=index)
            pending_files.append(PendingMedia(source_path=source_path, filename=original_name))
    except MediaIngestionError as exc:
        cleanup_media_job_dir(work_dir)
        raise _media_http_error(exc) from exc
    except Exception:
        cleanup_media_job_dir(work_dir)
        raise

    manager = get_task_manager()
    task = manager.start_transcription(
        task_id=task_id,
        model_name=model_config.model_name,
        language=resolved_language,
        precision=resolved_precision,
        output_dir=str(destination),
        total_files=len(pending_files),
        work_dir=str(work_dir),
    )
    publish_task(task)
    try:
        enqueue_transcription(
            task_id,
            run_transcription_job(
                task_id=task_id,
                files=pending_files,
                model_config=model_config,
                language=resolved_language,
                precision=resolved_precision,
                output_suffix=resolved_suffix,
                output_dir=destination,
                work_dir=work_dir,
                export_formats=resolved_formats,
                silence_paragraphs=silence_paragraphs,
            ),
        )
    except Exception as exc:
        cleanup_media_job_dir(work_dir)
        failed = manager.update_transcription(
            task_id, status="failed", stage="failed", error=str(exc)
        )
        if failed is not None:
            publish_task(failed)
        raise HTTPException(status_code=503, detail="Transcription queue is unavailable.") from exc

    return models.TranscriptionJobCreateResponse(task_id=task_id, status="queued")


@router.get(
    "/transcription/jobs/{task_id}", response_model=models.TranscriptionJobResponse
)
async def get_transcription_job(task_id: str):
    task = get_task_manager().get_transcription_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Transcription task not found.")
    return _job_response(task)


@router.get("/transcription/jobs/{task_id}/progress")
async def stream_transcription_job_progress(task_id: str):
    if get_task_manager().get_transcription_task(task_id) is None:
        raise HTTPException(status_code=404, detail="Transcription task not found.")
    return StreamingResponse(
        get_progress_manager().subscribe(task_id, include_current=True),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/transcription/jobs/{task_id}/cancel",
    response_model=models.TranscriptionJobResponse,
)
async def cancel_transcription_job(task_id: str):
    manager = get_task_manager()
    task = manager.get_transcription_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Transcription task not found.")
    if task.status in {"completed", "failed", "cancelled"}:
        return _job_response(task)

    result = cancel_transcription(task_id)
    if result == "queued":
        task = manager.update_transcription(
            task_id,
            status="cancelled",
            stage="cancelled",
            current_file=None,
            error=None,
        )
        if task is not None:
            publish_task(task)
        if task and task.work_dir:
            cleanup_media_job_dir(task.work_dir)
    elif result == "running":
        task = manager.update_transcription(
            task_id, status="cancelling", stage="cancelling"
        )
        if task is not None:
            publish_task(task)

    latest = manager.get_transcription_task(task_id)
    assert latest is not None
    return _job_response(latest)


@router.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
):
    """Compatibility endpoint routed through the same media ingestion boundary."""
    try:
        model_config = resolve_stt_config(model)
        resolved_language = resolve_job_language(model_config, language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_dir = create_media_job_dir()
    try:
        source_path, _ = await save_upload_to_job(file, job_dir)
        stt_backend, _ = transcribe.get_stt_model(model_config.model_name)
        already_loaded = (
            stt_backend.is_loaded()
            and getattr(stt_backend, "model_size", None) == model_config.model_size
        )
        if not already_loaded and not is_model_cached(model_config.hf_repo_id):
            progress_model_name = model_config.model_name
            task_manager = get_task_manager()

            async def download_stt_background():
                try:
                    await stt_backend.load_model(model_config.model_size)
                    task_manager.complete_download(progress_model_name)
                except Exception as exc:
                    task_manager.error_download(progress_model_name, str(exc))

            task_manager.start_download(progress_model_name)
            create_background_task(download_stt_background())
            raise HTTPException(
                status_code=202,
                detail={
                    "message": (
                        f"{model_config.display_name} is being downloaded. "
                        "Please wait and try again."
                    ),
                    "model_name": progress_model_name,
                    "downloading": True,
                },
            )

        async with ingest_media(
            source_path,
            model_config.audio_input,
            job_dir=job_dir,
            cleanup=False,
        ) as media:
            text, _ = await transcribe.transcribe_audio(
                str(media.audio_path), model_config.model_name, resolved_language
            )
            return models.TranscriptionResponse(
                text=text,
                duration=float(media.duration or 0.0),
            )
    except HTTPException:
        raise
    except MediaIngestionError as exc:
        raise _media_http_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        cleanup_media_job_dir(job_dir)
