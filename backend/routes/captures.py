"""Capture (voice input) endpoints."""

import asyncio
import json
import logging
import mimetypes
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..backends import get_llm_model_configs, is_model_config_cached, resolve_stt_config
from ..backends.base import is_model_cached
from ..database import Capture as DBCapture, get_db
from ..services import captures as captures_service
from ..services import task_queue
from ..services import transcribe as transcribe_service
from ..services import settings as settings_service
from ..services.media_ingestion import MediaIngestionError
from ..services.refinement import RefinementFlags

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
LIVE_SAMPLE_RATE = 16_000
MAX_LIVE_CHUNK_BYTES = LIVE_SAMPLE_RATE * 4 * 2


def _capture_session():
    from ..database import session as database_session

    session_factory = database_session.SessionLocal
    if session_factory is None:
        raise RuntimeError("Database session factory is not initialized")
    return session_factory()


async def _run_live_capture(
    websocket: WebSocket,
    *,
    operation_id: str,
    model_config,
    language: str | None,
    auto_refine: bool,
    allow_auto_paste: bool,
) -> None:
    """Run one true streaming dictation through the shared model lifecycle."""
    from ..services.model_lifecycle import stt_model_lifecycle

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=f"diarix-live-{operation_id[:8]}",
    )
    live_stream = None
    chunks: list[np.ndarray] = []
    try:
        async with stt_model_lifecycle.use_model(model_config) as backend:
            await backend.load_model(model_config.model_size)
            if not getattr(backend, "supports_live_streaming", lambda: False)():
                await websocket.send_json(
                    {
                        "type": "unsupported",
                        "reason": (
                            f"{model_config.display_name} does not expose a true "
                            "microphone-time stream"
                        ),
                    }
                )
                return

            live_stream = await loop.run_in_executor(
                executor,
                backend.open_live_stream,
                language,
            )
            await websocket.send_json(
                {
                    "type": "ready",
                    "operation_id": operation_id,
                    "model": model_config.model_name,
                    "sample_rate": LIVE_SAMPLE_RATE,
                }
            )

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))

                payload = message.get("bytes")
                if payload is not None:
                    if not payload or len(payload) > MAX_LIVE_CHUNK_BYTES:
                        raise ValueError("Invalid live PCM chunk size")
                    if len(payload) % 4:
                        raise ValueError("Live PCM chunks must contain float32 samples")
                    pcm = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=True)
                    chunks.append(pcm)
                    update = await loop.run_in_executor(
                        executor,
                        live_stream.feed,
                        pcm,
                    )
                    if update["changed"]:
                        await websocket.send_json({"type": "partial", **update})
                    continue

                text = message.get("text")
                if text is None:
                    continue
                control = json.loads(text)
                kind = control.get("type")
                if kind == "cancel":
                    live_stream.cancel()
                    await websocket.send_json({"type": "cancelled"})
                    return
                if kind != "stop":
                    continue

                final = await loop.run_in_executor(executor, live_stream.finalize)
                pcm = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
                db = _capture_session()
                try:
                    capture = captures_service.persist_live_capture(
                        pcm=pcm,
                        language=language,
                        transcript=final["full"],
                        stt_model=model_config.model_name,
                        db=db,
                    )
                finally:
                    db.close()
                response = models.CaptureCreateResponse(
                    **capture.model_dump(),
                    auto_refine=auto_refine,
                    allow_auto_paste=allow_auto_paste,
                )
                await websocket.send_json(
                    {
                        "type": "final",
                        "capture": response.model_dump(mode="json"),
                        **final,
                    }
                )
                return
    finally:
        if live_stream is not None:
            live_stream.cancel()
            try:
                await loop.run_in_executor(executor, live_stream.close)
            except Exception:
                logger.exception("Failed to close live dictation stream")
        executor.shutdown(wait=False, cancel_futures=True)


@router.websocket("/captures/live")
async def live_capture_endpoint(websocket: WebSocket):
    """Stream 16 kHz mono float32 PCM and return partial text while speaking."""
    await websocket.accept()
    operation_id = ""
    try:
        init = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if init.get("type") != "start":
            raise ValueError("The first live-capture message must be 'start'")
        operation_id = str(init.get("operation_id") or uuid.uuid4().hex).strip()
        if not operation_id:
            raise ValueError("Live capture operation ID is required")

        db = _capture_session()
        try:
            saved = settings_service.get_capture_settings(db)
            model_config = resolve_stt_config(init.get("stt_model") or saved.stt_model)
            requested_language = init.get("language")
            transcribe_language = None if saved.language == "auto" else saved.language
            if requested_language is not None:
                transcribe_language = (
                    None if requested_language == "auto" else str(requested_language)
                )
            resolved_language = transcribe_service.resolve_stt_language(
                model_config,
                transcribe_language,
            )
            language = None if resolved_language == "auto" else resolved_language
            auto_refine = bool(saved.auto_refine)
            allow_auto_paste = bool(saved.allow_auto_paste)
        finally:
            db.close()

        if model_config.engine != "transcribe_cpp" or "streaming" not in set(
            model_config.capabilities
        ):
            await websocket.send_json(
                {
                    "type": "unsupported",
                    "reason": (
                        f"{model_config.display_name} uses the standard completed-audio "
                        "dictation path"
                    ),
                }
            )
            return
        if not is_model_config_cached(model_config):
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"{model_config.display_name} is not downloaded",
                }
            )
            return

        await task_queue.run_capture_operation(
            operation_id,
            _run_live_capture(
                websocket,
                operation_id=operation_id,
                model_config=model_config,
                language=language,
                auto_refine=auto_refine,
                allow_auto_paste=allow_auto_paste,
            ),
        )
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        try:
            await websocket.send_json({"type": "cancelled"})
        except Exception:
            pass
        raise
    except Exception as exc:
        logger.exception("Live dictation %s failed", operation_id or "uninitialized")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.post("/capture/warm")
async def warm_capture_model_endpoint(db: Session = Depends(get_db)):
    """Warm the selected STT model while the user is still recording."""
    from ..services.model_lifecycle import stt_model_lifecycle

    saved = settings_service.get_capture_settings(db)
    try:
        config_model = resolve_stt_config(saved.stt_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not is_model_config_cached(config_model):
        raise HTTPException(
            status_code=409,
            detail=f"{config_model.display_name} is not downloaded",
        )

    try:
        await stt_model_lifecycle.warm_model(config_model)
    except Exception as exc:
        logger.exception("Failed to warm dictation model %s", config_model.model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "model_name": config_model.model_name,
        "display_name": config_model.display_name,
        "loaded": True,
    }


@router.post("/captures", response_model=models.CaptureCreateResponse)
async def create_capture_endpoint(
    file: UploadFile = File(...),
    source: str = Form("file"),
    language: str | None = Form(None),
    stt_model: str | None = Form(None),
    operation_id: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Upload audio, run STT, persist the capture."""
    chunks = []
    while chunk := await file.read(UPLOAD_CHUNK_SIZE):
        chunks.append(chunk)
    audio_bytes = b"".join(chunks)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    saved = settings_service.get_capture_settings(db)
    resolved_stt = stt_model or saved.stt_model
    if language is None:
        resolved_language = None if saved.language == "auto" else saved.language
    else:
        resolved_language = None if language == "auto" else language

    resolved_operation_id = (operation_id or uuid.uuid4().hex).strip()
    try:
        capture = await task_queue.run_capture_operation(
            resolved_operation_id,
            captures_service.create_capture(
                audio_bytes=audio_bytes,
                filename=file.filename or "capture.wav",
                source=source,
                language=resolved_language,
                stt_model=resolved_stt,
                db=db,
                should_stop=lambda: task_queue.is_capture_cancel_requested(
                    resolved_operation_id
                ),
            ),
        )
    except asyncio.CancelledError as exc:
        raise HTTPException(status_code=409, detail="Capture cancelled") from exc
    except (ValueError, MediaIngestionError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create capture")
        raise HTTPException(status_code=500, detail=str(e))

    return models.CaptureCreateResponse(
        **capture.model_dump(),
        auto_refine=bool(saved.auto_refine),
        allow_auto_paste=bool(saved.allow_auto_paste),
    )


@router.post("/captures/operations/{operation_id}/cancel")
async def cancel_capture_operation_endpoint(operation_id: str):
    """Cancel the active dictate/transcribe/refine operation behind the tray."""
    return {
        "operation_id": operation_id,
        "cancelled": task_queue.cancel_capture_operation(operation_id),
    }


@router.get("/captures", response_model=models.CaptureListResponse)
async def list_captures_endpoint(
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    db: Session = Depends(get_db),
):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    if search is not None and len(search) > 200:
        raise HTTPException(status_code=400, detail="search must be 200 characters or fewer")

    items, total = captures_service.list_captures(
        db, limit=limit, offset=offset, search=search
    )
    return models.CaptureListResponse(items=items, total=total)


@router.get("/captures/{capture_id}", response_model=models.CaptureResponse)
async def get_capture_endpoint(capture_id: str, db: Session = Depends(get_db)):
    capture = captures_service.get_capture(capture_id, db)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture


@router.get("/captures/{capture_id}/audio")
async def get_capture_audio_endpoint(capture_id: str, db: Session = Depends(get_db)):
    """Stream the original capture audio file."""
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Capture not found")

    audio_path = config.resolve_storage_path(row.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        audio_path,
        media_type=mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream",
        filename=(audio_path.name.split("__", 1)[1] if "__" in audio_path.name else audio_path.name),
    )


@router.delete("/captures/{capture_id}")
async def delete_capture_endpoint(capture_id: str, db: Session = Depends(get_db)):
    deleted = captures_service.delete_capture(capture_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Capture not found")
    return {"message": f"Capture {capture_id} deleted"}


@router.post("/captures/{capture_id}/refine", response_model=models.CaptureResponse)
async def refine_capture_endpoint(
    capture_id: str,
    request: models.CaptureRefineRequest,
    operation_id: str | None = Header(None, alias="X-Diarix-Operation-ID"),
    db: Session = Depends(get_db),
):
    saved = settings_service.get_capture_settings(db)
    if request.flags is not None:
        flags = RefinementFlags(
            smart_cleanup=request.flags.smart_cleanup,
            self_correction=request.flags.self_correction,
            preserve_technical=request.flags.preserve_technical,
            custom_instructions=request.flags.custom_instructions,
        )
    else:
        flags = RefinementFlags(
            smart_cleanup=saved.smart_cleanup,
            self_correction=saved.self_correction,
            preserve_technical=saved.preserve_technical,
            custom_instructions=saved.custom_instructions,
        )

    resolved_model = request.model_size or saved.llm_model
    resolved_operation_id = (operation_id or uuid.uuid4().hex).strip()

    try:
        capture = await task_queue.run_capture_operation(
            resolved_operation_id,
            captures_service.refine_capture(
                capture_id=capture_id,
                flags=flags,
                model_size=resolved_model,
                db=db,
            ),
        )
    except asyncio.CancelledError as exc:
        raise HTTPException(status_code=409, detail="Refinement cancelled") from exc
    except Exception as e:
        logger.exception("Refinement failed for capture %s", capture_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture


@router.get("/capture/readiness", response_model=models.CaptureReadinessResponse)
async def capture_readiness_endpoint(db: Session = Depends(get_db)):
    """Whether the STT and LLM models the user has selected are downloaded.

    The frontend gates the global hotkey on this — pressing the chord with
    a missing model would otherwise produce a stuck "transcribing" pill that
    waits forever for a download to finish. Checks on-disk cache, not RAM
    load, so the answer survives backend restarts.
    """
    saved = settings_service.get_capture_settings(db)

    try:
        stt_cfg = resolve_stt_config(saved.stt_model)
    except ValueError:
        stt_cfg = None
    llm_cfg = next(
        (c for c in get_llm_model_configs() if c.model_size == saved.llm_model),
        None,
    )

    if stt_cfg is None or llm_cfg is None:
        # Should be impossible — both fields are pattern-validated against
        # known sizes — but bail loudly rather than return half a response.
        raise HTTPException(
            status_code=500,
            detail=f"No model config for stt={saved.stt_model} or llm={saved.llm_model}",
        )

    return models.CaptureReadinessResponse(
        stt=models.ModelReadiness(
            ready=is_model_config_cached(stt_cfg),
            model_name=stt_cfg.model_name,
            display_name=stt_cfg.display_name,
            size=stt_cfg.model_size,
            size_mb=stt_cfg.size_mb or None,
            live_supported=(
                stt_cfg.engine == "transcribe_cpp"
                and "streaming" in set(stt_cfg.capabilities)
            ),
        ),
        llm=models.ModelReadiness(
            ready=is_model_cached(llm_cfg.hf_repo_id),
            model_name=llm_cfg.model_name,
            display_name=llm_cfg.display_name,
            size=llm_cfg.model_size,
            size_mb=llm_cfg.size_mb or None,
        ),
    )


@router.post("/captures/{capture_id}/retranscribe", response_model=models.CaptureResponse)
async def retranscribe_capture_endpoint(
    capture_id: str,
    request: models.CaptureRetranscribeRequest,
    db: Session = Depends(get_db),
):
    saved = settings_service.get_capture_settings(db)
    resolved_stt = request.model or saved.stt_model
    if request.language is None:
        resolved_language = None if saved.language == "auto" else saved.language
    else:
        resolved_language = request.language

    try:
        capture = await captures_service.retranscribe_capture(
            capture_id=capture_id,
            stt_model=resolved_stt,
            language=resolved_language,
            db=db,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=410, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Retranscribe failed for capture %s", capture_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture
