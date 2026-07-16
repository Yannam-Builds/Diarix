"""
Captures service — persists raw audio alongside its STT transcript and,
optionally, an LLM-refined version.

A capture is a single voice input event (dictation, long-form recording, or
uploaded file). Storage mirrors the generations flow: audio lives under
``data/captures/<id>.wav`` and rows live in the ``captures`` table.
"""

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from .. import config
from ..backends import resolve_stt_config
from ..database import Capture as DBCapture
from ..models import CaptureResponse, RefinementFlagsModel
from .refinement import RefinementFlags, refine_transcript
from . import transcribe
from .media_ingestion import ingest_media, sanitize_upload_filename

logger = logging.getLogger(__name__)


VALID_SOURCES = {"dictation", "recording", "file"}


def _to_response(row: DBCapture) -> CaptureResponse:
    flags_model: Optional[RefinementFlagsModel] = None
    if row.refinement_flags:
        try:
            flags_model = RefinementFlagsModel(**json.loads(row.refinement_flags))
        except (ValueError, TypeError):
            flags_model = None

    return CaptureResponse(
        id=row.id,
        audio_path=row.audio_path,
        source=row.source,
        language=row.language,
        duration_ms=row.duration_ms,
        transcript_raw=row.transcript_raw or "",
        transcript_refined=row.transcript_refined,
        stt_model=row.stt_model,
        llm_model=row.llm_model,
        refinement_flags=flags_model,
        created_at=row.created_at,
    )


async def create_capture(
    *,
    audio_bytes: bytes,
    filename: str,
    source: str,
    language: Optional[str],
    stt_model: Optional[str],
    db: Session,
) -> CaptureResponse:
    """Persist raw audio, run STT, store the row."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source '{source}'. Must be one of {sorted(VALID_SOURCES)}")

    capture_id = str(uuid.uuid4())
    safe_name = sanitize_upload_filename(filename, "capture.media")[:180]
    raw_path = config.get_captures_dir() / f"{capture_id}__{safe_name}"
    written_files: list[Path] = []

    try:
        raw_path.write_bytes(audio_bytes)
        written_files.append(raw_path)

        model_config = resolve_stt_config(stt_model)
        async with ingest_media(raw_path, model_config.audio_input) as media:
            duration_ms = (
                max(0, round(media.duration * 1000)) if media.duration is not None else None
            )
            transcript, resolved_stt = await transcribe.transcribe_audio(
                str(media.audio_path), model_config.model_name, language
            )

        row = DBCapture(
            id=capture_id,
            audio_path=config.to_storage_path(raw_path),
            source=source,
            language=language,
            duration_ms=duration_ms,
            transcript_raw=transcript,
            stt_model=resolved_stt,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception:
        # Anything between the first write and the commit means the audio on
        # disk has no row pointing at it — clean up so data/captures doesn't
        # accumulate orphan blobs across failed transcribes.
        for path in written_files:
            try:
                path.unlink()
            except OSError:
                pass
        raise

    return _to_response(row)


def list_captures(
    db: Session, limit: int = 50, offset: int = 0, search: str | None = None
) -> tuple[list[CaptureResponse], int]:
    query = db.query(DBCapture)
    if search and search.strip():
        # Case-insensitive substring match over both transcript columns.
        # SQLite LIKE is already case-insensitive for ASCII; ilike() keeps
        # the intent explicit and portable. Escape LIKE wildcards so a
        # user searching for "100%" matches literally.
        term = search.strip().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
        pattern = f"%{term}%"
        query = query.filter(
            DBCapture.transcript_raw.ilike(pattern, escape="\\")
            | DBCapture.transcript_refined.ilike(pattern, escape="\\")
        )
    total = query.count()
    rows = (
        query.order_by(DBCapture.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    return [_to_response(r) for r in rows], total


def get_capture(capture_id: str, db: Session) -> Optional[CaptureResponse]:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    return _to_response(row) if row else None


def delete_capture(capture_id: str, db: Session) -> bool:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return False

    resolved = config.resolve_storage_path(row.audio_path)
    if resolved and resolved.exists():
        try:
            resolved.unlink()
        except OSError:
            logger.exception("Failed to remove capture audio %s", resolved)

    db.delete(row)
    db.commit()
    return True


async def refine_capture(
    capture_id: str,
    flags: RefinementFlags,
    model_size: Optional[str],
    db: Session,
) -> Optional[CaptureResponse]:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return None

    refined, llm_size = await refine_transcript(
        row.transcript_raw or "",
        flags,
        model_size=model_size,
    )

    row.transcript_refined = refined
    row.llm_model = llm_size
    row.refinement_flags = json.dumps(flags.to_dict())
    db.commit()
    db.refresh(row)
    return _to_response(row)


async def retranscribe_capture(
    capture_id: str,
    stt_model: Optional[str],
    language: Optional[str],
    db: Session,
) -> Optional[CaptureResponse]:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return None

    resolved = config.resolve_storage_path(row.audio_path)
    if not resolved or not resolved.exists():
        raise FileNotFoundError(f"Audio for capture {capture_id} is missing")

    model_config = resolve_stt_config(stt_model or row.stt_model)
    async with ingest_media(resolved, model_config.audio_input) as media:
        transcript, resolved_stt = await transcribe.transcribe_audio(
            str(media.audio_path), model_config.model_name, language
        )
        if media.duration is not None:
            row.duration_ms = max(0, round(media.duration * 1000))

    row.transcript_raw = transcript
    row.stt_model = resolved_stt
    if language:
        row.language = language
    # Refined text is stale after a fresh STT pass — force a re-refine.
    row.transcript_refined = None
    row.llm_model = None
    row.refinement_flags = None
    db.commit()
    db.refresh(row)
    return _to_response(row)


def persist_completed_transcription(
    *,
    source_path: str | Path,
    filename: str,
    language: Optional[str],
    duration: float,
    transcript: str,
    stt_model: str,
) -> str:
    """Retain uploaded media and surface a completed job in capture history."""
    # Import the session module, not the package-level SessionLocal re-export.
    # The re-export is bound to None when the database package is first imported,
    # before init_db() installs the real session factory.
    from ..database import session as database_session

    capture_id = str(uuid.uuid4())
    safe_name = sanitize_upload_filename(filename)[:180]
    retained_path = config.get_captures_dir() / f"{capture_id}__{safe_name}"
    session_factory = database_session.SessionLocal
    if session_factory is None:
        raise RuntimeError("Database session factory is not initialized.")
    db = session_factory()
    try:
        shutil.copy2(Path(source_path), retained_path)
        row = DBCapture(
            id=capture_id,
            audio_path=config.to_storage_path(retained_path),
            source="file",
            language=None if language == "auto" else language,
            duration_ms=max(0, round(duration * 1000)),
            transcript_raw=transcript,
            stt_model=stt_model,
        )
        db.add(row)
        db.commit()
        return capture_id
    except Exception:
        db.rollback()
        retained_path.unlink(missing_ok=True)
        raise
    finally:
        db.close()
