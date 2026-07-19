"""POST /speak — local REST speech-generation endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models
from ..database import get_db
from ..services import speak_events
from ..services.profile_resolution import resolve_profile


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/speak", response_model=models.GenerationResponse)
async def speak(
    data: models.SpeakRequest,
    db: Session = Depends(get_db),
):
    """Generate speech with an explicit or globally configured profile.

    Response shape matches POST /generate — a ``GenerationResponse`` with
    ``status="generating"`` and an ``id`` the caller polls at
    ``GET /generate/{id}/status``.
    """
    profile = resolve_profile(data.profile, db)
    if profile is None:
        if data.profile:
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile '{data.profile}' not found.",
            )
        raise HTTPException(
            status_code=400,
            detail=(
                "No voice profile resolved. Pass `profile` (name or id), "
                "or configure a default playback voice in Diarix settings."
            ),
        )

    from .generations import generate_speech

    generation = await generate_speech(
        models.GenerationRequest(
            profile_id=profile.id,
            text=data.text,
            language=data.language or "en",
            engine=data.engine,
            personality=bool(data.personality),
        ),
        db,
    )

    speak_events.publish(
        "speak-start",
        {
            "generation_id": getattr(generation, "id", None),
            "profile_name": profile.name,
            "source": "rest",
        },
    )
    return generation
