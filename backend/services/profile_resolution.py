"""Resolve the voice profile used by local REST playback."""

from sqlalchemy.orm import Session

from ..database import VoiceProfile as DBVoiceProfile
from ..database.models import CaptureSettings
from .profiles import get_profile_orm_by_name_or_id as _lookup_profile


def resolve_profile(explicit: str | None, db: Session) -> DBVoiceProfile | None:
    """Resolve an explicit profile, then the user's global playback default."""
    if explicit:
        return _lookup_profile(explicit, db)

    settings = db.query(CaptureSettings).filter(CaptureSettings.id == 1).first()
    if settings and settings.default_playback_voice_id:
        return _lookup_profile(settings.default_playback_voice_id, db)
    return None
