"""Model-free tests for completed transcription persistence."""

from datetime import datetime
from pathlib import Path

import numpy as np

from backend.database import session as database_session
from backend.services import captures


def test_completed_transcription_uses_initialized_session_factory(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"media")
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()

    stored_rows = []

    class FakeSession:
        def add(self, row) -> None:
            stored_rows.append(row)

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            raise AssertionError("Persistence should not roll back")

        def close(self) -> None:
            return None

    monkeypatch.setattr(database_session, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(captures.config, "get_captures_dir", lambda: capture_dir)
    monkeypatch.setattr(captures.config, "to_storage_path", lambda path: str(path))

    capture_id = captures.persist_completed_transcription(
        source_path=source,
        filename="example.mp4",
        language="auto",
        duration=1.25,
        transcript="hello",
        stt_model="whisper-base",
    )

    retained = capture_dir / f"{capture_id}__example.mp4"
    assert retained.read_bytes() == b"media"
    assert len(stored_rows) == 1
    assert stored_rows[0].audio_path == str(retained)
    assert stored_rows[0].duration_ms == 1250
    assert stored_rows[0].language is None


def test_live_capture_persists_normalized_wav_without_second_inference(
    monkeypatch, tmp_path: Path
) -> None:
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    stored_rows = []

    class FakeSession:
        def add(self, row) -> None:
            stored_rows.append(row)

        def commit(self) -> None:
            return None

        def refresh(self, row) -> None:
            row.created_at = datetime(2026, 7, 16)

        def rollback(self) -> None:
            raise AssertionError("Persistence should not roll back")

    monkeypatch.setattr(captures.config, "get_captures_dir", lambda: capture_dir)
    monkeypatch.setattr(captures.config, "to_storage_path", lambda path: str(path))

    response = captures.persist_live_capture(
        pcm=np.linspace(-0.25, 0.25, 16_000, dtype=np.float32),
        language="en",
        transcript="live text",
        stt_model="moonshine-streaming-tiny-gguf",
        db=FakeSession(),
    )

    retained = Path(response.audio_path)
    assert retained.exists()
    assert retained.suffix == ".wav"
    assert response.duration_ms == 1000
    assert response.transcript_raw == "live text"
    assert response.stt_model == "moonshine-streaming-tiny-gguf"
    assert len(stored_rows) == 1
