"""Model-free tests for the batch transcription job contract."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.backends import resolve_stt_config
from backend.services import transcription_jobs
from backend.services.media_ingestion import create_media_job_dir
from backend.utils.tasks import TaskManager


def test_job_options_accept_default_without_claiming_adapter_precision() -> None:
    config, precision = transcription_jobs.resolve_job_options("whisper-base", "default")
    assert config.model_name == "whisper-base"
    assert precision == "default"


def test_job_options_reject_unknown_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        transcription_jobs.resolve_job_options("whisper-base", "int4")


def test_job_language_rejects_model_unsupported_language() -> None:
    config = resolve_stt_config("faster-distil-whisper-large-v3")
    with pytest.raises(ValueError, match="not supported"):
        transcription_jobs.resolve_job_language(config, "fr")


def test_job_language_requires_explicit_choice_without_detection() -> None:
    config = resolve_stt_config("nvidia-canary-180m-flash")
    with pytest.raises(ValueError, match="requires an explicit language"):
        transcription_jobs.resolve_job_language(config, "auto")


def test_job_language_keeps_auto_for_whisper_detection() -> None:
    config = resolve_stt_config("whisper-turbo")
    assert transcription_jobs.resolve_job_language(config, "auto") == "auto"


def test_task_manager_returns_detached_job_snapshots() -> None:
    manager = TaskManager()
    created = manager.start_transcription(
        task_id="task-1",
        model_name="whisper-base",
        language="auto",
        precision="default",
        output_dir="C:/transcripts",
        total_files=2,
        work_dir="C:/cache/task-1",
    )
    created.status = "tampered"
    assert manager.get_transcription_task("task-1").status == "queued"


@pytest.mark.asyncio
async def test_batch_runs_sequentially_writes_results_and_cleans_work_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DIARIX_MEDIA_CACHE_DIR", str(tmp_path / "jobs"))
    work_dir = create_media_job_dir("batch")
    first = work_dir / "0000-first.mp4"
    second = work_dir / "0001-second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    manager = TaskManager()
    monkeypatch.setattr(transcription_jobs, "get_task_manager", lambda: manager)
    progress_events: list[dict] = []
    monkeypatch.setattr(
        transcription_jobs,
        "publish_task",
        lambda task: progress_events.append(transcription_jobs.task_to_public_dict(task)),
    )
    manager.start_transcription(
        task_id="batch",
        model_name="whisper-base",
        language="auto",
        precision="default",
        output_dir=str(output_dir),
        total_files=2,
        work_dir=str(work_dir),
    )

    class FakeBackend:
        async def load_model(self, _model_size: str) -> None:
            return None

    monkeypatch.setattr(
        transcription_jobs.transcribe,
        "get_stt_model",
        lambda _model: (FakeBackend(), object()),
    )
    transcribed: list[str] = []

    async def fake_transcribe(
        path: str,
        model: str,
        language: str | None,
        progress_callback=None,
        should_stop=None,
        partial_callback=None,
        segments_callback=None,
    ):
        assert language is None
        assert should_stop is not None
        assert not should_stop()
        transcribed.append(Path(path).name)
        progress_callback(0.25)
        progress_callback(0.5)
        progress_callback(0.5)  # Duplicate callbacks must not move the task.
        progress_callback(1.0)
        partial_callback(f"partial text for {Path(path).name}")
        segments_callback([])
        return f"text for {Path(path).name}", model

    monkeypatch.setattr(
        transcription_jobs.transcribe, "transcribe_audio", fake_transcribe
    )

    @asynccontextmanager
    async def fake_ingest(source_path, _spec, **_kwargs):
        yield SimpleNamespace(
            audio_path=Path(source_path),
            duration=1.25,
        )

    monkeypatch.setattr(transcription_jobs, "ingest_media", fake_ingest)
    retained: list[str] = []
    monkeypatch.setattr(
        transcription_jobs.captures,
        "persist_completed_transcription",
        lambda **kwargs: retained.append(kwargs["filename"]) or "capture-id",
    )

    config = resolve_stt_config("whisper-base")
    await transcription_jobs.run_transcription_job(
        task_id="batch",
        files=[
            transcription_jobs.PendingMedia(first, "first.mp4"),
            transcription_jobs.PendingMedia(second, "second.wav"),
        ],
        model_config=config,
        language="auto",
        precision="default",
        output_suffix="_transcript",
        output_dir=output_dir,
        work_dir=work_dir,
    )

    finished = manager.get_transcription_task("batch")
    assert finished.status == "completed"
    assert finished.completed_files == 2
    assert finished.progress == 100.0
    assert [result.filename for result in finished.results] == ["first.mp4", "second.wav"]
    assert transcribed == ["0000-first.mp4", "0001-second.wav"]
    assert retained == ["first.mp4", "second.wav"]
    assert (output_dir / "first_transcript.txt").read_text(encoding="utf-8")
    assert (output_dir / "second_transcript.txt").read_text(encoding="utf-8")
    assert not work_dir.exists()
    assert progress_events[-1]["stage"] == "completed"
    transcribing_progress = [
        event["progress"]
        for event in progress_events
        if event["stage"] == "transcribing"
    ]
    # Partial-text events intentionally repeat the current percentage while
    # replacing the live transcript. Progress must remain monotonic, but it
    # does not need to be unique across those content-only updates.
    assert transcribing_progress == sorted(transcribing_progress)
    assert any(event["partial_text"] for event in progress_events)
    assert any(17.5 < value < 45.0 for value in transcribing_progress)


@pytest.mark.asyncio
async def test_transcription_cancel_while_waiting_for_inference_lock_cleans_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from backend.services import task_queue
    from backend.utils.tasks import get_task_manager

    task_queue.init_queue(force=True)
    monkeypatch.setenv("DIARIX_MEDIA_CACHE_DIR", str(tmp_path / "jobs"))
    manager = get_task_manager()
    manager.clear_all()
    running_started = asyncio.Event()
    release_running = asyncio.Event()
    queued_ran = asyncio.Event()

    async def running_generation():
        running_started.set()
        await release_running.wait()

    async def queued_transcription():
        queued_ran.set()

    task_queue.enqueue_generation("holds-inference", running_generation())
    await asyncio.wait_for(running_started.wait(), timeout=1)
    work_dir = create_media_job_dir("waiting-for-lock")
    manager.start_transcription(
        task_id="queued-stt",
        model_name="whisper-base",
        language="auto",
        precision="default",
        output_dir=str(tmp_path),
        total_files=1,
        work_dir=str(work_dir),
    )
    task_queue.enqueue_transcription("queued-stt", queued_transcription())
    await asyncio.sleep(0.05)
    assert task_queue.cancel_transcription("queued-stt") == "running"
    for _ in range(50):
        if manager.get_transcription_task("queued-stt").status == "cancelled":
            break
        await asyncio.sleep(0.01)
    assert manager.get_transcription_task("queued-stt").status == "cancelled"
    assert not work_dir.exists()
    release_running.set()
    await asyncio.sleep(0.1)
    assert not queued_ran.is_set()
