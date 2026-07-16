"""
Serial generation queue — ensures only one TTS inference runs at a time
to avoid GPU contention.
"""

import asyncio
import traceback
from dataclasses import dataclass
from typing import Coroutine, Literal

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()


@dataclass
class GenerationJob:
    """Queued generation work plus the generation ID it belongs to."""

    generation_id: str
    coro: Coroutine


@dataclass
class TranscriptionJob:
    """Queued transcription work plus its public task ID."""

    task_id: str
    coro: Coroutine


# Generation queue — serializes TTS inference to avoid GPU contention
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
_generation_worker_task: asyncio.Task | None = None
_queued_generation_ids: set[str] = set()
_running_generation_tasks: dict[str, asyncio.Task] = {}
_cancelled_generation_ids: set[str] = set()

_transcription_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
_transcription_worker_task: asyncio.Task | None = None
_queued_transcription_ids: set[str] = set()
_running_transcription_tasks: dict[str, asyncio.Task] = {}
_cancelled_transcription_ids: set[str] = set()
_inference_lock: asyncio.Lock = None  # type: ignore  # initialized at startup


def create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _run_with_inference_lock(coro: Coroutine) -> None:
    """Run one inference coroutine and close it if cancelled before entry."""
    entered = False
    try:
        async with _inference_lock:
            entered = True
            await coro
    finally:
        if not entered:
            coro.close()


async def _generation_worker():
    """Worker that processes generation tasks one at a time."""
    while True:
        job = await _generation_queue.get()
        try:
            if job.generation_id in _cancelled_generation_ids:
                _cancelled_generation_ids.discard(job.generation_id)
                job.coro.close()
                continue

            task = asyncio.create_task(_run_with_inference_lock(job.coro))
            _running_generation_tasks[job.generation_id] = task
            _queued_generation_ids.discard(job.generation_id)
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        except Exception:
            traceback.print_exc()
            await _force_fail_if_active(
                job.generation_id,
                "Worker exited without writing terminal status",
            )
        finally:
            _running_generation_tasks.pop(job.generation_id, None)
            _queued_generation_ids.discard(job.generation_id)
            _generation_queue.task_done()


async def _transcription_worker():
    """Process transcription batches through the shared inference lock."""
    while True:
        job = await _transcription_queue.get()
        try:
            if job.task_id in _cancelled_transcription_ids:
                _cancelled_transcription_ids.discard(job.task_id)
                job.coro.close()
                continue

            task = asyncio.create_task(_run_with_inference_lock(job.coro))
            _running_transcription_tasks[job.task_id] = task
            _queued_transcription_ids.discard(job.task_id)
            try:
                await task
            except asyncio.CancelledError:
                await _force_cancel_transcription_if_active(job.task_id)
                if not task.cancelled():
                    raise
        except Exception as exc:
            traceback.print_exc()
            await _force_fail_transcription_if_active(job.task_id, str(exc))
        finally:
            _running_transcription_tasks.pop(job.task_id, None)
            _queued_transcription_ids.discard(job.task_id)
            _transcription_queue.task_done()


async def _force_fail_transcription_if_active(task_id: str, error: str) -> None:
    """Best-effort terminal state if a batch escapes its own error boundary."""
    try:
        from ..utils.progress import get_progress_manager
        from ..utils.tasks import get_task_manager

        manager = get_task_manager()
        task = manager.get_transcription_task(task_id)
        if task is None or task.status in {"completed", "failed", "cancelled"}:
            return
        snapshot = manager.update_transcription(
            task_id, status="failed", stage="failed", error=error
        )
        if snapshot is not None:
            get_progress_manager().update_task_progress(
                task_id,
                status=snapshot.status,
                progress=snapshot.progress,
                stage=snapshot.stage,
                error=snapshot.error,
            )
    except Exception:
        traceback.print_exc()


async def _force_cancel_transcription_if_active(task_id: str) -> None:
    """Finalize cancellation when work was stopped before its coroutine started."""
    try:
        from .media_ingestion import cleanup_media_job_dir
        from .transcription_jobs import publish_task
        from ..utils.tasks import get_task_manager

        manager = get_task_manager()
        task = manager.get_transcription_task(task_id)
        if task is None or task.status in {"completed", "failed", "cancelled"}:
            return
        snapshot = manager.update_transcription(
            task_id,
            status="cancelled",
            stage="cancelled",
            current_file=None,
            error=None,
        )
        if snapshot is not None:
            publish_task(snapshot)
            if snapshot.work_dir:
                cleanup_media_job_dir(snapshot.work_dir)
    except Exception:
        traceback.print_exc()


async def _force_fail_if_active(generation_id: str, error: str) -> None:
    """Best-effort recovery — flip an active row to failed if the worker
    bailed before writing a terminal status. Catches the case where the gen
    coroutine's own status-write raised (e.g. SQLite lock contention)."""
    try:
        from ..database import Generation as DBGeneration, get_db
        from . import history

        db = next(get_db())
        try:
            gen = db.query(DBGeneration).filter_by(id=generation_id).first()
            if gen is None:
                return
            if (gen.status or "completed") not in ("loading_model", "generating"):
                return
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=db,
                error=error,
            )
        finally:
            db.close()
    except Exception:
        traceback.print_exc()


def enqueue_generation(generation_id: str, coro):
    """Add a generation coroutine to the serial queue."""
    if _generation_queue is None:
        raise RuntimeError("Generation queue has not been initialized")

    _queued_generation_ids.add(generation_id)
    _generation_queue.put_nowait(GenerationJob(generation_id=generation_id, coro=coro))


def cancel_generation(generation_id: str) -> Literal["queued", "running"] | None:
    """Cancel a queued or running generation if it is still active."""
    running_task = _running_generation_tasks.get(generation_id)
    if running_task is not None:
        running_task.cancel()
        return "running"

    if generation_id in _queued_generation_ids:
        _queued_generation_ids.discard(generation_id)
        _cancelled_generation_ids.add(generation_id)
        return "queued"

    return None


def enqueue_transcription(task_id: str, coro) -> None:
    """Add a batch transcription coroutine to the shared inference queue."""
    if _transcription_queue is None:
        raise RuntimeError("Transcription queue has not been initialized")
    _queued_transcription_ids.add(task_id)
    _transcription_queue.put_nowait(TranscriptionJob(task_id=task_id, coro=coro))


def cancel_transcription(task_id: str) -> Literal["queued", "running"] | None:
    """Cancel queued work immediately or signal a running batch task."""
    running_task = _running_transcription_tasks.get(task_id)
    if running_task is not None:
        running_task.cancel()
        return "running"
    if task_id in _queued_transcription_ids:
        _queued_transcription_ids.discard(task_id)
        _cancelled_transcription_ids.add(task_id)
        return "queued"
    return None


def is_transcription_cancel_requested(task_id: str) -> bool:
    """Whether a running batch's task has an outstanding cancellation request.

    STT adapters poll this between audio chunks so a stop request cuts the
    job short after the in-flight chunk instead of after the whole file —
    asyncio.shield() (see ``transcribe.await_stt_operation``) means the
    inference coroutine keeps running regardless of ``.cancel()``, so this
    cooperative check is what actually bounds how long a cancel takes to
    free the model.
    """
    task = _running_transcription_tasks.get(task_id)
    return task is not None and task.cancelling() > 0


def init_queue(force: bool = False):
    """Initialize the generation queue and start the worker.

    Must be called once during application startup (inside a running event loop).
    """
    global _generation_queue, _generation_worker_task
    global _queued_generation_ids, _running_generation_tasks, _cancelled_generation_ids
    global _transcription_queue, _transcription_worker_task
    global _queued_transcription_ids, _running_transcription_tasks, _cancelled_transcription_ids
    global _inference_lock

    if _generation_worker_task is not None and not _generation_worker_task.done():
        if not force:
            return
        _generation_worker_task.cancel()
        for task in list(_running_generation_tasks.values()):
            task.cancel()

    if _transcription_worker_task is not None and not _transcription_worker_task.done():
        if not force:
            return
        _transcription_worker_task.cancel()
        for task in list(_running_transcription_tasks.values()):
            task.cancel()

    _generation_queue = asyncio.Queue()
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _transcription_queue = asyncio.Queue()
    _queued_transcription_ids = set()
    _running_transcription_tasks = {}
    _cancelled_transcription_ids = set()
    _inference_lock = asyncio.Lock()
    _generation_worker_task = create_background_task(_generation_worker())
    _transcription_worker_task = create_background_task(_transcription_worker())
