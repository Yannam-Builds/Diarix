"""Regression coverage for prompt cancellation of running transcription jobs.

Guards against the bug where asyncio.shield() in transcribe.await_stt_operation
let a cancelled batch job keep running every remaining audio chunk (and
therefore keep the model loaded) before the task reported itself cancelled.
"""

import asyncio

import pytest

from backend.backends.stt.common import stop_requested
from backend.services import task_queue


def test_stop_requested_is_false_without_a_callback() -> None:
    assert stop_requested(None) is False


def test_stop_requested_reflects_the_callback() -> None:
    assert stop_requested(lambda: False) is False
    assert stop_requested(lambda: True) is True


@pytest.mark.asyncio
async def test_cancel_requested_flips_immediately_on_a_running_task() -> None:
    """cancel_transcription() must mark the task as cancel-requested before
    the coroutine gets a chance to observe a CancelledError, since STT
    adapters poll is_transcription_cancel_requested() cooperatively between
    chunks rather than relying on exception delivery."""
    task_queue.init_queue(force=True)

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_job() -> None:
        started.set()
        await release.wait()

    task_id = "cancel-flag-test"
    task_queue.enqueue_transcription(task_id, slow_job())
    await asyncio.wait_for(started.wait(), timeout=2)

    assert task_queue.is_transcription_cancel_requested(task_id) is False

    outcome = task_queue.cancel_transcription(task_id)
    assert outcome == "running"
    assert task_queue.is_transcription_cancel_requested(task_id) is True

    release.set()
    await asyncio.sleep(0)
