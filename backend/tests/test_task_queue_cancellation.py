import asyncio

import pytest

from backend.services import task_queue


@pytest.mark.asyncio
async def test_cancel_queued_generation_skips_execution():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    release_running = asyncio.Event()
    queued_ran = asyncio.Event()

    async def running_job():
        running_started.set()
        await release_running.wait()

    async def queued_job():
        queued_ran.set()

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    task_queue.enqueue_generation("gen-queued", queued_job())
    assert task_queue.cancel_generation("gen-queued") == "queued"

    release_running.set()
    await asyncio.sleep(0.1)

    assert not queued_ran.is_set()


@pytest.mark.asyncio
async def test_cancel_running_generation_cancels_task():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    running_cancelled = asyncio.Event()

    async def running_job():
        running_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            running_cancelled.set()
            raise

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    assert task_queue.cancel_generation("gen-running") == "running"
    await asyncio.wait_for(running_cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_cancel_capture_operation_sets_cooperative_stop_and_cleans_registry():
    task_queue.init_queue(force=True)

    operation_id = "dictation-cancel"
    started = asyncio.Event()
    stop_observed = asyncio.Event()

    async def native_inference():
        started.set()
        while not task_queue.is_capture_cancel_requested(operation_id):
            await asyncio.sleep(0)
        stop_observed.set()

    async def capture_operation():
        inference = asyncio.create_task(native_inference())
        try:
            await asyncio.shield(inference)
        except asyncio.CancelledError:
            await inference
            raise

    running = asyncio.create_task(
        task_queue.run_capture_operation(operation_id, capture_operation())
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    assert task_queue.cancel_capture_operation(operation_id) is True
    assert task_queue.is_capture_cancel_requested(operation_id) is True
    await asyncio.wait_for(stop_observed.wait(), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await running

    assert task_queue.cancel_capture_operation(operation_id) is False
