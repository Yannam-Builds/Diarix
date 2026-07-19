"""In-memory pub/sub for speaking-pill SSE broadcasts."""

import asyncio
from typing import Any


_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue[dict[str, Any]]) -> None:
    _subscribers.discard(queue)


def publish(kind: str, payload: dict[str, Any]) -> None:
    """Fan out independent event objects without blocking a publisher."""
    for queue in list(_subscribers):
        try:
            queue.put_nowait({"kind": kind, **payload})
        except asyncio.QueueFull:
            pass
