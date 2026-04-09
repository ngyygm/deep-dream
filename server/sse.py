"""SSE (Server-Sent Events) utilities for streaming responses."""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Generator

from flask import Response


def sse_event(event_type: str, data: Any) -> str:
    """Format a single SSE event.

    Args:
        event_type: The event name (e.g. 'thought', 'tool_call', 'done').
        data: Payload — will be JSON-encoded if not a string.

    Returns:
        A formatted SSE string with ``event:`` and ``data:`` lines.
    """
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"event: {event_type}\ndata: {payload}\n\n"


def sse_response(gen: Generator) -> Response:
    """Wrap a generator as a Flask SSE streaming response.

    Sets appropriate headers:
    - ``Content-Type: text/event-stream``
    - ``Cache-Control: no-cache``
    - ``X-Accel-Buffering: no`` (for nginx)
    """
    return Response(
        gen,
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def queue_to_generator(q: queue.Queue, sentinel=None) -> Generator[str, None, None]:
    """Drain a thread-safe queue, yielding SSE strings until *sentinel* is received."""
    while True:
        try:
            item = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is sentinel:
            break
        yield item
    yield sse_event("done", {"status": "completed"})

