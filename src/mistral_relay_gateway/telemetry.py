from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestEvent:
    id: int
    ts: float
    method: str
    path: str
    model: str
    mistral: bool
    route_mode: str  # chat, responses_native, responses_convert, fallback
    stream: bool
    status: int
    latency_ms: int
    body_kb: int
    images_total: int = 0
    images_dropped: int = 0
    stripped_fields: list[str] = field(default_factory=list)
    normalized_chunks: int = 0
    account: str = ""
    error: str = ""


class TelemetryStore:
    """Thread/event-loop-safe in-memory ring buffer for the built-in Web Console."""

    def __init__(self, max_events: int = 200):
        self._events: deque[RequestEvent] = deque(maxlen=max_events)
        self._seq: int = 0
        self.started_at: float = time.time()
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_images_dropped: int = 0
        self.total_fields_stripped: int = 0
        self.total_normalized_chunks: int = 0

    def record(
        self,
        *,
        method: str,
        path: str,
        model: str,
        mistral: bool,
        route_mode: str,
        stream: bool,
        status: int,
        latency_ms: int,
        body_kb: int,
        images_total: int = 0,
        images_dropped: int = 0,
        stripped_fields: list[str] | None = None,
        normalized_chunks: int = 0,
        account: str = "",
        error: str = "",
    ) -> RequestEvent:
        self._seq += 1
        fields = list(stripped_fields or [])
        ev = RequestEvent(
            id=self._seq,
            ts=time.time(),
            method=method,
            path=path,
            model=model or "-",
            mistral=mistral,
            route_mode=route_mode,
            stream=stream,
            status=status,
            latency_ms=latency_ms,
            body_kb=body_kb,
            images_total=images_total,
            images_dropped=images_dropped,
            stripped_fields=fields,
            normalized_chunks=normalized_chunks,
            account=account,
            error=error[:240] if error else "",
        )
        self._events.appendleft(ev)
        self.total_requests += 1
        if status >= 400 or error:
            self.total_errors += 1
        self.total_images_dropped += images_dropped
        self.total_fields_stripped += len(fields)
        self.total_normalized_chunks += normalized_chunks
        return ev

    def recent(self, limit: int = 80) -> list[dict[str, Any]]:
        out = []
        for idx, ev in enumerate(self._events):
            if idx >= limit:
                break
            out.append(asdict(ev))
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "uptime_seconds": int(time.time() - self.started_at),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_images_dropped": self.total_images_dropped,
            "total_fields_stripped": self.total_fields_stripped,
            "total_normalized_chunks": self.total_normalized_chunks,
        }
