from __future__ import annotations

import asyncio
from typing import Any


class ModelLimiter:
    """
    Hot-reloadable per-model concurrency limiter.

    Holds a slot for the full lifetime of a streaming or non-streaming request,
    queuing up to `wait_seconds` before rejecting with 429.
    """

    def __init__(self, limits: dict[str, int], wait_seconds: float):
        self.limits: dict[str, int] = dict(limits)
        self.wait_seconds: float = wait_seconds
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._sem_caps: dict[str, int] = {}
        self._inflight: dict[str, int] = {}
        self._rejected: int = 0

    def reconfigure(self, limits: dict[str, int], wait_seconds: float) -> None:
        self.limits = dict(limits)
        self.wait_seconds = wait_seconds
        # Rebuild semaphore for any model whose cap changed and currently has no inflight
        for k, cap in list(self._sem_caps.items()):
            new_cap = self.limits.get(k)
            if new_cap != cap and self._inflight.get(k, 0) == 0:
                self._semaphores.pop(k, None)
                self._sem_caps.pop(k, None)

    def limit_for(self, model: str | None) -> int | None:
        if not model:
            return None
        return self.limits.get(model.strip().lower())

    def _semaphore(self, key: str, cap: int) -> asyncio.Semaphore:
        sem = self._semaphores.get(key)
        if sem is None:
            sem = asyncio.Semaphore(cap)
            self._semaphores[key] = sem
            self._sem_caps[key] = cap
        return sem

    async def acquire_slot(self, model: str | None) -> tuple[str | None, int | None, asyncio.Semaphore | None]:
        """
        Returns (slot_key, cap, sem_handle).
        - slot_key is None -> no concurrency limit configured for this model.
        - cap is None (with non-None slot_key) -> limit exceeded after timeout.
        """
        cap = self.limit_for(model)
        if cap is None:
            return None, None, None
        key = (model or "").strip().lower()
        sem = self._semaphore(key, cap)
        try:
            await asyncio.wait_for(sem.acquire(), timeout=self.wait_seconds)
        except asyncio.TimeoutError:
            self._rejected += 1
            return key, None, None
        self._inflight[key] = self._inflight.get(key, 0) + 1
        return key, cap, sem

    def release_slot(self, slot_key: str | None, sem: asyncio.Semaphore | None) -> None:
        if slot_key is None:
            return
        n = self._inflight.get(slot_key, 0) - 1
        if n <= 0:
            self._inflight.pop(slot_key, None)
        else:
            self._inflight[slot_key] = n
        if sem is not None:
            sem.release()

    def inflight(self, key: str) -> int:
        return self._inflight.get(key.strip().lower(), 0)

    def stats(self) -> dict[str, Any]:
        return {
            "model_limits": dict(self.limits),
            "wait_seconds": self.wait_seconds,
            "total_rejected": self._rejected,
            "inflight": {k: v for k, v in self._inflight.items() if v > 0},
        }
