"""Scheduler: semaphore-based concurrency control."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import Agent


@dataclass
class Scheduler:
    max_concurrent: int = 50
    _semaphore: asyncio.Semaphore = field(init=False)
    _active: int = field(default=0, init=False)
    _waiting: int = field(default=0, init=False)

    def __post_init__(self):
        if self.max_concurrent <= 0:
            self.max_concurrent = 1_000_000
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def acquire(self):
        self._waiting += 1
        await self._semaphore.acquire()
        self._waiting -= 1
        self._active += 1

    def release(self):
        self._active -= 1
        self._semaphore.release()

    @property
    def stats(self) -> dict:
        return {
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "waiting": self._waiting,
            "available": self._semaphore._value,
        }
