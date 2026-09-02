"""Bounded concurrent scheduling that isolates individual collector failures."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from osint.limits import BudgetExceeded
from osint.models import CollectorOutcome


class Scheduler:
    def __init__(self, max_parallel_tasks: int) -> None: self._semaphore = asyncio.Semaphore(max_parallel_tasks)

    async def run(self, jobs: list[Callable[[], Awaitable[CollectorOutcome]]]) -> list[CollectorOutcome]:
        async def guarded(job: Callable[[], Awaitable[CollectorOutcome]]) -> CollectorOutcome:
            async with self._semaphore:
                try: return await job()
                except BudgetExceeded as exc: return CollectorOutcome("budget", False, errors=[str(exc)])
                except Exception as exc: return CollectorOutcome("scheduler", False, errors=[f"Collector crashed: {exc}"])
        return list(await asyncio.gather(*(guarded(job) for job in jobs)))
