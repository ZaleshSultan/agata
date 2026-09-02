"""Hard termination budgets for an investigation."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import config


class BudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class InvestigationLimits:
    max_runtime_seconds: float = config.OSINT_MAX_RUNTIME_SECONDS
    max_requests: int = config.OSINT_MAX_REQUESTS
    max_entities: int = config.OSINT_MAX_ENTITIES
    max_pages: int = config.OSINT_MAX_PAGES
    max_depth: int = config.OSINT_MAX_DEPTH
    max_parallel_tasks: int = config.OSINT_MAX_PARALLEL_TASKS
    max_ai_calls: int = config.OSINT_MAX_AI_CALLS
    max_image_ai_calls: int = config.OSINT_MAX_IMAGE_AI_CALLS
    max_reverse_image_calls: int = config.OSINT_MAX_REVERSE_IMAGE_CALLS


class Budget:
    def __init__(self, limits: InvestigationLimits) -> None:
        self.limits, self.started_at = limits, monotonic()
        self.requests = self.pages = 0
        self.ai_calls = self.image_ai_calls = self.reverse_image_calls = 0

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started_at

    def check_time(self) -> None:
        if self.elapsed_seconds > self.limits.max_runtime_seconds:
            raise BudgetExceeded("Investigation runtime budget exhausted.")

    def consume_request(self, pages: int = 0) -> None:
        self.check_time()
        if self.requests >= self.limits.max_requests:
            raise BudgetExceeded("Investigation request budget exhausted.")
        if self.pages + pages > self.limits.max_pages:
            raise BudgetExceeded("Investigation page budget exhausted.")
        self.requests += 1; self.pages += pages

    def consume_ai(self, image: bool = False) -> None:
        if self.ai_calls >= self.limits.max_ai_calls or (image and self.image_ai_calls >= self.limits.max_image_ai_calls):
            raise BudgetExceeded("AI analysis budget exhausted.")
        self.ai_calls += 1; self.image_ai_calls += int(image)
