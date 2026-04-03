from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Iterable


@dataclass
class JobCandidate:
    id: str
    location_lat: float
    location_lng: float
    priority_level: int
    missed_count: int
    sla_urgency: int
    days_since_last_wash: int
    time_window_start: time | None
    time_window_end: time | None
    estimated_duration_min: int


@dataclass
class WorkerCandidate:
    id: str
    max_jobs_per_day: int
    shift_start: datetime
    shift_end: datetime
    start_lat: float
    start_lng: float


class SchedulingRepository:
    """Abstraction layer for DB operations used by scheduler service."""

    def load_schedulable_jobs(self, target_date: date) -> list[JobCandidate]:
        raise NotImplementedError

    def load_available_workers(self, target_date: date) -> list[WorkerCandidate]:
        raise NotImplementedError

    def save_assignments(self, assignments: dict[str, list[str]], target_date: date) -> None:
        raise NotImplementedError

    def save_unassigned(self, unassigned: Iterable[tuple[str, str]], target_date: date) -> None:
        raise NotImplementedError
