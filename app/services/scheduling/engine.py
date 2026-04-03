from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from math import asin, cos, radians, sin, sqrt

from app.repositories.scheduling_repository import JobCandidate, SchedulingRepository, WorkerCandidate


@dataclass
class SchedulerConfig:
    weight_customer_priority: int = 30
    weight_missed_jobs: int = 25
    weight_sla_urgency: int = 20
    weight_days_since_wash: int = 10
    weight_flex_window_bonus: int = 5
    late_penalty_per_min: float = 1.2
    travel_penalty_per_km: float = 1.0
    imbalance_penalty: float = 0.8


@dataclass
class WorkerState:
    worker: WorkerCandidate
    remaining_capacity: int
    current_time: datetime
    current_lat: float
    current_lng: float
    route: list[str] = field(default_factory=list)
    total_minutes: int = 0


class SchedulingEngine:
    """Heuristic scheduler for daily PH7 jobs.

    Algorithm:
    1. Load pending + missed jobs and available workers.
    2. Score jobs using priority formula.
    3. Greedy assignment using minimum weighted cost (travel + lateness + load balance).
    4. Persist assignments and unassigned diagnostics.
    """

    def __init__(self, repository: SchedulingRepository, config: SchedulerConfig | None = None):
        self.repository = repository
        self.config = config or SchedulerConfig()

    def run(self, target_date: date) -> dict:
        jobs = self.repository.load_schedulable_jobs(target_date)
        workers = self.repository.load_available_workers(target_date)

        if not workers:
            self.repository.save_unassigned(((j.id, "no_workers") for j in jobs), target_date)
            return {"date": str(target_date), "assigned_jobs": 0, "unassigned_jobs": len(jobs)}

        worker_state = {
            w.id: WorkerState(
                worker=w,
                remaining_capacity=w.max_jobs_per_day,
                current_time=w.shift_start,
                current_lat=w.start_lat,
                current_lng=w.start_lng,
            )
            for w in workers
        }

        sorted_jobs = sorted(jobs, key=self._job_priority_score, reverse=True)
        unassigned: list[tuple[str, str]] = []

        for job in sorted_jobs:
            candidates = [s for s in worker_state.values() if self._is_feasible(job, s)]
            if not candidates:
                unassigned.append((job.id, "no_feasible_worker"))
                continue

            best_state = min(candidates, key=lambda state: self._assignment_cost(job, state, worker_state))
            self._assign(job, best_state)

        assignments = {wid: state.route for wid, state in worker_state.items()}
        self.repository.save_assignments(assignments, target_date)
        self.repository.save_unassigned(unassigned, target_date)

        assigned_jobs = sum(len(state.route) for state in worker_state.values())
        return {
            "date": str(target_date),
            "assigned_jobs": assigned_jobs,
            "unassigned_jobs": len(unassigned),
        }

    def _job_priority_score(self, job: JobCandidate) -> int:
        # Weighted priority formula tuned for doorstep operations.
        return (
            job.priority_level * self.config.weight_customer_priority
            + job.missed_count * self.config.weight_missed_jobs
            + job.sla_urgency * self.config.weight_sla_urgency
            + job.days_since_last_wash * self.config.weight_days_since_wash
            - (0 if job.time_window_start and job.time_window_end else self.config.weight_flex_window_bonus)
        )

    def _is_feasible(self, job: JobCandidate, state: WorkerState) -> bool:
        if state.remaining_capacity <= 0:
            return False

        travel_km = self._distance_km(state.current_lat, state.current_lng, job.location_lat, job.location_lng)
        travel_minutes = int(travel_km * 3.2)  # rough city average
        projected_start = state.current_time + timedelta(minutes=travel_minutes)
        projected_end = projected_start + timedelta(minutes=job.estimated_duration_min)
        return projected_end <= state.worker.shift_end

    def _assignment_cost(self, job: JobCandidate, state: WorkerState, all_states: dict[str, WorkerState]) -> float:
        travel_km = self._distance_km(state.current_lat, state.current_lng, job.location_lat, job.location_lng)
        travel_cost = travel_km * self.config.travel_penalty_per_km

        avg_jobs = sum(len(s.route) for s in all_states.values()) / max(len(all_states), 1)
        load_cost = max(0.0, len(state.route) - avg_jobs) * self.config.imbalance_penalty

        late_cost = 0.0
        if job.time_window_end:
            projected_start = state.current_time + timedelta(minutes=int(travel_km * 3.2))
            if projected_start.time() > job.time_window_end:
                delta_minutes = (
                    datetime.combine(projected_start.date(), projected_start.time())
                    - datetime.combine(projected_start.date(), job.time_window_end)
                ).seconds / 60
                late_cost = delta_minutes * self.config.late_penalty_per_min

        priority_discount = self._job_priority_score(job) * 0.01
        return travel_cost + load_cost + late_cost - priority_discount

    def _assign(self, job: JobCandidate, state: WorkerState) -> None:
        travel_km = self._distance_km(state.current_lat, state.current_lng, job.location_lat, job.location_lng)
        travel_minutes = int(travel_km * 3.2)

        state.current_time = state.current_time + timedelta(minutes=travel_minutes + job.estimated_duration_min)
        state.current_lat = job.location_lat
        state.current_lng = job.location_lng
        state.remaining_capacity -= 1
        state.total_minutes += travel_minutes + job.estimated_duration_min
        state.route.append(job.id)

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return 6371 * c
