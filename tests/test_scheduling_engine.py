from datetime import date, datetime, time

from app.repositories.scheduling_repository import JobCandidate, SchedulingRepository, WorkerCandidate
from app.services.scheduling.engine import SchedulingEngine


class InMemorySchedulingRepo(SchedulingRepository):
    def __init__(self, jobs, workers):
        self._jobs = jobs
        self._workers = workers
        self.assignments = {}
        self.unassigned = []

    def load_schedulable_jobs(self, target_date: date):
        return self._jobs

    def load_available_workers(self, target_date: date):
        return self._workers

    def save_assignments(self, assignments, target_date: date):
        self.assignments = assignments

    def save_unassigned(self, unassigned, target_date: date):
        self.unassigned = list(unassigned)


def _job(job_id: str, lat: float, lng: float, duration: int = 30, tw_end: time | None = None):
    return JobCandidate(
        id=job_id,
        location_lat=lat,
        location_lng=lng,
        priority_level=1,
        missed_count=0,
        sla_urgency=1,
        days_since_last_wash=2,
        time_window_start=None,
        time_window_end=tw_end,
        estimated_duration_min=duration,
    )


def _worker(worker_id: str, max_jobs: int = 2):
    return WorkerCandidate(
        id=worker_id,
        max_jobs_per_day=max_jobs,
        shift_start=datetime(2026, 4, 3, 9, 0),
        shift_end=datetime(2026, 4, 3, 18, 0),
        start_lat=12.9716,
        start_lng=77.5946,
    )


def test_scheduler_assigns_jobs_when_workers_available():
    repo = InMemorySchedulingRepo(
        jobs=[_job("j1", 12.972, 77.595), _job("j2", 12.975, 77.599)],
        workers=[_worker("w1", 2)],
    )
    engine = SchedulingEngine(repo)

    summary = engine.run(date(2026, 4, 3))

    assert summary["assigned_jobs"] == 2
    assert summary["unassigned_jobs"] == 0
    assert repo.assignments["w1"] == ["j1", "j2"]


def test_scheduler_marks_unassigned_when_no_workers():
    repo = InMemorySchedulingRepo(jobs=[_job("j1", 12.972, 77.595)], workers=[])
    engine = SchedulingEngine(repo)

    summary = engine.run(date(2026, 4, 3))

    assert summary["assigned_jobs"] == 0
    assert summary["unassigned_jobs"] == 1
    assert repo.unassigned == [("j1", "no_workers")]


def test_scheduler_respects_worker_capacity():
    repo = InMemorySchedulingRepo(
        jobs=[_job("j1", 12.972, 77.595), _job("j2", 12.973, 77.596), _job("j3", 12.974, 77.597)],
        workers=[_worker("w1", 2)],
    )
    engine = SchedulingEngine(repo)

    summary = engine.run(date(2026, 4, 3))

    assert summary["assigned_jobs"] == 2
    assert summary["unassigned_jobs"] == 1
