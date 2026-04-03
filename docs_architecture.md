# PH7 Autocare: Migration and Scheduling Implementation Notes

## Alembic rollout plan

1. Create extensions and enum types.
2. Create foundational tables: `users`, `locations`, `services`, `plans`.
3. Create operational entities: `workers`, `customers`, `customer_subscriptions`, `jobs`.
4. Create tracking entities: `attendance`, `job_events`, `wix_sync_log`.
5. Add performance indexes:
   - `jobs(scheduled_date, status)`
   - `jobs(assigned_worker_id, scheduled_date)`
   - `customers(phone)`
   - `attendance(worker_id, date)` unique
6. Load normalized data from Excel using an ETL script.

## Scheduler pseudocode

```text
function RUN_SCHEDULER(target_date):
  jobs <- load pending(target_date) + missed(last 7 days)
  workers <- load active workers with attendance present

  if workers empty:
    mark all jobs unassigned(no_workers)
    return

  for job in jobs:
    job.score <- weighted_priority(job)

  sort jobs descending by score
  init worker states(capacity, clock, location, route)

  for each job in sorted jobs:
    feasible <- filter workers by capacity + shift feasibility
    if feasible empty:
      unassigned += (job, no_feasible_worker)
      continue

    best_worker <- argmin(travel + lateness + load_imbalance - priority_discount)
    assign job to best_worker; update worker clock/location/capacity

  persist assignments and unassigned reasons
  return summary
```

## Weighted priority formula

```text
score =
  customer_priority * 30
+ missed_count * 25
+ sla_urgency * 20
+ days_since_last_wash * 10
- flexibility_bonus * 5
```

## Next hardening milestones

- Add route reordering pass per worker.
- Add schedule versioning and replay.
- Replace travel estimation with distance matrix API.
- Add OR-Tools optimization mode for high-volume days.
