# PH7 Autocare Backend Blueprint

This repository now includes an implementation-ready backend foundation for PH7 Autocare's scheduling, workforce, and operations stack.

## Included artifacts

- SQLAlchemy ORM models with normalized PostgreSQL entities (`app/models/`).
- Initial Alembic migration with enums, tables, and indexes (`alembic/versions/0001_ph7_initial.py`).
- Heuristic scheduling engine suitable for MVP (`app/services/scheduling/engine.py`).
- Repository abstraction for scheduler I/O (`app/repositories/scheduling_repository.py`).
- Scheduler pseudocode and migration notes (`docs_architecture.md`).
- Unit tests for scheduler baseline behaviors (`tests/test_scheduling_engine.py`).

## Run tests

```bash
python -m pytest tests/test_scheduling_engine.py
```

## Next steps

1. Wire engine to actual database repository methods.
2. Expose `/scheduling/run` via FastAPI endpoint.
3. Add attendance-aware worker availability checks.
4. Add OR-Tools mode for high-volume optimization.
