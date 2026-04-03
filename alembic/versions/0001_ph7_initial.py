"""Initial PH7 Autocare schema.

Revision ID: 0001_ph7_initial
Revises:
Create Date: 2026-04-03
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_ph7_initial"
down_revision = None
branch_labels = None
depends_on = None


def _enum(name: str, *values: str):
    return sa.Enum(*values, name=name)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    user_role = _enum("user_role", "worker", "manager", "admin")
    worker_status = _enum("worker_status", "active", "inactive")
    generic_status = _enum("generic_status", "active", "inactive")
    subscription_status = _enum("subscription_status", "active", "paused", "cancelled")
    job_status = _enum("job_status", "pending", "assigned", "in_progress", "completed", "missed", "rescheduled", "cancelled")
    attendance_status = _enum("attendance_status", "present", "absent", "late", "leave")
    job_event_type = _enum("job_event_type", "assigned", "reassigned", "started", "completed", "missed", "rescheduled")
    sync_direction = _enum("sync_direction", "inbound", "outbound")
    sync_status = _enum("sync_status", "success", "failed", "retry")

    for e in [
        user_role,
        worker_status,
        generic_status,
        subscription_status,
        job_status,
        attendance_status,
        job_event_type,
        sync_direction,
        sync_status,
    ]:
        e.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("role", user_role, nullable=False),
        sa.Column("phone", sa.String(32), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"], unique=False)

    op.create_table(
        "locations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("address_line1", sa.String(255), nullable=False),
        sa.Column("address_line2", sa.String(255), nullable=True),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("state", sa.String(100), nullable=False),
        sa.Column("pincode", sa.String(20), nullable=True),
        sa.Column("latitude", sa.Numeric(10, 7), nullable=True),
        sa.Column("longitude", sa.Numeric(10, 7), nullable=True),
        sa.Column("geo_hash", sa.String(32), nullable=True),
    )
    op.create_index("ix_locations_city", "locations", ["city"], unique=False)
    op.create_index("ix_locations_pincode", "locations", ["pincode"], unique=False)
    op.create_index("ix_locations_geo_hash", "locations", ["geo_hash"], unique=False)

    op.create_table(
        "services",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("estimated_duration_min", sa.Integer(), nullable=False),
        sa.Column("base_price", sa.Numeric(10, 2), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_index("ix_services_name", "services", ["name"], unique=True)

    op.create_table(
        "plans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("frequency_type", sa.String(32), nullable=False),
        sa.Column("frequency_value", sa.Integer(), nullable=False),
        sa.Column("service_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("services.id"), nullable=False),
        sa.Column("price", sa.Numeric(10, 2), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_index("ix_plans_name", "plans", ["name"], unique=True)

    op.create_table(
        "workers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("employee_code", sa.String(64), nullable=False),
        sa.Column("default_shift_start", sa.Time(), nullable=True),
        sa.Column("default_shift_end", sa.Time(), nullable=True),
        sa.Column("max_jobs_per_day", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("home_location_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("locations.id"), nullable=True),
        sa.Column("status", worker_status, nullable=False, server_default="active"),
    )
    op.create_index("ix_workers_employee_code", "workers", ["employee_code"], unique=True)
    op.create_index("ix_workers_status", "workers", ["status"], unique=False)
    op.create_unique_constraint("uq_workers_user_id", "workers", ["user_id"])

    op.create_table(
        "customers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("wix_customer_id", sa.String(128), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("phone", sa.String(32), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("preferred_time_window_start", sa.String(8), nullable=True),
        sa.Column("preferred_time_window_end", sa.String(8), nullable=True),
        sa.Column("priority_level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", generic_status, nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_customers_wix_customer_id", "customers", ["wix_customer_id"], unique=True)
    op.create_index("ix_customers_phone", "customers", ["phone"], unique=False)
    op.create_index("ix_customers_email", "customers", ["email"], unique=False)
    op.create_index("ix_customers_priority_level", "customers", ["priority_level"], unique=False)

    op.create_table(
        "customer_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("customers.id"), nullable=False),
        sa.Column("plan_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("plans.id"), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("remaining_washes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", subscription_status, nullable=False, server_default="active"),
    )
    op.create_index("ix_customer_subscriptions_customer_id", "customer_subscriptions", ["customer_id"], unique=False)
    op.create_index("ix_customer_subscriptions_plan_id", "customer_subscriptions", ["plan_id"], unique=False)
    op.create_index("ix_customer_subscriptions_status", "customer_subscriptions", ["status"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("customer_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("customers.id"), nullable=False),
        sa.Column("subscription_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("customer_subscriptions.id"), nullable=True),
        sa.Column("service_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("services.id"), nullable=False),
        sa.Column("location_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("locations.id"), nullable=False),
        sa.Column("scheduled_date", sa.Date(), nullable=False),
        sa.Column("time_window_start", sa.Time(), nullable=True),
        sa.Column("time_window_end", sa.Time(), nullable=True),
        sa.Column("assigned_worker_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workers.id"), nullable=True),
        sa.Column("status", job_status, nullable=False, server_default="pending"),
        sa.Column("priority_score", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_duration_min", sa.Integer(), nullable=False),
        sa.Column("actual_start_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_end_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_from", sa.String(50), nullable=False, server_default="system"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_jobs_scheduled_date", "jobs", ["scheduled_date"], unique=False)
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_priority_score", "jobs", ["priority_score"], unique=False)
    op.create_index("ix_jobs_assigned_worker_id", "jobs", ["assigned_worker_id"], unique=False)
    op.create_index("ix_jobs_date_status", "jobs", ["scheduled_date", "status"], unique=False)
    op.create_index("ix_jobs_worker_date", "jobs", ["assigned_worker_id", "scheduled_date"], unique=False)

    op.create_table(
        "attendance",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("worker_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workers.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("check_in_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("check_out_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", attendance_status, nullable=False, server_default="present"),
    )
    op.create_index("ix_attendance_worker_id", "attendance", ["worker_id"], unique=False)
    op.create_index("ix_attendance_date", "attendance", ["date"], unique=False)
    op.create_unique_constraint("uq_attendance_worker_date", "attendance", ["worker_id", "date"])

    op.create_table(
        "job_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id"), nullable=False),
        sa.Column("event_type", job_event_type, nullable=False),
        sa.Column("old_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"], unique=False)
    op.create_index("ix_job_events_event_type", "job_events", ["event_type"], unique=False)

    op.create_table(
        "wix_sync_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(128), nullable=False),
        sa.Column("direction", sync_direction, nullable=False),
        sa.Column("status", sync_status, nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_wix_sync_log_entity_type", "wix_sync_log", ["entity_type"], unique=False)
    op.create_index("ix_wix_sync_log_entity_id", "wix_sync_log", ["entity_id"], unique=False)
    op.create_index("ix_wix_sync_log_direction", "wix_sync_log", ["direction"], unique=False)
    op.create_index("ix_wix_sync_log_status", "wix_sync_log", ["status"], unique=False)


def downgrade() -> None:
    op.drop_table("wix_sync_log")
    op.drop_table("job_events")
    op.drop_table("attendance")
    op.drop_table("jobs")
    op.drop_table("customer_subscriptions")
    op.drop_table("customers")
    op.drop_table("workers")
    op.drop_table("plans")
    op.drop_table("services")
    op.drop_table("locations")
    op.drop_table("users")

    for enum_name in [
        "sync_status",
        "sync_direction",
        "job_event_type",
        "attendance_status",
        "job_status",
        "subscription_status",
        "generic_status",
        "worker_status",
        "user_role",
    ]:
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)
