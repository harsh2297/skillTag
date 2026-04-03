from __future__ import annotations

import uuid
from datetime import date, datetime, time

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDMixin
from app.models.enums import (
    AttendanceStatus,
    GenericStatus,
    JobEventType,
    JobStatus,
    SubscriptionStatus,
    SyncDirection,
    SyncStatus,
    UserRole,
    WorkerStatus,
)


class User(Base, UUIDMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), index=True, nullable=False)
    phone: Mapped[str | None] = mapped_column(String(32))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    worker_profile: Mapped[Worker | None] = relationship(back_populates="user", uselist=False)


class Location(Base, UUIDMixin):
    __tablename__ = "locations"

    address_line1: Mapped[str] = mapped_column(String(255), nullable=False)
    address_line2: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    pincode: Mapped[str | None] = mapped_column(String(20), index=True)
    latitude: Mapped[float | None] = mapped_column(Numeric(10, 7))
    longitude: Mapped[float | None] = mapped_column(Numeric(10, 7))
    geo_hash: Mapped[str | None] = mapped_column(String(32), index=True)


class Worker(Base, UUIDMixin):
    __tablename__ = "workers"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False)
    employee_code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    default_shift_start: Mapped[time | None] = mapped_column(Time)
    default_shift_end: Mapped[time | None] = mapped_column(Time)
    max_jobs_per_day: Mapped[int] = mapped_column(Integer, default=8, nullable=False)
    home_location_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("locations.id"))
    status: Mapped[WorkerStatus] = mapped_column(
        Enum(WorkerStatus, name="worker_status"), default=WorkerStatus.active, nullable=False, index=True
    )

    user: Mapped[User] = relationship(back_populates="worker_profile")


class Customer(Base, UUIDMixin):
    __tablename__ = "customers"

    wix_customer_id: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), index=True)
    preferred_time_window_start: Mapped[str | None] = mapped_column(String(8))
    preferred_time_window_end: Mapped[str | None] = mapped_column(String(8))
    priority_level: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    status: Mapped[GenericStatus] = mapped_column(
        Enum(GenericStatus, name="generic_status"), default=GenericStatus.active, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Service(Base, UUIDMixin):
    __tablename__ = "services"

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    estimated_duration_min: Mapped[int] = mapped_column(Integer, nullable=False)
    base_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Plan(Base, UUIDMixin):
    __tablename__ = "plans"

    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    frequency_type: Mapped[str] = mapped_column(String(32), nullable=False)
    frequency_value: Mapped[int] = mapped_column(Integer, nullable=False)
    service_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id"), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class CustomerSubscription(Base, UUIDMixin):
    __tablename__ = "customer_subscriptions"

    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False, index=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("plans.id"), nullable=False, index=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date | None] = mapped_column(Date)
    remaining_washes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus, name="subscription_status"),
        default=SubscriptionStatus.active,
        nullable=False,
        index=True,
    )


class Job(Base, UUIDMixin):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_date_status", "scheduled_date", "status"),
        Index("ix_jobs_worker_date", "assigned_worker_id", "scheduled_date"),
    )

    customer_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("customers.id"), nullable=False, index=True)
    subscription_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("customer_subscriptions.id"))
    service_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("services.id"), nullable=False)
    location_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("locations.id"), nullable=False)
    scheduled_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    time_window_start: Mapped[time | None] = mapped_column(Time)
    time_window_end: Mapped[time | None] = mapped_column(Time)
    assigned_worker_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("workers.id"), index=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status"), default=JobStatus.pending, nullable=False, index=True
    )
    priority_score: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    estimated_duration_min: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actual_end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_from: Mapped[str] = mapped_column(String(50), default="system", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Attendance(Base, UUIDMixin):
    __tablename__ = "attendance"
    __table_args__ = (UniqueConstraint("worker_id", "date", name="uq_attendance_worker_date"),)

    worker_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("workers.id"), nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    check_in_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    check_out_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[AttendanceStatus] = mapped_column(
        Enum(AttendanceStatus, name="attendance_status"), default=AttendanceStatus.present, nullable=False
    )


class JobEvent(Base, UUIDMixin):
    __tablename__ = "job_events"

    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, index=True)
    event_type: Mapped[JobEventType] = mapped_column(Enum(JobEventType, name="job_event_type"), nullable=False, index=True)
    old_value: Mapped[dict | None] = mapped_column(JSONB)
    new_value: Mapped[dict | None] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WixSyncLog(Base, UUIDMixin):
    __tablename__ = "wix_sync_log"

    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    direction: Mapped[SyncDirection] = mapped_column(Enum(SyncDirection, name="sync_direction"), nullable=False, index=True)
    status: Mapped[SyncStatus] = mapped_column(Enum(SyncStatus, name="sync_status"), nullable=False, index=True)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
