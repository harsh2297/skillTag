import enum


class UserRole(str, enum.Enum):
    worker = "worker"
    manager = "manager"
    admin = "admin"


class WorkerStatus(str, enum.Enum):
    active = "active"
    inactive = "inactive"


class GenericStatus(str, enum.Enum):
    active = "active"
    inactive = "inactive"


class SubscriptionStatus(str, enum.Enum):
    active = "active"
    paused = "paused"
    cancelled = "cancelled"


class JobStatus(str, enum.Enum):
    pending = "pending"
    assigned = "assigned"
    in_progress = "in_progress"
    completed = "completed"
    missed = "missed"
    rescheduled = "rescheduled"
    cancelled = "cancelled"


class AttendanceStatus(str, enum.Enum):
    present = "present"
    absent = "absent"
    late = "late"
    leave = "leave"


class JobEventType(str, enum.Enum):
    assigned = "assigned"
    reassigned = "reassigned"
    started = "started"
    completed = "completed"
    missed = "missed"
    rescheduled = "rescheduled"


class SyncDirection(str, enum.Enum):
    inbound = "inbound"
    outbound = "outbound"


class SyncStatus(str, enum.Enum):
    success = "success"
    failed = "failed"
    retry = "retry"
