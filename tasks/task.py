"""
tasks/task.py

The unit of work the coordinator allocates. Replaces the old "just a
goal point" model from coordinator v1 with something that actually
supports the resume claim "adapt task prioritization" -- priority has
to exist as data before anything can adapt to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

Point3 = Tuple[float, float, float]


class Priority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class Task:
    task_id: str
    location: Point3
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    # simple sector/zone tag -- used later for heterogeneous routing
    # (e.g. "aerial" vs "ground") but harmless to carry now.
    task_type: str = "aerial"

    def assign(self, agent_name: str) -> None:
        self.assigned_agent = agent_name
        self.status = TaskStatus.ASSIGNED

    def start(self) -> None:
        if self.status != TaskStatus.ASSIGNED:
            raise ValueError(f"Task {self.task_id} must be ASSIGNED before it can start "
                              f"(currently {self.status})")
        self.status = TaskStatus.IN_PROGRESS

    def complete(self) -> None:
        self.status = TaskStatus.COMPLETE

    def fail_and_release(self) -> None:
        """Called when the assigned agent drops out -- the task goes back
        to the pending pool so the coordinator can hand it to someone else."""
        self.assigned_agent = None
        self.status = TaskStatus.PENDING

    def __repr__(self) -> str:
        return (f"Task({self.task_id}, prio={self.priority.name}, "
                f"status={self.status.value}, agent={self.assigned_agent})")
