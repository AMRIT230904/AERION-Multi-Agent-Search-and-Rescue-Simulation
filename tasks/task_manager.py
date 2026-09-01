"""
tasks/task_manager.py

Owns the mission's task pool. This is the "Mission -> Task1, Task2..."
layer -- the coordinator asks it for ranked pending tasks, and reports
back completions/failures so status stays accurate.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from tasks.task import Task, TaskStatus, Priority
from tasks.prioritizer import Prioritizer


class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self.prioritizer = Prioritizer()

    def add_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task
        self.prioritizer.register(task)

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    def pending_ranked(self) -> List[Task]:
        return self.prioritizer.rank_pending(self.all_tasks())

    def mark_assigned(self, task_id: str, agent_name: str) -> None:
        self._tasks[task_id].assign(agent_name)

    def mark_in_progress(self, task_id: str) -> None:
        self._tasks[task_id].start()

    def mark_complete(self, task_id: str) -> None:
        self._tasks[task_id].complete()

    def release_agent_tasks(self, agent_name: str) -> List[Task]:
        """Called on agent failure: find every task assigned to this
        agent, release it back to PENDING, and return the released
        tasks so the caller can log/react to them."""
        released = []
        for task in self._tasks.values():
            if task.assigned_agent == agent_name and task.status in (
                TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS
            ):
                task.fail_and_release()
                released.append(task)
        return released

    def status_summary(self) -> str:
        lines = [repr(t) for t in self._tasks.values()]
        return "\n".join(lines)


if __name__ == "__main__":
    tm = TaskManager()
    tm.add_task(Task("T1", (10.0, 0.0, -5.0), Priority.LOW))
    tm.add_task(Task("T2", (0.0, 10.0, -5.0), Priority.HIGH))
    tm.add_task(Task("T3", (5.0, 5.0, -5.0), Priority.CRITICAL))

    ranked = tm.pending_ranked()
    print("Ranked pending tasks:")
    for t in ranked:
        print(" ", t)

    tm.mark_assigned("T3", "Drone1")
    tm.mark_in_progress("T3")
    released = tm.release_agent_tasks("Drone1")
    print("\nReleased after Drone1 failure:", released)
    print("\nFull status:")
    print(tm.status_summary())
