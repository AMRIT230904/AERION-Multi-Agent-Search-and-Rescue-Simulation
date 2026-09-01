"""
tasks/prioritizer.py

Decides WHICH pending tasks get considered first when there are more
tasks than available agents. Kept separate from TaskManager so the
ordering policy can be swapped/tested independently.

Current policy: sort by Priority (descending), tie-break by how long
the task has been waiting (oldest-pending-first), so a HIGH task
doesn't get starved by a stream of newer HIGH tasks.
"""

from __future__ import annotations

from typing import List

from tasks.task import Task


class Prioritizer:
    def __init__(self):
        # tracks insertion order for tie-breaking staleness
        self._order: List[str] = []

    def register(self, task: Task) -> None:
        if task.task_id not in self._order:
            self._order.append(task.task_id)

    def rank_pending(self, tasks: List[Task]) -> List[Task]:
        """Return pending tasks sorted highest-priority-first, with
        older tasks (registered earlier) breaking ties."""
        def sort_key(t: Task):
            staleness_rank = self._order.index(t.task_id) if t.task_id in self._order else 0
            return (-t.priority.value, staleness_rank)

        pending = [t for t in tasks if t.status.value == "PENDING"]
        return sorted(pending, key=sort_key)
