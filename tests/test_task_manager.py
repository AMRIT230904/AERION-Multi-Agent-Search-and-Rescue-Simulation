"""tests/test_task_manager.py"""

from tasks.task import Task, Priority, TaskStatus
from tasks.task_manager import TaskManager


def make_manager():
    tm = TaskManager()
    tm.add_task(Task("T1", (0, 0, 0), Priority.LOW))
    tm.add_task(Task("T2", (0, 0, 0), Priority.HIGH))
    tm.add_task(Task("T3", (0, 0, 0), Priority.CRITICAL))
    return tm


def test_pending_ranked_orders_by_priority_desc():
    tm = make_manager()
    ranked = tm.pending_ranked()
    assert [t.task_id for t in ranked] == ["T3", "T2", "T1"]


def test_pending_ranked_tie_break_is_insertion_order():
    tm = TaskManager()
    tm.add_task(Task("A", (0, 0, 0), Priority.HIGH))
    tm.add_task(Task("B", (0, 0, 0), Priority.HIGH))
    ranked = tm.pending_ranked()
    # A was registered first, so among equal-priority tasks it should
    # come first (oldest-pending-first tie-break).
    assert [t.task_id for t in ranked] == ["A", "B"]


def test_assigned_task_drops_out_of_pending():
    tm = make_manager()
    tm.mark_assigned("T3", "Drone1")
    ranked = tm.pending_ranked()
    assert "T3" not in [t.task_id for t in ranked]


def test_release_agent_tasks_only_releases_that_agent():
    tm = make_manager()
    tm.mark_assigned("T1", "Drone1")
    tm.mark_assigned("T2", "Drone2")
    released = tm.release_agent_tasks("Drone1")
    assert [t.task_id for t in released] == ["T1"]
    assert tm.get("T1").status == TaskStatus.PENDING
    assert tm.get("T2").status == TaskStatus.ASSIGNED  # untouched


def test_release_agent_tasks_returns_empty_if_agent_has_nothing():
    tm = make_manager()
    released = tm.release_agent_tasks("DroneGhost")
    assert released == []


def test_released_task_reappears_in_pending_ranked():
    tm = make_manager()
    tm.mark_assigned("T3", "Drone1")
    assert "T3" not in [t.task_id for t in tm.pending_ranked()]
    tm.release_agent_tasks("Drone1")
    assert "T3" in [t.task_id for t in tm.pending_ranked()]
