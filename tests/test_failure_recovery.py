"""tests/test_failure_recovery.py

This directly tests the bug that was flagged in the v1 review: that
reassignment must actually change the surviving agents' assigned task,
not just compute a value that's thrown away.
"""

from tasks.task import Task, Priority, TaskStatus
from tasks.task_manager import TaskManager
from coordinator import TaskCoordinator


def make_scenario():
    tm = TaskManager()
    tm.add_task(Task("T1", (0.0, 0.0, 0.0), Priority.MEDIUM))
    tm.add_task(Task("T2", (10.0, 0.0, 0.0), Priority.HIGH))
    tm.add_task(Task("T3", (20.0, 0.0, 0.0), Priority.CRITICAL))
    coord = TaskCoordinator(agent_names=["Drone1", "Drone2", "Drone3"], task_manager=tm)
    positions = {
        "Drone1": (0.0, 0.0, 0.0),
        "Drone2": (10.0, 0.0, 0.0),
        "Drone3": (20.0, 0.0, 0.0),
    }
    coord.assign_pending(positions)
    return coord, tm, positions


def test_failed_agents_task_returns_to_pending():
    coord, tm, positions = make_scenario()
    failed_agent_task_id = None
    for t in tm.all_tasks():
        if t.assigned_agent == "Drone2":
            failed_agent_task_id = t.task_id
    coord.handle_agent_failure("Drone2", {})
    assert tm.get(failed_agent_task_id).status == TaskStatus.PENDING
    assert tm.get(failed_agent_task_id).assigned_agent is None


def test_failure_with_no_idle_agents_leaves_task_pending_not_lost():
    """This is the exact case the v1 code silently mishandled: a failure
    with nobody idle to take the task shouldn't crash and shouldn't drop
    the task -- it should sit PENDING until someone frees up."""
    coord, tm, positions = make_scenario()
    result = coord.handle_agent_failure("Drone2", {})
    assert result == {}
    pending_ids = [t.task_id for t in tm.pending_ranked()]
    assert len(pending_ids) == 1  # Drone2's task, waiting


def test_failure_reassignment_actually_applies_when_agent_idle():
    """Regression test for the v1 bug: reassign_on_failure's return value
    must reflect a real, appliable assignment -- verified here by
    confirming the returned Task's assigned_agent field is updated to
    the new agent, not left pointing at nothing or the failed agent."""
    coord, tm, positions = make_scenario()

    for t in tm.all_tasks():
        if t.assigned_agent == "Drone2":
            orphaned_task_id = t.task_id

    # Drone1 becomes free (e.g. finished its own task) and can pick up
    # Drone2's orphaned task.
    tm.mark_complete([t.task_id for t in tm.all_tasks() if t.assigned_agent == "Drone1"][0])
    reassignment = coord.handle_agent_failure("Drone2", {"Drone1": positions["Drone1"]})

    assert "Drone1" in reassignment
    assert reassignment["Drone1"].task_id == orphaned_task_id
    # the task object itself must reflect the new owner -- this is what
    # a caller applying `.set_task()` would read
    assert tm.get(orphaned_task_id).assigned_agent == "Drone1"
    assert tm.get(orphaned_task_id).status == TaskStatus.ASSIGNED


def test_failed_agent_excluded_from_its_own_reassignment_pool():
    coord, tm, positions = make_scenario()
    # even if caller mistakenly includes the failed agent's own position,
    # it must not be offered the task it just failed
    reassignment = coord.handle_agent_failure("Drone2", {"Drone2": positions["Drone2"]})
    assert "Drone2" not in reassignment
