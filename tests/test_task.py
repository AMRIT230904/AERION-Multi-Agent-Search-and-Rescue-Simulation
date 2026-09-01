"""tests/test_task.py — pure logic, no AirSim/ROS2 needed."""

import pytest
from tasks.task import Task, Priority, TaskStatus


def test_task_starts_pending():
    t = Task("T1", (0, 0, 0))
    assert t.status == TaskStatus.PENDING
    assert t.assigned_agent is None


def test_assign_sets_agent_and_status():
    t = Task("T1", (0, 0, 0))
    t.assign("Drone1")
    assert t.assigned_agent == "Drone1"
    assert t.status == TaskStatus.ASSIGNED


def test_start_requires_assigned_first():
    t = Task("T1", (0, 0, 0))
    with pytest.raises(ValueError):
        t.start()  # can't start a task nobody's assigned to


def test_start_after_assign_succeeds():
    t = Task("T1", (0, 0, 0))
    t.assign("Drone1")
    t.start()
    assert t.status == TaskStatus.IN_PROGRESS


def test_complete_sets_status():
    t = Task("T1", (0, 0, 0))
    t.assign("Drone1")
    t.start()
    t.complete()
    assert t.status == TaskStatus.COMPLETE


def test_fail_and_release_clears_agent():
    t = Task("T1", (0, 0, 0))
    t.assign("Drone1")
    t.start()
    t.fail_and_release()
    assert t.status == TaskStatus.PENDING
    assert t.assigned_agent is None


def test_default_priority_is_medium():
    t = Task("T1", (0, 0, 0))
    assert t.priority == Priority.MEDIUM
