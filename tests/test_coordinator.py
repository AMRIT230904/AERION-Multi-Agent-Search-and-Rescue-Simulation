"""tests/test_coordinator.py"""

from tasks.task import Task, Priority, TaskStatus
from tasks.task_manager import TaskManager
from coordinator import TaskCoordinator


def make_coordinator():
    tm = TaskManager()
    tm.add_task(Task("T1", (0.0, 0.0, 0.0), Priority.MEDIUM))
    tm.add_task(Task("T2", (100.0, 0.0, 0.0), Priority.HIGH))
    coord = TaskCoordinator(agent_names=["Drone1", "Drone2"], task_manager=tm)
    return coord, tm


def test_assign_pending_matches_nearest_agent_to_task():
    coord, tm = make_coordinator()
    positions = {"Drone1": (1.0, 0.0, 0.0), "Drone2": (99.0, 0.0, 0.0)}
    assignment = coord.assign_pending(positions)
    assert assignment["Drone1"].task_id == "T1"
    assert assignment["Drone2"].task_id == "T2"


def test_assign_pending_updates_task_status():
    coord, tm = make_coordinator()
    positions = {"Drone1": (1.0, 0.0, 0.0), "Drone2": (99.0, 0.0, 0.0)}
    coord.assign_pending(positions)
    assert tm.get("T1").status == TaskStatus.ASSIGNED
    assert tm.get("T2").status == TaskStatus.ASSIGNED


def test_assign_pending_with_no_pending_tasks_returns_empty():
    coord, tm = make_coordinator()
    tm.mark_assigned("T1", "X")
    tm.mark_assigned("T2", "Y")
    assignment = coord.assign_pending({"Drone1": (0, 0, 0)})
    assert assignment == {}


def test_assign_pending_with_no_idle_agents_returns_empty():
    coord, tm = make_coordinator()
    assignment = coord.assign_pending({})
    assert assignment == {}


def test_more_agents_than_tasks_only_assigns_available_tasks():
    coord, tm = make_coordinator()
    positions = {
        "Drone1": (0.0, 0.0, 0.0),
        "Drone2": (100.0, 0.0, 0.0),
    }
    # only 2 tasks exist, 2 agents -- both should get one
    assignment = coord.assign_pending(positions)
    assert len(assignment) == 2
