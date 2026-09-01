"""
coordinator.py

A lightweight hierarchical coordinator for AERION.

Responsibility split (this is the actual "hierarchy"):
  - The coordinator NEVER flies a drone or picks low-level actions.
  - It only solves task allocation: given N drones and M goal points,
    decide which drone is responsible for which goal, minimizing total
    travel distance (a min-cost bipartite matching / assignment problem).
  - Each drone's own DQN policy (see multi_agent_train.py) is still the
    thing deciding velocity commands step-to-step. The coordinator just
    tells each agent "your goal this episode is X".

This keeps the split honest: task allocation is centralized/hierarchical,
low-level control/navigation is decentralized (independent per-agent RL).

Assignment solver:
  - Uses scipy's linear_sum_assignment (Hungarian algorithm) when available.
  - Falls back to a greedy nearest-goal assignment if scipy isn't installed,
    so this still runs in a stripped-down sim environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.task import Task
    from tasks.task_manager import TaskManager

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


Point3 = Tuple[float, float, float]


def _dist(a: Point3, b: Point3) -> float:
    return math.dist(a, b)


@dataclass
class DispatchRecord:
    """One coordinator decision, kept for logging / after-the-fact audits."""
    episode: int
    agent_name: str
    goal: Point3
    cost_at_assignment: float


@dataclass
class TaskCoordinator:
    """
    Hierarchical coordinator: owns the pool of pending goals and decides,
    once per episode (or on-demand mid-episode via reassign()), which
    drone gets which goal.

    `task_manager` is optional: it's only required by the Task-aware API
    (assign_pending / handle_agent_failure) used by the tasks/ package.
    The plain point-based API (assign / reassign_on_failure) below it
    works without one, e.g. for the raw goal-list flow in
    multi_agent_train.py.
    """

    agent_names: List[str]
    task_manager: Optional["TaskManager"] = None  # noqa: F821 (string annotation, see module docstring)
    history: List[DispatchRecord] = field(default_factory=list)
    _episode: int = 0

    def _build_cost_matrix(
        self,
        agent_positions: Dict[str, Point3],
        goals: List[Point3],
    ):
        agents = self.agent_names
        cost = [[_dist(agent_positions[a], g) for g in goals] for a in agents]
        return cost

    def assign(
        self,
        agent_positions: Dict[str, Point3],
        goals: List[Point3],
    ) -> Dict[str, Point3]:
        """
        Assign each agent exactly one goal, minimizing summed travel
        distance. Requires len(goals) >= len(agents); if there are more
        agents than goals, the extras get the (duplicated) nearest
        remaining goal.
        """
        self._episode += 1
        agents = self.agent_names
        n_agents = len(agents)

        if len(goals) < n_agents:
            # pad by repeating goals so every agent gets an assignment
            goals = goals + [goals[i % len(goals)] for i in range(n_agents - len(goals))]

        cost = self._build_cost_matrix(agent_positions, goals)

        assignment: Dict[str, Point3] = {}

        if _HAS_SCIPY:
            cost_np = np.array(cost)
            row_idx, col_idx = linear_sum_assignment(cost_np)
            for r, c in zip(row_idx, col_idx):
                agent = agents[r]
                goal = goals[c]
                assignment[agent] = goal
                self.history.append(
                    DispatchRecord(self._episode, agent, goal, float(cost_np[r][c]))
                )
        else:
            # Greedy fallback: repeatedly pick the globally cheapest
            # remaining (agent, goal) pair. Not globally optimal like
            # Hungarian, but a fine degraded mode.
            remaining_agents = list(agents)
            remaining_goal_idx = list(range(len(goals)))
            while remaining_agents:
                best = None  # (cost, agent, goal_idx)
                for a in remaining_agents:
                    for gi in remaining_goal_idx:
                        c = cost[agents.index(a)][gi]
                        if best is None or c < best[0]:
                            best = (c, a, gi)
                _, a, gi = best
                assignment[a] = goals[gi]
                self.history.append(
                    DispatchRecord(self._episode, a, goals[gi], best[0])
                )
                remaining_agents.remove(a)
                if len(remaining_goal_idx) > 1:
                    remaining_goal_idx.remove(gi)

        return assignment

    def assign_pending(
        self,
        agent_positions: Dict[str, Point3],
    ) -> Dict[str, "Task"]:
        """
        Task-aware sibling of assign(): pulls the highest-priority PENDING
        tasks straight from self.task_manager and hands them to whichever
        agents are currently idle (i.e. present in `agent_positions`),
        nearest-agent-to-task first, capped at
        min(#idle agents, #pending tasks) -- no padding, unlike assign().

        Mutates task state via task_manager.mark_assigned() and returns
        {agent_name: Task} for every pair it actually assigned. Returns
        {} (and touches nothing) if there are no idle agents or no
        pending tasks.
        """
        if self.task_manager is None:
            raise ValueError("assign_pending() requires this TaskCoordinator "
                              "to be constructed with a task_manager")

        if not agent_positions:
            return {}

        pending = self.task_manager.pending_ranked()
        if not pending:
            return {}

        self._episode += 1
        idle_agents = list(agent_positions.keys())
        cost = [[_dist(agent_positions[a], t.location) for t in pending] for a in idle_agents]

        pairs: List[Tuple[str, int]] = []  # (agent, task_index)

        if _HAS_SCIPY:
            row_idx, col_idx = linear_sum_assignment(np.array(cost))
            pairs = [(idle_agents[r], c) for r, c in zip(row_idx, col_idx)]
        else:
            # Greedy fallback: repeatedly take the cheapest remaining
            # (agent, task) pair until either side runs out.
            remaining_agents = list(idle_agents)
            remaining_task_idx = list(range(len(pending)))
            while remaining_agents and remaining_task_idx:
                best = None  # (cost, agent, task_idx)
                for a in remaining_agents:
                    for ti in remaining_task_idx:
                        c = cost[idle_agents.index(a)][ti]
                        if best is None or c < best[0]:
                            best = (c, a, ti)
                _, a, ti = best
                pairs.append((a, ti))
                remaining_agents.remove(a)
                remaining_task_idx.remove(ti)

        assignment: Dict[str, "Task"] = {}
        for agent, task_idx in pairs:
            task = pending[task_idx]
            self.task_manager.mark_assigned(task.task_id, agent)
            assignment[agent] = self.task_manager.get(task.task_id)
            self.history.append(
                DispatchRecord(self._episode, agent, task.location, cost[idle_agents.index(agent)][task_idx])
            )

        return assignment

    def handle_agent_failure(
        self,
        failed_agent: str,
        idle_positions: Dict[str, Point3],
    ) -> Dict[str, "Task"]:
        """
        Task-aware failure hook: releases every task owned by
        `failed_agent` back to PENDING via task_manager, then tries to
        hand pending work to whoever is idle right now.

        `idle_positions` is the caller's view of who's free -- the failed
        agent is always excluded from its own reassignment pool even if
        the caller includes it by mistake. If nobody is idle, released
        tasks are simply left PENDING (not lost, not crashed on) until
        the next call finds someone free.
        """
        if self.task_manager is None:
            raise ValueError("handle_agent_failure() requires this TaskCoordinator "
                              "to be constructed with a task_manager")

        self.task_manager.release_agent_tasks(failed_agent)

        live_positions = {a: p for a, p in idle_positions.items() if a != failed_agent}
        if not live_positions:
            return {}

        return self.assign_pending(live_positions)

    def reassign_on_failure(
        self,
        failed_agent: str,
        agent_positions: Dict[str, Point3],
        pending_goals: List[Point3],
    ) -> Dict[str, Point3]:
        """
        Failure-handling hook: if an agent drops out mid-mission (e.g. lost
        connection, collision), redistribute its pending goal(s) among the
        remaining live agents. Returns a fresh assignment for everyone else.
        """
        live_agents = [a for a in self.agent_names if a != failed_agent]
        if not live_agents:
            return {}
        sub_coord = TaskCoordinator(agent_names=live_agents)
        live_positions = {a: agent_positions[a] for a in live_agents}
        return sub_coord.assign(live_positions, pending_goals)

    def summary(self) -> str:
        lines = [f"Episode {r.episode}: {r.agent_name} -> goal {r.goal} "
                 f"(assignment cost {r.cost_at_assignment:.2f})"
                 for r in self.history[-len(self.agent_names):]]
        return "\n".join(lines)


if __name__ == "__main__":
    # Tiny smoke test you can run without AirSim to sanity-check the
    # allocation logic in isolation.
    coord = TaskCoordinator(agent_names=["Drone1", "Drone2", "Drone3"])
    positions = {
        "Drone1": (0.0, 0.0, -5.0),
        "Drone2": (50.0, 0.0, -5.0),
        "Drone3": (0.0, 50.0, -5.0),
    }
    goals = [(-34.6, -6.24, -1.0), (48.0, 5.0, -3.0), (2.0, 47.0, -2.0)]
    result = coord.assign(positions, goals)
    for agent, goal in result.items():
        print(f"{agent} -> {goal}")
    print("\n--- history ---")
    print(coord.summary())
