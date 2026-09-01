"""
multi_agent_train.py

Decentralized multi-agent extension of train_sandbox.py.

Design (be able to explain this in an interview):
  - N drones, each with its OWN QNetwork, OWN optimizer, and OWN replay
    buffer. There is no shared/joint policy and no centralized critic --
    this is Independent Q-Learning (IQL), the standard "decentralized RL"
    MARL baseline. Each agent only ever sees its own observation and
    only ever updates its own weights.
  - The TaskCoordinator (coordinator.py) is the ONLY centralized piece:
    once per episode it looks at all drones' start positions and the
    active goal list, and assigns each drone a goal via Hungarian
    matching. After that, agents act independently for the whole episode.
  - This mirrors your existing single-agent AirSimDroneEnv almost line
    for line -- MultiDroneEnv just wraps K of them and steps them together.

Known limitations (be ready to state these honestly):
  - AirSim/Unreal RPC calls for K drones are issued sequentially inside
    asyncio.gather per tick, so wall-clock step time grows with agent
    count -- this does not scale to large swarms without a proper
    batched/async sim client.
  - No collision-avoidance *between* drones is modeled beyond each
    agent's own LiDAR (a drone doesn't inherently see AirSim's other
    drone actors as obstacles unless they show up in its point cloud).
  - Independent Q-learning is known to be non-stationary from each
    agent's point of view (other agents' policies are shifting under
    it) -- it can be less stable than centralized-training/
    decentralized-execution methods like QMIX. That's a deliberate
    scope trade-off for a small, explainable v1, not an oversight.
"""

import asyncio
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.buffers import ReplayBuffer

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log

from coordinator import TaskCoordinator


AGENT_NAMES = ["Drone1", "Drone2", "Drone3"]

# Candidate goal points -- must be >= len(AGENT_NAMES). The coordinator
# assigns each agent one of these (nearest-cost matching) at every reset.
GOAL_POOL = [
    (-34.6, -6.24, -1.0),
    (48.0, 5.0, -3.0),
    (2.0, 47.0, -2.0),
]


class SingleDroneHandle:
    """
    Thin per-drone observation/action logic, reused across agents.
    Mirrors AirSimDroneEnv from train_sandbox.py but without owning its
    own asyncio loop or gym.Env identity -- MultiDroneEnv drives all of
    these together so drones step in lockstep each tick.
    """

    NUM_SECTORS = 36
    MAX_LIDAR_RANGE = 20.0
    Z_CEILING = -20.0

    def __init__(self, client: ProjectAirSimClient, world: World, name: str):
        self.name = name
        self.drone = Drone(client, world, name)
        self.start_pose = self.drone.get_ground_truth_pose()
        self.goal = (0.0, 0.0, 0.0)
        self.previous_distance = 0.0
        self.latest_lidar_data: List = []

        client.subscribe(
            self.drone.sensors["Lidar1"]["lidar"],
            self._update_lidar_cache,
        )

    def _update_lidar_cache(self, topic, lidar_msg):
        self.latest_lidar_data = lidar_msg.get("point_cloud", [])

    def _process_lidar(self) -> np.ndarray:
        binned = np.full(self.NUM_SECTORS, self.MAX_LIDAR_RANGE, dtype=np.float32)
        points = self.latest_lidar_data
        if len(points) < 3:
            return binned

        pts = np.array(points, dtype=np.float32).reshape(-1, 3)
        for x, y, z in pts:
            if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x):
                continue
            distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            angle_deg = (math.degrees(math.atan2(y, x)) + 360) % 360
            sector = int(angle_deg // (360 / self.NUM_SECTORS))
            if distance < binned[sector]:
                binned[sector] = distance
        return np.nan_to_num(binned, nan=self.MAX_LIDAR_RANGE,
                              posinf=self.MAX_LIDAR_RANGE, neginf=0.0)

    def build_observation(self, pos: Dict[str, float]) -> np.ndarray:
        lidar_obs = self._process_lidar()
        dx = self.goal[0] - pos["x"]
        dy = self.goal[1] - pos["y"]
        dz = self.goal[2] - pos["z"]
        distance = math.dist([pos["x"], pos["y"], pos["z"]], list(self.goal))
        spatial = np.array([dx / 200.0, dy / 200.0, dz / 200.0, distance / 200.0],
                            dtype=np.float32)
        return np.concatenate([lidar_obs, spatial])

    async def reset(self, goal):
        self.goal = goal
        self.drone.enable_api_control()
        self.drone.arm()
        self.drone.set_pose(self.start_pose)
        await asyncio.sleep(0.1)
        await self.drone.move_by_velocity_z_async(
            0.0, 0.0, self.start_pose["translation"]["z"], duration=1.0
        )
        state = self.drone.get_ground_truth_pose()
        pos = state["translation"]
        self.previous_distance = math.dist([pos["x"], pos["y"], pos["z"]], list(self.goal))
        return self.build_observation(pos)

    async def step(self, action: int):
        v_x = v_y = v_z = 0.0
        if action == 0: v_x = 4.0
        elif action == 1: v_x = -4.0
        elif action == 2: v_y = 4.0
        elif action == 3: v_y = -4.0
        elif action == 4: v_z = -6.0
        elif action == 5: v_z = 4.0

        await self.drone.move_by_velocity_async(v_x, v_y, v_z, duration=0.5)

        state = self.drone.get_ground_truth_pose()
        pos = state["translation"]
        obs = self.build_observation(pos)

        reward = 0.0
        terminated = False
        current_distance = math.dist([pos["x"], pos["y"], pos["z"]], list(self.goal))
        reward += (self.previous_distance - current_distance) * 5.0
        self.previous_distance = current_distance

        if current_distance < 2.0:
            reward += 100.0
            terminated = True

        if pos["z"] < self.Z_CEILING:
            reward -= (abs(pos["z"] - self.Z_CEILING) ** 2) * 0.25

        min_lidar = float(np.min(obs[:self.NUM_SECTORS]))
        if min_lidar < 1.0:
            reward -= 50.0
            terminated = True

        return obs, reward, terminated, pos


class MultiDroneEnv:
    """
    Owns N SingleDroneHandle objects and the TaskCoordinator. Not a
    gym.Env itself (multi-agent step/reset don't fit gym's single-agent
    API cleanly) -- exposes dict-keyed reset()/step() instead, which the
    training loop below consumes directly.
    """

    def __init__(self, agent_names: List[str]):
        self.agent_names = agent_names
        self.loop = asyncio.get_event_loop()
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_multi_drone.jsonc")
        self.agents = {
            name: SingleDroneHandle(self.client, self.world, name)
            for name in agent_names
        }
        self.coordinator = TaskCoordinator(agent_names=agent_names)
        self.max_steps = 500
        self.step_count = 0
        self.active = {name: True for name in agent_names}

    def _agent_positions(self) -> Dict[str, tuple]:
        positions = {}
        for name, handle in self.agents.items():
            t = handle.drone.get_ground_truth_pose()["translation"]
            positions[name] = (t["x"], t["y"], t["z"])
        return positions

    def reset(self):
        self.step_count = 0
        self.active = {name: True for name in self.agent_names}
        assignment = self.coordinator.assign(self._agent_positions(), GOAL_POOL)

        async def _reset_all():
            return await asyncio.gather(*[
                self.agents[name].reset(assignment[name]) for name in self.agent_names
            ])

        obs_list = self.loop.run_until_complete(_reset_all())
        return dict(zip(self.agent_names, obs_list))

    def step(self, actions: Dict[str, int]):
        live = [n for n in self.agent_names if self.active[n]]

        async def _step_live():
            return await asyncio.gather(*[
                self.agents[n].step(actions[n]) for n in live
            ])

        results = self.loop.run_until_complete(_step_live())
        self.step_count += 1

        obs, rewards, terms, truncs = {}, {}, {}, {}
        for name, (o, r, t, pos) in zip(live, results):
            obs[name] = o
            rewards[name] = r
            terms[name] = t
            truncs[name] = self.step_count >= self.max_steps
            if t:
                self.active[name] = False
                # Failure-handling: redistribute this agent's goal among
                # the survivors instead of leaving it orphaned.
                remaining_goals = [self.agents[n].goal for n in self.agent_names
                                    if self.active[n]]
                if remaining_goals and any(self.active.values()):
                    self.coordinator.reassign_on_failure(
                        name, self._agent_positions(), remaining_goals
                    )

        for name in self.agent_names:
            if name not in obs:
                obs[name] = None  # agent already terminated earlier this episode

        return obs, rewards, terms, truncs

    def close(self):
        self.client.disconnect()


class QNetwork(nn.Module):
    """Same architecture as train_sandbox.py -- kept identical on purpose
    so single-agent and multi-agent runs are directly comparable."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class IndependentAgent:
    """One fully independent learner: own network, own target network,
    own optimizer, own replay buffer. No parameter sharing across agents."""

    def __init__(self, name: str, obs_dim: int, n_actions: int, device):
        self.name = name
        self.device = device
        self.q = QNetwork(obs_dim, n_actions).to(device)
        self.target = QNetwork(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=2.5e-4)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.Discrete(n_actions)
        self.buffer = ReplayBuffer(10000, obs_space, act_space, device,
                                    handle_timeout_termination=False)

    def act(self, obs: np.ndarray, epsilon: float, n_actions: int):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        with torch.no_grad():
            q_values = self.q(torch.Tensor(obs).to(self.device))
        return int(torch.argmax(q_values).cpu().numpy())

    def learn(self, batch_size: int, gamma: float):
        if self.buffer.size() < batch_size:
            return None
        data = self.buffer.sample(batch_size)
        with torch.no_grad():
            target_max, _ = self.target(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())
        old_val = self.q(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self, tau: float = 1.0):
        for tp, p in zip(self.target.parameters(), self.q.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


if __name__ == "__main__":
    projectairsim_log().info("Starting decentralized multi-agent DQN training...")

    OBS_DIM = 40
    N_ACTIONS = 6
    total_timesteps = 2_000_000
    batch_size = 128
    gamma = 0.99
    start_e, end_e, exploration_fraction = 1.0, 0.05, 0.5
    learning_starts = 5000
    train_frequency = 4
    target_network_frequency = 500

    device = torch.device("cpu")
    env = MultiDroneEnv(AGENT_NAMES)

    agents = {
        name: IndependentAgent(name, OBS_DIM, N_ACTIONS, device)
        for name in AGENT_NAMES
    }

    writer = SummaryWriter(f"runs/marl_iql_{int(time.time())}")
    os.makedirs("./models/multi_agent_checkpoints/", exist_ok=True)

    obs = env.reset()

    try:
        for global_step in range(total_timesteps):
            epsilon = max(end_e, start_e - (global_step / (exploration_fraction * total_timesteps))
                          * (start_e - end_e))

            actions = {}
            for name in AGENT_NAMES:
                if obs[name] is not None:
                    actions[name] = agents[name].act(obs[name], epsilon, N_ACTIONS)
                else:
                    actions[name] = 0  # inert action for a terminated agent

            next_obs, rewards, terms, truncs = env.step(actions)

            for name in AGENT_NAMES:
                if obs[name] is not None and next_obs[name] is not None:
                    agents[name].buffer.add(
                        obs[name], next_obs[name], actions[name],
                        rewards[name], terms[name], {}
                    )

            obs = next_obs

            if all(terms[n] or truncs[n] or obs[n] is None for n in AGENT_NAMES):
                obs = env.reset()

            if global_step > learning_starts and global_step % train_frequency == 0:
                for name in AGENT_NAMES:
                    loss = agents[name].learn(batch_size, gamma)
                    if loss is not None and global_step % 100 == 0:
                        writer.add_scalar(f"losses/{name}_td_loss", loss, global_step)

            if global_step % target_network_frequency == 0:
                for name in AGENT_NAMES:
                    agents[name].update_target()

            if global_step > 0 and global_step % 5000 == 0:
                for name in AGENT_NAMES:
                    torch.save(
                        agents[name].q.state_dict(),
                        f"./models/multi_agent_checkpoints/{name}_{global_step}.pt",
                    )

    except KeyboardInterrupt:
        projectairsim_log().info("Interrupted -- saving weights for all agents.")
        for name in AGENT_NAMES:
            torch.save(agents[name].q.state_dict(),
                       f"./models/{name}_interrupted.pt")
    finally:
        env.close()
        writer.close()

    for name in AGENT_NAMES:
        torch.save(agents[name].q.state_dict(), f"./models/{name}_final.pt")
    projectairsim_log().info("Multi-agent training complete.")
