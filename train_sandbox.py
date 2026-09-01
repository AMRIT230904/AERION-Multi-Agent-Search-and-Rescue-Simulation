import asyncio
import math
import numpy as np
import random
import os
import time
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# CleanRL often relies on SB3's replay buffer for simplicity!
from stable_baselines3.common.buffers import ReplayBuffer

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log


# =====================================================================
# 1. ENVIRONMENT (Unchanged from your implementation)
# =====================================================================
class AirSimDroneEnv(gym.Env):
    """Custom Gymnasium Environment for Project AirSim Drone with LiDAR"""

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        self.loop = asyncio.get_event_loop()
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_rl_train.jsonc")
        self.drone = Drone(self.client, self.world, "Drone1")
        self.start_pose = self.drone.get_ground_truth_pose()

        self.num_sectors = 36
        self.max_lidar_range = 20.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_sectors + 4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        self.goal_x = -34.6
        self.goal_y = -6.24
        self.goal_z = -1

        #       self.goal_x = self.start_pose['translation']['x'] - 201.8
        #       self.goal_y = self.start_pose['translation']['y'] - 92.6
        #       self.goal_z = self.start_pose['translation']['z']

        self.z_ceiling = -20.0

        self.previous_distance = 0.0
        self.step_count = 0
        self.max_steps = 500

        self.latest_lidar_data = []
        self.client.subscribe(
            self.drone.sensors["Lidar1"]["lidar"],
            self._update_lidar_cache
        )

    def _build_observation(self, pos):
        """Combines LiDAR data with normalized spatial awareness"""

        # 1. Get the 36 LiDAR sectors
        lidar_obs = self._process_lidar()

        # 2. Calculate the relative vector to the goal
        dx = self.goal_x - pos['x']
        dy = self.goal_y - pos['y']
        dz = self.goal_z - pos['z']

        # 3. Calculate total 3D distance
        distance = math.dist([pos['x'], pos['y'], pos['z']], [self.goal_x, self.goal_y, self.goal_z])

        # 4. Normalize the spatial data (divide by estimated max map size)
        # This keeps the numbers small (mostly between -1 and 1) so the network stays stable
        spatial_obs = np.array([
            dx / 200.0,
            dy / 200.0,
            dz / 200.0,
            distance / 200.0
        ], dtype=np.float32)

        # 5. Stitch them together into a single 40-element array
        full_obs = np.concatenate([lidar_obs, spatial_obs])
        return full_obs

    def set_max_steps(self, new_max):
        self.max_steps = new_max

    def _update_lidar_cache(self, topic, lidar_msg):
        self.latest_lidar_data = lidar_msg.get('point_cloud', [])

    def _process_lidar(self):
        binned_distances = np.full(self.num_sectors, self.max_lidar_range, dtype=np.float32)
        points = self.latest_lidar_data

        if len(points) < 3:
            return binned_distances

        points_3d = np.array(points, dtype=np.float32).reshape(-1, 3)

        for point in points_3d:
            x, y, z = point
            if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x):
                continue

            distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            angle = math.atan2(y, x)
            angle_degrees = (math.degrees(angle) + 360) % 360
            sector_idx = int(angle_degrees // (360 / self.num_sectors))

            if distance < binned_distances[sector_idx]:
                binned_distances[sector_idx] = distance

        clean_obs = np.nan_to_num(
            binned_distances,
            nan=self.max_lidar_range,
            posinf=self.max_lidar_range,
            neginf=0.0
        )
        return clean_obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.target_z = self.start_pose['translation']['z']

        async def _reset_drone():
            self.drone.enable_api_control()
            self.drone.arm()
            self.drone.set_pose(self.start_pose)
            await asyncio.sleep(0.1)
            # hover_async does not engage altitude hold after set_pose — drone free-falls.
            # An explicit position command kicks the controller into altitude-hold mode.
            await self.drone.move_by_velocity_z_async(
                0.0, 0.0, self.target_z, duration=1.0
            )

        self.loop.run_until_complete(_reset_drone())

        state = self.drone.get_ground_truth_pose()
        pos = state['translation']
        self.previous_distance = math.dist([pos['x'], pos['y']], [self.goal_x, self.goal_y])

        obs = self._build_observation(pos)
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1
        v_x, v_y, v_z = 0.0, 0.0, 0.0

        if action == 0:
            v_x = 4.0  # Forward
        elif action == 1:
            v_x = -4.0  # Backward
        elif action == 2:
            v_y = 4.0  # Right
        elif action == 3:
            v_y = -4.0  # Left
        elif action == 4:
            v_z = -6.0  # UP (Negative Z in AirSim)
        elif action == 5:
            v_z = 4.0  # DOWN (Positive Z in AirSim)

        async def _move():
            # Go back to using the full 3D velocity command
            await self.drone.move_by_velocity_async(v_x, v_y, v_z, duration=0.5)
            # await self.drone.move_by_velocity_z_async(v_x, v_y, self.start_pose['translation']['z'],duration=0.5)

        self.loop.run_until_complete(_move())
        # time.sleep(0.5) # Keep this to prevent RPC flooding!

        state = self.drone.get_ground_truth_pose()
        pos = state['translation']
        obs = self._build_observation(pos)

        state = self.drone.get_ground_truth_pose()
        pos = state['translation']

        reward = 0.0
        terminated = False
        truncated = False

        current_distance = math.dist(
            [pos['x'], pos['y'], pos['z']],
            [self.goal_x, self.goal_y, self.goal_z]
        )
        reward += (self.previous_distance - current_distance) * 5.0
        self.previous_distance = current_distance

        if current_distance < 2.0:
            reward += 100.0
            terminated = True
            projectairsim_log().info("Goal Reached!")

        if pos['z'] < self.z_ceiling:
            overage = abs(pos['z'] - self.z_ceiling) ** 2
            reward -= (overage * 0.25)

        min_distance = np.min(obs[:self.num_sectors])

        if min_distance < 1.0:
            reward -= 50.0
            terminated = True
            projectairsim_log().info("Collision Avoidance Triggered!")

        if self.step_count >= self.max_steps:
            truncated = True

        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.client.disconnect()


# =====================================================================
# 2. Q-NETWORK (CleanRL Style)
# =====================================================================
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x)


# =====================================================================
# 3. HYPERPARAMETERS & SETUP
# =====================================================================
if __name__ == "__main__":
    projectairsim_log().info("Starting CleanRL DQN Training Setup...")

    episode_count = 0
    curr_episode_freq = 50  # Increase difficulty every 50 episodes

    # Hyperparameters
    total_timesteps = 5000000
    learning_rate = 2.5e-4
    buffer_size = 10000
    batch_size = 128
    gamma = 0.99
    tau = 1.0
    target_network_frequency = 500
    start_e = 1.0
    end_e = 0.05
    exploration_fraction = 0.5
    learning_starts = 10000
    train_frequency = 4

    # Curriculum Settings
    curr_max_limit = 2000
    curr_increase_by = 100
    curr_freq = 25000

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Training on device: {device}")

    env = AirSimDroneEnv()

    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    rb = ReplayBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )

    writer = SummaryWriter(f"runs/dqn_airsim_{int(time.time())}")
    os.makedirs("./models/checkpoints/", exist_ok=True)

    obs, _ = env.reset()

    # =====================================================================
    # 4. EXPLICIT TRAINING LOOP (Replaces model.learn and Callbacks)
    # =====================================================================
    projectairsim_log().info("Beginning Training. Press Ctrl+C to save and exit gracefully.")

    try:
        for global_step in range(total_timesteps):

            # --- ACTION LOGIC (Epsilon Greedy) ---
            epsilon = max(end_e, start_e - (global_step / (exploration_fraction * total_timesteps)) * (start_e - end_e))
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(torch.Tensor(obs).to(device))
                    if torch.isnan(q_values).any():
                        print(f"CRITICAL: NaNs detected in Q-values at step {global_step}!")
                        break
                    action = torch.argmax(q_values).cpu().numpy()

            # --- ENVIRONMENT STEP ---
            next_obs, reward, terminated, truncated, info = env.step(action)

            # --- RECORD TRANSITION ---
            rb.add(obs, next_obs, action, reward, terminated, info)
            obs = next_obs

            if terminated or truncated:
                obs, _ = env.reset()

                episode_count += 1

                # --- EPISODE-BASED CURRICULUM ---
                if episode_count % curr_episode_freq == 0:
                    if env.max_steps < curr_max_limit:
                        env.set_max_steps(env.max_steps + curr_increase_by)
                        print(
                            f"\n[Curriculum] Reached {episode_count} episodes. Increased max steps to {env.max_steps}")

            # --- TRAINING NETWORK ---
            if global_step > learning_starts and global_step % train_frequency == 0:
                data = rb.sample(batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())

                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- UPDATE TARGET NETWORK ---
            if global_step % target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                    )

            # --- CUSTOM LOGIC: CHECKPOINTS (Replaces CheckpointCallback) ---
            if global_step > 0 and global_step % 5000 == 0:
                torch.save(q_network.state_dict(), f"./models/checkpoints/dqn_lidar_drone_{global_step}.pt")

    except KeyboardInterrupt:
        projectairsim_log().info("\nTraining interrupted by user! Saving current weights...")
        torch.save(q_network.state_dict(), "./models/dqn_lidar_drone_interrupted.pt")
        projectairsim_log().info("Model saved successfully. Exiting.")

    finally:
        env.close()
        writer.close()

    # Save final model
    torch.save(q_network.state_dict(), "./models/dqn_lidar_drone_final.pt")
    projectairsim_log().info("Training complete. Final model saved.")