import asyncio
import math
import numpy as np
import time
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log


# =====================================================================
# 1. ENVIRONMENT (Identical to your fixed training env)
# =====================================================================
class AirSimDroneEnv(gym.Env):
    """Custom Gymnasium Environment for Project AirSim Drone Evaluation"""

    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        self.loop = asyncio.get_event_loop()
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_rl_train.jsonc")
        self.drone = Drone(self.client, self.world, "Drone1")
        self.start_pose = self.drone.get_ground_truth_pose()

        '''
        self.start_pose['translation']['x'] = 149.1
        self.start_pose['translation']['y'] = 35.0
        self.start_pose['translation']['z'] = -7.0
        '''
        self.num_sectors = 36
        self.max_lidar_range = 20.0

        # 40 Elements: 36 LiDAR + 4 Spatial
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_sectors + 4,), dtype=np.float32
        )

        # 6 Actions: 4 Horizontal, 2 Vertical
        self.action_space = spaces.Discrete(6)

        self.goal_x = -52.7
        self.goal_y = -57.6
        self.goal_z = -5

        self.z_ceiling = -50.0
        self.step_count = 0
        self.max_steps = 1000  # Give it plenty of time to reach the goal

        self.latest_lidar_data = []
        self.client.subscribe(
            self.drone.sensors["Lidar1"]["lidar"],
            self._update_lidar_cache
        )

    def _build_observation(self, pos):
        lidar_obs = self._process_lidar()
        dx = self.goal_x - pos['x']
        dy = self.goal_y - pos['y']
        dz = self.goal_z - pos['z']
        distance = math.dist([pos['x'], pos['y'], pos['z']], [self.goal_x, self.goal_y, self.goal_z])

        spatial_obs = np.array([
            dx / 200.0, dy / 200.0, dz / 200.0, distance / 200.0
        ], dtype=np.float32)

        return np.concatenate([lidar_obs, spatial_obs])

    def _update_lidar_cache(self, topic, lidar_msg):
        self.latest_lidar_data = lidar_msg.get('point_cloud', [])

    def _process_lidar(self):
        binned_distances = np.full(self.num_sectors, self.max_lidar_range, dtype=np.float32)
        points = self.latest_lidar_data
        if len(points) < 3: return binned_distances

        points_3d = np.array(points, dtype=np.float32).reshape(-1, 3)
        for point in points_3d:
            x, y, z = point
            if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x): continue

            distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            angle = math.atan2(y, x)
            angle_degrees = (math.degrees(angle) + 360) % 360
            sector_idx = int(angle_degrees // (360 / self.num_sectors))
            if distance < binned_distances[sector_idx]:
                binned_distances[sector_idx] = distance

        return np.nan_to_num(binned_distances, nan=self.max_lidar_range, posinf=self.max_lidar_range, neginf=0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        async def _reset_drone():
            # Soft stop before teleport
            await self.drone.move_by_velocity_async(0, 0, 0, duration=0.1)
            self.drone.enable_api_control()
            self.drone.disarm()
            self.drone.set_pose(self.start_pose)
            self.drone.arm()
            await self.drone.takeoff_async()

        self.loop.run_until_complete(_reset_drone())
        state = self.drone.get_ground_truth_pose()
        return self._build_observation(state['translation']), {}

    def step(self, action):
        self.step_count += 1
        v_x, v_y, v_z = 0.0, 0.0, 0.0

        if action == 0:
            v_x = 4.0
        elif action == 1:
            v_x = -4.0
        elif action == 2:
            v_y = 4.0
        elif action == 3:
            v_y = -4.0
        elif action == 4:
            v_z = -4.0
        elif action == 5:
            v_z = 4.0

        async def _move():
            await self.drone.move_by_velocity_async(v_x, v_y, v_z, duration=0.5)

        self.loop.run_until_complete(_move())

        # Highly recommend uncommenting this for evaluation so you can actually watch it fly smoothly
        # time.sleep(0.5)

        state = self.drone.get_ground_truth_pose()
        pos = state['translation']
        obs = self._build_observation(pos)

        reward = 0.0
        terminated = False
        truncated = False

        current_distance = math.dist([pos['x'], pos['y'], pos['z']], [self.goal_x, self.goal_y, self.goal_z])

        if current_distance < 2.0:
            reward += 100
            terminated = True
            projectairsim_log().info("Goal Reached!")

        if pos['z'] < self.z_ceiling:
            reward -= 10
            terminated = True
            projectairsim_log().info("Ceiling breached!")

        min_distance = np.min(obs[:self.num_sectors])
        if min_distance < 1.0:
            reward -= 10
            terminated = True
            projectairsim_log().info("Collision Avoidance Triggered!")

        if self.step_count >= self.max_steps:
            truncated = True

        # decay reward
        reward -= 0.03

        # distance reward
        reward -= current_distance

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.client.disconnect()


# =====================================================================
# 2. Q-NETWORK (Must match training architecture exactly)
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
# 3. EVALUATION LOOP
# =====================================================================
if __name__ == "__main__":
    projectairsim_log().info("Starting DQN Evaluation...")

    # Setup device (CPU is usually fine for single-agent inference)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = AirSimDroneEnv()

    # 1. Initialize the network
    q_network = QNetwork(env).to(device)

    # 2. Load the trained weights
    model_path = "./models/dqn_lidar_drone_final.pt"
    try:
        q_network.load_state_dict(torch.load(model_path, map_location=device))
        projectairsim_log().info(f"Successfully loaded weights from {model_path}")
    except FileNotFoundError:
        projectairsim_log().error(f"Could not find model at {model_path}. Please check the path.")
        exit()

    # 3. Set the network to Evaluation Mode
    q_network.eval()

    obs, _ = env.reset()
    done = False

    projectairsim_log().info("Running inference...")

    while not done:
        # torch.no_grad() disables backpropagation memory tracking, making inference much faster
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            q_values = q_network(obs_tensor)

            # Pure exploitation - always pick the highest Q-value action
            action = torch.argmax(q_values).cpu().numpy()

        # Execute the action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    projectairsim_log().info("Evaluation run complete.")
    env.close()