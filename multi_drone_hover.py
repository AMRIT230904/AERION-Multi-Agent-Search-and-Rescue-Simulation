"""
multi_drone_hover.py

Connects to Project AirSim, spawns 5 drones (Drone1..Drone5) from
scene_multi_drone.jsonc at 5 different spread-out points in the city,
arms them, takes off, and holds them in hover indefinitely.

Press Ctrl+C to land/disarm and disconnect cleanly.
"""

import asyncio
import random

import pynng

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log


DRONE_NAMES = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5"]


async def arm_and_takeoff(drone: Drone, name: str):
    drone.enable_api_control()
    drone.arm()

    # Log altitude BEFORE takeoff so we can confirm whether it actually climbs
    pose_before = drone.get_ground_truth_pose()
    z_before = pose_before["translation"]["z"]
    projectairsim_log().info(f"{name}: altitude before takeoff = {z_before:.2f} (NED, negative=up)")

    projectairsim_log().info(f"{name}: taking off...")
    await drone.takeoff_async()

    # Give the controller a brief moment to actually climb, then re-check
    await asyncio.sleep(1.0)
    pose_after = drone.get_ground_truth_pose()
    z_after = pose_after["translation"]["z"]
    projectairsim_log().info(f"{name}: altitude after takeoff = {z_after:.2f}")

    if abs(z_after - z_before) < 0.3:
        # takeoff_async did not actually move the drone -- fall back to an
        # explicit velocity-Z command to kick the controller into hold mode
        # (same trick used in train_sandbox.py's reset()).
        projectairsim_log().info(f"{name}: takeoff_async had no effect, forcing climb via velocity_z command")
        target_z = z_before - 5.0  # climb 5 meters (more negative = higher)
        await drone.move_by_velocity_z_async(0.0, 0.0, target_z, duration=2.0)
        await asyncio.sleep(0.5)
        pose_final = drone.get_ground_truth_pose()
        projectairsim_log().info(f"{name}: altitude after forced climb = {pose_final['translation']['z']:.2f}")

    projectairsim_log().info(f"{name}: airborne / hovering.")


async def random_move_tick(drone: Drone, name: str, duration: float = 0.5, speed: float = 1.5):
    # Small random horizontal velocity each tick -- enough to visibly
    # drift/wander without flying off, vz stays 0 so altitude is roughly held.
    vx = random.uniform(-speed, speed)
    vy = random.uniform(-speed, speed)
    try:
        await drone.move_by_velocity_async(vx, vy, 0.0, duration=duration)
    except pynng.exceptions.Timeout:
        # A single missed reply on one tick isn't fatal -- log it and
        # let the next tick try again.
        projectairsim_log().info(f"{name}: random_move_tick timed out, retrying next loop")


def print_positions(drones: dict):
    # Pull ground-truth pose for each drone and print as one array
    # so you can see all 5 positions update together every tick.
    positions = []
    for name, drone in drones.items():
        t = drone.get_ground_truth_pose()["translation"]
        positions.append((name, round(t["x"], 2), round(t["y"], 2), round(t["z"], 2)))
    print(positions)


async def main():
    client = ProjectAirSimClient()
    client.connect()

    # 5 robot actors at 5 different spawn points across the city
    world = World(client, "scene_multi_drone.jsonc")

    drones = {name: Drone(client, world, name) for name in DRONE_NAMES}

    # Take off all 5 concurrently
    await asyncio.gather(*[
        arm_and_takeoff(drone, name) for name, drone in drones.items()
    ])

    projectairsim_log().info("All 5 drones airborne. Press Ctrl+C to land and exit.")

    try:
        # Small random jitter every 0.5s, printing positions each tick
        # so you can visually confirm all 5 drones are actually moving.
        while True:
            await asyncio.gather(*[
                random_move_tick(drone, name, duration=0.5, speed=1.5)
                for name, drone in drones.items()
            ])
            print_positions(drones)
            await asyncio.sleep(0.5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        projectairsim_log().info("Stopping... landing all drones.")

        async def _land_and_disarm(name: str, drone: Drone):
            try:
                await drone.land_async()
            except pynng.exceptions.Timeout:
                projectairsim_log().info(f"{name}: land_async timed out, disarming anyway")
            try:
                drone.disarm()
            except pynng.exceptions.Timeout:
                projectairsim_log().info(f"{name}: disarm timed out")

        await asyncio.gather(*[
            _land_and_disarm(name, drone) for name, drone in drones.items()
        ])

    await asyncio.sleep(1.0)
    projectairsim_log().info("Disconnecting.")
    try:
        client.disconnect()
    except Exception as e:
        projectairsim_log().info(f"(non-fatal disconnect cleanup: {e})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass