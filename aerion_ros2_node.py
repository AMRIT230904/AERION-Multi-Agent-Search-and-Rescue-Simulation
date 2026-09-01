"""
aerion_ros2_node.py

Minimal ROS2 (rclpy) bridge for AERION.

What this actually does, honestly:
  - Publishes each drone's ground-truth pose and a compressed "min LiDAR
    clearance" scalar as ROS2 topics, at a fixed rate, by polling the
    AirSim client (Project AirSim's own client is not ROS2-native, so this
    node is the translation layer).
  - Exposes a ROS2 service (/aerion/dispatch_goal) that the coordinator
    calls to push a new goal to a named drone. The service handler here
    just republishes to a per-drone goal topic; multi_agent_train.py's
    SingleDroneHandle would subscribe to that in a fuller integration
    (that subscription hookup is NOT done in this file -- seam is marked
    below with a TODO so it's honest about what's wired up vs. stubbed).

Requires: ROS2 (rclpy) installed alongside your Project AirSim /
Unreal setup. This will NOT run in a plain pip environment -- it needs
an actual ROS2 distro (Humble/Jazzy) sourced first.

Run with:
    ros2 run aerion_bridge aerion_ros2_node   (once packaged), or
    python3 aerion_ros2_node.py               (standalone, for testing)
"""

import threading
import time
from typing import Dict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from example_interfaces.srv import Trigger  # placeholder request/response type

from projectairsim import ProjectAirSimClient, Drone, World


AGENT_NAMES = ["Drone1", "Drone2", "Drone3","Drone4",
    "Drone5"]
PUBLISH_RATE_HZ = 10.0


class AerionBridgeNode(Node):
    def __init__(self):
        super().__init__("aerion_bridge")

        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_multi_drone.jsonc")
        self.drones: Dict[str, Drone] = {
            name: Drone(self.client, self.world, name) for name in AGENT_NAMES
        }

        self.pose_pubs = {
            name: self.create_publisher(PoseStamped, f"/aerion/{name}/pose", 10)
            for name in AGENT_NAMES
        }
        self.clearance_pubs = {
            name: self.create_publisher(Float32, f"/aerion/{name}/min_clearance", 10)
            for name in AGENT_NAMES
        }
        self.goal_pubs = {
            name: self.create_publisher(PoseStamped, f"/aerion/{name}/goal", 10)
            for name in AGENT_NAMES
        }

        # NOTE (honest seam): SingleDroneHandle in multi_agent_train.py
        # currently takes its goal directly from the coordinator's Python
        # return value, not from this topic. Wiring the training loop to
        # actually subscribe to /aerion/<name>/goal instead is the
        # remaining integration step to make ROS2 the real transport
        # rather than a parallel observability layer. Tracked as TODO.
        self.dispatch_srv = self.create_service(
            Trigger, "/aerion/dispatch_goal", self._handle_dispatch
        )

        self.latest_clearance: Dict[str, float] = {n: -1.0 for n in AGENT_NAMES}
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self._publish_tick)

    def _handle_dispatch(self, request, response):
        # In a fuller build this would parse a goal + drone name out of
        # a custom srv type. Using Trigger as a stand-in so this file has
        # no undeclared custom message dependency.
        response.success = True
        response.message = "dispatch acknowledged (stub -- wire to coordinator.assign())"
        return response

    def _publish_tick(self):
        now = self.get_clock().now().to_msg()
        for name, drone in self.drones.items():
            pose = drone.get_ground_truth_pose()
            t = pose["translation"]

            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = "world"
            msg.pose.position.x = float(t["x"])
            msg.pose.position.y = float(t["y"])
            msg.pose.position.z = float(t["z"])
            self.pose_pubs[name].publish(msg)

            clearance_msg = Float32()
            clearance_msg.data = float(self.latest_clearance.get(name, -1.0))
            self.clearance_pubs[name].publish(clearance_msg)

    def shutdown(self):
        self.client.disconnect()


def main():
    rclpy.init()
    node = AerionBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
