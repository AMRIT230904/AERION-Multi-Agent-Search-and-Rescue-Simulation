"""
ros2_bridge/ros2_goal_listener.py

Companion to aerion_ros2_node.py. This is the piece that closes the loop
the v1 README flagged as missing: something on the training side has to
actually subscribe to /aerion/<name>/goal and update the agent's target,
instead of the goal only ever being set via a direct Python call.

Usage from multi_agent_train.py's SingleDroneHandle (sketch -- wiring
this into the class itself is the last integration step, left explicit
here rather than silently merged in, since it changes how the training
loop's asyncio loop and ROS2's own executor coexist):

    listener = GoalListener("Drone1")
    listener.start()          # runs rclpy spin on a background thread
    ...
    # each env.step() tick:
    latest = listener.get_latest_goal()
    if latest is not None:
        handle.current_task_location = latest   # or map to a Task if you
                                                  # want full Task semantics
                                                  # on this side too

Why a background thread: rclpy.spin() blocks, and the training loop
already owns its own asyncio event loop via run_until_complete(). Running
ROS2's executor on a separate thread avoids fighting over the same loop.
"""

import threading
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

Point3 = Tuple[float, float, float]


class _GoalSubscriberNode(Node):
    def __init__(self, agent_name: str, on_goal_update):
        super().__init__(f"aerion_goal_listener_{agent_name.lower()}")
        self._on_goal_update = on_goal_update
        self.create_subscription(
            PoseStamped,
            f"/aerion/{agent_name}/goal",
            self._callback,
            10,
        )

    def _callback(self, msg: PoseStamped):
        goal = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self._on_goal_update(goal)


class GoalListener:
    """Thread-safe holder for "the most recent goal this agent has been
    told about via ROS2." Poll get_latest_goal() from the training loop."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._lock = threading.Lock()
        self._latest_goal: Optional[Point3] = None
        self._node: Optional[_GoalSubscriberNode] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _set_goal(self, goal: Point3) -> None:
        with self._lock:
            self._latest_goal = goal

    def get_latest_goal(self) -> Optional[Point3]:
        with self._lock:
            return self._latest_goal

    def _spin(self):
        rclpy.init(args=None)
        self._node = _GoalSubscriberNode(self.agent_name, self._set_goal)
        while rclpy.ok() and not self._stop_event.is_set():
            rclpy.spin_once(self._node, timeout_sec=0.1)
        self._node.destroy_node()
        rclpy.shutdown()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
