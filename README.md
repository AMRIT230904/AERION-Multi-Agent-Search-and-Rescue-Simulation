# AERION — Multi-Agent Search-and-Rescue Simulation

AERION is a multi-agent reinforcement-learning simulation for autonomous aerial navigation and task allocation in Microsoft Project AirSim and Unreal Engine.

The system combines independent deep reinforcement-learning agents with a lightweight hierarchical coordinator that allocates navigation tasks based on priority and estimated travel cost.

---

## Overview

AERION explores decentralized agent execution with centralized high-level task allocation.

The current implementation provides:

- Project AirSim + Unreal Engine simulation
- LiDAR-based autonomous drone navigation
- 36-sector LiDAR observation processing
- Goal-directed DQN navigation
- Multiple simulated drone agents
- Independent Q-learning (IQL) agents with separate policies
- Priority-aware task management
- Hungarian-algorithm-based task assignment
- Agent failure detection and task reassignment logic
- ROS2 goal communication components
- Unit tests for task management, coordination, and failure recovery

The project is designed as a foundation for heterogeneous search-and-rescue missions involving aerial and ground agents.

---

# Architecture

```text
                       Search Mission
                             |
                             v
                    +------------------+
                    |   Task Manager   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   Prioritizer    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   Coordinator    |
                    |                  |
                    | Task Assignment |
                    +--------+---------+
                             |
                +------------+------------+
                |            |            |
                v            v            v
             Drone 1      Drone 2      Drone 3
                |            |            |
                v            v            v
             DQN Policy   DQN Policy   DQN Policy
                |            |            |
                +------------+------------+
                             |
                             v
                         Project AirSim
                             |
                             v
                       Unreal Engine
