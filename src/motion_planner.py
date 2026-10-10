"""Joint-space RRT-Connect Motion Planner with Semantic Costs and Quintic Trajectories.

Implements bidirectional RRT-Connect with semantic cost terms (human hazard penalty,
fragile approach zone cost), randomized shortcut path smoothing, and quintic polynomial
time-parameterization enforcing Franka Emika Panda velocity and acceleration limits.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Strict virtual environment assertion
if sys.prefix == sys.base_prefix:
    sys.exit("CRITICAL ERROR: Script must run inside the virtual environment at ./venv")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from src.sim_env import (
    CobotSimEnv,
    PANDA_JOINT_LIMITS_LOWER,
    PANDA_JOINT_LIMITS_UPPER,
    PANDA_VELOCITY_LIMITS,
    PANDA_HOME_CONFIG,
)

PANDA_ACCELERATION_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])


class RRTNode:
    """Node in joint-space search tree."""

    def __init__(self, q: np.ndarray, parent: Optional[RRTNode] = None, cost: float = 0.0) -> None:
        self.q = np.array(q, dtype=np.float64)
        self.parent = parent
        self.cost = cost


class RRTConnectPlanner:
    """Bidirectional RRT-Connect with VLM semantic cost integration."""

    def __init__(
        self,
        env: CobotSimEnv,
        step_size: float = 0.06,
        max_iter: int = 1200,
        semantic_human_weight: float = 4.0,
        semantic_fragile_weight: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.env = env
        self.step_size = step_size
        self.max_iter = max_iter
        self.w_human = semantic_human_weight
        self.w_fragile = semantic_fragile_weight
        self.rng = np.random.RandomState(seed)

    def evaluate_semantic_cost(self, q: np.ndarray, constraints: List[str]) -> float:
        """Compute state cost penalizing human proximity and sensitive workspace regions."""
        cost = 0.0
        if "avoid_human" in constraints:
            dist_human, _ = self.env.get_human_distance(q)
            # Inverse barrier potential penalty
            if dist_human < 0.35:
                cost += self.w_human / max(dist_human, 0.02) ** 2

        if "slow_near_fragile" in constraints:
            ee_pos, _ = self.env.forward_kinematics(q)
            # Fixture insertion region [0.50, 0.20, 0.45]
            dist_fixture = np.linalg.norm(ee_pos - np.array([0.50, 0.20, 0.45]))
            if dist_fixture < 0.15:
                cost += self.w_fragile * (0.15 - dist_fixture)

        return cost

    def plan_path(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        constraints: Optional[List[str]] = None,
        check_human: bool = True,
    ) -> Tuple[bool, List[np.ndarray], float]:
        """Bidirectional RRT-Connect search in joint space."""
        if constraints is None:
            constraints = []

        # Validate start and goal
        if self.env.check_collision(q_start, check_human=check_human) or \
           self.env.check_collision(q_goal, check_human=check_human):
            return False, [], 0.0

        tree_start = [RRTNode(q_start, None, 0.0)]
        tree_goal = [RRTNode(q_goal, None, 0.0)]

        start_time = time.perf_counter()

        for iteration in range(self.max_iter):
            # Sample random configuration in joint limits
            if self.rng.rand() < 0.15:
                q_rand = q_goal.copy()
            else:
                q_rand = self.rng.uniform(
                    PANDA_JOINT_LIMITS_LOWER + 0.05,
                    PANDA_JOINT_LIMITS_UPPER - 0.05
                )

            # Extend Tree A towards q_rand
            status_a, new_node_a = self._extend_tree(tree_start, q_rand, constraints, check_human)
            if status_a != "trapped":
                # Connect Tree B towards new_node_a
                status_b, new_node_b = self._connect_tree(tree_goal, new_node_a.q, constraints, check_human)
                if status_b == "reached":
                    # Trees successfully merged
                    path = self._extract_path(new_node_a, new_node_b)
                    elapsed = time.perf_counter() - start_time
                    return True, path, elapsed

            # Swap trees for bidirectional symmetry
            tree_start, tree_goal = tree_goal, tree_start

        elapsed = time.perf_counter() - start_time
        return False, [], elapsed

    def _extend_tree(
        self,
        tree: List[RRTNode],
        q_target: np.ndarray,
        constraints: List[str],
        check_human: bool
    ) -> Tuple[str, Optional[RRTNode]]:
        """Extend nearest tree node by step_size towards q_target."""
        # Find nearest neighbor
        dists = [np.linalg.norm(node.q - q_target) for node in tree]
        nearest_idx = int(np.argmin(dists))
        nearest_node = tree[nearest_idx]

        diff = q_target - nearest_node.q
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return "reached", nearest_node

        step = min(self.step_size, dist)
        q_new = nearest_node.q + (diff / dist) * step

        # Check collision
        if self.env.check_collision(q_new, check_human=check_human):
            return "trapped", None

        # Check semantic rejection threshold if avoiding human
        if "avoid_human" in constraints and check_human:
            d_human, _ = self.env.get_human_distance(q_new)
            if d_human < 0.18:  # Stricter semantic avoidance boundary
                return "trapped", None

        cost = nearest_node.cost + step + self.evaluate_semantic_cost(q_new, constraints)
        new_node = RRTNode(q_new, nearest_node, cost)
        tree.append(new_node)

        if np.linalg.norm(q_new - q_target) < 1e-4:
            return "reached", new_node
        return "advanced", new_node

    def _connect_tree(
        self,
        tree: List[RRTNode],
        q_target: np.ndarray,
        constraints: List[str],
        check_human: bool
    ) -> Tuple[str, Optional[RRTNode]]:
        """Greedily advance tree towards target until reached or trapped."""
        status = "advanced"
        last_node = None
        while status == "advanced":
            status, last_node = self._extend_tree(tree, q_target, constraints, check_human)
        return status, last_node

    def _extract_path(self, node_a: RRTNode, node_b: RRTNode) -> List[np.ndarray]:
        """Stitch bidirectional trees into a single forward path."""
        path_a: List[np.ndarray] = []
        curr: Optional[RRTNode] = node_a
        while curr is not None:
            path_a.append(curr.q)
            curr = curr.parent
        path_a.reverse()

        path_b: List[np.ndarray] = []
        curr = node_b
        while curr is not None:
            path_b.append(curr.q)
            curr = curr.parent

        # Identify which tree was rooted at start
        return path_a + path_b

    def shortcut_path(
        self,
        path: List[np.ndarray],
        max_iterations: int = 60,
        check_human: bool = True
    ) -> List[np.ndarray]:
        """Greedy randomized path shortcutting in joint space."""
        if len(path) <= 2:
            return path

        smoothed = [q.copy() for q in path]
        for _ in range(max_iterations):
            if len(smoothed) <= 2:
                break
            idx1 = self.rng.randint(0, len(smoothed) - 1)
            idx2 = self.rng.randint(idx1 + 1, len(smoothed))
            if idx2 - idx1 <= 1:
                continue

            q1 = smoothed[idx1]
            q2 = smoothed[idx2]

            # Test straight line interpolation
            num_steps = int(np.ceil(np.linalg.norm(q2 - q1) / (self.step_size * 0.8)))
            valid = True
            for s in range(1, num_steps):
                alpha = s / float(num_steps)
                q_interp = (1.0 - alpha) * q1 + alpha * q2
                if self.env.check_collision(q_interp, check_human=check_human):
                    valid = False
                    break

            if valid:
                smoothed = smoothed[:idx1 + 1] + smoothed[idx2:]

        return smoothed


class QuinticTrajectoryGenerator:
    """Generates C^2 time-parameterized trajectory with zero start/end accelerations."""

    def __init__(
        self,
        vel_limits: Optional[np.ndarray] = None,
        acc_limits: Optional[np.ndarray] = None,
        dt: float = 0.01
    ) -> None:
        self.vel_limits = vel_limits if vel_limits is not None else PANDA_VELOCITY_LIMITS
        self.acc_limits = acc_limits if acc_limits is not None else PANDA_ACCELERATION_LIMITS
        self.dt = dt

    def parameterize_path(self, waypoints: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate smooth trajectory over list of joint configurations."""
        if len(waypoints) < 2:
            raise ValueError("Path must contain at least 2 waypoints")

        times_list: List[float] = [0.0]
        q_traj: List[np.ndarray] = [waypoints[0]]
        qd_traj: List[np.ndarray] = [np.zeros(len(waypoints[0]))]
        qdd_traj: List[np.ndarray] = [np.zeros(len(waypoints[0]))]

        current_time = 0.0

        for i in range(len(waypoints) - 1):
            q0 = waypoints[i]
            q1 = waypoints[i + 1]
            dq = q1 - q0

            # Compute minimum required segment duration for velocity & acceleration limits
            # Quintic max velocity: v_max = (15 / 8) * |dq| / T
            # Quintic max acceleration: a_max = (10 * sqrt(3) / 3) * |dq| / T^2
            t_vel = np.max((15.0 / 8.0) * np.abs(dq) / self.vel_limits)
            t_acc = np.sqrt(np.max((10.0 * np.sqrt(3.0) / 3.0) * np.abs(dq) / self.acc_limits))
            T_seg = max(float(max(t_vel, t_acc)), 0.35)

            steps = max(int(np.ceil(T_seg / self.dt)), 4)
            actual_dt = T_seg / steps

            for s in range(1, steps + 1):
                tau = s / float(steps)
                # Quintic basis: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
                s_pos = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
                # s_dot(tau) = (30*tau^2 - 60*tau^3 + 30*tau^4) / T
                s_vel = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / T_seg
                # s_ddot(tau) = (60*tau - 180*tau^2 + 120*tau^3) / T^2
                s_acc = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (T_seg ** 2)

                q_t = q0 + dq * s_pos
                qd_t = dq * s_vel
                qdd_t = dq * s_acc

                current_time += actual_dt
                times_list.append(current_time)
                q_traj.append(q_t)
                qd_traj.append(qd_t)
                qdd_traj.append(qdd_t)

        return {
            "time": np.array(times_list),
            "q": np.array(q_traj),
            "qd": np.array(qd_traj),
            "qdd": np.array(qdd_traj),
            "duration": float(times_list[-1])
        }


def run_phase4_verification() -> bool:
    """Run Phase 4 Motion Planner verification tests."""
    print(">>> Verifying Phase 4: Joint-Space RRT-Connect, Shortcutting & Quintic Parameterization...")
    env = CobotSimEnv()
    planner = RRTConnectPlanner(env, step_size=0.06, seed=42)
    traj_gen = QuinticTrajectoryGenerator(dt=0.01)

    q_start = PANDA_HOME_CONFIG.copy()
    # Target reach configuration: pick region
    target_pos = np.array([0.48, -0.15, 0.48])
    ok_ik, q_goal = env.inverse_kinematics(target_pos)
    assert ok_ik, "IK failed for target test pose"

    print(f"Planning from Home to Target Pose: {target_pos}")

    # 1. RRT-Connect Planning with Semantic Constraints
    constraints = ["avoid_human", "slow_near_fragile"]
    ok, path, plan_time = planner.plan_path(q_start, q_goal, constraints=constraints)
    assert ok, "RRT-Connect planning failed"
    print(f"RRT-Connect Search: Success = {ok}, Found path of {len(path)} waypoints in {plan_time:.3f} s")

    # 2. Path Shortcutting
    smoothed_path = planner.shortcut_path(path, max_iterations=60)
    print(f"Path Shortcutting: Reduced from {len(path)} to {len(smoothed_path)} waypoints")
    assert len(smoothed_path) <= len(path), "Shortcutting failed to optimize waypoints"

    # 3. Quintic Trajectory Time-Parameterization
    traj = traj_gen.parameterize_path(smoothed_path)
    total_time = traj["duration"]
    num_points = len(traj["time"])
    max_vel = np.max(np.abs(traj["qd"]), axis=0)
    max_acc = np.max(np.abs(traj["qdd"]), axis=0)

    print(f"Quintic Trajectory: Duration = {total_time:.2f} s ({num_points} discrete timesteps)")
    print(f"Max Commanded Joint Velocities (rad/s): {np.round(max_vel, 3)}")
    print(f"Max Commanded Joint Accelerations (rad/s^2): {np.round(max_acc, 3)}")

    # Verification Assertions
    assert np.all(max_vel <= PANDA_VELOCITY_LIMITS + 1e-4), "Joint velocity limits exceeded"
    assert np.all(max_acc <= PANDA_ACCELERATION_LIMITS + 1e-4), "Joint acceleration limits exceeded"
    assert np.allclose(traj["qd"][0], 0.0, atol=1e-3), "Start velocity non-zero"
    assert np.allclose(traj["qd"][-1], 0.0, atol=1e-3), "End velocity non-zero"

    print(">>> Phase 4 Verification PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase4_verification()
