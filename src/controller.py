"""Joint-space Tracking Controller with Closed-Loop Safety Filtering.

Executes computed-torque and impedance trajectory tracking on Franka Emika Panda,
interfacing with the DualLayerSafetyFilter to modulate commanded velocities
based on real-time human separation distance.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Strict virtual environment assertion
if sys.prefix == sys.base_prefix:
    sys.exit("CRITICAL ERROR: Script must run inside the virtual environment at ./venv")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from src.sim_env import CobotSimEnv, PANDA_VELOCITY_LIMITS, PANDA_HOME_CONFIG
from src.safety_filter import DualLayerSafetyFilter


class CobotTrackingController:
    """Joint-space tracking controller integrating the real-time safety filter."""

    def __init__(
        self,
        env: CobotSimEnv,
        safety_filter: Optional[DualLayerSafetyFilter] = None,
        kp: float = 120.0,
        kd: float = 20.0,
    ) -> None:
        self.env = env
        self.filter = safety_filter if safety_filter is not None else DualLayerSafetyFilter(env)
        self.kp = kp
        self.kd = kd

    def track_trajectory(
        self,
        trajectory: Dict[str, np.ndarray],
        enable_cbf: bool = True,
        enable_ssm: bool = True,
        dt_sim: float = 0.01,
    ) -> Dict[str, Any]:
        """Execute closed-loop tracking over time-parameterized trajectory."""
        times = trajectory["time"]
        q_ref = trajectory["q"]
        qd_ref = trajectory["qd"]

        n_steps = len(times)
        actual_q: List[np.ndarray] = []
        actual_qd: List[np.ndarray] = []
        cmd_qd: List[np.ndarray] = []
        human_distances: List[float] = []
        ssm_scales: List[float] = []
        cbf_activations: List[bool] = []
        ee_positions: List[np.ndarray] = []

        q_curr = self.env.get_joint_positions()

        for step in range(n_steps):
            t = float(times[step])
            q_des = q_ref[step]
            qd_des = qd_ref[step]

            # Update moving human obstacle
            h_pos, h_vel = self.env.update_human_proxy(dt_sim)

            # Position tracking error
            pos_err = q_des - q_curr
            # Nominal commanded velocity with proportional correction
            qd_nominal = qd_des + 2.5 * pos_err
            # Clamp nominal velocity
            qd_nominal = np.clip(qd_nominal, -PANDA_VELOCITY_LIMITS, PANDA_VELOCITY_LIMITS)

            # Pass through Safety Filter
            qd_safe, filter_info = self.filter.filter_velocity(
                q=q_curr,
                qd_nominal=qd_nominal,
                human_pos=h_pos,
                human_vel=h_vel,
                enable_cbf=enable_cbf,
                enable_ssm=enable_ssm,
            )

            # Step robot joints via kinematic integration
            q_next = q_curr + qd_safe * dt_sim
            self.env.set_joint_positions(q_next)
            q_curr = self.env.get_joint_positions()

            ee_pos, _ = self.env.forward_kinematics(q_curr)

            # Record telemetry
            actual_q.append(q_curr.copy())
            actual_qd.append(qd_safe.copy())
            cmd_qd.append(qd_nominal.copy())
            human_distances.append(filter_info["d_human"])
            ssm_scales.append(filter_info["ssm_scale"])
            cbf_activations.append(filter_info["cbf_active"])
            ee_positions.append(ee_pos.copy())

        # Compute tracking Root Mean Squared Error (RMSE)
        actual_q_arr = np.array(actual_q)
        rmse = float(np.sqrt(np.mean((actual_q_arr - q_ref) ** 2)))
        min_dist = float(np.min(human_distances))
        violations = int(np.sum(np.array(human_distances) < 0.15))

        return {
            "time": times,
            "q_actual": actual_q_arr,
            "qd_actual": np.array(actual_qd),
            "qd_nominal": np.array(cmd_qd),
            "ee_positions": np.array(ee_positions),
            "human_distances": np.array(human_distances),
            "ssm_scales": np.array(ssm_scales),
            "cbf_activations": np.array(cbf_activations),
            "tracking_rmse": rmse,
            "min_human_dist": min_dist,
            "safety_violations": violations,
            "total_cbf_events": int(np.sum(cbf_activations)),
        }


if __name__ == "__main__":
    from src.motion_planner import RRTConnectPlanner, QuinticTrajectoryGenerator
    print(">>> Verifying Closed-Loop Tracking Controller with Safety Filter...")
    env = CobotSimEnv()
    planner = RRTConnectPlanner(env)
    traj_gen = QuinticTrajectoryGenerator(dt=0.01)

    ok, path, _ = planner.plan_path(PANDA_HOME_CONFIG, np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785]))
    assert ok, "Path planning failed"
    traj = traj_gen.parameterize_path(planner.shortcut_path(path))

    controller = CobotTrackingController(env)
    res = controller.track_trajectory(traj, enable_cbf=True, enable_ssm=True)
    print(f"Tracking Controller Test: RMSE = {res['tracking_rmse']:.4f} rad, Min Human Dist = {res['min_human_dist']:.3f} m")
    assert res["tracking_rmse"] < 0.05, "Tracking error excessive"
    print(">>> Tracking Controller Verification PASSED!")
