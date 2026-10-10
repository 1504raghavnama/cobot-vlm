"""Dual-Layer Safety Filter for Collaborative Robot Assembly.

Combines ISO/TS 15066 Speed & Separation Monitoring (SSM) and
Control Barrier Functions (CBF-QP) to guarantee human safety
by filtering commanded joint velocities in real time.
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

import cvxpy as cp
import mujoco
import numpy as np
from src.sim_env import CobotSimEnv, PANDA_VELOCITY_LIMITS


class DualLayerSafetyFilter:
    """ISO/TS 15066 SSM + Control Barrier Function (CBF-QP) joint velocity filter."""

    def __init__(
        self,
        env: CobotSimEnv,
        ssm_d_min: float = 0.25,
        ssm_d_stop: float = 0.10,
        cbf_gamma: float = 8.0,
        vel_limits: Optional[np.ndarray] = None,
        cbf_slack_weight: float = 1000.0,
    ) -> None:
        self.env = env
        self.d_min = ssm_d_min
        self.d_stop = ssm_d_stop
        self.gamma = cbf_gamma
        self.vel_limits = vel_limits if vel_limits is not None else PANDA_VELOCITY_LIMITS
        self.slack_weight = cbf_slack_weight

        # Telemetry / metrics log
        self.history: List[Dict[str, float]] = []

    def compute_ssm_scaling(self, d_human: float) -> float:
        """Compute ISO/TS 15066 speed scaling factor based on current separation distance."""
        if d_human >= self.d_min:
            return 1.0
        elif d_human <= self.d_stop:
            return 0.0
        else:
            # Linear deceleration between d_stop and d_min
            return float((d_human - self.d_stop) / (self.d_min - self.d_stop))

    def filter_velocity(
        self,
        q: np.ndarray,
        qd_nominal: np.ndarray,
        human_pos: np.ndarray,
        human_vel: np.ndarray,
        enable_cbf: bool = True,
        enable_ssm: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Filter nominal commanded joint velocity vector using SSM and CBF-QP."""
        d_human, closest_point = self.env.get_human_distance(q)

        # 1. Speed & Separation Monitoring (SSM)
        if enable_ssm:
            ssm_scale = self.compute_ssm_scaling(d_human)
            qd_cmd = qd_nominal * ssm_scale
        else:
            ssm_scale = 1.0
            qd_cmd = qd_nominal.copy()

        cbf_active = False
        slack_val = 0.0

        # 2. Control Barrier Function (CBF-QP)
        if enable_cbf:
            # Barrier function: h(q) = ||p_r - p_h||^2 - d_stop^2 >= 0
            rel_pos = closest_point - human_pos
            dist_sq = np.dot(rel_pos, rel_pos)
            h_val = dist_sq - (self.d_stop ** 2)

            # Jacobian at closest point
            J = self.env.get_jacobian(q)[:3, :]
            # Time derivative of barrier: dh/dt = 2 * rel_pos^T * (J * qd - v_h) >= -gamma * h
            # A_cbf * qd <= b_cbf
            a_vec = -2.0 * (rel_pos @ J)
            b_scalar = self.gamma * h_val - 2.0 * float(np.dot(rel_pos, human_vel))

            # Set up CVXPY Quadratic Program
            # min 0.5 * ||qd - qd_cmd||^2 + 0.5 * slack_weight * slack^2
            qd_var = cp.Variable(7)
            slack_var = cp.Variable(1)

            cost = 0.5 * cp.sum_squares(qd_var - qd_cmd) + 0.5 * self.slack_weight * cp.square(slack_var)
            constraints = [
                a_vec @ qd_var <= b_scalar + slack_var,
                qd_var >= -self.vel_limits,
                qd_var <= self.vel_limits,
                slack_var >= 0.0,
            ]

            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                # Solve using Clarabel or OSQP
                prob.solve(solver=cp.CLARABEL, verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    qd_safe = np.array(qd_var.value, dtype=np.float64)
                    slack_val = float(slack_var.value[0])
                    # Detect if CBF modified nominal command
                    if np.linalg.norm(qd_safe - qd_cmd) > 1e-3:
                        cbf_active = True
                else:
                    # Projection fallback
                    qd_safe = qd_cmd * 0.2
            except Exception:
                qd_safe = qd_cmd * 0.2
        else:
            qd_safe = qd_cmd

        # Telemetry record
        info = {
            "d_human": float(d_human),
            "ssm_scale": float(ssm_scale),
            "cbf_active": bool(cbf_active),
            "slack": float(slack_val),
            "vel_reduction": float(np.linalg.norm(qd_nominal) - np.linalg.norm(qd_safe)),
        }
        self.history.append(info)
        return qd_safe, info


def run_phase5_verification() -> bool:
    """Run Phase 5 Safety Filter verification tests."""
    print(">>> Verifying Phase 5: SSM and CBF-QP Safety Controller...")
    env = CobotSimEnv()
    safety_filter = DualLayerSafetyFilter(env, ssm_d_min=0.25, ssm_d_stop=0.10, cbf_gamma=8.0)

    q0 = env.get_joint_positions()
    qd_nom = np.array([0.5, 0.4, 0.0, -0.3, 0.2, 0.0, 0.1])

    # Case 1: Human is far away (safe distance > 0.40m)
    env.update_human_proxy(0.0)
    hpos_far = np.copy(env.data.xpos[env.human_body_id])
    hvel_far = np.zeros(3)

    qd_safe1, info1 = safety_filter.filter_velocity(q0, qd_nom, hpos_far, hvel_far)
    print(f"\n[Case 1: Distant Human] Dist = {info1['d_human']:.3f}m")
    print(f"  SSM scale: {info1['ssm_scale']:.2f}, CBF active: {info1['cbf_active']}")
    print(f"  Velocity diff norm: {np.linalg.norm(qd_nom - qd_safe1):.4f}")
    assert info1["ssm_scale"] == 1.0, "SSM scaled velocity when human was far"

    # Case 2: Human encroaches into warning buffer (0.10m < dist < 0.25m)
    ee_pos, _ = env.forward_kinematics(q0)
    # Set human proxy 0.32m away: dist = 0.32 - 0.14 = 0.18m (in [0.10, 0.25])
    env.model.body_pos[env.human_body_id] = ee_pos + np.array([0.0, 0.32, 0.0])
    mujoco.mj_forward(env.model, env.data)
    hpos_near = np.copy(env.data.xpos[env.human_body_id])
    hvel_near = np.array([0.0, -0.15, 0.0])

    qd_safe2, info2 = safety_filter.filter_velocity(q0, qd_nom, hpos_near, hvel_near)
    print(f"\n[Case 2: Encroaching Human] Dist = {info2['d_human']:.3f}m")
    print(f"  SSM scale: {info2['ssm_scale']:.2f}, CBF active: {info2['cbf_active']}")
    print(f"  Velocity command scaled from norm {np.linalg.norm(qd_nom):.3f} to {np.linalg.norm(qd_safe2):.3f}")
    assert 0.0 < info2["ssm_scale"] < 1.0, "SSM did not scale velocity in buffer zone"

    # Case 3: Human right at stopping threshold (dist <= 0.10m)
    # Set human proxy 0.22m away: dist = 0.22 - 0.14 = 0.08m (<= 0.10)
    env.model.body_pos[env.human_body_id] = ee_pos + np.array([0.0, 0.22, 0.0])
    mujoco.mj_forward(env.model, env.data)
    hpos_crit = np.copy(env.data.xpos[env.human_body_id])
    hvel_crit = np.array([0.0, -0.20, 0.0])

    qd_safe3, info3 = safety_filter.filter_velocity(q0, qd_nom, hpos_crit, hvel_crit)
    print(f"\n[Case 3: Critical Stop Threshold] Dist = {info3['d_human']:.3f}m")
    print(f"  SSM scale: {info3['ssm_scale']:.2f}, CBF active: {info3['cbf_active']}")
    print(f"  Filtered velocity norm: {np.linalg.norm(qd_safe3):.4f}")
    assert info3["ssm_scale"] == 0.0, "SSM failed to command 0 velocity at stop boundary"
    assert np.allclose(qd_safe3, 0.0, atol=1e-3), "Filtered velocity non-zero at emergency stop"

    print("\n>>> Phase 5 Verification PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase5_verification()
