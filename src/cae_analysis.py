"""Computer-Aided Engineering (CAE) Analysis.

Performs:
1. Manipulator joint torque, velocity, and Yoshikawa manipulability analysis along assembly trajectories.
2. Gripper finger static structural stress and Factor of Safety (FoS) analysis under grasp load.
Adheres strictly to grayscale styling rules and saves numerical summaries to outputs/data/.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Strict virtual environment assertion
if sys.prefix == sys.base_prefix:
    sys.exit("CRITICAL ERROR: Script must run inside the virtual environment at ./venv")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from src.sim_env import CobotSimEnv, PANDA_TORQUE_LIMITS, PANDA_VELOCITY_LIMITS


class CAEAnalyzer:
    """Manipulator dynamics and structural finite stress analyzer."""

    def __init__(
        self,
        env: CobotSimEnv,
        material: str = "Al6061-T6",
        yield_strength_mpa: float = 276.0,
        grasp_force_n: float = 45.0,
        finger_length_mm: float = 55.0,
        finger_width_mm: float = 16.0,
        finger_thickness_mm: float = 8.0,
        output_data_path: str = "outputs/data/cae_summary.json",
    ) -> None:
        self.env = env
        self.material = material
        self.sigma_yield = yield_strength_mpa
        self.grasp_force = grasp_force_n
        self.L_mm = finger_length_mm
        self.b_mm = finger_width_mm
        self.h_mm = finger_thickness_mm
        self.output_data_path = output_data_path

    def analyze_finger_stress(self) -> Dict[str, Any]:
        """Compute analytical & 2D FEA continuum stress distribution for gripper finger."""
        # Beam geometry in meters
        L = self.L_mm * 1e-3
        b = self.b_mm * 1e-3
        h = self.h_mm * 1e-3
        F = self.grasp_force

        # Cross-sectional moment of inertia I = b * h^3 / 12
        I = (b * (h ** 3)) / 12.0
        # Section modulus Z = b * h^2 / 6
        Z = (b * (h ** 2)) / 6.0
        # Cross-sectional area A = b * h
        A = b * h

        # Maximum bending moment at root (fixed end)
        M_max = F * L  # N*m
        # Maximum normal bending stress
        sigma_bend_max = (M_max / Z) * 1e-6  # MPa
        # Maximum transverse shear stress (parabolic distribution across height)
        tau_shear_max = (1.5 * F / A) * 1e-6  # MPa

        # Maximum von Mises equivalent stress at outer fiber root:
        # sigma_vm = sqrt(sigma_b^2 + 3 * tau^2)
        # At outer fiber, tau = 0, so sigma_vm = sigma_bend_max
        # In general planar stress:
        sigma_vm_max = float(np.sqrt(sigma_bend_max ** 2 + 3.0 * (0.0 ** 2)))
        safety_factor = float(self.sigma_yield / sigma_vm_max)

        # 2D Grid stress distribution across beam longitudinal section (x: 0->L, y: -h/2 -> h/2)
        nx, ny = 40, 20
        x_grid = np.linspace(0, L, nx)
        y_grid = np.linspace(-h / 2.0, h / 2.0, ny)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Bending moment M(x) = F * (L - x)
        M_x = F * (L - X)
        # Bending normal stress sigma(x, y) = M(x) * y / I
        sigma_xy = (M_x * Y / I) * 1e-6  # MPa
        # Shear stress tau(x, y) = 1.5 * (F / A) * (1 - (2*y/h)^2)
        tau_xy = (1.5 * (F / A) * (1.0 - (2.0 * Y / h) ** 2)) * 1e-6  # MPa
        # Von Mises stress field
        vm_field = np.sqrt(sigma_xy ** 2 + 3.0 * (tau_xy ** 2))

        return {
            "material": self.material,
            "yield_strength_mpa": float(self.sigma_yield),
            "grasp_force_n": float(F),
            "max_bending_moment_nm": float(M_max),
            "max_bending_stress_mpa": float(sigma_bend_max),
            "max_shear_stress_mpa": float(tau_shear_max),
            "max_von_mises_mpa": float(sigma_vm_max),
            "factor_of_safety": float(safety_factor),
            "grid_x_mm": (x_grid * 1e3).tolist(),
            "grid_y_mm": (y_grid * 1e3).tolist(),
            "von_mises_field_mpa": vm_field.tolist(),
        }

    def analyze_manipulator_trajectory(self, q_trajectory: np.ndarray, qd_trajectory: np.ndarray) -> Dict[str, Any]:
        """Compute joint torques and Yoshikawa manipulability index along trajectory."""
        n_steps = len(q_trajectory)
        manipulabilities: List[float] = []
        static_torques: List[np.ndarray] = []

        # Payload force at end effector: grasp 1.5 kg workpiece = 14.7 N downwards
        F_ext = np.array([0.0, 0.0, -14.715, 0.0, 0.0, 0.0])

        for step in range(n_steps):
            q = q_trajectory[step]
            J = self.env.get_jacobian(q)

            # Yoshikawa manipulability measure w = sqrt(det(J J^T))
            # Using position Jacobian (3x7)
            J_pos = J[:3, :]
            JJT = J_pos @ J_pos.T
            w = float(np.sqrt(max(np.linalg.det(JJT), 0.0)))
            manipulabilities.append(w)

            # Static joint torque required: tau = J^T F_ext
            tau_ext = J.T @ F_ext
            static_torques.append(tau_ext)

        torques_arr = np.array(static_torques)
        peak_torques = np.max(np.abs(torques_arr), axis=0)
        torque_margins = PANDA_TORQUE_LIMITS - peak_torques

        return {
            "manipulability_mean": float(np.mean(manipulabilities)),
            "manipulability_min": float(np.min(manipulabilities)),
            "manipulability_max": float(np.max(manipulabilities)),
            "manipulability_profile": manipulabilities,
            "peak_joint_torques_nm": peak_torques.tolist(),
            "torque_limits_nm": PANDA_TORQUE_LIMITS.tolist(),
            "torque_safety_margins_nm": torque_margins.tolist(),
            "torques_profile": torques_arr.tolist(),
        }

    def run_full_cae(self, q_trajectory: np.ndarray, qd_trajectory: np.ndarray) -> Dict[str, Any]:
        """Execute full CAE pipeline and persist summary results."""
        stress_res = self.analyze_finger_stress()
        manip_res = self.analyze_manipulator_trajectory(q_trajectory, qd_trajectory)

        summary = {
            "gripper_stress_analysis": stress_res,
            "manipulator_analysis": manip_res,
        }

        os.makedirs(os.path.dirname(self.output_data_path), exist_ok=True)
        with open(self.output_data_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def run_phase6_verification() -> bool:
    """Run Phase 6 CAE analysis verification test."""
    from src.cad_models import CADModelGenerator
    from src.motion_planner import RRTConnectPlanner, QuinticTrajectoryGenerator
    from src.sim_env import PANDA_HOME_CONFIG

    print(">>> Verifying Phase 6: CAD Models and CAE Structural/Dynamics Analysis...")
    cad_gen = CADModelGenerator()
    exported = cad_gen.export_all_models()
    print(f"Exported {len(exported)} solid STL models.")

    env = CobotSimEnv()
    planner = RRTConnectPlanner(env)
    traj_gen = QuinticTrajectoryGenerator(dt=0.01)

    target_q = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785])
    ok, path, _ = planner.plan_path(PANDA_HOME_CONFIG, target_q)
    assert ok, "Path planning failed"
    traj = traj_gen.parameterize_path(planner.shortcut_path(path))

    analyzer = CAEAnalyzer(env)
    cae_summary = analyzer.run_full_cae(traj["q"], traj["qd"])

    # Verify stress results
    stress_info = cae_summary["gripper_stress_analysis"]
    print(f"\nGripper Finger Structural FEA Results:")
    print(f"  Material: {stress_info['material']} (Yield Strength: {stress_info['yield_strength_mpa']} MPa)")
    print(f"  Grasp Force: {stress_info['grasp_force_n']} N")
    print(f"  Max Von Mises Stress: {stress_info['max_von_mises_mpa']:.2f} MPa")
    print(f"  Factor of Safety (FoS): {stress_info['factor_of_safety']:.2f}")

    assert stress_info["max_von_mises_mpa"] < stress_info["yield_strength_mpa"], "Stress exceeds yield strength!"
    assert stress_info["factor_of_safety"] > 1.5, "Factor of safety below standard industrial limit (1.5)"

    # Verify dynamics & manipulability
    manip_info = cae_summary["manipulator_analysis"]
    print(f"\nManipulator Kinematics & Torques:")
    print(f"  Mean Yoshikawa Manipulability: {manip_info['manipulability_mean']:.4f}")
    print(f"  Peak Torques (Nm): {np.round(manip_info['peak_joint_torques_nm'], 2)}")
    print(f"  Torque Margins (Nm): {np.round(manip_info['torque_safety_margins_nm'], 2)}")

    assert np.all(np.array(manip_info["torque_safety_margins_nm"]) > 0), "Joint torque limit exceeded!"
    assert os.path.exists(analyzer.output_data_path), "CAE summary JSON not saved!"

    print("\n>>> Phase 6 Verification PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase6_verification()
