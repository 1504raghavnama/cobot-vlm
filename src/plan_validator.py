"""Geometric Plan Validator and Repair Engine for Cobot Assembly.

Verifies VLM-generated task plans against physical robot kinematic reachability,
joint angle limits, and static/environmental collision constraints.
Provides automatic plan repair heuristics and logs all validation audits.
Blocks VLM hallucinations from causing unsafe robot motions.
"""

from __future__ import annotations

import json
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
from src.sim_env import (
    CobotSimEnv,
    PANDA_JOINT_LIMITS_LOWER,
    PANDA_JOINT_LIMITS_UPPER,
    PANDA_HOME_CONFIG,
)

# Standard top-down end-effector orientation rotation matrix (gripper pointing downwards)
TOP_DOWN_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0]
])


class PlanValidator:
    """Rigorous geometric validator and repair system for VLM plans."""

    def __init__(
        self,
        joint_margin: float = 0.05,
        table_z_min: float = 0.41,
        reach_max_radius: float = 0.82,
        audit_log_path: str = "outputs/data/validator_audit.json"
    ) -> None:
        self.joint_margin = joint_margin
        self.table_z_min = table_z_min
        self.reach_max_radius = reach_max_radius
        self.audit_log_path = audit_log_path
        self.audit_records: List[Dict[str, Any]] = []

    def validate_and_repair(
        self,
        raw_plan: Dict[str, Any],
        env: CobotSimEnv
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """Validate VLM plan geometrically, attempt repairs if needed, or reject."""
        plan = json.loads(json.dumps(raw_plan))  # Deep copy
        reasons: List[str] = []
        repaired_actions: List[str] = []
        is_valid = True

        # Extract coordinates
        pick_pos = np.array(plan["pick_pose_3d"])
        place_pos = np.array(plan["place_pose"])

        # Check 1: Workspace Radius Reachability
        robot_base_pos = np.array([0.0, 0.0, 0.40])
        dist_pick = np.linalg.norm(pick_pos - robot_base_pos)
        dist_place = np.linalg.norm(place_pos - robot_base_pos)

        if dist_pick > self.reach_max_radius:
            reasons.append(f"Pick position out of reach (dist {dist_pick:.3f}m > {self.reach_max_radius}m)")
            is_valid = False

        if dist_place > self.reach_max_radius:
            reasons.append(f"Place position out of reach (dist {dist_place:.3f}m > {self.reach_max_radius}m)")
            is_valid = False

        # Check 2: Tabletop Penetration & Clearance Repair
        if pick_pos[2] < self.table_z_min:
            repaired_z = self.table_z_min + 0.015
            repaired_actions.append(f"Repaired pick z from {pick_pos[2]:.3f}m to {repaired_z:.3f}m (tabletop clearance)")
            pick_pos[2] = repaired_z
            plan["pick_pose_3d"][2] = float(repaired_z)

        if place_pos[2] < self.table_z_min:
            repaired_z = self.table_z_min + 0.02
            repaired_actions.append(f"Repaired place z from {place_pos[2]:.3f}m to {repaired_z:.3f}m (fixture clearance)")
            place_pos[2] = repaired_z
            plan["place_pose"][2] = float(repaired_z)

        # Check 3: Approach Waypoint Synthesis
        approach_mode = plan.get("approach", "top")
        if approach_mode == "top":
            approach_offset = np.array([0.0, 0.0, 0.10])
        else:
            approach_offset = np.array([-0.08, 0.0, 0.05])

        pre_pick_pos = pick_pos + approach_offset
        lift_pos = pick_pos + np.array([0.0, 0.0, 0.12])
        pre_place_pos = place_pos + approach_offset

        # Check 4: Inverse Kinematics for Waypoints
        waypoints = {
            "pre_pick": pre_pick_pos,
            "pick": pick_pos,
            "lift": lift_pos,
            "pre_place": pre_place_pos,
            "place": place_pos,
        }

        waypoint_configs: Dict[str, List[float]] = {}
        q_seed = PANDA_HOME_CONFIG.copy()

        for name, wpt_pos in waypoints.items():
            ik_ok, q_sol = env.inverse_kinematics(
                target_pos=wpt_pos,
                target_rot=TOP_DOWN_ROTATION,
                q_init=q_seed,
                max_iter=150,
                tol=1e-3
            )
            if not ik_ok:
                reasons.append(f"IK failed to converge for waypoint '{name}' at {np.round(wpt_pos, 3)}")
                is_valid = False
                break

            # Check 5: Joint Limit Safety Margin
            lower_viol = q_sol < (PANDA_JOINT_LIMITS_LOWER + self.joint_margin)
            upper_viol = q_sol > (PANDA_JOINT_LIMITS_UPPER - self.joint_margin)
            if np.any(lower_viol) or np.any(upper_viol):
                reasons.append(f"Joint limit violated with {self.joint_margin} rad margin at waypoint '{name}'")
                is_valid = False
                break

            # Check 6: Static Workcell Collision
            if env.check_collision(q_sol, check_human=False):
                reasons.append(f"Collision detected at waypoint '{name}'")
                is_valid = False
                break

            waypoint_configs[name] = [float(x) for x in np.round(q_sol, 4)]
            q_seed = q_sol.copy()

        # Audit Record
        audit_entry = {
            "target_part": plan.get("target_part"),
            "instruction": plan.get("instruction", ""),
            "original_pick": raw_plan.get("pick_pose_3d"),
            "original_place": raw_plan.get("place_pose"),
            "repaired_actions": repaired_actions,
            "validated": is_valid,
            "rejection_reasons": reasons,
            "has_waypoint_configs": is_valid,
        }
        self.audit_records.append(audit_entry)
        self._save_audit_log()

        if is_valid:
            plan["waypoint_configs"] = waypoint_configs
            plan["repaired_actions"] = repaired_actions
            return True, plan, repaired_actions
        else:
            return False, raw_plan, reasons

    def _save_audit_log(self) -> None:
        """Persist validation audit history to JSON file."""
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        with open(self.audit_log_path, "w", encoding="utf-8") as f:
            json.dump(self.audit_records, f, indent=2)


def run_phase3_verification() -> bool:
    """Run Phase 3 Plan Validator verification tests."""
    from src.vlm_planner import VLMAssemblyPlanner

    print(">>> Verifying Phase 3: Plan Validator & Geometric Repair...")
    env = CobotSimEnv()
    rgb, depth = env.render_rgb_depth()
    vlm_planner = VLMAssemblyPlanner(use_mock=True)
    validator = PlanValidator()

    # Test 1: Valid Nominal Plan
    nominal_instr = "Grasp the shaft and insert it into the assembly fixture."
    plan_nom = vlm_planner.plan_task(nominal_instr, rgb, depth, env)
    ok_nom, repaired_nom, msgs_nom = validator.validate_and_repair(plan_nom, env)
    print(f"\n[Test 1] Nominal Plan: Valid = {ok_nom}, Repaired = {len(msgs_nom)} actions")
    if ok_nom:
        print(f"  Waypoints synthesized: {list(repaired_nom['waypoint_configs'].keys())}")
    assert ok_nom, "Nominal plan failed validation"

    # Test 2: Hallucinated / Out-of-Reach Target (Simulated VLM Hallucination)
    plan_hallucinated = json.loads(json.dumps(plan_nom))
    plan_hallucinated["pick_pose_3d"] = [1.85, -0.40, 0.45]  # 1.85m is far outside Panda's 0.85m workspace
    ok_hal, _, msgs_hal = validator.validate_and_repair(plan_hallucinated, env)
    print(f"\n[Test 2] Hallucinated Plan (Out-of-Reach): Valid = {ok_hal}")
    print(f"  Rejection Reason: {msgs_hal}")
    assert not ok_hal, "Validator failed to reject out-of-reach hallucinated plan"

    # Test 3: Table Penetration Target (Automatic Repair)
    plan_table_pen = json.loads(json.dumps(plan_nom))
    plan_table_pen["pick_pose_3d"] = [0.45, -0.20, 0.35]  # 0.35m is beneath 0.40m table surface
    ok_pen, repaired_pen, msgs_pen = validator.validate_and_repair(plan_table_pen, env)
    print(f"\n[Test 3] Low Tabletop Target: Valid = {ok_pen}")
    print(f"  Repair Actions: {msgs_pen}")
    print(f"  Corrected Pick Z: {repaired_pen['pick_pose_3d'][2]} m")
    assert ok_pen, "Validator failed to repair safe height target"
    assert repaired_pen["pick_pose_3d"][2] >= 0.41, "Target z not raised above tabletop"

    # Test 4: Verify Audit File
    assert os.path.exists(validator.audit_log_path), "Audit log file was not created"
    with open(validator.audit_log_path, "r", encoding="utf-8") as f:
        audits = json.load(f)
    print(f"\nAudit Log File: {validator.audit_log_path} contains {len(audits)} verified records.")

    print(">>> Phase 3 Verification PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase3_verification()
