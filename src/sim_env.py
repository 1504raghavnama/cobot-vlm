"""Simulation Environment for Cobot Assembly with Franka Emika Panda.

Simulates a 7-DOF manipulator, workcell table, fixture, 3 assembly parts,
an active human proxy, and a calibrated RGB-D camera using MuJoCo physics.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Strict virtual environment assertion
if sys.prefix == sys.base_prefix:
    sys.exit("CRITICAL ERROR: Script must run inside the virtual environment at ./venv")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

# Standard Franka Emika Panda kinematic & dynamic specifications
PANDA_JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
PANDA_VELOCITY_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
PANDA_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
PANDA_HOME_CONFIG = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785398])

# Complete standalone MJCF Model with Franka Panda, table, fixture, parts, camera, and human proxy
PANDA_MJCF_XML = """
<mujoco model="franka_panda_assembly">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" meshdir="."/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler">
    <flag contact="enable"/>
  </option>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <joint damping="1.0" armature="0.1"/>
    <geom contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.9 0.9 0.9" rgb2="0.8 0.8 0.8"/>
    <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.05"/>
    <material name="arm_mat" rgba="0.25 0.25 0.25 1"/>
    <material name="link_gray" rgba="0.65 0.65 0.65 1"/>
    <material name="table_mat" rgba="0.4 0.4 0.4 1"/>
    <material name="fixture_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="part_base_mat" rgba="0.15 0.15 0.15 1"/>
    <material name="part_shaft_mat" rgba="0.5 0.5 0.5 1"/>
    <material name="part_cap_mat" rgba="0.8 0.8 0.8 1"/>
    <material name="human_mat" rgba="0.3 0.3 0.3 0.85"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0.5 -1.0 2.0" dir="-0.2 0.5 -1.0" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <light directional="true" pos="-0.5 1.0 2.0" dir="0.2 -0.5 -1.0" diffuse="0.4 0.4 0.4"/>
    
    <!-- Floor -->
    <geom name="floor" type="plane" size="3 3 0.1" material="grid_mat"/>

    <!-- Assembly Workbench -->
    <body name="table" pos="0.45 0.0 0.2">
      <geom name="table_top" type="box" size="0.45 0.65 0.2" material="table_mat"/>
    </body>

    <!-- Assembly Fixture -->
    <body name="fixture" pos="0.50 0.20 0.42">
      <geom name="fixture_base" type="box" size="0.07 0.07 0.02" material="fixture_mat"/>
      <geom name="fixture_slot" type="cylinder" size="0.018 0.02" pos="0 0 0.01" material="part_base_mat"/>
    </body>

    <!-- 3 Assembly Parts: Base, Shaft, Cap -->
    <body name="part_base" pos="0.45 -0.22 0.425">
      <geom name="part_base_geom" type="box" size="0.04 0.04 0.025" material="part_base_mat"/>
    </body>

    <body name="part_shaft" pos="0.55 -0.15 0.45">
      <geom name="part_shaft_geom" type="cylinder" size="0.012 0.05" material="part_shaft_mat"/>
    </body>

    <body name="part_cap" pos="0.38 -0.15 0.42">
      <geom name="part_cap_geom" type="cylinder" size="0.025 0.015" material="part_cap_mat"/>
    </body>

    <!-- Moving Human Proxy Obstacle -->
    <body name="human_proxy" pos="0.40 0.45 0.45">
      <geom name="human_head" type="sphere" size="0.14" material="human_mat"/>
      <geom name="human_body" type="capsule" size="0.10 0.15" pos="0 0.10 -0.10" material="human_mat"/>
    </body>

    <!-- Franka Emika Panda 7-DOF Kinematic Chain -->
    <body name="panda_link0" pos="0 0 0.4">
      <geom name="link0_geom" type="cylinder" size="0.07 0.05" material="arm_mat"/>
      
      <body name="panda_link1" pos="0 0 0.14">
        <joint name="panda_joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom name="link1_geom" type="cylinder" size="0.06 0.10" material="link_gray"/>
        
        <body name="panda_link2" pos="0 0 0.19" quat="0.7071068 -0.7071068 0 0">
          <joint name="panda_joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom name="link2_geom" type="cylinder" size="0.055 0.10" material="arm_mat"/>
          
          <body name="panda_link3" pos="0 -0.19 0" quat="0.7071068 0.7071068 0 0">
            <joint name="panda_joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom name="link3_geom" type="cylinder" size="0.05 0.10" material="link_gray"/>
            
            <body name="panda_link4" pos="0.0825 0 0.12" quat="0.7071068 0.7071068 0 0">
              <joint name="panda_joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom name="link4_geom" type="cylinder" size="0.05 0.10" material="arm_mat"/>
              
              <body name="panda_link5" pos="-0.0825 0.12 0" quat="0.7071068 -0.7071068 0 0">
                <joint name="panda_joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom name="link5_geom" type="cylinder" size="0.045 0.10" material="link_gray"/>
                
                <body name="panda_link6" pos="0 0 0.18" quat="0.7071068 0.7071068 0 0">
                  <joint name="panda_joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom name="link6_geom" type="cylinder" size="0.045 0.08" material="arm_mat"/>
                  
                  <body name="panda_link7" pos="0.088 0 0" quat="0.7071068 0.7071068 0 0">
                    <joint name="panda_joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom name="link7_geom" type="cylinder" size="0.04 0.05" material="link_gray"/>
                    
                    <!-- End-Effector / Flange & Parallel Gripper -->
                    <body name="panda_hand" pos="0 0 0.107">
                      <geom name="hand_base" type="box" size="0.04 0.07 0.02" material="arm_mat"/>
                      
                      <!-- Left Finger -->
                      <body name="panda_leftfinger" pos="0 0.035 0.04">
                        <geom name="leftfinger_geom" type="box" size="0.008 0.01 0.03" material="link_gray"/>
                      </body>
                      <!-- Right Finger -->
                      <body name="panda_rightfinger" pos="0 -0.035 0.04">
                        <geom name="rightfinger_geom" type="box" size="0.008 0.01 0.03" material="link_gray"/>
                      </body>
                      
                      <!-- Grasp Center Site -->
                      <site name="ee_site" pos="0 0 0.065" size="0.01" rgba="0.1 0.1 0.1 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- RGB-D Workcell Camera -->
    <camera name="workcell_cam" pos="0.95 -0.65 0.95" fovy="45" mode="targetbody" target="table"/>
    <!-- Wide Overview Camera Showing Full Robot Arm, Table, Fixture, and Human Proxy -->
    <camera name="overview_cam" pos="1.85 -1.45 1.45" fovy="56" mode="targetbody" target="table"/>
  </worldbody>
</mujoco>
"""


class CobotSimEnv:
    """Complete simulation workcell environment for Franka Panda assembly."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize simulation world from configuration."""
        self.config: Dict[str, Any] = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        self.seed = self.config.get("seed", 42)
        np.random.seed(self.seed)

        # Load physics model
        self.model = mujoco.MjModel.from_xml_string(PANDA_MJCF_XML)
        self.data = mujoco.MjData(self.model)

        # Extract IDs
        self.joint_names = [f"panda_joint{i+1}" for i in range(7)]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.qpos_ids = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.human_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human_proxy")

        # Human motion generator parameters
        self.human_base_pos = np.array([0.40, 0.45, 0.45])
        self.human_time = 0.0
        self.human_velocity = np.zeros(3)

        # Camera calibration parameters
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "workcell_cam")
        self.cam_width = 640
        self.cam_height = 480
        fovy_rad = np.deg2rad(45.0)
        f = (self.cam_height / 2.0) / np.tan(fovy_rad / 2.0)
        self.K = np.array([
            [f, 0.0, self.cam_width / 2.0],
            [0.0, f, self.cam_height / 2.0],
            [0.0, 0.0, 1.0]
        ])

        # Initialize to home configuration
        self.reset()

    def reset(self, q_init: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset robot joint configuration and human trajectory."""
        mujoco.mj_resetData(self.model, self.data)
        if q_init is None:
            q_init = PANDA_HOME_CONFIG.copy()

        for idx, qid in enumerate(self.qpos_ids):
            self.data.qpos[qid] = q_init[idx]

        self.human_time = 0.0
        self.update_human_proxy(0.0)
        mujoco.mj_forward(self.model, self.data)
        return self.get_joint_positions()

    def get_joint_positions(self) -> np.ndarray:
        """Return current 7 joint angles (rad)."""
        return np.array([self.data.qpos[qid] for qid in self.qpos_ids])

    def get_joint_velocities(self) -> np.ndarray:
        """Return current 7 joint velocities (rad/s)."""
        return np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.joint_ids])

    def set_joint_positions(self, q: np.ndarray) -> None:
        """Directly position robot joints and update kinematics."""
        for idx, qid in enumerate(self.qpos_ids):
            self.data.qpos[qid] = q[idx]
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector position (3,) and 3x3 rotation matrix."""
        if q is not None:
            old_q = self.get_joint_positions()
            self.set_joint_positions(q)
            pos = np.copy(self.data.site_xpos[self.ee_site_id])
            rot = np.copy(self.data.site_xmat[self.ee_site_id].reshape(3, 3))
            self.set_joint_positions(old_q)
        else:
            pos = np.copy(self.data.site_xpos[self.ee_site_id])
            rot = np.copy(self.data.site_xmat[self.ee_site_id].reshape(3, 3))
        return pos, rot

    def get_jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute 6x7 geometric Jacobian for the end-effector site."""
        if q is not None:
            old_q = self.get_joint_positions()
            self.set_joint_positions(q)

        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)

        # Extract only Franka's 7 joint dofs
        dof_ids = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]
        J = np.vstack([jac_pos[:, dof_ids], jac_rot[:, dof_ids]])

        if q is not None:
            self.set_joint_positions(old_q)
        return J

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 150,
        tol: float = 1e-3,
    ) -> Tuple[bool, np.ndarray]:
        """Damped Least-Squares numerical IK with joint limit projection."""
        q = q_init.copy() if q_init is not None else self.get_joint_positions()
        lambda_damping = 0.05

        for _ in range(max_iter):
            cur_pos, cur_rot = self.forward_kinematics(q)
            pos_err = target_pos - cur_pos

            if target_rot is not None:
                # Orientation error vector via skew-symmetric matrix
                rot_err = 0.5 * (np.cross(cur_rot[:, 0], target_rot[:, 0]) +
                                 np.cross(cur_rot[:, 1], target_rot[:, 1]) +
                                 np.cross(cur_rot[:, 2], target_rot[:, 2]))
                e = np.concatenate([pos_err, rot_err])
            else:
                e = pos_err

            if np.linalg.norm(pos_err) < tol and (target_rot is None or np.linalg.norm(rot_err) < tol * 3):
                return True, q

            J = self.get_jacobian(q)
            if target_rot is None:
                J = J[:3, :]

            # Damped least-squares: dq = J^T (J J^T + lambda^2 I)^-1 e
            JT = J.T
            JJT = J @ JT
            A = JJT + (lambda_damping ** 2) * np.eye(J.shape[0])
            dq = JT @ np.linalg.solve(A, e)

            # Step and clamp within Franka joint limits
            q = q + 0.5 * dq
            q = np.clip(q, PANDA_JOINT_LIMITS_LOWER + 0.02, PANDA_JOINT_LIMITS_UPPER - 0.02)

        final_pos, _ = self.forward_kinematics(q)
        success = bool(np.linalg.norm(target_pos - final_pos) < (tol * 5))
        return success, q

    def update_human_proxy(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Update dynamic human proxy obstacle along realistic intrusion trajectory."""
        self.human_time += dt
        # Dynamic harmonic path intruding into assembly zone
        omega_y = 0.4
        omega_x = 0.2
        y = self.human_base_pos[1] - 0.25 * (1.0 - np.cos(omega_y * self.human_time))
        x = self.human_base_pos[0] + 0.12 * np.sin(omega_x * self.human_time)
        z = self.human_base_pos[2] + 0.05 * np.sin(0.6 * self.human_time)
        new_pos = np.array([x, y, z])

        # Velocity approximation
        self.human_velocity = (new_pos - self.data.xpos[self.human_body_id]) / max(dt, 1e-5) if dt > 0 else np.zeros(3)

        self.model.body_pos[self.human_body_id] = new_pos
        mujoco.mj_forward(self.model, self.data)
        return new_pos, self.human_velocity

    def get_human_distance(self, q: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """Compute minimum Euclidean distance between all robot links and human proxy."""
        if q is not None:
            old_q = self.get_joint_positions()
            self.set_joint_positions(q)

        human_pos = np.copy(self.data.xpos[self.human_body_id])
        # Human collision geometry radius
        human_effective_radius = 0.14

        min_dist = float("inf")
        closest_point = np.zeros(3)

        # Check links 3, 4, 5, 6, 7 and hand
        link_names = [f"panda_link{i}" for i in range(3, 8)] + ["panda_hand"]
        for lname in link_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, lname)
            link_pos = self.data.xpos[bid]
            d = np.linalg.norm(link_pos - human_pos) - human_effective_radius
            if d < min_dist:
                min_dist = d
                closest_point = link_pos

        if q is not None:
            self.set_joint_positions(old_q)

        return float(max(min_dist, 0.0)), closest_point

    def check_collision(self, q: np.ndarray, check_human: bool = True) -> bool:
        """Check if robot configuration q is in self-collision or table/obstacle collision."""
        old_q = self.get_joint_positions()
        self.set_joint_positions(q)

        # Check table collision (end-effector or links below table surface 0.40m)
        ee_pos, _ = self.forward_kinematics()
        if ee_pos[2] < 0.405:  # Collision with tabletop surface
            self.set_joint_positions(old_q)
            return True

        # Check human proxy obstacle
        if check_human:
            d_human, _ = self.get_human_distance()
            if d_human < 0.08:  # Encroaching physical human collision volume
                self.set_joint_positions(old_q)
                return True

        # Check internal contact pairs
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            # Self-collision between robot components or robot-table collision
            if ("panda" in geom1_name and "panda" in geom2_name) or ("table" in geom1_name and "panda" in geom2_name):
                self.set_joint_positions(old_q)
                return True

        self.set_joint_positions(old_q)
        return False

    def render_rgb_depth(self) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB image and synthetic depth buffer from calibrated workcell camera."""
        renderer = mujoco.Renderer(self.model, height=self.cam_height, width=self.cam_width)
        renderer.update_scene(self.data, camera="workcell_cam")
        rgb = renderer.render()

        # Depth buffer rendering
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()

        return rgb, depth

    def get_camera_extrinsic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return camera world position and 3x3 rotation matrix."""
        cam_pos = np.copy(self.data.cam_xpos[self.cam_id])
        cam_mat = np.copy(self.data.cam_xmat[self.cam_id].reshape(3, 3))
        return cam_pos, cam_mat

    def pixel_to_world(self, u: float, v: float, depth_val: float) -> np.ndarray:
        """Deproject camera pixel (u, v) and depth to world 3D coordinate."""
        cam_pos, cam_mat = self.get_camera_extrinsic()
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Camera frame point (OpenGL camera convention: -Z forward, +Y up, +X right)
        x_c = (u - cx) * depth_val / fx
        y_c = -(v - cy) * depth_val / fy
        z_c = -depth_val
        p_c = np.array([x_c, y_c, z_c])

        # World coordinate: p_w = R_cam * p_c + T_cam
        p_w = cam_mat @ p_c + cam_pos
        return p_w

    def world_to_pixel(self, p_w: np.ndarray) -> Tuple[int, int]:
        """Project world coordinate p_w onto camera image pixel (u, v)."""
        cam_pos, cam_mat = self.get_camera_extrinsic()
        # Relative camera frame: p_c = R_cam^T (p_w - T_cam)
        p_c = cam_mat.T @ (p_w - cam_pos)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        forward_depth = -p_c[2]
        if forward_depth <= 1e-4:
            return -1, -1

        u = int(np.clip(fx * p_c[0] / forward_depth + cx, 0, self.cam_width - 1))
        v = int(np.clip(cy - fy * p_c[1] / forward_depth, 0, self.cam_height - 1))
        return u, v

    def save_workcell_screenshot(self, output_path: str = "outputs/screenshots/workcell_scene.png") -> str:
        """Render high-resolution grayscale-styled snapshot of workcell and save PNG."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Render wide overview showing entire 7-DOF robot arm, table, parts, and human proxy
        renderer = mujoco.Renderer(self.model, height=600, width=800)
        renderer.update_scene(self.data, camera="overview_cam")
        rgb = renderer.render()

        # Convert to Grayscale adhering to strict monochrome style rule
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=300)
        ax.imshow(gray, cmap="gray")
        ax.set_title("Franka Emika Panda Assembly Workcell (MuJoCo Simulation)", fontsize=11, color="black", pad=10)
        ax.axis("off")

        # Project 3D labels through overview camera frame
        ov_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview_cam")
        cam_pos = np.copy(self.data.cam_xpos[ov_cam_id])
        cam_mat = np.copy(self.data.cam_xmat[ov_cam_id].reshape(3, 3))
        fovy_rad = np.deg2rad(56.0)
        f_ov = (600 / 2.0) / np.tan(fovy_rad / 2.0)
        cx_ov, cy_ov = 400.0, 300.0

        # Key components in workcell
        base_pos = np.array([0.45, -0.22, 0.425])
        shaft_pos = np.array([0.55, -0.15, 0.45])
        cap_pos = np.array([0.38, -0.15, 0.42])
        fixture_pos = np.array([0.50, 0.20, 0.42])
        human_pos = np.copy(self.data.xpos[self.human_body_id])
        robot_base = np.array([0.0, 0.0, 0.45])
        ee_pos, _ = self.forward_kinematics()

        labels = [
            (robot_base, "Franka Panda Base (7-DOF)"),
            (ee_pos, "Parallel Gripper"),
            (base_pos, "Part: Base"),
            (shaft_pos, "Part: Shaft"),
            (cap_pos, "Part: Cap"),
            (fixture_pos, "Assembly Fixture"),
            (human_pos, "Human Proxy Obstacle"),
        ]

        for pos, label in labels:
            p_c = cam_mat.T @ (pos - cam_pos)
            forward_d = -p_c[2]
            if forward_d > 0.05:
                u = int(f_ov * p_c[0] / forward_d + cx_ov)
                v = int(cy_ov - f_ov * p_c[1] / forward_d)
                if 20 <= u < 780 and 20 <= v < 580:
                    ax.plot(u, v, "o", color="white", markersize=5, markeredgecolor="black", markeredgewidth=1.4)
                    ax.text(u + 6, v - 6, label, color="black", fontsize=7.5, fontweight="bold",
                            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="black", alpha=0.88))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, facecolor="white", edgecolor="none")
        plt.close(fig)
        return output_path


def run_phase1_verification() -> bool:
    """Run Phase 1 simulation test verification."""
    print(">>> Verifying Phase 1: Simulation Workcell Setup...")
    env = CobotSimEnv()
    q0 = env.get_joint_positions()
    print(f"Initial Franka Joint Angles (rad): {np.round(q0, 3)}")

    # Test Forward Kinematics
    ee_pos, ee_rot = env.forward_kinematics(q0)
    print(f"End-Effector Home Pose: Pos = {np.round(ee_pos, 3)}, Det(R) = {np.linalg.det(ee_rot):.3f}")

    # Test Numerical Inverse Kinematics
    target_test = np.array([0.50, -0.10, 0.55])
    ik_success, q_ik = env.inverse_kinematics(target_test)
    ee_ik, _ = env.forward_kinematics(q_ik)
    ik_err = np.linalg.norm(target_test - ee_ik)
    print(f"IK Target: {target_test} -> Reached: {np.round(ee_ik, 3)}, Error = {ik_err:.4f} m (Success: {ik_success})")

    # Test Camera Rendering & 2D-3D Deprojection
    rgb, depth = env.render_rgb_depth()
    print(f"Camera Image Acquired: RGB shape = {rgb.shape}, Depth shape = {depth.shape}")

    # Physical part base deprojection
    part_base_world = np.array([0.45, -0.22, 0.43])
    u_part, v_part = env.world_to_pixel(part_base_world)
    cam_pos, R_cam = env.get_camera_extrinsic()
    exact_depth = -(R_cam.T @ (part_base_world - cam_pos))[2]
    deproj_exact = env.pixel_to_world(u_part, v_part, exact_depth)
    geom_err = np.linalg.norm(part_base_world - deproj_exact)
    
    # Sensor depth buffer deprojection
    buf_depth = float(depth[v_part, u_part])
    deproj_buf = env.pixel_to_world(u_part, v_part, buf_depth)
    sensor_err = np.linalg.norm(part_base_world - deproj_buf)
    print(f"2D-to-3D Projection Verification: Target {part_base_world} -> Exact Deprojected {np.round(deproj_exact, 3)} (Err: {geom_err:.4f} m), Sensor Surface Deprojected {np.round(deproj_buf, 3)} (Err: {sensor_err:.4f} m)")

    # Test Human Intrusion Motion & Distance
    hpos_init, _ = env.update_human_proxy(0.0)
    d_init, _ = env.get_human_distance(q0)
    hpos_intrude, hvel = env.update_human_proxy(1.5)
    d_intrude, _ = env.get_human_distance(q0)
    print(f"Human Dynamic Proxy: Init Pos = {np.round(hpos_init, 2)} (Dist: {d_init:.3f}m) -> Intrude Pos = {np.round(hpos_intrude, 2)} (Dist: {d_intrude:.3f}m)")

    # Save and verify screenshot
    out_img = env.save_workcell_screenshot("outputs/screenshots/workcell_scene.png")
    assert os.path.exists(out_img), f"Screenshot file not found at {out_img}"
    print(f"Workcell Screenshot Verified: Saved to {out_img} (Size: {os.path.getsize(out_img)} bytes)")

    assert ik_success, "IK failed for valid reachable workspace target"
    assert geom_err < 0.005, f"Geometric deprojection error {geom_err:.4f} m exceeds tolerance"
    assert sensor_err < 0.05, f"Sensor surface deprojection error {sensor_err:.4f} m exceeds tolerance"
    assert rgb.shape == (480, 640, 3), "Invalid RGB dimensions"
    print(">>> Phase 1 Verification PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase1_verification()
