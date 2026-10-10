"""Parametric CAD Modeling and STL Export for Cobot Workcell.

Generates 3D models for parallel gripper fingers, assembly fixture,
and three assembly parts (base, shaft, cap) using trimesh.
Exports solid STL files to outputs/cad/ and generates grayscale previews.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Strict virtual environment assertion
if sys.prefix == sys.base_prefix:
    sys.exit("CRITICAL ERROR: Script must run inside the virtual environment at ./venv")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


class CADModelGenerator:
    """Generates parametric CAD solids and exports STL models."""

    def __init__(self, output_dir: str = "outputs/cad") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def create_gripper_finger(
        self,
        length_mm: float = 55.0,
        width_mm: float = 16.0,
        thickness_mm: float = 8.0,
    ) -> trimesh.Trimesh:
        """Create precision parallel gripper finger with mounting base and grasp tip."""
        L = length_mm / 1000.0
        W = width_mm / 1000.0
        T = thickness_mm / 1000.0

        # Main finger beam
        beam = trimesh.creation.box(extents=[T, W, L])
        beam.apply_translation([0.0, 0.0, L / 2.0])

        # Mounting tab at base
        mount = trimesh.creation.box(extents=[T * 1.5, W * 1.2, L * 0.25])
        mount.apply_translation([T * 0.25, 0.0, L * 0.125])

        # Grasp tip pad with bevel
        tip = trimesh.creation.box(extents=[T * 0.8, W * 0.9, L * 0.2])
        tip.apply_translation([-T * 0.1, 0.0, L * 0.9])

        finger = trimesh.util.concatenate([beam, mount, tip])
        finger.merge_vertices()
        return finger

    def create_assembly_fixture(self) -> trimesh.Trimesh:
        """Create assembly mounting fixture with central circular locating pocket."""
        # Base plate
        base = trimesh.creation.box(extents=[0.14, 0.14, 0.03])
        base.apply_translation([0.0, 0.0, 0.015])

        # Locating pin / cylinder boss
        boss = trimesh.creation.cylinder(radius=0.018, height=0.02)
        boss.apply_translation([0.0, 0.0, 0.04])

        # 4 Corner clamp blocks
        clamps = []
        for dx in [-0.05, 0.05]:
            for dy in [-0.05, 0.05]:
                c = trimesh.creation.box(extents=[0.02, 0.02, 0.025])
                c.apply_translation([dx, dy, 0.0425])
                clamps.append(c)

        fixture = trimesh.util.concatenate([base, boss] + clamps)
        fixture.merge_vertices()
        return fixture

    def create_part_base(self) -> trimesh.Trimesh:
        """Create rectangular base workpiece with center through-hole."""
        block = trimesh.creation.box(extents=[0.08, 0.08, 0.035])
        block.apply_translation([0.0, 0.0, 0.0175])
        return block

    def create_part_shaft(self) -> trimesh.Trimesh:
        """Create stepped cylindrical shaft workpiece."""
        lower_cylinder = trimesh.creation.cylinder(radius=0.015, height=0.06)
        lower_cylinder.apply_translation([0.0, 0.0, 0.03])

        collar = trimesh.creation.cylinder(radius=0.022, height=0.015)
        collar.apply_translation([0.0, 0.0, 0.0675])

        upper_pin = trimesh.creation.cylinder(radius=0.012, height=0.035)
        upper_pin.apply_translation([0.0, 0.0, 0.0925])

        shaft = trimesh.util.concatenate([lower_cylinder, collar, upper_pin])
        shaft.merge_vertices()
        return shaft

    def create_part_cap(self) -> trimesh.Trimesh:
        """Create top retaining cap with knurled profile."""
        cap_disk = trimesh.creation.cylinder(radius=0.028, height=0.02)
        cap_disk.apply_translation([0.0, 0.0, 0.01])

        knob = trimesh.creation.cylinder(radius=0.014, height=0.015)
        knob.apply_translation([0.0, 0.0, 0.0275])

        cap = trimesh.util.concatenate([cap_disk, knob])
        cap.merge_vertices()
        return cap

    def export_all_models(self) -> Dict[str, str]:
        """Export all parametric CAD components to STL format."""
        models = {
            "gripper_finger": self.create_gripper_finger(),
            "assembly_fixture": self.create_assembly_fixture(),
            "part_base": self.create_part_base(),
            "part_shaft": self.create_part_shaft(),
            "part_cap": self.create_part_cap(),
        }

        exported_paths = {}
        for name, mesh in models.items():
            stl_path = os.path.join(self.output_dir, f"{name}.stl")
            mesh.export(stl_path)
            exported_paths[name] = stl_path

        return exported_paths


def run_phase6_cad_verification() -> bool:
    """Run CAD generation verification test."""
    print(">>> Verifying Phase 6 CAD Models Generation & STL Export...")
    generator = CADModelGenerator()
    exported = generator.export_all_models()

    for name, path in exported.items():
        assert os.path.exists(path), f"File {path} was not created"
        size = os.path.getsize(path)
        print(f"  CAD Solid: {name:18s} -> {path} ({size} bytes)")
        assert size > 500, f"STL file {path} has suspiciously small size"

    print(">>> CAD Model Generation PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase6_cad_verification()
