"""Vision-Language Model (VLM) Assembly Task Planner.

Interprets natural-language assembly instructions alongside RGB-D camera frames
to produce structured JSON motion plans with 3D pick/place targets and semantic constraints.
Includes both Hugging Face SmolVLM interface and an offline MockVLM engine.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Strict virtual environment assertion
if sys.prefix == sys.base_prefix:
    sys.exit("CRITICAL ERROR: Script must run inside the virtual environment at ./venv")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
from PIL import Image

SYSTEM_PROMPT = """You are a precision robotic assembly vision-language planner.
Analyze the camera image and natural language assembly instruction.
Output strictly a JSON object with this exact schema:
{
  "target_part": "base" | "shaft" | "cap",
  "pick_pixel": [u, v],
  "place_pose": [x, y, z],
  "approach": "top" | "side",
  "constraints": ["avoid_human", "slow_near_fragile", "approach_from_above"],
  "steps": [
    "move_pre_pick",
    "approach_pick",
    "grasp_part",
    "lift_part",
    "move_pre_place",
    "insert_or_place",
    "release_part",
    "retreat_home"
  ]
}
Do not output any reasoning text, explanations, or conversational filler. Output only valid JSON.
"""


class MockVLM:
    """Deterministic, rule-based VLM engine implementing identical JSON output schema."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        # Part locations in pixel space (u, v) and assembly fixture place poses
        self.part_catalog = {
            "base": {
                "pixel": [226, 203],
                "place_pose": [0.50, 0.20, 0.44],
                "approach": "top",
                "default_constraints": ["avoid_human", "approach_from_above"],
            },
            "shaft": {
                "pixel": [270, 185],
                "place_pose": [0.50, 0.20, 0.49],
                "approach": "top",
                "default_constraints": ["avoid_human", "slow_near_fragile", "approach_from_above"],
            },
            "cap": {
                "pixel": [205, 235],
                "place_pose": [0.50, 0.20, 0.53],
                "approach": "top",
                "default_constraints": ["avoid_human", "approach_from_above"],
            },
        }

    def generate_plan(self, instruction: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Generate structured assembly plan from natural language instruction."""
        instr_lower = instruction.lower()

        # Identify target part from instruction (checking primary target nouns)
        if "cap" in instr_lower or "lid" in instr_lower or "cover" in instr_lower:
            part_name = "cap"
        elif "base" in instr_lower or "block" in instr_lower:
            part_name = "base"
        elif "shaft" in instr_lower or "peg" in instr_lower or "cylinder" in instr_lower:
            part_name = "shaft"
        else:
            part_name = "shaft"  # default assembly part

        catalog = self.part_catalog[part_name]
        pixel = list(catalog["pixel"])
        # Slight realistic pixel jitter
        pixel[0] += int(self.rng.randint(-2, 3))
        pixel[1] += int(self.rng.randint(-2, 3))

        constraints = list(catalog["default_constraints"])
        if "careful" in instr_lower or "fragile" in instr_lower:
            if "slow_near_fragile" not in constraints:
                constraints.append("slow_near_fragile")
        if "human" in instr_lower or "person" in instr_lower or "operator" in instr_lower:
            if "avoid_human" not in constraints:
                constraints.append("avoid_human")

        plan = {
            "target_part": part_name,
            "pick_pixel": pixel,
            "place_pose": catalog["place_pose"],
            "approach": catalog["approach"],
            "constraints": constraints,
            "steps": [
                "move_pre_pick",
                "approach_pick",
                "grasp_part",
                "lift_part",
                "move_pre_place",
                "insert_or_place",
                "release_part",
                "retreat_home"
            ]
        }
        return plan


class HuggingFaceVLM:
    """Hugging Face Vision-Language Model interface for SmolVLM / Qwen2-VL."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> None:
        self.model_id = model_id
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """Load model and processor onto CPU."""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        print(f"Loading VLM model: {self.model_id} (CPU)...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("VLM model successfully loaded.")

    def generate_plan(self, instruction: str, image: Image.Image) -> Dict[str, Any]:
        """Query VLM with instruction and image, parsing structured JSON response."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{SYSTEM_PROMPT}\nInstruction: {instruction}"}
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

        import torch
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=False)

        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        return parse_and_validate_vlm_json(decoded)


def parse_and_validate_vlm_json(raw_text: str, max_retries: int = 3) -> Dict[str, Any]:
    """Extract, parse, and validate JSON plan against the assembly schema."""
    cleaned = raw_text.strip()

    # Match JSON block inside ```json ... ``` or first { ... }
    match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        json_str = cleaned

    try:
        data = json.loads(json_str)
    except Exception as exc:
        raise ValueError(f"Failed to parse VLM response as valid JSON: {exc}\nRaw: {raw_text}")

    # Enforce mandatory schema fields
    required_fields = ["target_part", "pick_pixel", "place_pose", "approach", "constraints", "steps"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field in VLM plan: '{field}'")

    # Validate types
    if data["target_part"] not in ["base", "shaft", "cap"]:
        raise ValueError(f"Invalid target_part '{data['target_part']}'")
    if len(data["pick_pixel"]) != 2:
        raise ValueError(f"pick_pixel must be 2D array [u, v], got {data['pick_pixel']}")
    if len(data["place_pose"]) != 3:
        raise ValueError(f"place_pose must be 3D array [x, y, z], got {data['place_pose']}")
    if data["approach"] not in ["top", "side"]:
        raise ValueError(f"approach must be 'top' or 'side', got {data['approach']}")
    if not isinstance(data["constraints"], list):
        raise ValueError("constraints must be a list of strings")

    return data


class VLMAssemblyPlanner:
    """High-level VLM assembly planner with pixel-to-3D projection and retry logic."""

    def __init__(self, use_mock: bool = True, model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> None:
        self.use_mock = use_mock
        if use_mock:
            self.engine: Union[MockVLM, HuggingFaceVLM] = MockVLM()
        else:
            try:
                self.engine = HuggingFaceVLM(model_id=model_id)
            except Exception as e:
                print(f"Warning: Failed to load Hugging Face model ({e}). Falling back to MockVLM.")
                self.engine = MockVLM()
                self.use_mock = True

    def plan_task(
        self,
        instruction: str,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        env: Any,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Generate verified assembly plan and convert pick_pixel to 3D pick_pose."""
        pil_image = Image.fromarray(rgb_image)
        plan: Optional[Dict[str, Any]] = None

        for attempt in range(max_retries):
            try:
                if self.use_mock:
                    raw_plan = self.engine.generate_plan(instruction, pil_image)
                else:
                    raw_plan = self.engine.generate_plan(instruction, pil_image)
                # Verify JSON structure
                plan = parse_and_validate_vlm_json(json.dumps(raw_plan))
                break
            except Exception as err:
                print(f"Plan parsing attempt {attempt+1}/{max_retries} failed: {err}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"VLM planner failed after {max_retries} retries: {err}")

        assert plan is not None

        # Convert 2D pick pixel + depth into calibrated 3D world coordinate
        u, v = int(plan["pick_pixel"][0]), int(plan["pick_pixel"][1])
        u_clamped = int(np.clip(u, 0, env.cam_width - 1))
        v_clamped = int(np.clip(v, 0, env.cam_height - 1))
        depth_val = float(depth_image[v_clamped, u_clamped])

        pick_3d = env.pixel_to_world(u_clamped, v_clamped, depth_val)

        # Build full executable plan object
        plan["pick_pose_3d"] = [float(x) for x in np.round(pick_3d, 4)]
        plan["raw_depth"] = float(np.round(depth_val, 4))
        plan["instruction"] = instruction
        return plan


def run_phase2_verification() -> bool:
    """Run Phase 2 VLM planner verification test."""
    from src.sim_env import CobotSimEnv

    print(">>> Verifying Phase 2: VLM Assembly Planner & Coordinate Deprojection...")
    env = CobotSimEnv()
    rgb, depth = env.render_rgb_depth()

    planner = VLMAssemblyPlanner(use_mock=True)

    # Test Case 1: Shaft assembly instruction with safety constraints
    instr1 = "Grasp the fragile cylindrical shaft and insert it into the assembly fixture, avoiding the human operator."
    plan1 = planner.plan_task(instr1, rgb, depth, env)

    print(f"Instruction 1: '{instr1}'")
    print(f"Target Part: {plan1['target_part']}")
    print(f"Pick Pixel: {plan1['pick_pixel']} -> 3D Pick World Pose: {plan1['pick_pose_3d']}")
    print(f"Place World Pose: {plan1['place_pose']}")
    print(f"Approach: {plan1['approach']}")
    print(f"Semantic Constraints: {plan1['constraints']}")
    print(f"Plan Steps ({len(plan1['steps'])} total): {plan1['steps']}")

    # Test Case 2: Base block instruction
    instr2 = "Pick the rectangular base block and mount it firmly onto the workbench fixture."
    plan2 = planner.plan_task(instr2, rgb, depth, env)
    print(f"\nInstruction 2: '{instr2}'")
    print(f"Target Part: {plan2['target_part']}, Pick 3D: {plan2['pick_pose_3d']}, Place 3D: {plan2['place_pose']}")

    # Test Case 3: Cap instruction
    instr3 = "Place the top cap carefully over the installed shaft."
    plan3 = planner.plan_task(instr3, rgb, depth, env)
    print(f"\nInstruction 3: '{instr3}'")
    print(f"Target Part: {plan3['target_part']}, Pick 3D: {plan3['pick_pose_3d']}, Place 3D: {plan3['place_pose']}")

    # Validations
    assert plan1["target_part"] == "shaft", "Failed to identify shaft"
    assert "avoid_human" in plan1["constraints"], "Failed to extract human avoidance constraint"
    assert "slow_near_fragile" in plan1["constraints"], "Failed to extract fragile constraint"
    assert len(plan1["pick_pose_3d"]) == 3, "Invalid 3D pick pose"
    assert len(plan1["place_pose"]) == 3, "Invalid place pose"
    assert plan2["target_part"] == "base", "Failed to identify base"
    assert plan3["target_part"] == "cap", "Failed to identify cap"

    print("\n>>> Phase 2 Verification PASSED successfully!")
    return True


if __name__ == "__main__":
    run_phase2_verification()
