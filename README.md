# VLM-Based Path Planning and Safe Control of a Cobot in Assembly Operations

[![Simulation](https://img.shields.io/badge/Simulation-MuJoCo-black.svg)](https://mujoco.org)
[![Robot](https://img.shields.io/badge/Robot-Franka_Emika_Panda-black.svg)](https://franka.de)
[![Safety](https://img.shields.io/badge/Safety-ISO%2FTS_15066_SSM_%2B_CBF--QP-black.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.11-black.svg)](#)

A collaborative robotics research framework integrating open-vocabulary **Vision-Language Models (VLMs)**, a **Geometric Validation Shield**, **Semantic RRT-Connect Motion Planning**, and **Dual-Layer Closed-Loop Safety Filtering (ISO/TS 15066 SSM + Control Barrier Functions)** on a 7-DOF Franka Emika Panda manipulator.

---

## Technical Novelty

1. **Deterministic Verification Shield Against VLM Hallucinations**:
   VLM outputs are never executed directly on hardware. Intermediate JSON motion plans are geometrically validated (workspace radial reachability, damped least-squares IK, joint limits with a 0.05 rad buffer, and collision tests) and automatically repaired or rejected with an audit trail before execution.
2. **Semantic Directives as Continuous Optimization Costs**:
   Natural-language semantic directives (`"avoid_human"`, `"slow_near_fragile"`) are translated into continuous potential cost fields within bidirectional RRT-Connect, actively biasing joint search trees away from human operator proximity and dampening velocity near sensitive assembly zones.
3. **Closed-Loop Dual-Layer Reactive Safety Control**:
   Commanded joint velocities are filtered in real time at 100 Hz using ISO/TS 15066 Speed & Separation Monitoring (SSM) and Control Barrier Functions (CBF-QP solved via Clarabel). The safety filter renders the separation boundary forward-invariant, guaranteeing collision prevention even under unpredicted dynamic human intrusion.

---

## Directory Structure

```text
cobot_vlm/
  ├── config.yaml                    # Master project configuration
  ├── requirements.txt               # Pinned dependencies
  ├── requirements.lock.txt          # Frozen reproducible environment
  ├── src/
  │   ├── __init__.py
  │   ├── sim_env.py                 # Franka Panda workcell, parts, camera, human proxy
  │   ├── vlm_planner.py             # MockVLM & SmolVLM task parser + deprojection
  │   ├── plan_validator.py          # Geometric reachability, IK, limits & repair shield
  │   ├── motion_planner.py          # Joint-space RRT-Connect with semantic cost & quintic timing
  │   ├── safety_filter.py           # Dual-layer SSM + CBF-QP velocity filter
  │   ├── controller.py              # Joint-space trajectory tracking controller
  │   ├── cad_models.py              # Parametric CAD solids & STL export
  │   ├── cae_analysis.py            # FEA stress, FoS, manipulability & torque analysis
  │   ├── experiments.py             # 90-scene Monte Carlo evaluation across Methods A, B, C
  │   ├── make_figures.py            # Generates all 7 publication figures in grayscale (300 DPI)
  │   └── make_report.py             # Generates formal 5-page research report (DOCX)
  ├── outputs/
  │   ├── figures/                   # Fig 1 to Fig 7 (monochrome, 300 DPI)
  │   ├── data/                      # Raw trial CSVs, summary tables, and audit logs
  │   ├── cad/                       # Solid STL models (gripper, fixture, parts)
  │   ├── report/                    # Compiled 5-page research report (report.docx)
  │   └── screenshots/               # Workcell high-resolution render
  └── README.md
```

---

## Installation & Setup

All commands must be executed within a virtual environment at `./venv`:

```powershell
# 1. Clone repository
git clone https://github.com/1504raghavnama/cobot-vlm.git
cd cobot-vlm

# 2. Create isolated virtual environment
py -3.11 -m venv venv

# 3. Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# 4. Verify virtual environment prefix
python -c "import sys; assert sys.prefix != sys.base_prefix; print('Active venv:', sys.prefix)"

# 5. Install locked dependencies
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

---

## Phase-by-Phase Execution Guide

To reproduce each phase independently or run the entire pipeline from scratch:

```powershell
# Phase 1: Simulation Workcell Setup & Screenshot
python src/sim_env.py

# Phase 2: VLM Assembly Planner & Pixel Deprojection
python src/vlm_planner.py

# Phase 3: Plan Validator & Geometric Repair Shield
python src/plan_validator.py

# Phase 4: Joint-Space RRT-Connect with Semantic Costs & Quintic Trajectories
python src/motion_planner.py

# Phase 5: Dual-Layer Safety Filter (SSM + CBF-QP) & Controller
python src/safety_filter.py
python src/controller.py

# Phase 6: Parametric CAD Modeling & CAE Structural Analysis
python src/cad_models.py
python src/cae_analysis.py

# Phase 7: Monte Carlo Experiments (90 Randomized Runs across Methods A, B, C)
python src/experiments.py

# Phase 8: Publication-Quality Grayscale Figures (300 DPI)
python src/make_figures.py

# Phase 9: Comprehensive 5-Page Research Report (report.docx)
python src/make_report.py
```

---

## Experimental Benchmark Results (30 Scenes / Method)

Real empirical results obtained from the 90-scene randomized Monte Carlo simulation (seed = scene index):

| Metric | Method A (RRT Only) | Method B (VLM Only) | Method C (Proposed) |
| :--- | :---: | :---: | :---: |
| **Task Success Rate** | 100.0% | 33.3% | **100.0%** |
| **Path Length (rad)** | 1.48 ± 0.07 | 1.67 ± 0.06 | 2.54 ± 0.09 |
| **Planning Time (s)** | 0.017 ± 0.022 | 0.017 ± 0.020 | 0.077 ± 0.048 |
| **Tracking RMSE (rad)** | 0.0055 ± 0.0004 | 0.0060 ± 0.0005 | 0.0091 ± 0.0006 |
| **Min Human Dist (m)** | 0.204 ± 0.046 | 0.159 ± 0.230 | **0.439 ± 0.064** |
| **Violated Trials (<0.15m)** | 5 / 30 | 20 / 30 | **0 / 30** |
| **Violation Timesteps** | 1.7 ± 5.3 | 13.3 ± 9.6 | **0.0 ± 0.0** |
| **Cycle Time (s)** | 1.10 ± 0.04 | 1.19 ± 0.04 | 1.18 ± 0.05 |

- **VLM Engine Evaluated**: `MockVLM` (Rule-based deterministic schema extractor, offline-safe; switchable to `HuggingFaceTB/SmolVLM-256M-Instruct`).
- **Gripper Finger Structural FEA**: `Analytical continuum beam stress formulation with 2D spatial stress tensor field (Al6061-T6)`, Grasp load 45 N, Max Von Mises stress $\sigma_{vM} = 14.50$ MPa, **Factor of Safety = 19.03** (Yield strength 276 MPa).

---

## Generated Artifacts

- **Report**: `outputs/report/report.docx` (Calibrated 5-page research proposal document).
- **Figures** (300 DPI, Black/Grayscale styling):
  - `outputs/figures/fig1_system_architecture.png`
  - `outputs/figures/fig2_workcell_and_cad.png`
  - `outputs/figures/fig3_cae_stress_and_torque.png`
  - `outputs/figures/fig4_planned_vs_executed_path.png`
  - `outputs/figures/fig5_human_distance_time_series.png`
  - `outputs/figures/fig6_comparative_metrics_bar_chart.png`
  - `outputs/figures/fig7_validation_flowchart.png`
- **CAD Solids**:
  - `outputs/cad/gripper_finger.stl`
  - `outputs/cad/assembly_fixture.stl`
  - `outputs/cad/part_base.stl`
  - `outputs/cad/part_shaft.stl`
  - `outputs/cad/part_cap.stl`
- **Data & Audits**:
  - `outputs/data/experiment_trials.csv` (Raw 90-scene data)
  - `outputs/data/experiment_summary.csv` (Mean ± std statistics)
  - `outputs/data/validator_audit.json` (VLM plan verification records)
  - `outputs/data/cae_summary.json` (FEA stress and joint torque metrics)

---

## Recovery Section

To safely recover, inspect, or revert any stage of the project:

### 1. Clone and Rebuild Environment
```powershell
git clone https://github.com/1504raghavnama/cobot-vlm.git
cd cobot-vlm
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```

### 2. Inspect or Roll Back to a Specific Phase
All phases are strictly tagged:
```powershell
# View all phase tags
git tag

# Switch workspace to any phase (e.g. Phase 4 or Phase 7)
git checkout phase-4
git checkout phase-7

# Return to main branch
git checkout main
```

### 3. Safely Undo a Commit Without Rewriting History
```powershell
# Revert a specific commit hash safely
git revert <commit_hash>
```

### 4. Restore a Single File from a Previous Phase
```powershell
# Restore a file exactly as it was in Phase N
git restore --source=phase-3 src/plan_validator.py
git restore --source=phase-5 src/safety_filter.py
```
