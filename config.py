"""
Shared simulation configuration.

This file is the single source of truth for:
- Movement class ordering and menu options
- Movement to MyoSuite environment routing
- Common runtime defaults used by controller/task registration
- NN integration defaults (used now as placeholders for future wiring)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple


# Movement classes used across controller, registration, and NN integration.
MOVEMENT_CLASSES: Tuple[str, ...] = (
	"No_Movement",
	"Wrist_Flexion",
	"Wrist_Extension",
	"Wrist_Pronation",
	"Wrist_Supination",
	"Chuck_Grip",
	"Hand_Open",
)


# Manual controller menu options. Last option exits the menu.
MOVEMENT_MENU_MAP: Dict[str, Optional[str]] = {
	"1": "No_Movement",
	"2": "Wrist_Flexion",
	"3": "Wrist_Extension",
	"4": "Wrist_Pronation",
	"5": "Wrist_Supination",
	"6": "Chuck_Grip",
	"7": "Hand_Open",
	"8": None,
}


# Single environment used for ALL movement tasks.
# All movements share this env; per-movement pose targets are stored in
# Output/joint_tuning/<Movement>.json and applied at runtime via joint qpos driving.
DEFAULT_ENV_ID: str = os.getenv("MYOSUITE_ENV", "myoHandReachFixed-v0")
TASK_ENV_ID: str = DEFAULT_ENV_ID  # explicit alias: the one env for all tasks


# Optional per-movement variant kwargs merged into a base env spec when registered
# via myosuite.envs.env_variants.register_env_variant.
# Keep these minimal until each class is validated against actuator behavior.
MOVEMENT_VARIANT_KWARGS: Dict[str, Dict[str, object]] = {
	"Wrist_Flexion": {},
	"Wrist_Extension": {},
	"Wrist_Pronation": {},
	"Wrist_Supination": {},
	"Chuck_Grip": {},
	"Hand_Open": {},
}


def get_movement_env_routing() -> Dict[str, str]:
	"""Return routing for all movement classes, all mapped to the single TASK_ENV_ID."""
	return {
		movement: TASK_ENV_ID
		for movement in MOVEMENT_CLASSES
		if movement != "No_Movement"
	}


# Debug flag consumed by mapping code.
DEBUG_MAPPING_DEFAULT: str = os.getenv("DEBUG_MAPPING", "1")


# Runtime defaults for simulation/controller.
DEFAULT_MANUAL_RUNTIME_SECONDS: int = int(os.getenv("SIM_MANUAL_RUNTIME_SECONDS", "5"))
MAIN_RANDOM_ENV_ID: str = os.getenv("MYOSUITE_MAIN_ENV", DEFAULT_ENV_ID)
DUMP_ACTUATORS_ON_START: bool = os.getenv("MYOSUITE_DUMP_ACTUATORS", "0") == "1"


# NN integration placeholders (future simulation + NN bridge reads from here).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NN_DEFAULT_MODEL_PATH: str = os.getenv(
	"NN_MODEL_PATH",
	str(PROJECT_ROOT / "NN" / "models" / "final_model_full.pth"),
)
NN_WINDOW_SIZE: int = int(os.getenv("NN_WINDOW_SIZE", "100"))
NN_INFERENCE_STRIDE: int = int(os.getenv("NN_INFERENCE_STRIDE", "25"))
