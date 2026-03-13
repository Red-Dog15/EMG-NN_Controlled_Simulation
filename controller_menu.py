"""
Simple CLI menu for choosing control mode and manual movement selection.
"""

import os
import shutil
import sys
import time
import json
from pathlib import Path

import numpy as np

from myosuite.utils import gym

from config import (
    DEBUG_MAPPING_DEFAULT,
    DEFAULT_ENV_ID,
    DEFAULT_MANUAL_RUNTIME_SECONDS,
    MOVEMENT_MENU_MAP,
)

SCRIPTS_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Scripts", "DATA")
)
if SCRIPTS_DATA_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DATA_DIR)

from Data_Mapping import get_MyoSuite_Movement_LUT


JOINT_TUNING_DIR = Path(__file__).resolve().parent.parent / "Output" / "joint_tuning"


def show_main_menu():
    print("\n=== Control Menu ===")
    print("1) NN implementation (placeholder)")
    print("2) Controller (manual movement selection)")
    print("3) Exit")


def nn_implementation_placeholder():
    print("\nNN implementation is not wired yet.")
    # Placeholder for future NN integration
    pass


def print_movement_guide():
    print("\n=== Movement Classes Guide ===")
    for key, movement in MOVEMENT_MENU_MAP.items():
        if movement is None:
            print(f"{key}) Exit")
        else:
            print(f"{key}) {movement}")


def get_movement_from_user():
    """
    Prompts for movement selection 1-8 and returns movement name or None for exit.
    """
    while True:
        print_movement_guide()
        choice = input("Select movement (1-8): ").strip()
        if choice in MOVEMENT_MENU_MAP:
            return MOVEMENT_MENU_MAP[choice]
        print("Invalid selection. Please choose 1-8.")


def _open_env(env_id):
    env = gym.make(env_id)
    env.reset()
    actuator_names = _get_actuator_names(env)
    return env, actuator_names


def _get_joint_name(model, idx):
    try:
        return model.joint(idx).name
    except Exception:
        pass
    try:
        return model.joint_id2name(idx)
    except Exception:
        return f"joint_{idx}"


def _load_exported_joint_targets(movement_name):
    """Load midpoint target qpos per joint from exported tuning JSON."""
    path = JOINT_TUNING_DIR / f"{movement_name}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ranges = payload.get("target_jnt_range") or payload.get("suggested_variant_kwargs", {}).get("target_jnt_range")
        if not isinstance(ranges, dict):
            return None

        targets = {}
        for joint_name, bounds in ranges.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                continue
            low, high = float(bounds[0]), float(bounds[1])
            targets[joint_name] = (low + high) / 2.0
        return targets if targets else None
    except Exception:
        return None


def _resolve_joint_qpos_targets(env, joint_targets):
    """Resolve exported joint-name targets to qpos-index targets for current env."""
    sim = env.unwrapped.sim
    model = sim.model
    resolved = {}

    for j in range(model.njnt):
        name = _get_joint_name(model, j)
        if name not in joint_targets:
            continue
        qidx = int(model.jnt_qposadr[j])
        qlow = float(model.jnt_range[j][0])
        qhigh = float(model.jnt_range[j][1])
        qtarget = max(qlow, min(qhigh, float(joint_targets[name])))
        resolved[qidx] = qtarget

    return resolved


def _apply_joint_targets_step(env, qpos_targets, blend=0.35):
    """Blend current qpos toward exported movement target pose."""
    sim = env.unwrapped.sim
    for qidx, target in qpos_targets.items():
        cur = float(sim.data.qpos[qidx])
        sim.data.qpos[qidx] = (1.0 - blend) * cur + blend * target
    sim.forward()


def _configure_git_executable():
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    git_exe = shutil.which("git")
    if not git_exe:
        common_git_paths = [
            "C:/Program Files/Git/bin/git.exe",
            "C:/Program Files/Git/cmd/git.exe",
            "C:/Program Files (x86)/Git/bin/git.exe",
            "C:/Program Files (x86)/Git/cmd/git.exe",
        ]
        git_exe = next((p for p in common_git_paths if os.path.isfile(p)), None)
    if git_exe:
        os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", git_exe)


def _get_actuator_names(env):
    """
    Docstring for _get_actuator_names
    
    :param env: Description
    """
    try:
        unwrapped = env.unwrapped
        model = getattr(unwrapped, "model", None)

        if model is None:
            sim = getattr(unwrapped, "sim", None)
            model = getattr(sim, "model", None) if sim is not None else None

        if model is None:
            return []

        if hasattr(model, "actuator_names"):
            return list(getattr(model, "actuator_names", []))

        try:
            import mujoco

            if hasattr(model, "id2name") and hasattr(model, "nu"):
                names = []
                for i in range(model.nu):
                    name = model.id2name(mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    names.append(name or "")
                return names
        except Exception:
            return []
    except Exception:
        return []

    return []


def run_manual_controller(time_value):
    _configure_git_executable()

    # Enable debug mode to inspect matching while you build custom task envs.
    os.environ["DEBUG_MAPPING"] = DEBUG_MAPPING_DEFAULT

    # Manual mode intentionally stays in a single environment for all movements.
    current_env_id = DEFAULT_ENV_ID
    env, actuator_names = _open_env(current_env_id)
    print("\n=== Manual Controller Environment ===")
    print(f"All movements run in: {current_env_id}")
    print("=" * 50)
    if not actuator_names:
        print("Warning: actuator names not found. Movement LUT may be empty.")
    else:
        print(f"Total actuators in {current_env_id}: {len(actuator_names)}")

    step_count = 0
    try:
        while True:
            movement = get_movement_from_user()
            if movement is None:
                print("Exiting controller.")
                return

            # Prefer exported joint tuning when available for this movement.
            exported_joint_targets = None
            resolved_qpos_targets = None
            if movement != "No_Movement":
                exported_joint_targets = _load_exported_joint_targets(movement)
                if exported_joint_targets:
                    resolved_qpos_targets = _resolve_joint_qpos_targets(env, exported_joint_targets)
                    if resolved_qpos_targets:
                        print(
                            f"Using exported joint tuning for '{movement}' "
                            f"({len(resolved_qpos_targets)} joints matched in {current_env_id})."
                        )
                    else:
                        print(
                            f"Export found for '{movement}', but no matching joints in '{current_env_id}'. "
                            "Falling back to legacy actuator LUT."
                        )

            action = None
            if not resolved_qpos_targets:
                action = get_MyoSuite_Movement_LUT(
                    movement_name=movement,
                    action_size=env.action_space.shape[0],
                    actuator_names=actuator_names,
                )

                if movement != "No_Movement" and not np.any(action):
                    print(f"Movement '{movement}' produced no active actuators in '{current_env_id}'.")
                    print("This movement has no mapped actuators in the current single-environment setup.")
                    continue

                action = np.array(action, dtype=float)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            print(f"Selected movement: {movement} (env: {current_env_id})")
            print(f"Running for {time_value} seconds... Close the window or press Ctrl+C to stop.")

            end_time = time.time() + time_value
            while time.time() < end_time:
                env.unwrapped.mj_render()
                if resolved_qpos_targets:
                    _apply_joint_targets_step(env, resolved_qpos_targets, blend=0.35)
                    # Advance one frame for dynamics/contacts while preserving pose drive.
                    env.unwrapped.sim.advance(substeps=1, render=False)
                else:
                    env.step(action)
                step_count += 1

                if step_count % 100 == 0:
                    print(f"Steps: {step_count}")

    except KeyboardInterrupt:
        print(f"\nStopped after {step_count} steps")
    finally:
        if env is not None:
            env.close()
        print("Environment closed")


def controller_menu():
    run_manual_controller(DEFAULT_MANUAL_RUNTIME_SECONDS)

def main():
    while True: # choice handler (loop)
        show_main_menu()
        choice = input("Select option (1-3): ").strip()
        if choice == "1":
            nn_implementation_placeholder()
        elif choice == "2":
            controller_menu()
        elif choice == "3":
            print("Goodbye.")
            return
        else:
            print("Invalid selection. Please choose 1-3.")


if __name__ == "__main__":
    main()
