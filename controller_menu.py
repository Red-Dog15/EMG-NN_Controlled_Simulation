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

try:
    import msvcrt
except ImportError:
    msvcrt = None

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
from run_nn import run_nn_mode
from viewer_utils import (
    close_passive_viewer,
    open_passive_viewer,
    run_viewer_submenu,
    sync_passive_viewer,
)


JOINT_TUNING_DIR = Path(__file__).resolve().parent.parent / "Output" / "joint_tuning"


def _poll_runtime_command():
    """Read non-blocking single-key runtime commands on Windows terminals."""
    if msvcrt is None:
        return None
    try:
        while msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key in {"\x00", "\xe0"}:
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            if key == "b":
                return "back"
            if key == "q":
                return "quit"
    except Exception:
        return None
    return None


def show_main_menu():
    print("\n=== Control Menu ===")
    print("1) NN implementation (placeholder)")
    print("2) Controller (manual movement selection)")
    print("3) Exit")


def nn_implementation_placeholder():
    print("\n=== NN-Driven Simulation ===")
    csv_input = input("Path to EMG CSV (Enter to use interactive menu): ").strip()
    csv_path = csv_input if csv_input else None
    run_nn_mode(csv_path=csv_path)


def print_movement_guide():
    print("\n=== Movement Classes Guide ===")
    for key, movement in MOVEMENT_MENU_MAP.items():
        if movement is None:
            print(f"{key}) Exit")
        else:
            print(f"{key}) {movement}")
    print("v) Viewer settings (skin / camera)")


def get_movement_from_user():
    """
    Prompts for movement selection 1-8 or 'v' for viewer. Returns movement name,
    None for exit, or '__viewer__' sentinel for viewer submenu.
    """
    while True:
        print_movement_guide()
        choice = input("Select movement (1-8) or v: ").strip().lower()
        if choice in MOVEMENT_MENU_MAP:
            return MOVEMENT_MENU_MAP[choice]
        if choice == "v":
            return "__viewer__"
        print("Invalid selection. Please choose 1-8 or v.")


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
    """Load target qpos per joint from exported tuning JSON.

        Preference order:
            1) exact joint snapshot from tuner (target_joint_qpos)
            2) midpoint of target_jnt_range (legacy fallback)
    """
    path = JOINT_TUNING_DIR / f"{movement_name}.json"
    
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        exact = payload.get("target_joint_qpos")
        if isinstance(exact, dict) and exact:
            targets = {}
            for joint_name, value in exact.items():
                try:
                    targets[joint_name] = float(value)
                except Exception:
                    continue
            if targets:
                return targets

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


def _smoothstep01(x: float) -> float:
    """Cubic smoothstep from 0..1 to 0..1 for gentle transition ramps."""
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _apply_joint_targets_interp(env, start_qpos, target_qpos, phase: float):
    """Interpolate joints from start pose to target pose using smooth phase."""
    sim = env.unwrapped.sim
    w = _smoothstep01(phase)
    for qidx, target in target_qpos.items():
        s = float(start_qpos.get(qidx, sim.data.qpos[qidx]))
        sim.data.qpos[qidx] = (1.0 - w) * s + w * float(target)
    # Zero velocity for the driven joints so interpolation is visually stable.
    if hasattr(sim.data, "qvel"):
        sim.data.qvel[:] = 0.0
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

    # Open passive viewer so user can adjust skin/camera while running.
    passive_viewer = open_passive_viewer(env)
    if passive_viewer is None:
        print("Note: passive viewer unavailable for this env backend; use MyoSuite's built-in render.")
    else:
        print("Passive viewer opened.  Press 'v' in the movement menu to adjust skin/camera.")
        print("Using passive viewer only (built-in env window disabled to avoid conflicting visuals).")

    step_count = 0
    # Keep action continuity across movement selections for smooth muscle transitions.
    prev_action = np.zeros(env.action_space.shape[0], dtype=float)
    try:
        while True:
            if passive_viewer is not None and not sync_passive_viewer(passive_viewer):
                passive_viewer = None
                print("Passive viewer was closed.")

            movement = get_movement_from_user()
            if movement is None:
                print("Exiting controller.")
                return

            if movement == "__viewer__":
                passive_viewer = run_viewer_submenu(env, passive_viewer)
                continue

            is_no_movement = movement == "No_Movement"

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
            if is_no_movement:
                # Explicit pass/idle state: no commanded movement.
                action = np.zeros(env.action_space.shape[0], dtype=float)
                resolved_qpos_targets = None
                print("No_Movement selected: issuing zero activation (pass state).")
            elif not resolved_qpos_targets:
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
            print(
                f"Running for {time_value} seconds... "
                "Press 'b' to abort to movement menu, 'q' to quit, or Ctrl+C to stop."
            )

            transition_seconds = max(0.4, min(2.0, float(time_value) * 0.5))
            seg_start_time = time.time()

            start_qpos_targets = None
            if resolved_qpos_targets:
                sim = env.unwrapped.sim
                start_qpos_targets = {qidx: float(sim.data.qpos[qidx]) for qidx in resolved_qpos_targets.keys()}

            target_action = action if action is not None else prev_action.copy()
            start_action = prev_action.copy()
            if is_no_movement:
                # Do not blend from previous movement into idle; stop immediately.
                start_action = target_action.copy()
                prev_action = target_action.copy()

            end_time = time.time() + time_value
            aborted_to_menu = False
            while time.time() < end_time:
                runtime_cmd = _poll_runtime_command()
                if runtime_cmd == "back":
                    print("Current movement aborted. Returning to movement menu.")
                    aborted_to_menu = True
                    break
                if runtime_cmd == "quit":
                    print("Quit requested.")
                    return
                if passive_viewer is None:
                    env.unwrapped.mj_render()
                if is_no_movement:
                    # Pass state: intentionally do not advance physics.
                    time.sleep(getattr(env, "dt", 0.01))
                elif resolved_qpos_targets:
                    phase = (time.time() - seg_start_time) / transition_seconds
                    _apply_joint_targets_interp(env, start_qpos_targets, resolved_qpos_targets, phase=phase)
                    env.unwrapped.sim.advance(substeps=1, render=False)
                else:
                    phase = (time.time() - seg_start_time) / transition_seconds
                    w = _smoothstep01(phase)
                    blended_action = (1.0 - w) * start_action + w * target_action
                    env.step(blended_action)
                    prev_action = blended_action
                sync_passive_viewer(passive_viewer)
                step_count += 1

                if step_count % 100 == 0:
                    print(f"Steps: {step_count}")

            if aborted_to_menu:
                continue

    except KeyboardInterrupt:
        print(f"\nStopped after {step_count} steps")
    finally:
        close_passive_viewer(passive_viewer)
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
