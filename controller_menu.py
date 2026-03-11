"""
Simple CLI menu for choosing control mode and manual movement selection.
"""

import os
import shutil
import sys
import time

import numpy as np

from myosuite.utils import gym

from config import (
    DEBUG_MAPPING_DEFAULT,
    DEFAULT_ENV_ID,
    DEFAULT_MANUAL_RUNTIME_SECONDS,
    MOVEMENT_MENU_MAP,
    get_movement_env_routing,
)

SCRIPTS_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Scripts", "DATA")
)
if SCRIPTS_DATA_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DATA_DIR)

from Data_Mapping import get_MyoSuite_Movement_LUT


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


def _resolve_env_for_movement(movement_name, default_env_id, movement_env_routing):
    if movement_name == "No_Movement":
        return default_env_id
    return movement_env_routing.get(movement_name, default_env_id)


def _open_env(env_id):
    env = gym.make(env_id)
    env.reset()
    actuator_names = _get_actuator_names(env)
    return env, actuator_names


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

    default_env_id = DEFAULT_ENV_ID
    movement_env_routing = get_movement_env_routing()
    env = None
    actuator_names = []
    current_env_id = None

    print("\n=== Movement -> Environment Routing ===")
    for movement, env_id in movement_env_routing.items():
        print(f"  {movement}: {env_id}")
    print(f"Default env (No_Movement): {default_env_id}")
    print("=" * 50)

    step_count = 0
    try:
        while True:
            movement = get_movement_from_user()
            if movement is None:
                print("Exiting controller.")
                return

            target_env_id = _resolve_env_for_movement(
                movement,
                default_env_id,
                movement_env_routing,
            )

            if env is None or target_env_id != current_env_id:
                if env is not None:
                    env.close()
                print(f"\nSwitching environment -> {target_env_id}")
                env, actuator_names = _open_env(target_env_id)
                current_env_id = target_env_id

                if not actuator_names:
                    print("Warning: actuator names not found. Movement LUT may be empty.")
                else:
                    print(f"Total actuators in {current_env_id}: {len(actuator_names)}")

            action = get_MyoSuite_Movement_LUT(
                movement_name=movement,
                action_size=env.action_space.shape[0],
                actuator_names=actuator_names,
            )

            if movement != "No_Movement" and not np.any(action):
                print(f"Movement '{movement}' produced no active actuators in '{current_env_id}'.")
                print("Change routing in simulation/config.py or update your custom task env mapping.")
                continue

            action = np.array(action, dtype=float)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            print(f"Selected movement: {movement} (env: {current_env_id})")
            print(f"Running for {time_value} seconds... Close the window or press Ctrl+C to stop.")

            end_time = time.time() + time_value
            while time.time() < end_time:
                env.unwrapped.mj_render()
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
