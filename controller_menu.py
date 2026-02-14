"""
Simple CLI menu for choosing control mode and manual movement selection.
"""

import os
import shutil
import sys
import time

import numpy as np

from myosuite.utils import gym

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
    print("1) No_Movement")
    print("2) Wrist_Flexion")
    print("3) Wrist_Extension")
    print("4) Wrist_Pronation")
    print("5) Wrist_Supination")
    print("6) Chuck_Grip")
    print("7) Hand_Open")
    print("8) Exit")


def get_movement_from_user():
    """
    Prompts for movement selection 1-8 and returns movement name or None for exit.
    """
    movement_map = {
        "1": "No_Movement",
        "2": "Wrist_Flexion",
        "3": "Wrist_Extension",
        "4": "Wrist_Pronation",
        "5": "Wrist_Supination",
        "6": "Chuck_Grip",
        "7": "Hand_Open",
        "8": None,
    }

    while True:
        print_movement_guide()
        choice = input("Select movement (1-8): ").strip()
        if choice in movement_map:
            return movement_map[choice]
        print("Invalid selection. Please choose 1-8.")


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


def run_manual_controller(time):
    _configure_git_executable()

    env_id = os.getenv("MYOSUITE_ENV", "myoArmReachFixed-v0")
    print(f"\nStarting MyoSuite env: {env_id}")

    env = gym.make(env_id)
    env.reset()

    actuator_names = _get_actuator_names(env)

    if not actuator_names:
        print("Warning: actuator names not found. Movement LUT may be empty.")

    step_count = 0
    try:
        while True:
            movement = get_movement_from_user()
            if movement is None:
                print("Exiting controller.")
                return

            action = get_MyoSuite_Movement_LUT(
                movement_name=movement,
                action_size=env.action_space.shape[0],
                actuator_names=actuator_names,
            )
            action = np.array(action, dtype=float)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            print(f"Selected movement: {movement}")
            print(f"Running for {time} seconds... Close the window or press Ctrl+C to stop.")

            end_time = time.time() + time
            while time.time() < end_time:
                env.unwrapped.mj_render()
                env.step(action)
                step_count += 1

                if step_count % 100 == 0:
                    print(f"Steps: {step_count}")

    except KeyboardInterrupt:
        print(f"\nStopped after {step_count} steps")
    finally:
        env.close()
        print("Environment closed")


def controller_menu():
    run_manual_controller(5) # run for 5 seconds

def main():
    while True:
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
