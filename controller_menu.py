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

from Data_Mapping import get_MyoSuite_Movement_LUT, get_supported_movement_classes


def show_main_menu():
    print("\n=== Control Menu ===")
    print("1) NN implementation (placeholder)")
    print("2) Controller (manual movement selection)")
    print("3) Exit")


def nn_implementation_placeholder():
    print("\nNN implementation is not wired yet.")
    # Placeholder for future NN integration
    pass


def print_movement_guide(supported_movements=None):
    if supported_movements is None:
        supported_movements = set()

    def _label(name):
        if not supported_movements or name in supported_movements or name == "No_Movement":
            return name
        return f"{name} [unsupported in current env]"

    print("\n=== Movement Classes Guide ===")
    print(f"1) {_label('No_Movement')}")
    print(f"2) {_label('Wrist_Flexion')}")
    print(f"3) {_label('Wrist_Extension')}")
    print(f"4) {_label('Wrist_Pronation')}")
    print(f"5) {_label('Wrist_Supination')}")
    print(f"6) {_label('Chuck_Grip')}")
    print(f"7) {_label('Hand_Open')}")
    print("8) Exit")


def get_movement_from_user(supported_movements=None):
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
        print_movement_guide(supported_movements)
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
    
    # Enable debug mode to see actuator matching
    os.environ["DEBUG_MAPPING"] = "1"

    env_id = os.getenv("MYOSUITE_ENV", "myoHandReachFixed-v0")
    print(f"\nStarting MyoSuite env: {env_id}")

    env = gym.make(env_id)
    env.reset()

    actuator_names = _get_actuator_names(env)
    profile_name, supported_movements = get_supported_movement_classes(actuator_names)

    if not actuator_names:
        print("Warning: actuator names not found. Movement LUT may be empty.")
    else:
        print(f"\n=== ACTUATOR DEBUG INFO ===")
        print(f"Detected profile: {profile_name}")
        print(f"Total actuators: {len(actuator_names)}")
        print("Available actuators:")
        for i, name in enumerate(actuator_names):
            print(f"  {i:3d}: {name}")
        print(f"Supported movements: {sorted(supported_movements)}")
        print("=" * 50 + "\n")

    step_count = 0
    try:
        while True:
            movement = get_movement_from_user(supported_movements)
            if movement is None:
                print("Exiting controller.")
                return

            if movement not in supported_movements and movement != "No_Movement":
                print(f"Movement '{movement}' is not supported by env '{env_id}' (profile: {profile_name}).")
                print("Choose a different movement or switch to a different MyoSuite environment.")
                continue

            action = get_MyoSuite_Movement_LUT(
                movement_name=movement,
                action_size=env.action_space.shape[0],
                actuator_names=actuator_names,
            )

            if movement != "No_Movement" and not np.any(action):
                print(f"Movement '{movement}' produced no active actuators in env '{env_id}'.")
                print("This environment does not expose a matching control for that movement.")
                continue

            action = np.array(action, dtype=float)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            print(f"Selected movement: {movement}")
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
        env.close()
        print("Environment closed")


def controller_menu():
    run_manual_controller(5) # run for 5 seconds

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
