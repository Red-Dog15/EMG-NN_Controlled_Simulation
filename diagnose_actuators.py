"""
Diagnostic script to examine actuator names in MyoSuite environment
and test movement mappings.
"""

import os
import sys
import json
import shutil

SCRIPTS_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Scripts", "DATA")
)
if SCRIPTS_DATA_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DATA_DIR)

from myosuite.utils import gym


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


def scan_environments():
    """Scan multiple MyoSuite environments to find hand/wrist actuators."""
    _configure_git_executable()
    
    # Scan enviornments to see which has all actuators necassary
    envs_to_check = [
        "myoArmReachFixed-v0",
        "myoArmReachRandom-v0",
        "myoHandReachFixed-v0",
        "myoHandReachRandom-v0",
        "myoHandPoseFixed-v0",
        "myoHandPoseRandom-v0",
        "myoHandKeyTurnFixed-v0",
        "myoHandKeyTurnRandom-v0",
        "myoHandObjHoldFixed-v0",
        "myoHandObjHoldRandom-v0",
        "myoHandPenTwirlFixed-v0",
        "myoHandPenTwirlRandom-v0",
    ]
    
    print("Scanning MyoSuite environments for hand/wrist actuators...")
    print("=" * 70)
    
    for env_id in envs_to_check:
        try:
            print(f"\n{env_id}:")
            env = gym.make(env_id)
            env.reset()
            actuator_names = _get_actuator_names(env)
            non_empty = [name for name in actuator_names if name]
            print(f"  Total: {len(actuator_names)}, Non-empty: {len(non_empty)}")
            if non_empty:
                print(f"  Sample actuators: {non_empty[:5]}")
            env.close()
        except Exception as e:
            print(f"  ✗ Not available: {str(e)[:50]}")
    
    print("\n" + "=" * 70)


def main():
    _configure_git_executable()

    env_id = os.getenv("MYOSUITE_ENV", "myoArmReachFixed-v0")
    print(f"Initializing MyoSuite environment: {env_id}\n")

    env = gym.make(env_id)
    env.reset()

    actuator_names = _get_actuator_names(env)

    print(f"Total actuators found: {len(actuator_names)}\n")
    print("Actuator names:")
    print("-" * 50)
    for i, name in enumerate(actuator_names):
        print(f"{i:3d}: {name}")

    # Save to JSON file
    output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Output", "actuators.json")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"actuators": actuator_names}, f, indent=2)
    
    print(f"\nActuator list saved to: {output_path}")

    # Test movement mappings
    print("\n" + "=" * 50)
    print("Testing movement substring matches:")
    print("=" * 50)
    
    # Updated mappings based on environment type
    if "hand" in env_id.lower():
        # MyoHand environment mappings
        substrings_map = {
            "Wrist_Flexion": ["wristflexor"],  # WristFlexor_wrap
            "Wrist_Extension": ["edc"],  # Extensor Digitorum Communis
            "Wrist_Pronation": ["pt"],  # Pronator Teres (if present)
            "Wrist_Supination": ["sup"],  # Supinator (if present)
            "Chuck_Grip": ["proximal_thumb", "mcp5_flexion"],  # Thumb and pinky
            "Hand_Open": ["edc", "2proxph"],  # Extensors and proximal phalanx
        }
    else:
        # MyoArm environment mappings
        substrings_map = {
            "Wrist_Flexion": ["cmc_flexion"],  # CMC flexion for wrist/thumb flexion
            "Wrist_Extension": ["trilong"],  # Triceps tendon may assist extension
            "Wrist_Pronation": ["pecm"],  # Pectoralis may assist pronation
            "Wrist_Supination": ["sup"],  # SUP muscle
            "Chuck_Grip": ["firstmc", "cmc_flexion"],  # Thumb metacarpal and CMC flexion
            "Hand_Open": ["delt"],  # Deltoid may assist in opening/extension
        }
    
    for movement, substrings in substrings_map.items():
        print(f"\n{movement}:")
        print(f"  Looking for: {substrings}")
        matched = []
        for name in actuator_names:
            name_l = name.lower()
            if any(s in name_l for s in [sub.lower() for sub in substrings]):
                matched.append(name)
        if matched:
            print(f"  ✓ Matched: {matched}")
        else:
            print(f"  ✗ NO MATCHES FOUND")

    env.close()
    print("\n" + "=" * 50)
    print("Diagnosis complete!")


if __name__ == "__main__":
    import sys
    if "--scan" in sys.argv:
        scan_environments()
    else:
        main()
