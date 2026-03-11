"""
Test script to verify muscle mappings work correctly for both environments.
"""

import os
import sys
import shutil

SCRIPTS_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Scripts", "DATA")
)
if SCRIPTS_DATA_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DATA_DIR)

from myosuite.utils import gym
from config import MOVEMENT_CLASSES
from Data_Mapping import get_MyoSuite_Movement_LUT


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


def test_environment(env_id):
    """Test muscle mappings for a specific environment."""
    print(f"\n{'='*70}")
    print(f"Testing: {env_id}")
    print(f"{'='*70}")
    
    env = gym.make(env_id)
    env.reset()
    actuator_names = _get_actuator_names(env)
    
    print(f"\nTotal actuators: {len(actuator_names)}")
    non_empty = [name for name in actuator_names if name]
    print(f"Non-empty actuators: {non_empty[:10]}")
    
    # Enable debug mode
    os.environ["DEBUG_MAPPING"] = "1"
    
    movements = list(MOVEMENT_CLASSES)
    
    print(f"\n{'='*70}")
    print("Testing Movement Mappings:")
    print(f"{'='*70}")
    
    for movement in movements:
        activation = get_MyoSuite_Movement_LUT(
            movement_name=movement,
            action_size=len(actuator_names),
            actuator_names=actuator_names
        )
        
        # Count active actuators (non-zero values)
        active_count = sum(1 for val in activation if val != 0.0)
        active_indices = [i for i, val in enumerate(activation) if val != 0.0]
        active_names = [actuator_names[i] for i in active_indices if actuator_names[i]]
        
        if active_count > 0:
            print(f"  ✓ {movement}: {active_count} actuators active -> {active_names}")
        else:
            print(f"  ✗ {movement}: NO ACTUATORS ACTIVE")
    
    env.close()
    print()


def main():
    _configure_git_executable()
    
    # Disable debug for cleaner output
    os.environ["DEBUG_MAPPING"] = "0"
    
    print("\n" + "="*70)
    print("MUSCLE MAPPING TEST SUITE")
    print("="*70)
    
    # Test both environment types
    test_environment("myoArmReachFixed-v0")
    test_environment("myoHandReachFixed-v0")
    
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
