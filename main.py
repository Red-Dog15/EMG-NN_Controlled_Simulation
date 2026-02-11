import json
import os
import shutil

os.environ["GIT_PYTHON_REFRESH"] = "quiet"  # Suppress git executable warning

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

from myosuite.utils import gym

ENV_ID = os.getenv("MYOSUITE_ENV", "myoArmReachFixed-v0")
PRINT_ACTUATORS = os.getenv("MYOSUITE_DUMP_ACTUATORS", "0") == "1"

print("Initializing myosuite environment...")
env = gym.make(ENV_ID)
env.reset()

if PRINT_ACTUATORS:
    print(f"Environment: {ENV_ID}")
    print(f"Action space: {env.action_space}")
    try:
        sim = getattr(env.unwrapped, "sim", None)
        model = getattr(sim, "model", None) if sim is not None else None
        actuator_names = list(getattr(model, "actuator_names", [])) if model is not None else []
    except Exception:
        actuator_names = []

    if actuator_names:
        print(f"Actuators ({len(actuator_names)}):")
        for i, name in enumerate(actuator_names):
            print(f"  {i}: {name}")
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Output"))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "actuators.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"env_id": ENV_ID, "actuators": actuator_names}, f, indent=2)
        print(f"Saved actuator list to {output_path}")
    else:
        print("No actuator names found on env model.")

print("Running simulation with random actions...")
print("Close the window or press Ctrl+C to stop")

step_count = 0

try:
    while True:  # Run indefinitely until window closed
        env.unwrapped.mj_render()  # Use unwrapped to avoid deprecation warning
        env.step(env.action_space.sample())
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Steps: {step_count}")
            
except KeyboardInterrupt:
    print(f"\nStopped after {step_count} steps")
finally:
    env.close()
    print("Environment closed")