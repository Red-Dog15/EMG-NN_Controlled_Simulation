"""
Interactive joint tuning sandbox for MyoSuite movement-class calibration.

Run this from the simulation folder with your conda MyoSuite env:
    conda run -n myosuite python joint_tuning_sandbox.py --env myoHandPoseFixed-v0

Use it to:
- Inspect joint names/ranges/current qpos values
- Manually set individual joints and visualize posture changes
- Build target_jnt_range values and export config-ready JSON/snippets
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from config import MOVEMENT_CLASSES, DEFAULT_ENV_ID as _CONFIG_ENV_ID
except Exception:
    MOVEMENT_CLASSES = (
        "No_Movement",
        "Wrist_Flexion",
        "Wrist_Extension",
        "Wrist_Pronation",
        "Wrist_Supination",
        "Chuck_Grip",
        "Hand_Open",
    )
    _CONFIG_ENV_ID = "myoHandReachFixed-v0"

from viewer_utils import (
    CAMERA_PRESETS,
    _get_mj_model_data,
    apply_camera_preset,
    close_passive_viewer,
    open_passive_viewer,
    set_viewer_activation,
    set_viewer_joints,
    set_viewer_skin,
    set_viewer_tendon,
    sync_passive_viewer,
)


@dataclass
class JointInfo:
    index: int
    name: str
    qpos_addr: int
    qpos_value: float
    qrange: Tuple[float, float]


def _configure_git_executable() -> None:
    """Set GitPython environment variables so MyoSuite imports are stable on Windows."""
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


def _get_joint_name(model, idx: int) -> str:
    try:
        return model.joint(idx).name
    except Exception:
        pass

    try:
        return model.joint_id2name(idx)
    except Exception:
        return f"joint_{idx}"


def collect_joint_info(env) -> List[JointInfo]:
    sim = env.unwrapped.sim
    model = sim.model
    data = sim.data
    joints: List[JointInfo] = []

    for j in range(model.njnt):
        name = _get_joint_name(model, j)
        qaddr = int(model.jnt_qposadr[j])
        qval = float(data.qpos[qaddr])
        qrange = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1]))
        joints.append(
            JointInfo(
                index=j,
                name=name,
                qpos_addr=qaddr,
                qpos_value=qval,
                qrange=qrange,
            )
        )

    return joints


def print_joint_table(env, filter_text: str | None = None) -> None:
    joints = collect_joint_info(env)
    print("\n=== Joint Table ===")
    print("idx | qpos_adr | name                 | qpos      | range")
    print("-" * 72)

    for j in joints:
        if filter_text and filter_text.lower() not in j.name.lower():
            continue
        low, high = j.qrange
        print(f"{j.index:3d} | {j.qpos_addr:8d} | {j.name:20.20s} | {j.qpos_value:9.5f} | ({low:7.3f}, {high:7.3f})")


def resolve_joint_by_name_or_index(env, token: str) -> JointInfo:
    joints = collect_joint_info(env)

    if token.isdigit():
        idx = int(token)
        for j in joints:
            if j.index == idx:
                return j
        raise ValueError(f"Joint index '{idx}' not found")

    for j in joints:
        if j.name == token:
            return j

    # fallback substring match if exact name not found
    matches = [j for j in joints if token.lower() in j.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches[:8])
        raise ValueError(f"Ambiguous token '{token}'. Matches: {names}")

    raise ValueError(f"Joint '{token}' not found")


def set_joint_value(env, token: str, value: float) -> None:
    info = resolve_joint_by_name_or_index(env, token)
    qidx = info.qpos_addr
    sim = env.unwrapped.sim
    sim.data.qpos[qidx] = value
    sim.forward()
    print(f"Set {info.name} (idx={info.index}, qpos={qidx}) -> {value:.5f}")


def render_steps(env, steps: int) -> None:
    """Render current joint state without stepping physics.

    Unlike env.step(), this just visualizes the qpos you've set with 'set' or 'neutral'.
    """
    for _ in range(steps):
        env.unwrapped.mj_render()


def simulate_steps(env, steps: int) -> None:
    """Advance physics so GUI actuator controls can move joints.

    This is the mode to use when adjusting muscle/control sliders in the viewer.
    """
    sim = env.unwrapped.sim
    for _ in range(steps):
        # Advance physics directly from current sim controls (e.g., GUI sliders)
        # so env.step(action) does not overwrite them.
        sim.advance(substeps=1, render=True)


def live_simulation(env, seconds: float = 20.0, sleep_s: float = 0.0) -> None:
    """Run realtime-ish stepping so GUI slider edits are easy to apply.

    Use this when tuning with the right-hand control panel. The simulation will
    keep stepping for the requested wall-clock duration.
    """
    sim = env.unwrapped.sim
    end_t = time.time() + max(0.1, float(seconds))
    print(f"Live simulation for {seconds:.1f}s. Move sliders while this runs...")
    while time.time() < end_t:
        sim.advance(substeps=1, render=True)
        if sleep_s > 0:
            time.sleep(sleep_s)


def print_control_summary(env, limit: int = 12) -> None:
    """Print current actuator control values to diagnose slider input flow."""
    sim = env.unwrapped.sim
    ctrl = sim.data.ctrl
    if ctrl is None or len(ctrl) == 0:
        print("No actuator controls found in sim.data.ctrl")
        return

    active = [(i, float(v)) for i, v in enumerate(ctrl) if abs(float(v)) > 1e-6]
    print(f"Control size: {len(ctrl)} | active: {len(active)}")
    if not active:
        print("  All controls are currently ~0")
        return

    # Show largest-magnitude controls first.
    active.sort(key=lambda x: abs(x[1]), reverse=True)
    show = active[: max(1, int(limit))]
    for idx, val in show:
        print(f"  ctrl[{idx:3d}] = {val: .5f}")


def _capture_joint_qpos_map(env) -> Dict[str, float]:
    """Capture current qpos value per joint name."""
    joints = collect_joint_info(env)
    return {j.name: j.qpos_value for j in joints}


def apply_key_qpos(env, qpos_values: List[float]) -> None:
    """Apply a full qpos vector (copied from MuJoCo key pose)."""
    sim = env.unwrapped.sim
    expected = len(sim.data.qpos)
    got = len(qpos_values)
    if got != expected:
        raise ValueError(f"Expected {expected} qpos values, got {got}")
    sim.data.qpos[:] = qpos_values
    sim.forward()
    print(f"Applied key pose with {got} qpos values.")


def auto_mark_from_baseline(
    env,
    baseline_qpos: Dict[str, float],
    marked_ranges: Dict[str, Tuple[float, float]],
    threshold: float = 0.02,
    pad: float = 0.03,
) -> int:
    """Auto-mark only joints that moved enough from baseline.

    Range for each moved joint is [min(base, cur)-pad, max(base, cur)+pad],
    clamped to the model's joint limits.
    """
    count = 0
    for j in collect_joint_info(env):
        if j.name not in baseline_qpos:
            continue
        base = baseline_qpos[j.name]
        cur = j.qpos_value
        if abs(cur - base) < threshold:
            continue

        lo = min(base, cur) - pad
        hi = max(base, cur) + pad
        qlo, qhi = j.qrange
        lo = max(lo, qlo)
        hi = min(hi, qhi)
        marked_ranges[j.name] = (float(lo), float(hi))
        count += 1
    return count


def set_neutral_pose(env) -> None:
    """Zero all joint positions for a flat neutral hand pose."""
    sim = env.unwrapped.sim
    sim.data.qpos[:] = 0.0
    sim.forward()
    print("Neutral pose applied (all qpos = 0).")


def launch_interactive_viewer(env) -> None:
    """Open the full MuJoCo interactive viewer with body dragging.

    Use:
      - Left-click + drag:   Orbit camera
      - Right-click + drag:  Pan camera
      - Scroll wheel:        Zoom
      - Shift + left-click + drag:  Drag the hand body to pose it.
      - Ctrl + left-click:   Pick a joint to show its range slider
      - F1:                  Help/keybindings

    Close the window when done. Joint positions will be printed.
    """
    backend, model, data = _get_mj_model_data(env)
    if backend is None:
        print("Could not access raw MuJoCo model from this env wrapper.")
        print("Use 'set <joint> <value>' + 'render' to adjust joints manually.")
        return

    print("\nLaunching interactive viewer...")
    print(f"  Backend: {backend}")
    print("  Left click-drag = rotate view")
    print("  Right click-drag = pan view")
    print("  Scroll = zoom")
    if backend == "mujoco":
        print("  Shift + left-drag = body perturb/drag")
    elif backend == "dm_control":
        print("  Use 'simulate' for live muscle-control stepping in this backend")
    else:
        print("  Ctrl + left/right drag = perturb selected body (mujoco_py)")
    print("  Close window when done.\n")

    try:
        if backend == "mujoco":
            import mujoco.viewer as mj_viewer
            mj_viewer.launch(model, data)
        elif backend == "dm_control":
            print("dm_control backend does not expose the standalone MuJoCo viewer here.")
            print("Use 'simulate [steps]' to move joints with the muscle control panel.")
            return
        else:
            # Older MyoSuite installs expose mujoco_py simulators.
            import glfw
            from mujoco_py import MjViewer

            viewer = MjViewer(env.sim)
            while not glfw.window_should_close(viewer.window):
                viewer.render()
                time.sleep(0.016)
    except Exception as exc:
        print(f"Interactive viewer error: {exc}")
        return

    # Refresh env state after viewer manipulation and report active joints
    print("\nJoint positions after interactive session:")
    joints = collect_joint_info(env)
    active = [(j.name, j.qpos_value) for j in joints if abs(j.qpos_value) > 0.005]
    if active:
        for name, val in active:
            print(f"  {name}: {val:.5f}")
    else:
        print("  (all near zero)")
    print("\nUse 'mark <joint> <low> <high>' to record ranges, then 'export' to save.")



def ensure_output_dir() -> str:
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Output", "joint_tuning"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def export_movement_ranges(
    movement: str,
    ranges: Dict[str, Tuple[float, float]],
    exact_qpos: Dict[str, float] | None = None,
) -> str:
    out_dir = ensure_output_dir()
    out_path = os.path.join(out_dir, f"{movement}.json")
    payload = {
        "movement": movement,
        "target_jnt_range": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "target_joint_qpos": {
            k: float(exact_qpos[k]) for k in ranges.keys() if exact_qpos is not None and k in exact_qpos
        },
        "suggested_variant_kwargs": {
            "target_jnt_range": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
            "target_type": "generate",
            "reset_type": "random",
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def print_help() -> None:
    print("\nCommands:")
    print("  help")
    print("  tasks                         -> list all movement classes")
    print("  task <movement_class>         -> select which movement you are tuning (no env switch)")
    print("  list [filter]                 -> show joints and ranges")
    print("  set <joint_name_or_idx> <v>   -> set one qpos value")
    print("  neutral                       -> zero all joints (flat neutral pose)")
    print("  render [steps]                -> render current state (no physics, default 200)")
    print("  simulate [seconds]            -> LIVE stepping for slider tuning, default 20")
    print("  live [seconds]                -> alias for simulate")
    print("  step [frames]                 -> fixed-frame physics stepping, default 200")
    print("  ctrl [n]                      -> show top n active actuator controls")
    print("  baseline                      -> store current pose as baseline for auto-mark")
    print("  keyqpos <vals...>             -> apply full qpos list (copy from <key qpos='...'>)")
    print("  auto_mark [th] [pad]          -> mark joints moved vs baseline")
    print("  viewer                        -> open interactive MuJoCo viewer (blocking, drag hand to pose)")
    print("  viewer_open [camera] [skin]   -> keep passive viewer open while using commands")
    print("  viewer_close                  -> close passive viewer")
    print("  camera <preset>               -> change passive viewer camera  (" + ", ".join(CAMERA_PRESETS) + ")")
    print("  skin <on|off>                 -> transparent mode: off = see-through geoms to reveal muscles")
    print("  activation <on|off>           -> colour muscles by activation level")
    print("  tendon <on|off>               -> show tendon paths")
    print("  joints <on|off>               -> show joint axes")
    print("  mark <joint> <low> <high>     -> add target_jnt_range candidate")
    print("  unmark <joint>                -> remove candidate")
    print("  marked                        -> show marked ranges")
    print("  export [movement_class]       -> write JSON for current/specified movement")
    print("  reset                         -> env.reset()")
    print("  quit")


def main() -> None:
    parser = argparse.ArgumentParser(description="MyoSuite joint tuning sandbox")
    parser.add_argument("--env", default=_CONFIG_ENV_ID, help="MyoSuite env id (default from config)")
    parser.add_argument("--movement", default=None, help="Movement class to tune (can change with 'task' command)")
    parser.add_argument("--show-on-start", action="store_true", help="Print full joint list at startup")
    args = parser.parse_args()

    _configure_git_executable()

    from myosuite.utils import gym

    print(f"Opening environment: {args.env}  [single env, all tasks]")
    env = gym.make(args.env)
    env.reset()
    marked_ranges: Dict[str, Tuple[float, float]] = {}
    baseline_qpos = _capture_joint_qpos_map(env)

    # Track which movement task is currently being tuned without switching env.
    current_movement: Optional[str] = args.movement
    if current_movement and current_movement not in MOVEMENT_CLASSES:
        print(f"Warning: --movement '{current_movement}' not in MOVEMENT_CLASSES. Ignored.")
        current_movement = None
    if current_movement:
        print(f"Tuning movement: {current_movement}")
    else:
        print("No movement selected. Use 'task <name>' to set one, or pass --movement.")

    if args.show_on_start:
        print_joint_table(env)

    print_help()
    passive_viewer = None

    try:
        while True:
            # Keep passive viewer alive if it has been opened.
            if passive_viewer is not None and not sync_passive_viewer(passive_viewer):
                passive_viewer = None
                print("Passive viewer closed.")

            prompt = f"\ntune[{current_movement}]> " if current_movement else "\ntune> "
            raw = input(prompt).strip()
            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in {"quit", "exit", "q"}:
                break

            if cmd == "help":
                print_help()
                continue

            if cmd == "list":
                filt = " ".join(parts[1:]) if len(parts) > 1 else None
                print_joint_table(env, filt)
                continue

            if cmd == "set":
                if len(parts) != 3:
                    print("Usage: set <joint_name_or_idx> <value>")
                    continue
                set_joint_value(env, parts[1], float(parts[2]))
                sync_passive_viewer(passive_viewer)
                continue

            if cmd == "render":
                steps = int(parts[1]) if len(parts) > 1 else 200
                render_steps(env, steps)
                sync_passive_viewer(passive_viewer)
                continue

            if cmd in {"simulate", "sim", "live"}:
                seconds = float(parts[1]) if len(parts) > 1 else 20.0
                live_simulation(env, seconds=seconds, sleep_s=0.0)
                sync_passive_viewer(passive_viewer)
                continue

            if cmd == "step":
                steps = int(parts[1]) if len(parts) > 1 else 200
                simulate_steps(env, steps)
                sync_passive_viewer(passive_viewer)
                continue

            if cmd == "ctrl":
                limit = int(parts[1]) if len(parts) > 1 else 12
                print_control_summary(env, limit)
                continue

            if cmd == "tasks":
                print("\nAvailable movement classes:")
                for m in MOVEMENT_CLASSES:
                    marker = " <-- current" if m == current_movement else ""
                    print(f"  {m}{marker}")
                continue

            if cmd == "task":
                if len(parts) != 2:
                    print("Usage: task <movement_class>")
                    continue
                name = parts[1]
                if name not in MOVEMENT_CLASSES:
                    print(f"Unknown movement '{name}'. Use 'tasks' to list options.")
                    continue
                if name == "No_Movement":
                    print("No_Movement does not have a pose to tune. Choose a movement class.")
                    continue
                current_movement = name
                print(f"Now tuning: {current_movement}  (env unchanged: {args.env})")
                continue

            if cmd == "baseline":
                baseline_qpos = _capture_joint_qpos_map(env)
                print(f"Baseline captured ({len(baseline_qpos)} joints).")
                continue

            if cmd == "keyqpos":
                payload = raw[len(parts[0]):].strip()
                payload = payload.replace("<key", " ").replace("/>", " ").replace("qpos=", " ")
                payload = payload.replace("'", " ").replace('"', " ").strip()
                if not payload:
                    print("Usage: keyqpos <v1 v2 ... vN>")
                    continue
                try:
                    vals = [float(tok) for tok in payload.split()]
                    apply_key_qpos(env, vals)
                except Exception as exc:
                    print(f"keyqpos parse/apply failed: {exc}")
                continue

            if cmd == "auto_mark":
                threshold = float(parts[1]) if len(parts) > 1 else 0.02
                pad = float(parts[2]) if len(parts) > 2 else 0.03
                moved = auto_mark_from_baseline(
                    env,
                    baseline_qpos=baseline_qpos,
                    marked_ranges=marked_ranges,
                    threshold=threshold,
                    pad=pad,
                )
                print(f"Auto-marked {moved} joints (threshold={threshold}, pad={pad}).")
                continue

            if cmd == "mark":
                if len(parts) != 4:
                    print("Usage: mark <joint_name_or_idx> <low> <high>")
                    continue
                info = resolve_joint_by_name_or_index(env, parts[1])
                low = float(parts[2])
                high = float(parts[3])
                marked_ranges[info.name] = (low, high)
                print(f"Marked {info.name}: ({low}, {high})")
                continue

            if cmd == "unmark":
                if len(parts) != 2:
                    print("Usage: unmark <joint_name_or_idx>")
                    continue
                info = resolve_joint_by_name_or_index(env, parts[1])
                marked_ranges.pop(info.name, None)
                print(f"Unmarked {info.name}")
                continue

            if cmd == "marked":
                print("\nMarked target_jnt_range:")
                if not marked_ranges:
                    print("  (none)")
                else:
                    for jn, (lo, hi) in marked_ranges.items():
                        print(f"  {jn}: ({lo}, {hi})")
                continue

            if cmd == "export":
                if len(parts) == 2:
                    movement = parts[1]
                elif current_movement:
                    movement = current_movement
                else:
                    print("Usage: export <movement_class>  (or set a current task with 'task <name>')")
                    continue
                if movement not in MOVEMENT_CLASSES:
                    print(f"Unknown movement '{movement}'. Expected one of: {', '.join(MOVEMENT_CLASSES)}")
                    continue
                if movement == "No_Movement":
                    print("No_Movement does not need target_jnt_range export.")
                    continue
                if not marked_ranges:
                    print("No marked ranges. Use 'mark' or 'auto_mark' first.")
                    continue

                out = export_movement_ranges(
                    movement,
                    marked_ranges,
                    exact_qpos=_capture_joint_qpos_map(env),
                )
                print(f"Saved: {out}")
                print("\nPaste into config.py -> MOVEMENT_VARIANT_KWARGS:")
                print(f"\"{movement}\": {{")
                print("    \"target_jnt_range\": {")
                for joint_name, (lo, hi) in marked_ranges.items():
                    print(f"        \"{joint_name}\": ({lo}, {hi}),")
                print("    },")
                print('    "target_type": "generate",')
                print('    "reset_type": "random",')
                print("},")
                continue

            if cmd == "neutral":
                set_neutral_pose(env)
                baseline_qpos = _capture_joint_qpos_map(env)
                sync_passive_viewer(passive_viewer)
                print("Baseline updated to neutral pose.")
                continue

            if cmd == "viewer":
                launch_interactive_viewer(env)
                sync_passive_viewer(passive_viewer)
                continue

            if cmd == "viewer_open":
                camera = parts[1] if len(parts) > 1 else "default"
                skin_token = parts[2].lower() if len(parts) > 2 else "on"
                if skin_token not in {"on", "off"}:
                    print("Usage: viewer_open [camera] [skin]  — skin must be on or off")
                    continue
                close_passive_viewer(passive_viewer)
                passive_viewer = open_passive_viewer(env, skin=(skin_token == "on"), camera=camera)
                if passive_viewer is None:
                    print("Passive viewer requires native mujoco backend (not available for this env).")
                else:
                    print(f"Passive viewer opened  camera={camera}  skin={skin_token}")
                continue

            if cmd == "viewer_close":
                close_passive_viewer(passive_viewer)
                passive_viewer = None
                print("Passive viewer closed.")
                continue

            if cmd == "camera":
                if len(parts) != 2:
                    print("Usage: camera <" + "|".join(CAMERA_PRESETS) + ">")
                    continue
                if passive_viewer is None:
                    print("No viewer open. Use 'viewer_open' first.")
                    continue
                if apply_camera_preset(passive_viewer, parts[1]):
                    print(f"Camera set to '{parts[1]}'.")
                else:
                    print(f"Unknown preset '{parts[1]}'. Available: {', '.join(CAMERA_PRESETS)}")
                continue

            if cmd == "skin":
                if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                    print("Usage: skin <on|off>")
                    continue
                if passive_viewer is None:
                    print("No viewer open. Use 'viewer_open' first.")
                    continue
                enabled = parts[1].lower() == "on"
                if set_viewer_skin(passive_viewer, enabled):
                    label = "solid (skin on)" if enabled else "transparent (skin off — see-through to muscles)"
                    print(f"Skin: {label}")
                else:
                    print("Skin toggle failed for this viewer.")
                continue

            if cmd == "activation":
                if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                    print("Usage: activation <on|off>")
                    continue
                if passive_viewer is None:
                    print("No viewer open. Use 'viewer_open' first.")
                    continue
                enabled = parts[1].lower() == "on"
                if set_viewer_activation(passive_viewer, enabled):
                    print(f"Muscle activation colour {'enabled' if enabled else 'disabled'}.")
                else:
                    print("Activation toggle failed for this viewer.")
                continue

            if cmd == "tendon":
                if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                    print("Usage: tendon <on|off>")
                    continue
                if passive_viewer is None:
                    print("No viewer open. Use 'viewer_open' first.")
                    continue
                enabled = parts[1].lower() == "on"
                if set_viewer_tendon(passive_viewer, enabled):
                    print(f"Tendon paths {'enabled' if enabled else 'disabled'}.")
                else:
                    print("Tendon toggle failed for this viewer.")
                continue

            if cmd == "joints":
                if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                    print("Usage: joints <on|off>")
                    continue
                if passive_viewer is None:
                    print("No viewer open. Use 'viewer_open' first.")
                    continue
                enabled = parts[1].lower() == "on"
                if set_viewer_joints(passive_viewer, enabled):
                    print(f"Joint axes {'enabled' if enabled else 'disabled'}.")
                else:
                    print("Joints toggle failed for this viewer.")
                continue

            if cmd == "reset":
                env.reset()
                baseline_qpos = _capture_joint_qpos_map(env)
                print("Environment reset; baseline updated.")
                continue

            print(f"Unknown command: {cmd}")
            print("Type 'help' for commands.")

    except KeyboardInterrupt:
        print("\nInterrupted")
    except EOFError:
        print("\nInput stream closed")
    finally:
        close_passive_viewer(passive_viewer)
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    main()
