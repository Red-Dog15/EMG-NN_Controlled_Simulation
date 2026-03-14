"""
NN-driven MyoSuite simulation entry point.

Runs under the myosuite Conda environment.
NN inference is handled by a subprocess (Scripts/NN/inference_worker.py) that
runs under the Scripts/.venv environment where PyTorch lives.
Communication between the two processes is via newline-delimited JSON over
stdin/stdout — no intermediate files required.

Usage:
    python simulation/run_nn.py
    python simulation/run_nn.py --csv <path/to/emg.csv>
    python simulation/run_nn.py --csv <path> --model <path/to/model.pth>
    python simulation/run_nn.py --stride 50 --steps-per-window 50

Default CSV  : Scripts/DATA/Example_data/S1_Hard_C7_R1.csv
Default model: NN_DEFAULT_MODEL_PATH from simulation/config.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SIM_DIR = Path(__file__).resolve().parent          # simulation/
_PROJ_ROOT = _SIM_DIR.parent                        # program/
_SCRIPTS_DIR = _PROJ_ROOT / "Scripts"
_SCRIPTS_DATA_DIR = _SCRIPTS_DIR / "DATA"

# simulation/ must be first so simulation/config.py shadows Scripts/config.py
if str(_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_DIR))
for _p in (_SCRIPTS_DIR, _SCRIPTS_DATA_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)

# ── local imports ─────────────────────────────────────────────────────────────
from config import (
    DEFAULT_ENV_ID,
    NN_DEFAULT_MODEL_PATH,
    NN_INFERENCE_STRIDE,
    NN_INFERENCE_WORKER_PATH,
    NN_WINDOW_SIZE,
    VENV_PYTHON_PATH,
)
from Data_Mapping import results_to_action
from viewer_utils import close_passive_viewer, open_passive_viewer, run_viewer_submenu, sync_passive_viewer

# ── movement / severity catalogue (mirrors Data_Conversion.py) ───────────────
_MOVEMENT_LABELS = {
    0: "No_Movement",
    1: "Wrist_Flexion",
    2: "Wrist_Extension",
    3: "Wrist_Pronation",
    4: "Wrist_Supination",
    5: "Chuck_Grip",
    6: "Hand_Open",
}
_SEVERITY_LABELS = {0: "Light", 1: "Medium", 2: "Hard"}
_EXAMPLE_DATA_DIR = _SCRIPTS_DATA_DIR / "Example_data"
JOINT_TUNING_DIR = _PROJ_ROOT / "Output" / "joint_tuning"


def _csv_for(movement_idx: int, severity: str) -> str:
    """Resolve Example_data filename: S1_{Severity}_C{idx+1}_R1.csv"""
    return str(_EXAMPLE_DATA_DIR / f"S1_{severity}_C{movement_idx + 1}_R1.csv")


def _pick_from_menu(title: str, options: dict) -> str:
    """Print a numbered menu and return the chosen value."""
    print(f"\n=== {title} ===")
    for key, label in options.items():
        print(f"  {key}) {label}")
    while True:
        choice = input("Select: ").strip()
        if choice in options:
            return options[choice]
        print("Invalid selection, try again.")


_MODELS_DIR = _SCRIPTS_DIR / "NN" / "models"

_MODEL_ARCH_MENU: dict[str, tuple[str, str]] = {
    "1": ("NN-A  Full CNN+GRU", "full"),
    "3": ("NN-B  Standard CNN", "standard_cnn"),
    "5": ("NN-C  Lightweight CNN", "lightweight"),
}

_MODEL_VARIANT_MENU: dict[str, str] = {
    "1": "best",
    "2": "final",
}

_MODEL_FILENAME_MAP: dict[tuple[str, str], str] = {
    ("full", "best"): "best_model_full.pth",
    ("full", "final"): "final_model_full.pth",
    ("standard_cnn", "best"): "best_model_standard_cnn.pth",
    ("standard_cnn", "final"): "final_model_standard_cnn.pth",
    ("lightweight", "best"): "best_model_lightweight.pth",
    ("lightweight", "final"): "final_model_lightweight.pth",
}


def _pick_model() -> str:
    """Menu to select which trained model checkpoint to load. Returns absolute path."""
    print("\n=== Select Model Architecture ===")
    for key, (label, model_key) in _MODEL_ARCH_MENU.items():
        best_exists = (_MODELS_DIR / _MODEL_FILENAME_MAP[(model_key, "best")]).is_file()
        final_exists = (_MODELS_DIR / _MODEL_FILENAME_MAP[(model_key, "final")]).is_file()
        exists = "✓" if (best_exists or final_exists) else "missing"
        print(f"  {key}) {label:<28} {exists}")
    while True:
        choice = input("Select architecture (1/3/5): ").strip()
        if choice in _MODEL_ARCH_MENU:
            break
        print("Invalid selection, try again.")

    label, model_key = _MODEL_ARCH_MENU[choice]
    print(f"\n=== Select Checkpoint Variant for {label} ===")
    for key, variant in _MODEL_VARIANT_MENU.items():
        fname = _MODEL_FILENAME_MAP[(model_key, variant)]
        exists = "✓" if (_MODELS_DIR / fname).is_file() else "missing"
        print(f"  {key}) {variant:<5} {exists}")

    while True:
        variant_choice = input("Select variant (1=best, 2=final): ").strip()
        if variant_choice in _MODEL_VARIANT_MENU:
            variant = _MODEL_VARIANT_MENU[variant_choice]
            fname = _MODEL_FILENAME_MAP[(model_key, variant)]
            return str(_MODELS_DIR / fname)
        print("Invalid selection, try again.")


def _prompt_csv() -> str:
    """Interactive menu: returns a CSV path based on mode/movement/severity choice."""
    mode = _pick_from_menu(
        "NN Mode",
        {"1": "Input mode   — specify a CSV path",
         "2": "Control mode — select movement class + severity"},
    )

    if "Input" in mode:
        path = input("CSV path: ").strip()
        return path

    # Control mode
    movement_name = _pick_from_menu(
        "Select Movement Class",
        {str(k): v for k, v in _MOVEMENT_LABELS.items()},
    )
    movement_idx = next(k for k, v in _MOVEMENT_LABELS.items() if v == movement_name)

    severity = _pick_from_menu(
        "Select Severity",
        {str(k): v for k, v in _SEVERITY_LABELS.items()},
    )

    csv_path = _csv_for(movement_idx, severity)
    print(f"[run_nn] Resolved CSV: {csv_path}")
    return csv_path


# ── helpers ───────────────────────────────────────────────────────────────────

def _configure_git_executable():
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    git_exe = shutil.which("git")
    if not git_exe:
        candidates = [
            "C:/Program Files/Git/bin/git.exe",
            "C:/Program Files/Git/cmd/git.exe",
            "C:/Program Files (x86)/Git/bin/git.exe",
        ]
        git_exe = next((p for p in candidates if os.path.isfile(p)), None)
    if git_exe:
        os.environ.setdefault("GIT_PYTHON_GIT_EXECUTABLE", git_exe)


def _get_actuator_names(env) -> list[str]:
    try:
        model = getattr(env.unwrapped, "model", None)
        if model is None:
            sim = getattr(env.unwrapped, "sim", None)
            model = getattr(sim, "model", None) if sim else None
        if model is None:
            return []
        if hasattr(model, "actuator_names"):
            return list(model.actuator_names)
        try:
            import mujoco
            if hasattr(model, "nu"):
                return [
                    model.id2name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
                    for i in range(model.nu)
                ]
        except Exception:
            pass
    except Exception:
        pass
    return []


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
    """Load joint pose targets exported by the tuning workflow for one movement."""
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

    for joint_index in range(model.njnt):
        name = _get_joint_name(model, joint_index)
        if name not in joint_targets:
            continue
        qidx = int(model.jnt_qposadr[joint_index])
        qlow = float(model.jnt_range[joint_index][0])
        qhigh = float(model.jnt_range[joint_index][1])
        qtarget = max(qlow, min(qhigh, float(joint_targets[name])))
        resolved[qidx] = qtarget

    return resolved


def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _apply_joint_targets_interp(env, start_qpos, target_qpos, phase: float):
    """Interpolate joints from start pose to target pose using a smooth ramp."""
    sim = env.unwrapped.sim
    w = _smoothstep01(phase)
    for qidx, target in target_qpos.items():
        start_value = float(start_qpos.get(qidx, sim.data.qpos[qidx]))
        sim.data.qpos[qidx] = (1.0 - w) * start_value + w * float(target)
    if hasattr(sim.data, "qvel"):
        sim.data.qvel[:] = 0.0
    sim.forward()


def _load_emg_windows(csv_path: str, window_size: int, stride: int) -> list[list]:
    """Load EMG CSV and return a list of (window_size × 8) Python lists for JSON serialisation."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        # Skip header row if present (non-numeric first cell)
        rows = list(reader)
    try:
        float(rows[0][0])
        data_rows = rows
    except (ValueError, IndexError):
        data_rows = rows[1:]

    emg = np.array([[float(v) for v in row] for row in data_rows if row], dtype=np.float32)
    n = emg.shape[0]
    windows = [
        emg[start : start + window_size].tolist()
        for start in range(0, n - window_size + 1, stride)
    ]
    if not windows:
        pad = np.zeros((window_size - n, emg.shape[1]), dtype=np.float32)
        windows = [np.concatenate([pad, emg], axis=0).tolist()]
    return windows


# ── inference worker subprocess ───────────────────────────────────────────────

def _start_worker(model_path: str, window_size: int) -> subprocess.Popen:
    """Spawn inference_worker.py under .venv Python and wait for the ready signal."""
    if not os.path.isfile(VENV_PYTHON_PATH):
        raise FileNotFoundError(
            f"[run_nn] .venv Python not found: {VENV_PYTHON_PATH}\n"
            "Set the NN_VENV_PYTHON environment variable to the correct path."
        )
    if not os.path.isfile(NN_INFERENCE_WORKER_PATH):
        raise FileNotFoundError(
            f"[run_nn] inference_worker.py not found: {NN_INFERENCE_WORKER_PATH}"
        )

    print(f"[run_nn] Starting inference worker ({VENV_PYTHON_PATH})...")
    proc = subprocess.Popen(
        [
            VENV_PYTHON_PATH,
            NN_INFERENCE_WORKER_PATH,
            "--model", model_path,
            "--window-size", str(window_size),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,   # worker stderr flows directly to this terminal
        text=True,
        bufsize=1,     # line-buffered
    )

    # Wait for {"status": "ready"} before returning
    ready_line = proc.stdout.readline()
    try:
        msg = json.loads(ready_line.strip())
        if msg.get("status") != "ready":
            raise RuntimeError(f"Unexpected first message from worker: {ready_line!r}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Worker did not send ready signal. Got: {ready_line!r}")

    print("[run_nn] Inference worker ready.")
    return proc


def _predict(proc: subprocess.Popen, window: list) -> dict:
    """Send one window to the worker and return the parsed prediction dict."""
    proc.stdin.write(json.dumps({"window": window}) + "\n")
    proc.stdin.flush()
    raw = proc.stdout.readline()
    return json.loads(raw.strip())


def _stop_worker(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.write(json.dumps({"command": "exit"}) + "\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


# ── main NN loop ──────────────────────────────────────────────────────────────

def run_nn_mode(
    csv_path: str | None = None,
    model_path: str | None = None,
    stride: int | None = None,
    steps_per_window: int | None = None,
    print_every: int = 50,
):
    """
    Load model (via subprocess), replay EMG CSV through it, drive MyoSuite.
    After each CSV finishes, prompts to run the same movement again, switch to
    a different movement, or quit — without restarting the environment or worker.
    All parameters fall back to config.py defaults when not supplied.
    """
    # ── resolve model ─────────────────────────────────────────────────────────
    model_path = model_path or _pick_model()
    if not os.path.isfile(model_path):
        print(f"[run_nn] Model not found: {model_path}")
        return

    stride = stride if stride is not None else NN_INFERENCE_STRIDE
    steps_per_window = steps_per_window if steps_per_window is not None else NN_INFERENCE_STRIDE
    window_size = NN_WINDOW_SIZE
    print_every = max(1, int(print_every))

    _configure_git_executable()

    # ── start inference worker in .venv (started once, reused across movements)
    worker = _start_worker(model_path, window_size)

    # ── build MyoSuite environment (started once) ─────────────────────────────
    from myosuite.utils import gym  # deferred — heavy import

    print(f"[run_nn] Initialising environment: {DEFAULT_ENV_ID}")
    env = gym.make(DEFAULT_ENV_ID)
    env.reset()
    actuator_names = _get_actuator_names(env)
    action_size = env.action_space.shape[0]

    if not actuator_names:
        print("[run_nn] Warning: could not read actuator names — LUT matching disabled.")

    passive_viewer = open_passive_viewer(env)
    if passive_viewer is None:
        print("[run_nn] Note: passive viewer unavailable, using mj_render fallback.")
    else:
        print("[run_nn] Passive viewer opened. Close it or press Ctrl+C to stop.")

    prev_action = np.zeros(action_size, dtype=float)
    step_count = 0
    current_csv = csv_path  # None → will prompt on first iteration

    try:
        while True:
            # ── pick movement for this pass ───────────────────────────────────
            if current_csv is None:
                current_csv = _prompt_csv()

            if not os.path.isfile(current_csv):
                print(f"[run_nn] CSV not found: {current_csv}")
                current_csv = None
                continue

            windows = _load_emg_windows(current_csv, window_size, stride)
            print(
                f"\n[run_nn] Running {len(windows)} windows from: {os.path.basename(current_csv)}"
                f"  (printing every {print_every})\n"
            )

            # ── inference + sim loop ──────────────────────────────────────────
            viewer_closed = False
            for window_idx, window in enumerate(windows):
                if passive_viewer is not None and not sync_passive_viewer(passive_viewer):
                    print("[run_nn] Viewer closed — stopping.")
                    viewer_closed = True
                    break

                prediction = _predict(worker, window)
                movement_name = prediction["movement_name"]
                confidence = prediction["movement_confidence"]

                # Throttled print: first window, every Nth, and last window
                is_milestone = (
                    window_idx == 0
                    or (window_idx + 1) % print_every == 0
                    or window_idx + 1 == len(windows)
                )
                if is_milestone:
                    print(
                        f"  window {window_idx + 1:>4}/{len(windows)} | "
                        f"{movement_name:<20} | conf={confidence:.2f}"
                    )

                exported_joint_targets = None
                resolved_qpos_targets = None
                if movement_name != "No_Movement":
                    exported_joint_targets = _load_exported_joint_targets(movement_name)
                    if exported_joint_targets:
                        resolved_qpos_targets = _resolve_joint_qpos_targets(env, exported_joint_targets)

                action = results_to_action(
                    {"movement_name": movement_name},
                    actuator_names,
                    action_size=action_size,
                )
                action = np.clip(
                    np.array(action, dtype=float),
                    env.action_space.low,
                    env.action_space.high,
                )

                transition_steps = max(1, steps_per_window // 4)
                for sub_step in range(steps_per_window):
                    if passive_viewer is None:
                        env.unwrapped.mj_render()
                    if resolved_qpos_targets:
                        if sub_step == 0:
                            sim = env.unwrapped.sim
                            start_qpos_targets = {
                                qidx: float(sim.data.qpos[qidx])
                                for qidx in resolved_qpos_targets.keys()
                            }
                        phase = (sub_step + 1) / max(1, steps_per_window)
                        _apply_joint_targets_interp(env, start_qpos_targets, resolved_qpos_targets, phase=phase)
                        env.unwrapped.sim.advance(substeps=1, render=False)
                    else:
                        if sub_step < transition_steps:
                            w = (sub_step + 1) / transition_steps
                            blended = (1.0 - w) * prev_action + w * action
                        else:
                            blended = action
                        env.step(blended)
                    sync_passive_viewer(passive_viewer)
                    step_count += 1

                prev_action = action

            if viewer_closed:
                break

            # ── end-of-CSV prompt ─────────────────────────────────────────────
            print(f"\n[run_nn] Finished. Total sim steps so far: {step_count}")
            print("  s) Same movement again")
            print("  n) New movement / severity")
            print("  v) Visual settings")
            print("  q) Quit")
            while True:
                choice = input("Choice [s/n/v/q]: ").strip().lower()
                if choice == "v":
                    passive_viewer = run_viewer_submenu(env, passive_viewer)
                    print("\n[run_nn] Viewer settings updated.")
                    print("  s) Same movement again")
                    print("  n) New movement / severity")
                    print("  v) Visual settings")
                    print("  q) Quit")
                    continue
                if choice in ("s", "n", "q"):
                    break
                print("  Enter s, n, v, or q.")

            if choice == "q":
                break
            elif choice == "n":
                current_csv = None   # will re-prompt on next iteration
            # "s" → current_csv stays set, loop runs same file again

    except KeyboardInterrupt:
        print(f"\n[run_nn] Interrupted after {step_count} steps.")
    finally:
        _stop_worker(worker)
        close_passive_viewer(passive_viewer)
        env.close()
        print(f"[run_nn] Environment closed. Total steps: {step_count}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NN-driven MyoSuite simulation (EMG CSV replay, two-process mode)"
    )
    parser.add_argument("--csv", default=None, help="Path to EMG CSV file")
    parser.add_argument("--model", default=None, help="Path to trained model .pth checkpoint")
    parser.add_argument("--stride", type=int, default=None, help="EMG window stride in samples")
    parser.add_argument(
        "--steps-per-window", type=int, default=None, dest="steps_per_window",
        help="Sim steps to hold each action",
    )
    parser.add_argument(
        "--print-every", type=int, default=50, dest="print_every",
        help="Print a status line every N windows (default: 50)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_nn_mode(
        csv_path=args.csv,
        model_path=args.model,
        stride=args.stride,
        steps_per_window=args.steps_per_window,
        print_every=args.print_every,
    )

