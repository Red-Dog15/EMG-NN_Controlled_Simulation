"""
Shared passive viewer utilities for MyoSuite simulation tools.

Provides skin toggle, camera presets, and a non-blocking passive viewer
that can be used from the joint tuning sandbox, the manual controller,
and main.py without duplicating code.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


_VIEWER_MODEL_MAP: dict[int, Any] = {}
_VIEWER_SKIN_ALPHA_MAP: dict[int, Any] = {}
_VIEWER_SITEGROUP_MAP: dict[int, Any] = {}


# Available named camera presets for apply_camera_preset().
CAMERA_PRESETS: dict[str, dict] = {
    "default": {"azimuth": 140.0, "elevation": -20.0, "distance": 1.8},
    "front":   {"azimuth": 180.0, "elevation": -10.0, "distance": 1.5},
    "side":    {"azimuth":  90.0, "elevation": -15.0, "distance": 1.5},
    "top":     {"azimuth": 180.0, "elevation": -89.0, "distance": 1.7},
    "close":   {"azimuth": 140.0, "elevation": -20.0, "distance": 0.8},
}


def _get_mj_model_data(env) -> Tuple[Optional[str], Any, Any]:
    """Extract raw MuJoCo model/data from a MyoSuite/gym env wrapper.

    Returns (backend, model, data) where backend is:
      "mujoco"     - native mujoco Python bindings
      "dm_control" - dm_control wrappers used by MyoSuite
      "mujoco_py"  - legacy mujoco_py wrappers
      None         - could not detect a compatible pair
    """
    raw = env.unwrapped
    sim = getattr(raw, "sim", None) or getattr(env, "sim", None)

    candidates = [(getattr(raw, "model", None), getattr(raw, "data", None))]
    if sim is not None:
        candidates.append((getattr(sim, "model", None), getattr(sim, "data", None)))

    # Native mujoco bindings take priority.
    try:
        import mujoco as _mj
        for m, d in candidates:
            if isinstance(m, _mj.MjModel) and isinstance(d, _mj.MjData):
                return "mujoco", m, d
    except ImportError:
        pass

    # dm_control wrappers used by MyoSuite.
    # The wrapper hides native mujoco._structs.MjModel/MjData under ._model / ._data,
    # which are exactly what mujoco.viewer.launch_passive expects.
    try:
        import mujoco as _mj
        for m, d in candidates:
            mod = type(m).__module__ if m is not None else ""
            if m is not None and d is not None and mod.startswith("dm_control."):
                raw_m = getattr(m, "_model", None)
                raw_d = getattr(d, "_data", None)
                if isinstance(raw_m, _mj.MjModel) and isinstance(raw_d, _mj.MjData):
                    return "mujoco", raw_m, raw_d
                # Fall back to returning the wrapper so callers at least know the backend.
                return "dm_control", m, d
    except ImportError:
        pass

    # Legacy mujoco_py duck-type detection.
    for m, d in candidates:
        if m is not None and d is not None and hasattr(m, "njnt") and hasattr(d, "qpos"):
            return "mujoco_py", m, d

    return None, None, None


# ---------------------------------------------------------------------------
# Passive viewer controls
# ---------------------------------------------------------------------------

def set_viewer_skin(viewer, enabled: bool) -> bool:
    """Toggle 'skin' visibility.

    Because MyoSuite hand environments have no embedded skin meshes (nskin=0),
    mjVIS_SKIN has no visual effect.  Instead we map:
      skin ON  -> transparent OFF (solid, normal view)
      skin OFF -> transparent ON  (see-through geoms, reveals inner muscles/tendons)

    This matches the intuitive meaning of "turn skin off to see what's under it".
    """
    if viewer is None:
        return False
    try:
        import mujoco

        # MyoSuite reference approach: deglove by making skin geom group transparent.
        model = _VIEWER_MODEL_MAP.get(id(viewer))
        if model is not None and hasattr(model, "geom_group") and hasattr(model, "geom_rgba"):
            skin_group_mask = (model.geom_group == 1)
            if skin_group_mask.any():
                if enabled:
                    saved_alpha = _VIEWER_SKIN_ALPHA_MAP.get(id(viewer))
                    if saved_alpha is not None and len(saved_alpha) == int(skin_group_mask.sum()):
                        model.geom_rgba[skin_group_mask, 3] = saved_alpha
                    else:
                        model.geom_rgba[skin_group_mask, 3] = 1.0
                else:
                    _VIEWER_SKIN_ALPHA_MAP[id(viewer)] = model.geom_rgba[skin_group_mask, 3].copy()
                    model.geom_rgba[skin_group_mask, 3] = 0.0

        # Keep transparent flag in sync as a visual fallback.
        viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)] = int(not enabled)
        viewer.sync()
        return True
    except Exception:
        return False


def set_viewer_activation(viewer, enabled: bool) -> bool:
    """Toggle muscle activation colour overlay (mjVIS_ACTIVATION)."""
    if viewer is None:
        return False
    try:
        import mujoco
        viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_ACTIVATION)] = int(enabled)
        viewer.sync()
        return True
    except Exception:
        return False


def set_viewer_tendon(viewer, enabled: bool) -> bool:
    """Toggle tendon path visualisation (mjVIS_TENDON)."""
    if viewer is None:
        return False
    try:
        import mujoco
        viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = int(enabled)
        viewer.sync()
        return True
    except Exception:
        return False


def set_viewer_joints(viewer, enabled: bool) -> bool:
    """Toggle joint axis visualisation (mjVIS_JOINT)."""
    if viewer is None:
        return False
    try:
        import mujoco
        viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_JOINT)] = int(enabled)
        viewer.sync()
        return True
    except Exception:
        return False


def set_viewer_markers(viewer, enabled: bool) -> bool:
    """Toggle site markers (small colored balls).

    Uses sitegroup visibility because some MuJoCo builds (including yours)
    do not expose mjVIS_SITE.
    """
    if viewer is None:
        return False
    try:
        key = id(viewer)
        if enabled:
            saved = _VIEWER_SITEGROUP_MAP.get(key)
            if saved is not None and hasattr(viewer.opt, "sitegroup") and len(saved) == len(viewer.opt.sitegroup):
                viewer.opt.sitegroup[:] = saved
            elif hasattr(viewer.opt, "sitegroup"):
                viewer.opt.sitegroup[:] = 1
        else:
            if hasattr(viewer.opt, "sitegroup"):
                _VIEWER_SITEGROUP_MAP[key] = viewer.opt.sitegroup.copy()
                viewer.opt.sitegroup[:] = 0
        viewer.sync()
        return True
    except Exception:
        return False


def apply_camera_preset(viewer, preset: str) -> bool:
    """Apply a named camera preset to a passive MuJoCo viewer.

    Returns True on success, False if the preset name is unknown or the
    viewer backend does not support it.
    """
    if viewer is None:
        return False
    try:
        import mujoco
        config = CAMERA_PRESETS.get(preset.strip().lower())
        if config is None:
            return False
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = config["azimuth"]
        viewer.cam.elevation = config["elevation"]
        viewer.cam.distance = config["distance"]
        if hasattr(viewer.cam, "lookat"):
            viewer.cam.lookat[:] = 0.0
        viewer.sync()
        return True
    except Exception:
        return False


def open_passive_viewer(env, skin: bool = True, camera: str = "default"):
    """Open a non-blocking MuJoCo passive viewer window.

    Returns the viewer handle on success, or None when the native mujoco
    backend is not available (e.g. dm_control wrapper).
    """
    backend, model, data = _get_mj_model_data(env)
    if backend != "mujoco":
        return None
    import mujoco.viewer as mj_viewer
    viewer = mj_viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)
    _VIEWER_MODEL_MAP[id(viewer)] = model
    set_viewer_skin(viewer, skin)
    # Hide small site-marker spheres by default for a cleaner model view.
    set_viewer_markers(viewer, False)
    apply_camera_preset(viewer, camera)
    return viewer


def sync_passive_viewer(viewer) -> bool:
    """Sync viewer state to the latest sim data.

    Returns True while the viewer window is still open, False when it has
    been closed or viewer is None.
    """
    if viewer is None:
        return False
    try:
        if not viewer.is_running():
            return False
        viewer.sync()
        return True
    except Exception:
        return False


def close_passive_viewer(viewer) -> None:
    """Close the passive MuJoCo viewer safely (no-op if already closed)."""
    if viewer is None:
        return
    try:
        _VIEWER_MODEL_MAP.pop(id(viewer), None)
        _VIEWER_SKIN_ALPHA_MAP.pop(id(viewer), None)
        _VIEWER_SITEGROUP_MAP.pop(id(viewer), None)
        viewer.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Viewer settings sub-menu (used in controller and main)
# ---------------------------------------------------------------------------

def print_viewer_menu(viewer) -> None:
    status = "open" if (viewer is not None and sync_passive_viewer(viewer)) else "closed"
    print(f"\n=== Viewer Settings (viewer is {status}) ===")
    print("  open [camera] [skin]   -> open passive viewer  (cameras: " + ", ".join(CAMERA_PRESETS) + ")")
    print("  close                  -> close passive viewer")
    print("  skin <on|off>          -> transparent mode: skin off = see-through geoms to reveal muscles")
    print("  activation <on|off>    -> colour muscles by activation level")
    print("  tendon <on|off>        -> show tendon paths")
    print("  joints <on|off>        -> show joint axes")
    print("  markers <on|off>       -> show/hide small site marker balls")
    print("  camera <preset>        -> change camera angle")
    print("  back                   -> return to movement menu")


def run_viewer_submenu(env, viewer):
    """Interactive viewer settings sub-menu.

    Returns the (possibly new) viewer handle so callers can track it.
    """
    print_viewer_menu(viewer)
    while True:
        raw = input("viewer> ").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in {"back", "b", "q", "exit"}:
            break

        if cmd == "open":
            camera = parts[1] if len(parts) > 1 else "default"
            skin_token = parts[2].lower() if len(parts) > 2 else "on"
            if skin_token not in {"on", "off"}:
                print("Usage: open [camera] [skin]  — skin must be on or off")
                continue
            close_passive_viewer(viewer)
            viewer = open_passive_viewer(env, skin=(skin_token == "on"), camera=camera)
            if viewer is None:
                print("Passive viewer requires native mujoco backend (not available for this env).")
            else:
                print(f"Passive viewer opened  camera={camera}  skin={skin_token}")

        elif cmd == "close":
            close_passive_viewer(viewer)
            viewer = None
            print("Passive viewer closed.")

        elif cmd == "skin":
            if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                print("Usage: skin <on|off>")
                continue
            if viewer is None:
                print("No viewer open. Use 'open' first.")
                continue
            enabled = parts[1].lower() == "on"
            if set_viewer_skin(viewer, enabled):
                label = "solid (skin on)" if enabled else "transparent (skin off — see-through to muscles)"
                print(f"Skin: {label}")
            else:
                print("Skin toggle failed for this viewer.")

        elif cmd == "activation":
            if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                print("Usage: activation <on|off>")
                continue
            if viewer is None:
                print("No viewer open. Use 'open' first.")
                continue
            enabled = parts[1].lower() == "on"
            if set_viewer_activation(viewer, enabled):
                print(f"Muscle activation colour {'enabled' if enabled else 'disabled'}.")
            else:
                print("Activation toggle failed for this viewer.")

        elif cmd == "tendon":
            if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                print("Usage: tendon <on|off>")
                continue
            if viewer is None:
                print("No viewer open. Use 'open' first.")
                continue
            enabled = parts[1].lower() == "on"
            if set_viewer_tendon(viewer, enabled):
                print(f"Tendon paths {'enabled' if enabled else 'disabled'}.")
            else:
                print("Tendon toggle failed for this viewer.")

        elif cmd == "joints":
            if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                print("Usage: joints <on|off>")
                continue
            if viewer is None:
                print("No viewer open. Use 'open' first.")
                continue
            enabled = parts[1].lower() == "on"
            if set_viewer_joints(viewer, enabled):
                print(f"Joint axes {'enabled' if enabled else 'disabled'}.")
            else:
                print("Joints toggle failed for this viewer.")

        elif cmd in {"markers", "marker"}:
            if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                print("Usage: markers <on|off>")
                continue
            if viewer is None:
                print("No viewer open. Use 'open' first.")
                continue
            enabled = parts[1].lower() == "on"
            if set_viewer_markers(viewer, enabled):
                print(f"Site markers {'enabled' if enabled else 'disabled'}.")
            else:
                print("Markers toggle failed for this viewer.")

        elif cmd == "camera":
            if len(parts) != 2:
                print("Usage: camera <" + "|".join(CAMERA_PRESETS) + ">")
                continue
            if viewer is None:
                print("No viewer open. Use 'open' first.")
                continue
            if apply_camera_preset(viewer, parts[1]):
                print(f"Camera set to '{parts[1]}'.")
            else:
                print(f"Unknown preset '{parts[1]}'. Available: {', '.join(CAMERA_PRESETS)}")

        else:
            print_viewer_menu(viewer)

    return viewer
