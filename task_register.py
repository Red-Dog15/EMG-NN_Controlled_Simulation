"""
Config-driven task routing and optional variant registration helpers.

This module keeps environment selection in sync with simulation/config.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from config import DEFAULT_ENV_ID, MOVEMENT_VARIANT_KWARGS, get_movement_env_routing


def _get_register_env_variant():
    """Lazily resolve myosuite variant registration API.

    Importing myosuite too early can fail on some Windows/git setups, so we
    resolve this only when registration is actually needed.
    """
    try:
        from myosuite.envs.env_variants import register_env_variant as _register_env_variant

        return _register_env_variant
    except Exception:
        return None


def get_tasks() -> Dict[str, str]:
    """Return movement->env mapping including No_Movement fallback."""
    tasks = {"No_Movement": DEFAULT_ENV_ID}
    tasks.update(get_movement_env_routing())
    return tasks


def print_task_summary() -> None:
    """Print currently configured movement routing."""
    tasks = get_tasks()
    print("\n=== Configured Simulation Tasks ===")
    for movement, env_id in tasks.items():
        print(f"{movement}: {env_id}")


def _load_exported_variant_kwargs() -> Dict[str, Dict[str, Any]]:
    """Load per-movement variant kwargs from Output/joint_tuning/*.json."""
    export_dir = Path(__file__).resolve().parent.parent / "Output" / "joint_tuning"
    merged: Dict[str, Dict[str, Any]] = {}

    if not export_dir.is_dir():
        return merged

    for path in export_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        movement = payload.get("movement")
        if not movement or movement == "No_Movement":
            continue

        # Preferred field from sandbox export.
        variant_kwargs = payload.get("suggested_variant_kwargs")
        if not isinstance(variant_kwargs, dict):
            # Fallback for minimal payloads.
            target_jnt_range = payload.get("target_jnt_range")
            if not isinstance(target_jnt_range, dict):
                continue
            variant_kwargs = {
                "target_jnt_range": target_jnt_range,
                "target_type": "generate",
                "reset_type": "random",
            }

        merged[movement] = variant_kwargs

    return merged


def register_custom_variant(base_env_id: str, variants: Dict[str, Any], variant_id: str | None = None) -> bool:
    """
    Register custom task variants when MyoSuite is installed.

    Args:
        base_env_id: Base MyoSuite environment id to extend.
        variants: Variant spec dictionary expected by MyoSuite.

    Returns:
        True when registration call succeeds, otherwise False.
    """
    register_env_variant = _get_register_env_variant()
    if register_env_variant is None:
        return False

    try:
        register_env_variant(env_id=base_env_id, variants=variants, variant_id=variant_id)
        return True
    except Exception:
        return False


def register_movement_tasks(prefix: str = "myoMove") -> Dict[str, str]:
    """
    Register one env-variant per movement class and return resulting routing.

    Variant ids are generated as: {prefix}<MovementNameNoUnderscores>-v0
    Example: myoMoveWristFlexion-v0
    """
    routing = get_tasks()
    registered = dict(routing)
    exported_variant_kwargs = _load_exported_variant_kwargs()
    register_env_variant = _get_register_env_variant()

    if register_env_variant is None:
        print("Warning: MyoSuite variant registration unavailable; using base env routing.")
        return registered

    for movement, base_env in routing.items():
        if movement == "No_Movement":
            continue

        suffix = movement.replace("_", "")
        variant_id = f"{prefix}{suffix}-v0"
        # Exported tuning JSON overrides static config defaults when present.
        variant_kwargs = dict(MOVEMENT_VARIANT_KWARGS.get(movement, {}))
        if movement in exported_variant_kwargs:
            variant_kwargs = dict(exported_variant_kwargs[movement])

        try:
            register_env_variant(env_id=base_env, variants=variant_kwargs, variant_id=variant_id, silent=True)
            registered[movement] = variant_id
        except Exception as exc:
            # Keep base mapping when registration fails.
            print(f"Warning: failed to register variant for {movement} on {base_env}: {exc}")
            registered[movement] = base_env

    return registered


if __name__ == "__main__":
    print_task_summary()
