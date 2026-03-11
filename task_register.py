"""
Config-driven task routing and optional variant registration helpers.

This module keeps environment selection in sync with simulation/config.py.
"""

from __future__ import annotations

from typing import Any, Dict

from config import DEFAULT_ENV_ID, MOVEMENT_VARIANT_KWARGS, get_movement_env_routing

try:
    from myosuite.envs.env_variants import register_env_variant
except Exception:  # pragma: no cover - handled gracefully when myosuite is unavailable
    register_env_variant = None


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


def register_custom_variant(base_env_id: str, variants: Dict[str, Any], variant_id: str | None = None) -> bool:
    """
    Register custom task variants when MyoSuite is installed.

    Args:
        base_env_id: Base MyoSuite environment id to extend.
        variants: Variant spec dictionary expected by MyoSuite.

    Returns:
        True when registration call succeeds, otherwise False.
    """
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

    if register_env_variant is None:
        return registered

    for movement, base_env in routing.items():
        if movement == "No_Movement":
            continue

        suffix = movement.replace("_", "")
        variant_id = f"{prefix}{suffix}-v0"
        variant_kwargs = dict(MOVEMENT_VARIANT_KWARGS.get(movement, {}))

        try:
            register_env_variant(env_id=base_env, variants=variant_kwargs, variant_id=variant_id, silent=True)
            registered[movement] = variant_id
        except Exception:
            # Keep base mapping when registration fails.
            registered[movement] = base_env

    return registered


if __name__ == "__main__":
    print_task_summary()
