"""
Config-driven task routing and optional variant registration helpers.

This module keeps environment selection in sync with simulation/config.py.
"""

from __future__ import annotations

from typing import Any, Dict

from config import DEFAULT_ENV_ID, get_movement_env_routing

try:
    from myosuite.envs.env_variants import register_env_variants
except Exception:  # pragma: no cover - handled gracefully when myosuite is unavailable
    register_env_variants = None


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


def register_custom_variants(base_env_id: str, variants: Dict[str, Any]) -> bool:
    """
    Register custom task variants when MyoSuite is installed.

    Args:
        base_env_id: Base MyoSuite environment id to extend.
        variants: Variant spec dictionary expected by MyoSuite.

    Returns:
        True when registration call succeeds, otherwise False.
    """
    if register_env_variants is None:
        return False

    try:
        register_env_variants(base_env_id, variants=variants)
        return True
    except TypeError:
        # Compatibility path for installations that use positional arg style.
        register_env_variants(base_env_id, variants)
        return True


if __name__ == "__main__":
    print_task_summary()
