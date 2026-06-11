from __future__ import annotations

from typing import Sequence

_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "hand_count",
    "schema_",
    "chunk_",
    "sig_",
)

_BEHAVIORAL_MARKERS: tuple[str, ...] = (
    "_ratio",
    "_rate",
    "_share",
    "_entropy",
    "_flag",
    "_cv",
    "aggression",
    "passive",
    "decisive",
    "vpip",
    "terminal",
    "hero_",
    "actor_",
    "street_",
    "showdown",
    "first_",
    "last_",
    "preflop_",
    "flop_",
    "turn_",
    "river_",
    "unique_actor",
    "player_count",
    "street_depth",
    "distinct_action",
    "action_count",
    "meaningful_action",
    "coverage",
    "switch",
    "run_max",
)

_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "total_pot",
    "stack_mean",
    "stack_max",
    "stack_min",
    "stack_spread",
    "stack_std",
    "mean_amount_bb",
    "max_amount_bb",
    "q90_amount_bb",
    "std_amount_bb",
    "mean_pot_after",
    "max_pot_after",
    "q90_pot_after",
    "mean_pot_growth",
    "max_pot_growth",
    "q90_pot_growth",
    "std_pot_growth",
    "schema_amount_mean",
    "schema_amount_max",
    "schema_amount_q90",
    "schema_amount_std",
    "schema_pot_before_mean",
    "schema_pot_after_mean",
    "schema_pot_delta_mean",
    "schema_pot_growth",
    "schema_starting_stack_mean",
    "schema_starting_stack_std",
    "schema_starting_stack_iqr",
    "sig_amount_bb_per_hand",
    "sig_pot_after_per_hand",
)


def is_robust_feature_name(name: str) -> bool:
    lowered = str(name).strip().lower()
    if not lowered:
        return False
    if any(token in lowered for token in _EXCLUDE_SUBSTRINGS):
        return False
    if any(token in lowered for token in _INCLUDE_SUBSTRINGS):
        return True
    return any(marker in lowered for marker in _BEHAVIORAL_MARKERS)


def filter_robust_feature_names(names: Sequence[str]) -> list[str]:
    return sorted(name for name in names if is_robust_feature_name(name))


def summarize_robust_filter(
    all_names: Sequence[str],
    kept: Sequence[str],
) -> dict[str, int | list[str]]:
    dropped = [name for name in all_names if name not in set(kept)]
    return {
        "total": len(all_names),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_sample": dropped[:12],
    }
