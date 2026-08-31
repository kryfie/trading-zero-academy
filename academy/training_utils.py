from __future__ import annotations


def should_validate(blocks_completed: int, interval_blocks: int, at_run_end: bool = False) -> bool:
    interval_blocks = max(1, int(interval_blocks))
    return bool(at_run_end or (int(blocks_completed) > 0 and int(blocks_completed) % interval_blocks == 0))


def update_candidate_streak(previous_streak: int, is_candidate: bool) -> int:
    return int(previous_streak) + 1 if is_candidate else 0


def is_better_validation(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    c = (
        float(candidate.get("median_return_pct", float("-inf"))),
        float(candidate.get("profit_factor", float("-inf"))),
        -float(candidate.get("median_max_drawdown_pct", float("inf"))),
    )
    i = (
        float(incumbent.get("median_return_pct", float("-inf"))),
        float(incumbent.get("profit_factor", float("-inf"))),
        -float(incumbent.get("median_max_drawdown_pct", float("inf"))),
    )
    return c > i
