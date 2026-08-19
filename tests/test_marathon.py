from academy.marathon import (
    is_better_validation,
    remaining_timesteps,
    should_validate,
    update_candidate_streak,
)


def test_checkpoint_validation_cadence():
    assert not should_validate(1, 4)
    assert not should_validate(3, 4)
    assert should_validate(4, 4)
    assert should_validate(8, 4)
    assert should_validate(9, 4, at_run_end=True)


def test_candidate_streak_resets():
    assert update_candidate_streak(0, True) == 1
    assert update_candidate_streak(2, True) == 3
    assert update_candidate_streak(3, False) == 0


def test_remaining_timesteps_never_negative():
    assert remaining_timesteps(3_000_000, 123_000_000) == 120_000_000
    assert remaining_timesteps(124_000_000, 123_000_000) == 0


def test_best_validation_ranking():
    old = {"median_return_pct": -10, "profit_factor": 0.9, "median_max_drawdown_pct": 30}
    new = {"median_return_pct": -5, "profit_factor": 0.8, "median_max_drawdown_pct": 40}
    assert is_better_validation(new, old)
    assert not is_better_validation(old, new)
