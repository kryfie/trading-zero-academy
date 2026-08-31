from academy.cohort import student_seed
from academy.evaluate import passes_candidate_gate
from academy.training_utils import is_better_validation, update_candidate_streak


def test_gen2_seed_mapping_can_match_gen1_raw_mtf_students():
    # Gen2 #001 uses reference ID 2, matching old Gen1 RAW_MTF Student #002 seed.
    assert student_seed(42, 2) == 2_000_048


def test_candidate_gate_is_unchanged():
    cfg = {"evaluation": {
        "candidate_min_episodes": 20,
        "candidate_min_median_return_pct": 2.0,
        "candidate_max_median_drawdown_pct": 20.0,
        "candidate_min_profitable_episode_ratio": 0.55,
        "candidate_min_profit_factor": 1.05,
    }}
    good = {
        "episodes": 30,
        "median_return_pct": 3.0,
        "median_max_drawdown_pct": 15.0,
        "profitable_episode_ratio": 0.6,
        "profit_factor": 1.1,
    }
    assert passes_candidate_gate(good, cfg)
    assert update_candidate_streak(2, True) == 3
    assert update_candidate_streak(2, False) == 0


def test_best_ranking_unchanged():
    old = {"median_return_pct": -10, "profit_factor": 0.8, "median_max_drawdown_pct": 20}
    new = {"median_return_pct": -8, "profit_factor": 0.7, "median_max_drawdown_pct": 25}
    assert is_better_validation(new, old)
