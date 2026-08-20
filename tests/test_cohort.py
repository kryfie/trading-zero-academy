from pathlib import Path

from academy.cohort import student_id_str, student_seed, student_paths
from academy.policy_audit import decode_action, compare_audits


def test_student_paths_are_isolated_and_padded(tmp_path: Path):
    p1 = student_paths(tmp_path, 1)
    p2 = student_paths(tmp_path, 2)
    assert student_id_str(1) == "001"
    assert p1["latest"] == tmp_path / "students" / "001" / "models" / "latest.zip"
    assert p2["latest"] == tmp_path / "students" / "002" / "models" / "latest.zip"
    assert p1["latest"] != p2["latest"]


def test_student_seeds_are_stable_and_independent():
    assert student_seed(42, 1) == student_seed(42, 1)
    assert student_seed(42, 1) != student_seed(42, 2)


def test_decode_action_matches_flat_long_short_logic():
    assert decode_action([0.0, 1.0], 10.0) == (0.0, 0.0)
    side, lev = decode_action([1.0, 1.0], 10.0)
    assert side == 1.0 and lev == 10.0
    side, lev = decode_action([-1.0, -1.0], 10.0)
    assert side == -1.0 and lev == 1.0


def test_autopsy_comparison_is_descriptive():
    best = {
        "median_return_pct": -5,
        "median_max_drawdown_pct": 10,
        "flat_bar_pct": 80,
        "avg_active_leverage": 2,
        "entries_per_episode": 5,
        "reversals_per_episode": 1,
        "turnover_leverage_units_per_episode": 20,
        "fees_pct_per_episode": 2,
        "slippage_pct_per_episode": 1,
        "funding_paid_pct_per_episode": 0.1,
        "gross_market_pnl_pct_per_episode": 4,
        "avg_holding_bars": 20,
    }
    latest = {
        "median_return_pct": -30,
        "median_max_drawdown_pct": 35,
        "flat_bar_pct": 20,
        "avg_active_leverage": 8,
        "entries_per_episode": 20,
        "reversals_per_episode": 8,
        "turnover_leverage_units_per_episode": 100,
        "fees_pct_per_episode": 10,
        "slippage_pct_per_episode": 6,
        "funding_paid_pct_per_episode": 0.2,
        "gross_market_pnl_pct_per_episode": -2,
        "avg_holding_bars": 5,
    }
    c = compare_audits(best, latest)
    text = " ".join(c["evidence"])
    assert "fees" in text
    assert "flat" in text
    assert "leverage" in text
    assert "turnover" in text
