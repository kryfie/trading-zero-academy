import numpy as np
import pandas as pd

from academy.gen2_env import Generation2TradingEnv, build_target_levels, decode_discrete_action
from academy.mtf_data import resample_market


def make_m1(n=7000, start=1_700_000_000_000):
    ts = start + np.arange(n) * 60_000
    close = np.full(n, 100.0)
    return pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": np.ones(n),
        "funding_rate": np.zeros(n),
        "symbol": "TEST",
    })


def make_env():
    m1 = make_m1()
    frames = {"TEST": {
        tf: resample_market(m1, tf, "TEST")
        for tf in ["1m", "5m", "15m", "1H", "4H"]
    }}
    rules = {
        "max_episode_bars": 50,
        "starting_equity": 100.0,
        "max_leverage": 10,
        "taker_fee_rate": 0.0008,
        "base_slippage_bps": 1.0,
        "volatility_slippage_multiplier": 0.15,
        "min_equity": 1.0,
        "max_position_fraction": 1.0,
    }
    windows = {"1m": 10, "5m": 10, "15m": 5, "1H": 3, "4H": 2}
    levels = build_target_levels(-10, 10, 1)
    return Generation2TradingEnv(
        frames, rules, ["1m", "5m", "15m", "1H", "4H"],
        windows, "5m", levels, seed=1, random_start=False
    ), levels


def test_discrete_levels_are_exactly_minus10_to_plus10():
    levels = build_target_levels(-10, 10, 1)
    assert len(levels) == 21
    assert levels[0] == -10
    assert levels[10] == 0
    assert levels[20] == 10
    assert decode_discrete_action(10, levels) == (0.0, 0.0, 0)
    assert decode_discrete_action(14, levels) == (1.0, 4.0, 4)
    assert decode_discrete_action(6, levels) == (-1.0, 4.0, -4)


def test_repeating_same_target_is_true_hold_without_new_costs():
    env, levels = make_env()
    obs, _ = env.reset()
    assert env.action_space.n == 21

    # +4 target
    _, _, _, _, info1 = env.step(14)
    fees1 = env.total_fees
    slip1 = env.total_slippage
    assert info1["turnover_this_step"] == 4.0
    assert fees1 > 0
    assert slip1 > 0

    # +4 again => no resize, no new fee/slippage
    _, _, _, _, info2 = env.step(14)
    assert info2["turnover_this_step"] == 0.0
    assert env.total_fees == fees1
    assert env.total_slippage == slip1


def test_reversal_charges_full_position_change():
    env, _ = make_env()
    env.reset()
    env.step(14)  # +4
    _, _, _, _, info = env.step(6)  # -4
    assert info["turnover_this_step"] == 8.0
