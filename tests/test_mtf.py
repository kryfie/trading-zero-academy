import numpy as np
import pandas as pd

from academy.mtf_data import resample_market
from academy.mtf_env import MultiTimeframeTradingEnv


def make_m1(n=2000, start=1_700_000_000_000):
    ts = start + np.arange(n) * 60_000
    close = 100 + np.arange(n) * 0.001
    return pd.DataFrame({
        "ts": ts,
        "open": close - 0.01,
        "high": close + 0.02,
        "low": close - 0.02,
        "close": close,
        "volume": np.ones(n),
        "funding_rate": np.zeros(n),
        "symbol": "TEST",
    })


def test_resample_shapes_and_funding_sum():
    m1 = make_m1(120)
    m1.loc[10, "funding_rate"] = 0.001
    m5 = resample_market(m1, "5m", "TEST")
    assert len(m5) >= 20
    assert abs(float(m5["funding_rate"].sum()) - 0.001) < 1e-12


def test_mtf_env_fixed_observation_and_step():
    m1 = make_m1(5000)
    frames = {"TEST": {
        "1m": resample_market(m1, "1m", "TEST"),
        "5m": resample_market(m1, "5m", "TEST"),
        "15m": resample_market(m1, "15m", "TEST"),
        "1H": resample_market(m1, "1H", "TEST"),
        "4H": resample_market(m1, "4H", "TEST"),
    }}
    rules = {
        "max_episode_bars": 50,
        "starting_equity": 100.0,
        "max_leverage": 10.0,
        "taker_fee_rate": 0.0008,
        "base_slippage_bps": 1.0,
        "volatility_slippage_multiplier": 0.15,
        "min_equity": 1.0,
        "max_position_fraction": 1.0,
    }
    windows = {"1m": 10, "5m": 10, "15m": 5, "1H": 3, "4H": 2}
    env = MultiTimeframeTradingEnv(
        frames, rules, ["1m", "5m", "15m", "1H", "4H"], windows,
        decision_bar="5m", seed=1, random_start=False
    )
    obs, info = env.reset()
    expected = (10 + 10 + 5 + 3 + 2) * 7 + 5
    assert obs.shape == (expected,)
    obs2, reward, terminated, truncated, info2 = env.step(np.array([0.0, 0.0]))
    assert obs2.shape == obs.shape
    assert np.isfinite(reward)
    assert info["mode"] == "RAW_MTF"
