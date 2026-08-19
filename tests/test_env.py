import numpy as np
import pandas as pd
from academy.env import TradingAcademyEnv


def fake_frame(n=500):
    ts = np.arange(n) * 300000
    close = 100 * np.cumprod(np.ones(n) * 1.0001)
    return pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": np.ones(n) * 100,
        "funding_rate": np.where(np.arange(n) % 96 == 0, 0.0001, 0.0),
        "symbol": "TEST",
    })


def rules():
    return {
        "starting_equity": 100.0,
        "max_leverage": 10.0,
        "taker_fee_rate": 0.0008,
        "base_slippage_bps": 1.0,
        "volatility_slippage_multiplier": 0.15,
        "liquidation_buffer": 0.003,
        "min_equity": 1.0,
        "max_episode_bars": 100,
        "observation_window": 20,
        "max_position_fraction": 1.0,
    }


def test_env_runs():
    env = TradingAcademyEnv({"TEST": fake_frame()}, rules(), random_start=False)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert info["equity"] > 0


def test_fee_is_charged_on_turnover():
    env = TradingAcademyEnv({"TEST": fake_frame()}, rules(), random_start=False)
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert info["fees"] > 0
