from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from .okx_data import bar_to_milliseconds


def _features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].to_numpy(dtype=np.float64)
    prev = np.r_[close[0], close[:-1]]
    eps = 1e-12
    vol = df["volume"].to_numpy(dtype=np.float64)
    arr = np.column_stack([
        np.log((df["open"].to_numpy(float) + eps) / (prev + eps)),
        np.log((df["high"].to_numpy(float) + eps) / (prev + eps)),
        np.log((df["low"].to_numpy(float) + eps) / (prev + eps)),
        np.log((close + eps) / (prev + eps)),
        np.r_[0.0, np.diff(np.log1p(vol))],
        (df["high"].to_numpy(float) - df["low"].to_numpy(float)) / (close + eps),
        df.get("funding_rate", pd.Series(0.0, index=df.index)).to_numpy(float),
    ])
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_target_levels(min_target: int, max_target: int, step: int = 1) -> tuple[int, ...]:
    if step <= 0:
        raise ValueError("action step must be > 0")
    if min_target >= 0 or max_target <= 0:
        raise ValueError("discrete target range must include negative, zero and positive positions")
    levels = tuple(range(int(min_target), int(max_target) + 1, int(step)))
    if 0 not in levels:
        raise ValueError("discrete target levels must include FLAT=0")
    return levels


def decode_discrete_action(action, target_levels: tuple[int, ...]) -> tuple[float, float, int]:
    idx = int(np.asarray(action).item())
    if idx < 0 or idx >= len(target_levels):
        raise ValueError(f"Action index {idx} outside 0..{len(target_levels)-1}")
    signed_target = int(target_levels[idx])
    if signed_target == 0:
        return 0.0, 0.0, signed_target
    return (1.0 if signed_target > 0 else -1.0), float(abs(signed_target)), signed_target


class Generation2TradingEnv(gym.Env):
    """Generation 2: same raw MTF market world, discrete target-position actions.

    Observation is deliberately kept equivalent to Gen1 RAW_MTF:
      raw M1/M5/M15/H1/H4 features + portfolio state.

    The only strategic experiment is the action representation. The policy chooses
    one target signed leverage from -10..+10. Repeating the current target is HOLD:
    no resize, no turnover, no fee, no slippage.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frames: dict[str, dict[str, pd.DataFrame]],
        rules: dict,
        timeframes: list[str],
        windows: dict[str, int],
        decision_bar: str,
        target_levels: tuple[int, ...],
        seed: int = 42,
        random_start: bool = True,
    ):
        super().__init__()
        self.frames = frames
        self.symbols = list(frames)
        self.rules = rules
        self.timeframes = list(timeframes)
        self.windows = {tf: int(windows[tf]) for tf in self.timeframes}
        self.decision_bar = decision_bar
        self.target_levels = tuple(int(x) for x in target_levels)
        if decision_bar not in self.timeframes:
            raise ValueError("decision_bar must be present in timeframes")
        self.decision_ms = bar_to_milliseconds(decision_bar)

        self.max_steps = int(rules["max_episode_bars"])
        self.starting_equity = float(rules["starting_equity"])
        self.max_leverage = float(rules["max_leverage"])
        if max(abs(x) for x in self.target_levels) > self.max_leverage:
            raise ValueError("target action exceeds max_leverage")
        self.fee = float(rules["taker_fee_rate"])
        self.base_slip = float(rules["base_slippage_bps"]) / 10000.0
        self.vol_slip_mult = float(rules["volatility_slippage_multiplier"])
        self.min_equity = float(rules["min_equity"])
        self.max_position_fraction = float(rules.get("max_position_fraction", 1.0))
        self.random_start = random_start
        self.rng = np.random.default_rng(seed)

        self.market = {}
        for sym, tfmap in frames.items():
            self.market[sym] = {}
            for tf in self.timeframes:
                df = tfmap[tf].sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
                self.market[sym][tf] = {
                    "df": df,
                    "ts": df["ts"].to_numpy(dtype=np.int64),
                    "feat": _features(df),
                }

        n_features = 7
        portfolio_features = 5
        obs_len = sum(self.windows[tf] * n_features for tf in self.timeframes) + portfolio_features
        self.observation_space = spaces.Box(-20, 20, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.target_levels))

        self.symbol = ""
        self.df = None
        self.i = 0
        self.end_i = 0
        self.equity = self.starting_equity
        self.peak_equity = self.starting_equity
        self.position = 0.0
        self.leverage = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.total_slippage = 0.0
        self.total_market_pnl = 0.0
        self.trade_pnls = []

    def _enough_context(self, sym: str, base_idx: int) -> bool:
        base_ts = int(self.market[sym][self.decision_bar]["ts"][base_idx])
        decision_end = base_ts + self.decision_ms
        for tf in self.timeframes:
            tf_ms = bar_to_milliseconds(tf)
            ts = self.market[sym][tf]["ts"]
            last_closed = np.searchsorted(ts, decision_end - tf_ms, side="right") - 1
            if last_closed + 1 < self.windows[tf]:
                return False
        return True

    def _choose_episode(self):
        self.symbol = str(self.rng.choice(self.symbols))
        base = self.market[self.symbol][self.decision_bar]["df"]
        self.df = base
        min_start = 2
        while min_start < len(base) - 2 and not self._enough_context(self.symbol, min_start):
            min_start += 1
        max_start = max(min_start, len(base) - self.max_steps - 2)
        if self.random_start and max_start > min_start:
            self.i = int(self.rng.integers(min_start, max_start + 1))
        else:
            self.i = min_start
        self.end_i = min(len(base) - 2, self.i + self.max_steps)

    def _obs(self) -> np.ndarray:
        base_ts = int(self.market[self.symbol][self.decision_bar]["ts"][self.i])
        decision_end = base_ts + self.decision_ms
        chunks = []
        for tf in self.timeframes:
            tf_ms = bar_to_milliseconds(tf)
            item = self.market[self.symbol][tf]
            last_closed = np.searchsorted(item["ts"], decision_end - tf_ms, side="right") - 1
            win = self.windows[tf]
            start = last_closed - win + 1
            if start < 0:
                pad = np.zeros((-start, item["feat"].shape[1]), dtype=np.float32)
                body = item["feat"][0:last_closed + 1]
                x = np.vstack([pad, body])
            else:
                x = item["feat"][start:last_closed + 1]
            if len(x) != win:
                padded = np.zeros((win, item["feat"].shape[1]), dtype=np.float32)
                padded[-len(x):] = x[-win:]
                x = padded
            chunks.append(x.ravel())

        dd = 1.0 - self.equity / max(self.peak_equity, 1e-9)
        portfolio = np.array([
            self.position,
            self.leverage / max(self.max_leverage, 1e-9),
            0.0,
            math.log(max(self.equity, 1e-9) / self.starting_equity),
            dd,
        ], dtype=np.float32)
        return np.clip(np.concatenate(chunks + [portfolio]), -20, 20).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._choose_episode()
        self.equity = self.starting_equity
        self.peak_equity = self.starting_equity
        self.position = 0.0
        self.leverage = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.total_slippage = 0.0
        self.total_market_pnl = 0.0
        self.trade_pnls = []
        return self._obs(), {
            "symbol": self.symbol,
            "mode": "GEN2_RAW_MTF_DISCRETE",
            "action_levels": list(self.target_levels),
        }

    def step(self, action):
        desired_side, desired_lev, signed_target = decode_discrete_action(action, self.target_levels)

        row = self.df.iloc[self.i]
        nxt = self.df.iloc[self.i + 1]
        current_close = float(row["close"])
        next_open = float(nxt["open"])
        next_close = float(nxt["close"])
        candle_vol = abs(float(row["high"] - row["low"])) / max(current_close, 1e-12)
        slip = self.base_slip + self.vol_slip_mult * candle_vol

        equity_before = self.equity
        old_notional_frac = self.position * self.leverage
        new_notional_frac = desired_side * desired_lev * self.max_position_fraction
        turnover = abs(new_notional_frac - old_notional_frac)

        fee_cost = self.equity * turnover * self.fee
        slip_cost = self.equity * turnover * slip
        self.equity -= fee_cost + slip_cost
        self.total_fees += fee_cost
        self.total_slippage += slip_cost

        self.position = desired_side
        self.leverage = desired_lev

        if desired_side != 0:
            market_ret = (next_close - next_open) / max(next_open, 1e-12)
            pnl = self.equity * self.position * self.leverage * self.max_position_fraction * market_ret
            self.equity += pnl
            self.total_market_pnl += pnl
            self.trade_pnls.append(float(pnl - fee_cost - slip_cost))

        funding_rate = float(nxt.get("funding_rate", 0.0))
        if self.position != 0 and funding_rate != 0:
            signed_notional = self.equity * self.position * self.leverage * self.max_position_fraction
            funding_payment = signed_notional * funding_rate
            self.equity -= funding_payment
            self.total_funding += funding_payment

        self.peak_equity = max(self.peak_equity, self.equity)
        log_growth = (
            math.log(max(self.equity, 1e-9) / max(equity_before, 1e-9))
            if self.equity > 0 else -10.0
        )
        dd = max(0.0, 1.0 - self.equity / max(self.peak_equity, 1e-9))
        reward = float(np.clip(log_growth - 0.002 * dd, -5, 5))

        self.i += 1
        terminated = bool(self.equity <= self.min_equity)
        truncated = bool(self.i >= self.end_i)
        info = {
            "symbol": self.symbol,
            "equity": self.equity,
            "return_pct": 100 * (self.equity / self.starting_equity - 1),
            "drawdown_pct": 100 * dd,
            "fees": self.total_fees,
            "funding": self.total_funding,
            "slippage": self.total_slippage,
            "gross_market_pnl": self.total_market_pnl,
            "turnover_this_step": float(turnover),
            "target_position": int(signed_target),
            "mode": "GEN2_RAW_MTF_DISCRETE",
        }
        return self._obs(), reward, terminated, truncated, info
