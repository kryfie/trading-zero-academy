from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingAcademyEnv(gym.Env):
    """Long/short perpetual-futures simulator.

    Action = [target_position, target_leverage]
      target_position in [-1, 1]: short .. flat .. long
      target_leverage in [-1, 1] mapped to [0, max_leverage]

    The agent is not given SL/TP/RR/indicators. Position changes execute at next-bar open
    plus a deterministic volatility-aware slippage model. Fees are charged on turnover.
    Funding is charged at recorded funding events against signed notional.
    """

    metadata = {"render_modes": []}

    def __init__(self, frames: dict[str, pd.DataFrame], rules: dict, seed: int = 42, random_start: bool = True):
        super().__init__()
        self.frames = frames
        self.symbols = list(frames)
        self.rules = rules
        self.random_start = random_start
        self.window = int(rules["observation_window"])
        self.max_steps = int(rules["max_episode_bars"])
        self.starting_equity = float(rules["starting_equity"])
        self.max_leverage = float(rules["max_leverage"])
        self.fee = float(rules["taker_fee_rate"])
        self.base_slip = float(rules["base_slippage_bps"]) / 10000.0
        self.vol_slip_mult = float(rules["volatility_slippage_multiplier"])
        self.min_equity = float(rules["min_equity"])
        self.max_position_fraction = float(rules.get("max_position_fraction", 1.0))
        self.rng = np.random.default_rng(seed)

        # each row: log returns OHLC vs previous close, log volume change, candle range, funding event
        n_features = 7
        portfolio_features = 5  # position, leverage, unrealized proxy, equity ratio, drawdown
        self.observation_space = spaces.Box(-20, 20, shape=(self.window * n_features + portfolio_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.df: pd.DataFrame | None = None
        self.symbol = ""
        self.i = 0
        self.end_i = 0
        self.equity = self.starting_equity
        self.peak_equity = self.starting_equity
        self.position = 0.0
        self.leverage = 0.0
        self.entry_price = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.trade_pnls: list[float] = []

    def _choose_episode(self):
        self.symbol = str(self.rng.choice(self.symbols))
        self.df = self.frames[self.symbol]
        min_start = self.window + 2
        max_start = max(min_start, len(self.df) - self.max_steps - 2)
        if self.random_start and max_start > min_start:
            self.i = int(self.rng.integers(min_start, max_start + 1))
        else:
            self.i = min_start
        self.end_i = min(len(self.df) - 2, self.i + self.max_steps)

    def _obs(self) -> np.ndarray:
        assert self.df is not None
        w = self.df.iloc[self.i - self.window + 1 : self.i + 1].copy()
        close = w["close"].to_numpy(dtype=float)
        prev = np.r_[close[0], close[:-1]]
        eps = 1e-12
        feats = np.column_stack([
            np.log((w["open"].to_numpy(float)+eps)/(prev+eps)),
            np.log((w["high"].to_numpy(float)+eps)/(prev+eps)),
            np.log((w["low"].to_numpy(float)+eps)/(prev+eps)),
            np.log((close+eps)/(prev+eps)),
            np.r_[0.0, np.diff(np.log1p(w["volume"].to_numpy(float)))],
            (w["high"].to_numpy(float)-w["low"].to_numpy(float))/(close+eps),
            w["funding_rate"].to_numpy(float),
        ])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        dd = 1.0 - self.equity / max(self.peak_equity, 1e-9)
        portfolio = np.array([
            self.position,
            self.leverage / max(self.max_leverage, 1e-9),
            0.0,
            math.log(max(self.equity, 1e-9) / self.starting_equity),
            dd,
        ])
        obs = np.concatenate([feats.ravel(), portfolio])
        return np.clip(obs, -20, 20).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._choose_episode()
        self.equity = self.starting_equity
        self.peak_equity = self.starting_equity
        self.position = 0.0
        self.leverage = 0.0
        self.entry_price = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.trade_pnls = []
        return self._obs(), {"symbol": self.symbol}

    def step(self, action):
        assert self.df is not None
        action = np.asarray(action, dtype=float)
        desired_side = float(np.clip(action[0], -1, 1))
        # dead zone lets the network choose true flat
        if abs(desired_side) < 0.20:
            desired_side = 0.0
        else:
            desired_side = 1.0 if desired_side > 0 else -1.0
        lev_raw = (float(np.clip(action[1], -1, 1)) + 1.0) / 2.0
        desired_lev = 0.0 if desired_side == 0 else max(1.0, lev_raw * self.max_leverage)

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

        # Set desired portfolio at next open, then mark to next close.
        self.position = desired_side
        self.leverage = desired_lev
        if desired_side != 0:
            market_ret = (next_close - next_open) / max(next_open, 1e-12)
            pnl = self.equity * self.position * self.leverage * self.max_position_fraction * market_ret
            self.equity += pnl
            self.trade_pnls.append(float(pnl - fee_cost - slip_cost))

        # Recorded funding event applies to position held across that event.
        funding_rate = float(nxt.get("funding_rate", 0.0))
        if self.position != 0 and funding_rate != 0:
            signed_notional = self.equity * self.position * self.leverage * self.max_position_fraction
            funding_payment = signed_notional * funding_rate
            self.equity -= funding_payment
            self.total_funding += funding_payment

        self.peak_equity = max(self.peak_equity, self.equity)
        # Log-equity growth encourages compounding; a small drawdown penalty discourages ruin
        # without prescribing a human trading strategy.
        log_growth = math.log(max(self.equity, 1e-9) / max(equity_before, 1e-9)) if self.equity > 0 else -10.0
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
        }
        return self._obs(), reward, terminated, truncated, info
