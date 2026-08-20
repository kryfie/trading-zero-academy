from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from .env import TradingAcademyEnv
from .mtf_env import MultiTimeframeTradingEnv
from .metrics import summarize_episode_infos


def decode_action(action, max_leverage: float) -> tuple[float, float]:
    """Decode a raw policy action exactly like TradingAcademyEnv.step()."""
    action = np.asarray(action, dtype=float)
    desired_side = float(np.clip(action[0], -1, 1))
    if abs(desired_side) < 0.20:
        desired_side = 0.0
    else:
        desired_side = 1.0 if desired_side > 0 else -1.0
    lev_raw = (float(np.clip(action[1], -1, 1)) + 1.0) / 2.0
    desired_lev = 0.0 if desired_side == 0 else max(1.0, lev_raw * float(max_leverage))
    return desired_side, desired_lev


def audit_model(model: PPO, frames: dict, rules: dict, episodes: int = 10, seed: int = 770_000, env_kind: str = 'M5_ONLY', mtf_kwargs: dict | None = None) -> dict:
    """Behavioral autopsy on validation only; never touches FINAL TEST.

    This is observational. It does not alter weights or feed results back into learning.
    """
    episode_results = []
    counts = defaultdict(float)
    holding_lengths: list[int] = []
    leverage_samples: list[float] = []

    for ep in range(int(episodes)):
        if env_kind == "RAW_MTF":
            kw = mtf_kwargs or {}
            env = MultiTimeframeTradingEnv(
                frames, rules,
                timeframes=kw["timeframes"],
                windows=kw["windows"],
                decision_bar=kw["decision_bar"],
                seed=seed + ep,
                random_start=True,
            )
        else:
            env = TradingAcademyEnv(frames, rules, seed=seed + ep, random_start=True)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        max_dd = 0.0
        final_info = {}
        prev_side = 0.0
        prev_lev = 0.0
        holding = 0
        episode_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            side, lev = decode_action(action, float(rules["max_leverage"]))

            episode_steps += 1
            if side > 0:
                counts["long_bars"] += 1
            elif side < 0:
                counts["short_bars"] += 1
            else:
                counts["flat_bars"] += 1

            if side != 0:
                leverage_samples.append(float(lev))

            old_notional = prev_side * prev_lev
            new_notional = side * lev
            counts["turnover_leverage_units"] += abs(new_notional - old_notional)

            if prev_side == 0 and side != 0:
                counts["entries"] += 1
                holding = 1
            elif prev_side != 0 and side == 0:
                counts["exits"] += 1
                if holding > 0:
                    holding_lengths.append(holding)
                holding = 0
            elif prev_side * side < 0:
                counts["reversals"] += 1
                if holding > 0:
                    holding_lengths.append(holding)
                holding = 1
            elif side != 0:
                holding += 1
                if prev_side == side and not math.isclose(prev_lev, lev, rel_tol=0.0, abs_tol=0.05):
                    counts["leverage_rebalances"] += 1

            obs, _, terminated, truncated, info = env.step(action)
            max_dd = max(max_dd, float(info["drawdown_pct"]))
            final_info = info
            done = terminated or truncated
            prev_side, prev_lev = side, lev

        if holding > 0:
            holding_lengths.append(holding)

        counts["steps"] += episode_steps
        episode_results.append({
            "return_pct": float(final_info.get("return_pct", -100.0)),
            "max_drawdown_pct": max_dd,
            "fees": float(final_info.get("fees", 0.0)),
            "funding": float(final_info.get("funding", 0.0)),
            "trade_pnls": list(env.trade_pnls),
        })
        counts["fees_pct"] += 100.0 * float(final_info.get("fees", 0.0)) / float(rules["starting_equity"])
        counts["funding_paid_pct"] += 100.0 * float(final_info.get("funding", 0.0)) / float(rules["starting_equity"])
        counts["slippage_pct"] += 100.0 * float(final_info.get("slippage", 0.0)) / float(rules["starting_equity"])
        counts["gross_market_pnl_pct"] += 100.0 * float(final_info.get("gross_market_pnl", 0.0)) / float(rules["starting_equity"])

    core = summarize_episode_infos(episode_results)
    total_steps = max(1.0, counts["steps"])
    n = max(1, int(episodes))
    core.update({
        "flat_bar_pct": 100.0 * counts["flat_bars"] / total_steps,
        "long_bar_pct": 100.0 * counts["long_bars"] / total_steps,
        "short_bar_pct": 100.0 * counts["short_bars"] / total_steps,
        "avg_active_leverage": float(np.mean(leverage_samples)) if leverage_samples else 0.0,
        "median_active_leverage": float(np.median(leverage_samples)) if leverage_samples else 0.0,
        "entries_per_episode": counts["entries"] / n,
        "exits_per_episode": counts["exits"] / n,
        "reversals_per_episode": counts["reversals"] / n,
        "leverage_rebalances_per_episode": counts["leverage_rebalances"] / n,
        "turnover_leverage_units_per_episode": counts["turnover_leverage_units"] / n,
        "avg_holding_bars": float(np.mean(holding_lengths)) if holding_lengths else 0.0,
        "fees_pct_per_episode": counts["fees_pct"] / n,
        "slippage_pct_per_episode": counts["slippage_pct"] / n,
        "funding_paid_pct_per_episode": counts["funding_paid_pct"] / n,
        "gross_market_pnl_pct_per_episode": counts["gross_market_pnl_pct"] / n,
    })
    return core


def compare_audits(best: dict, latest: dict) -> dict:
    """Deterministic evidence summary; descriptive, not causal."""
    deltas = {}
    keys = [
        "median_return_pct",
        "median_max_drawdown_pct",
        "flat_bar_pct",
        "avg_active_leverage",
        "entries_per_episode",
        "reversals_per_episode",
        "turnover_leverage_units_per_episode",
        "fees_pct_per_episode",
        "slippage_pct_per_episode",
        "funding_paid_pct_per_episode",
        "gross_market_pnl_pct_per_episode",
        "avg_holding_bars",
    ]
    for key in keys:
        deltas[key] = float(best.get(key, 0.0)) - float(latest.get(key, 0.0))

    evidence = []
    if deltas["fees_pct_per_episode"] < -0.5:
        evidence.append("BEST paid materially less fees per episode than LATEST.")
    if deltas["slippage_pct_per_episode"] < -0.5:
        evidence.append("BEST lost materially less to modeled slippage than LATEST.")
    if deltas["flat_bar_pct"] > 10.0:
        evidence.append("BEST stayed flat materially more often than LATEST.")
    if deltas["avg_active_leverage"] < -1.0:
        evidence.append("BEST used materially lower leverage while active than LATEST.")
    if deltas["turnover_leverage_units_per_episode"] < -10.0:
        evidence.append("BEST had materially lower turnover than LATEST.")
    if deltas["gross_market_pnl_pct_per_episode"] > 1.0:
        evidence.append("BEST generated better gross market PnL before explicit costs than LATEST.")
    if not evidence:
        evidence.append("No single large behavioral difference crossed the deterministic evidence thresholds.")

    return {"best_minus_latest": deltas, "evidence": evidence}
