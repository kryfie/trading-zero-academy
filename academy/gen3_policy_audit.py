from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .gen3_env import Generation3TradingEnv, decode_discrete_action
from .metrics import summarize_episode_infos
from .recurrent_utils import recurrent_predict


def audit_recurrent_model(
    model,
    frames: dict,
    rules: dict,
    timeframes: list[str],
    windows: dict[str, int],
    decision_bar: str,
    target_levels: tuple[int, ...],
    episodes: int,
    seed: int,
) -> dict:
    episode_results = []
    counts = defaultdict(float)
    holding_lengths: list[int] = []
    leverage_samples: list[float] = []
    for ep in range(int(episodes)):
        env = Generation3TradingEnv(
            frames, rules,
            timeframes=timeframes, windows=windows,
            decision_bar=decision_bar, target_levels=target_levels,
            seed=seed + ep, random_start=True,
        )
        obs, _ = env.reset(seed=seed + ep)
        done = False
        max_dd = 0.0
        final_info = {}
        prev_side = 0.0
        prev_lev = 0.0
        holding = 0
        episode_steps = 0
        lstm_state = None
        first_step = True
        while not done:
            action, lstm_state = recurrent_predict(
                model, obs, lstm_state=lstm_state,
                episode_start=first_step, deterministic=True,
            )
            first_step = False
            side, lev, _ = decode_discrete_action(action, target_levels)
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
            turnover = abs(new_notional - old_notional)
            counts["turnover_leverage_units"] += turnover
            if math.isclose(turnover, 0.0, abs_tol=1e-12):
                counts["hold_actions"] += 1
            else:
                counts["position_change_actions"] += 1
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
                if prev_side == side and not math.isclose(prev_lev, lev, abs_tol=1e-12):
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
        "hold_action_pct": 100.0 * counts["hold_actions"] / total_steps,
        "position_change_action_pct": 100.0 * counts["position_change_actions"] / total_steps,
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
    keys = [
        "median_return_pct", "median_max_drawdown_pct", "flat_bar_pct",
        "hold_action_pct", "position_change_action_pct", "avg_active_leverage",
        "entries_per_episode", "reversals_per_episode",
        "turnover_leverage_units_per_episode", "fees_pct_per_episode",
        "slippage_pct_per_episode", "funding_paid_pct_per_episode",
        "gross_market_pnl_pct_per_episode", "avg_holding_bars",
    ]
    deltas = {k: float(best.get(k, 0.0)) - float(latest.get(k, 0.0)) for k in keys}
    evidence = []
    if deltas["turnover_leverage_units_per_episode"] < -10:
        evidence.append("BEST had materially lower turnover than LATEST.")
    if deltas["fees_pct_per_episode"] < -0.5:
        evidence.append("BEST paid materially less fees per episode than LATEST.")
    if deltas["hold_action_pct"] > 10:
        evidence.append("BEST repeated its existing target position materially more often than LATEST.")
    if deltas["gross_market_pnl_pct_per_episode"] > 1:
        evidence.append("BEST generated better gross market PnL before explicit costs than LATEST.")
    if not evidence:
        evidence.append("No single large behavioral difference crossed the deterministic evidence thresholds.")
    return {"best_minus_latest": deltas, "evidence": evidence}
