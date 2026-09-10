from __future__ import annotations

from .gen3_env import Generation3TradingEnv
from .metrics import summarize_episode_infos
from .recurrent_utils import recurrent_predict


def evaluate_recurrent_model(
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
    results = []
    for ep in range(int(episodes)):
        env = Generation3TradingEnv(
            frames,
            rules,
            timeframes=timeframes,
            windows=windows,
            decision_bar=decision_bar,
            target_levels=target_levels,
            seed=seed + ep,
            random_start=True,
        )
        obs, _ = env.reset(seed=seed + ep)
        done = False
        max_dd = 0.0
        final_info = {}
        lstm_state = None
        first_step = True
        while not done:
            action, lstm_state = recurrent_predict(
                model, obs, lstm_state=lstm_state,
                episode_start=first_step, deterministic=True,
            )
            first_step = False
            obs, _, terminated, truncated, info = env.step(action)
            max_dd = max(max_dd, float(info["drawdown_pct"]))
            final_info = info
            done = terminated or truncated
        results.append({
            "return_pct": float(final_info.get("return_pct", -100.0)),
            "max_drawdown_pct": max_dd,
            "fees": float(final_info.get("fees", 0.0)),
            "funding": float(final_info.get("funding", 0.0)),
            "trade_pnls": list(env.trade_pnls),
        })
    return summarize_episode_infos(results)


def passes_candidate_gate(metrics: dict, cfg: dict) -> bool:
    e = cfg["evaluation"]
    return (
        metrics["episodes"] >= int(e["candidate_min_episodes"])
        and metrics["median_return_pct"] >= float(e["candidate_min_median_return_pct"])
        and metrics["median_max_drawdown_pct"] <= float(e["candidate_max_median_drawdown_pct"])
        and metrics["profitable_episode_ratio"] >= float(e["candidate_min_profitable_episode_ratio"])
        and metrics["profit_factor"] >= float(e["candidate_min_profit_factor"])
    )
