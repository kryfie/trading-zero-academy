from __future__ import annotations
import numpy as np
from stable_baselines3 import PPO
from .env import TradingAcademyEnv
from .mtf_env import MultiTimeframeTradingEnv
from .metrics import summarize_episode_infos


def _evaluate(env_builder, model: PPO, episodes: int, seed: int) -> dict:
    results = []
    for ep in range(episodes):
        env = env_builder(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        max_dd = 0.0
        final_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
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


def evaluate_model(model: PPO, frames: dict, rules: dict, episodes: int, seed: int = 1000) -> dict:
    return _evaluate(
        lambda s: TradingAcademyEnv(frames, rules, seed=s, random_start=True),
        model, episodes, seed,
    )


def evaluate_mtf_model(
    model: PPO,
    frames: dict,
    rules: dict,
    timeframes: list[str],
    windows: dict[str, int],
    decision_bar: str,
    episodes: int,
    seed: int = 1000,
) -> dict:
    return _evaluate(
        lambda s: MultiTimeframeTradingEnv(
            frames, rules, timeframes=timeframes, windows=windows,
            decision_bar=decision_bar, seed=s, random_start=True
        ),
        model, episodes, seed,
    )


def passes_candidate_gate(metrics: dict, cfg: dict) -> bool:
    e = cfg["evaluation"]
    return (
        metrics["episodes"] >= int(e["candidate_min_episodes"])
        and metrics["median_return_pct"] >= float(e["candidate_min_median_return_pct"])
        and metrics["median_max_drawdown_pct"] <= float(e["candidate_max_median_drawdown_pct"])
        and metrics["profitable_episode_ratio"] >= float(e["candidate_min_profitable_episode_ratio"])
        and metrics["profit_factor"] >= float(e["candidate_min_profit_factor"])
    )
