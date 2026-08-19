from __future__ import annotations
import numpy as np


def summarize_episode_infos(episodes: list[dict]) -> dict:
    rets = np.array([x["return_pct"] for x in episodes], dtype=float)
    dds = np.array([x["max_drawdown_pct"] for x in episodes], dtype=float)
    pnls = np.array([p for x in episodes for p in x.get("trade_pnls", [])], dtype=float)
    gross_profit = pnls[pnls > 0].sum() if pnls.size else 0.0
    gross_loss = -pnls[pnls < 0].sum() if pnls.size else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    return {
        "episodes": len(episodes),
        "median_return_pct": float(np.median(rets)) if rets.size else 0.0,
        "mean_return_pct": float(np.mean(rets)) if rets.size else 0.0,
        "median_max_drawdown_pct": float(np.median(dds)) if dds.size else 0.0,
        "profitable_episode_ratio": float(np.mean(rets > 0)) if rets.size else 0.0,
        "profit_factor": pf,
        "worst_return_pct": float(np.min(rets)) if rets.size else 0.0,
        "best_return_pct": float(np.max(rets)) if rets.size else 0.0,
    }
