from __future__ import annotations

from typing import Optional

from .gen2_env import Generation2TradingEnv, build_target_levels, decode_discrete_action


class Generation3TradingEnv(Generation2TradingEnv):
    """Generation 3 market environment.

    Dynamics intentionally inherit Generation 2 unchanged. The experiment changes
    the policy from MLP to recurrent PPO/LSTM, not the market world or reward.
    """

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["mode"] = "GEN3_RAW_MTF_DISCRETE_RECURRENT"
        info["experiment_id"] = "GEN3_SEQUENCE_MEMORY_V1"
        return obs, info


__all__ = [
    "Generation3TradingEnv",
    "build_target_levels",
    "decode_discrete_action",
]
