from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

TELEMETRY_FIELDS = [
    "cohort_round_index", "block_index", "block_start_timesteps", "total_timesteps",
    "block_elapsed_seconds", "steps_per_second",
    "learning_rate", "entropy_loss", "policy_entropy_estimate",
    "policy_gradient_loss", "value_loss", "loss", "approx_kl", "clip_fraction",
    "explained_variance", "n_updates", "clip_range",
    "policy_parameter_norm_l2", "last_minibatch_gradient_norm_l2",
    "parameter_delta_l2", "relative_parameter_delta_l2",
    "actor_lstm_parameter_norm_l2", "critic_lstm_parameter_norm_l2",
]


def capture_parameter_state(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.policy.named_parameters()}


def _norm_of_named(model, predicate=lambda name: True, gradients: bool = False) -> float:
    total = 0.0
    for name, p in model.policy.named_parameters():
        if not predicate(name):
            continue
        x = p.grad if gradients else p.detach()
        if x is None:
            continue
        total += float(torch.sum(x.detach().float() ** 2).cpu())
    return math.sqrt(total)


def parameter_delta_norm(model, before: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, p in model.policy.named_parameters():
        if name not in before:
            continue
        delta = p.detach().cpu().float() - before[name].float()
        total += float(torch.sum(delta ** 2))
    return math.sqrt(total)


def _logger_float(model, key: str):
    value = getattr(model.logger, "name_to_value", {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def collect_training_telemetry(
    model,
    before: dict[str, torch.Tensor],
    cohort_round_index: int,
    block_index: int,
    block_start_timesteps: int,
    elapsed_seconds: float,
) -> dict:
    param_norm = _norm_of_named(model)
    delta_norm = parameter_delta_norm(model, before)
    entropy_loss = _logger_float(model, "train/entropy_loss")
    total_steps = int(model.num_timesteps)
    gained = max(0, total_steps - int(block_start_timesteps))
    return {
        "cohort_round_index": int(cohort_round_index),
        "block_index": int(block_index),
        "block_start_timesteps": int(block_start_timesteps),
        "total_timesteps": total_steps,
        "block_elapsed_seconds": float(elapsed_seconds),
        "steps_per_second": float(gained / elapsed_seconds) if elapsed_seconds > 0 else None,
        "learning_rate": _logger_float(model, "train/learning_rate"),
        "entropy_loss": entropy_loss,
        "policy_entropy_estimate": (-entropy_loss if entropy_loss is not None else None),
        "policy_gradient_loss": _logger_float(model, "train/policy_gradient_loss"),
        "value_loss": _logger_float(model, "train/value_loss"),
        "loss": _logger_float(model, "train/loss"),
        "approx_kl": _logger_float(model, "train/approx_kl"),
        "clip_fraction": _logger_float(model, "train/clip_fraction"),
        "explained_variance": _logger_float(model, "train/explained_variance"),
        "n_updates": _logger_float(model, "train/n_updates"),
        "clip_range": _logger_float(model, "train/clip_range"),
        "policy_parameter_norm_l2": param_norm,
        "last_minibatch_gradient_norm_l2": _norm_of_named(model, gradients=True),
        "parameter_delta_l2": delta_norm,
        "relative_parameter_delta_l2": (delta_norm / param_norm if param_norm > 0 else None),
        "actor_lstm_parameter_norm_l2": _norm_of_named(model, lambda n: "lstm_actor" in n),
        "critic_lstm_parameter_norm_l2": _norm_of_named(model, lambda n: "lstm_critic" in n),
    }


def append_telemetry(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TELEMETRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in TELEMETRY_FIELDS})


def read_latest_telemetry(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    last = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last = row
    return last
