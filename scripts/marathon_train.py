from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from academy.config import load_config, ROOT
from academy.dataset import (
    load_frames,
    ensure_split_manifest,
    build_academy_partitions,
    partition_summary,
    SPLIT_MANIFEST_NAME,
)
from academy.env import TradingAcademyEnv
from academy.evaluate import evaluate_model, passes_candidate_gate
from academy.marathon import (
    is_better_validation,
    remaining_timesteps,
    should_validate,
    update_candidate_streak,
)

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
learner = cfg["learner"]
runtime = cfg["runtime"]
base_seed = int(cfg["project"]["seed"])

data_dir = ROOT / "data" / "processed"
manifest_path = data_dir / SPLIT_MANIFEST_NAME
frames = load_frames(data_dir, m["symbols"], m["bar"])
manifest = ensure_split_manifest(
    frames,
    float(e["train_fraction"]),
    float(e["validation_fraction"]),
    manifest_path,
)
train_frames, val_frames, frozen_final_frames, live_shadow_frames, validation_source = build_academy_partitions(
    frames,
    manifest,
    live_shadow_days=int(e["live_shadow_days"]),
    train_maturation_days=int(e["train_maturation_days"]),
    min_rolling_validation_bars=int(e["min_rolling_validation_bars"]),
)

checkpoint = ROOT / runtime["checkpoint_path"]
checkpoint.parent.mkdir(parents=True, exist_ok=True)
state_path = ROOT / runtime["training_state_path"]
state_path.parent.mkdir(parents=True, exist_ok=True)
status_path = ROOT / runtime["status_path"]
status_path.parent.mkdir(parents=True, exist_ok=True)
progress_path = ROOT / runtime["progress_path"]
progress_path.parent.mkdir(parents=True, exist_ok=True)
history_path = ROOT / runtime.get("learning_history_path", "reports/learning_history.csv")
history_path.parent.mkdir(parents=True, exist_ok=True)
best_checkpoint = ROOT / runtime.get("best_checkpoint_path", "models/best_validation.zip")
best_metrics_path = ROOT / runtime.get("best_metrics_path", "models/best_validation.json")

if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
else:
    state = {}

marathon_run_index = int(state.get("marathon_run_index", 0)) + 1
run_seed = base_seed + marathon_run_index * 100_003


def env_fn():
    return TradingAcademyEnv(train_frames, rules, seed=run_seed, random_start=True)


venv = make_vec_env(env_fn, n_envs=4, seed=run_seed)
if checkpoint.exists():
    model = PPO.load(checkpoint, env=venv)
    print(f"Continuing Student #1 from {checkpoint}; existing timesteps={model.num_timesteps:,}")
else:
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=float(learner["learning_rate"]),
        n_steps=int(learner["n_steps"]),
        batch_size=int(learner["batch_size"]),
        gamma=float(learner["gamma"]),
        gae_lambda=float(learner["gae_lambda"]),
        ent_coef=float(learner["ent_coef"]),
        clip_range=float(learner["clip_range"]),
        policy_kwargs={"net_arch": list(learner["policy_layers"])},
        verbose=1,
        seed=run_seed,
    )

block_steps = int(learner.get("timesteps_per_block", 500_000))
max_steps_this_run = int(learner.get("marathon_steps_per_run", 10_000_000))
validation_interval_blocks = int(learner.get("marathon_validation_interval_blocks", 4))
master_streak_required = int(learner.get("marathon_master_candidate_streak", 3))
stop_on_master_streak = bool(learner.get("marathon_stop_on_master_streak", True))
config_target = int(learner.get("marathon_target_total_timesteps", 123_000_000))
target_total = int(os.environ.get("ACADEMY_TARGET_TIMESTEPS") or config_target)

start_total = int(model.num_timesteps)
run_target = min(target_total, start_total + max_steps_this_run)
candidate_streak = int(state.get("candidate_streak", 0))
validation_index = int(state.get("validation_index", 0))
blocks_completed = 0
run_started = time.time()
last_metrics = state.get("last_validation")
last_candidate = bool(state.get("last_candidate", False))
graduation_ready = candidate_streak >= master_streak_required

HISTORY_FIELDS = [
    "timestamp_utc",
    "marathon_run_index",
    "validation_index",
    "total_timesteps",
    "status",
    "candidate_streak",
    "validation_source",
    "episodes",
    "median_return_pct",
    "mean_return_pct",
    "median_max_drawdown_pct",
    "profitable_episode_ratio",
    "profit_factor",
    "worst_return_pct",
    "best_return_pct",
]


def append_history(metrics: dict, candidate: bool) -> None:
    new_file = not history_path.exists() or history_path.stat().st_size == 0
    row = {
        "timestamp_utc": int(time.time()),
        "marathon_run_index": marathon_run_index,
        "validation_index": validation_index,
        "total_timesteps": int(model.num_timesteps),
        "status": "MASTER_CANDIDATE" if candidate else "LEARNING",
        "candidate_streak": candidate_streak,
        "validation_source": validation_source,
        **{k: metrics.get(k) for k in HISTORY_FIELDS if k in metrics},
    }
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in HISTORY_FIELDS})


def maybe_archive_best(metrics: dict) -> None:
    incumbent = None
    if best_metrics_path.exists():
        try:
            incumbent = json.loads(best_metrics_path.read_text(encoding="utf-8")).get("validation")
        except Exception:
            incumbent = None
    if is_better_validation(metrics, incumbent):
        model.save(best_checkpoint)
        payload = {
            "timestamp_utc": int(time.time()),
            "total_timesteps": int(model.num_timesteps),
            "validation_source": validation_source,
            "validation": metrics,
            "note": "Archive only. This selection does not feed back into learner reward/weights.",
        }
        best_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"New best validation archive at {model.num_timesteps:,} steps")


def live_shadow_metrics():
    min_live_rows = int(rules["observation_window"]) + 20
    if not all(len(df) >= min_live_rows for df in live_shadow_frames.values()):
        return None
    try:
        return evaluate_model(
            model,
            live_shadow_frames,
            rules,
            int(e.get("live_shadow_episodes", 10)),
            seed=700000,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def write_state_and_status(metrics: dict | None, candidate: bool, block_number: int) -> None:
    reached_target = int(model.num_timesteps) >= target_total
    state_payload = {
        "run_index": int(state.get("run_index", 0)),
        "marathon_run_index": marathon_run_index,
        "run_seed": run_seed,
        "validation_index": validation_index,
        "candidate_streak": candidate_streak,
        "last_candidate": candidate,
        "last_validation": metrics,
        "completed_blocks_this_marathon_run": block_number,
        "timesteps_per_block_target": block_steps,
        "total_timesteps": int(model.num_timesteps),
        "marathon_target_total_timesteps": target_total,
        "updated_at_utc": int(time.time()),
    }
    state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

    progress = {
        **state_payload,
        "start_timesteps_this_run": start_total,
        "run_target_timesteps": run_target,
        "remaining_to_marathon_target": remaining_timesteps(model.num_timesteps, target_total),
        "elapsed_seconds_this_run": round(time.time() - run_started, 2),
    }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    if metrics is None:
        return

    status_payload = {
        "timestamp_utc": int(time.time()),
        "status": "MASTER_CANDIDATE" if candidate else "LEARNING",
        "student": 1,
        "marathon_run_index": marathon_run_index,
        "run_seed": run_seed,
        "validation_index": validation_index,
        "candidate_streak": candidate_streak,
        "master_candidate_streak_required": master_streak_required,
        "graduation_ready": candidate_streak >= master_streak_required,
        "total_timesteps": int(model.num_timesteps),
        "validation_source": validation_source,
        "validation": metrics,
        "live_shadow": live_shadow_metrics(),
        "partitions": {
            "train": partition_summary(train_frames),
            "validation": partition_summary(val_frames),
            "live_shadow": partition_summary(live_shadow_frames),
            "frozen_final": partition_summary(frozen_final_frames),
        },
        "marathon": {
            "target_total_timesteps": target_total,
            "start_timesteps_this_run": start_total,
            "run_target_timesteps": run_target,
            "max_steps_this_run": max_steps_this_run,
            "checkpoint_every_requested_steps": block_steps,
            "validation_every_requested_steps": block_steps * validation_interval_blocks,
            "reached_target": reached_target,
            "remaining_timesteps": remaining_timesteps(model.num_timesteps, target_total),
            "stop_on_master_candidate_streak": stop_on_master_streak,
            "stopped_for_graduation": bool(stop_on_master_streak and candidate_streak >= master_streak_required),
        },
        "data_policy": {
            "live_shadow_days": int(e["live_shadow_days"]),
            "train_maturation_days": int(e["train_maturation_days"]),
            "rule": "Post-launch data: 0-7d LIVE SHADOW, 7-30d rolling validation, >=30d TRAIN. Frozen final never trains.",
        },
        "best_validation_archive": str(best_checkpoint.relative_to(ROOT)) if best_checkpoint.exists() else None,
        "learning_history": str(history_path.relative_to(ROOT)),
        "final_test_locked": True,
        "note": "Validation is observation/model-selection only. FINAL TEST remains locked and never influences learning.",
    }
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")


print("=== TRADING ZERO ACADEMY — MARATHON MODE ===")
print(f"Current: {start_total:,} steps")
print(f"This run may train until: {run_target:,}")
print(f"Marathon target: {target_total:,}")
print(f"Checkpoint every requested: {block_steps:,}")
print(f"Validation every requested: {block_steps * validation_interval_blocks:,}")

# If the model is already at/above target, still write a fresh validation/status once.
if start_total >= target_total:
    validation_index += 1
    last_metrics = evaluate_model(model, val_frames, rules, int(e["validation_episodes"]), seed=50000)
    last_candidate = passes_candidate_gate(last_metrics, cfg)
    candidate_streak = update_candidate_streak(candidate_streak, last_candidate)
    append_history(last_metrics, last_candidate)
    maybe_archive_best(last_metrics)
    write_state_and_status(last_metrics, last_candidate, 0)
else:
    while int(model.num_timesteps) < run_target and not graduation_ready:
        blocks_completed += 1
        before = int(model.num_timesteps)
        request_steps = min(block_steps, remaining_timesteps(model.num_timesteps, target_total))
        if request_steps <= 0:
            break

        print(
            f"=== MARATHON BLOCK {blocks_completed}: +{request_steps:,} requested "
            f"from {before:,}/{target_total:,} ==="
        )
        model.learn(total_timesteps=request_steps, reset_num_timesteps=False, progress_bar=False)
        model.save(checkpoint)
        after = int(model.num_timesteps)
        print(f"Checkpoint saved: {after:,} total steps (+{after-before:,} actual)")

        # Save local state after every 500k block. GitHub cache persists at the end of each chained run.
        write_state_and_status(last_metrics, last_candidate, blocks_completed)

        at_run_end = after >= run_target or after >= target_total
        if should_validate(blocks_completed, validation_interval_blocks, at_run_end=at_run_end):
            validation_index += 1
            print(f"--- VALIDATION #{validation_index} at {after:,} steps ---")
            last_metrics = evaluate_model(model, val_frames, rules, int(e["validation_episodes"]), seed=50000)
            last_candidate = passes_candidate_gate(last_metrics, cfg)
            candidate_streak = update_candidate_streak(candidate_streak, last_candidate)
            append_history(last_metrics, last_candidate)
            maybe_archive_best(last_metrics)
            write_state_and_status(last_metrics, last_candidate, blocks_completed)
            print(
                f"Validation median={last_metrics['median_return_pct']:.2f}% "
                f"PF={last_metrics['profit_factor']:.3f} "
                f"DD={last_metrics['median_max_drawdown_pct']:.2f}% "
                f"candidate={last_candidate} streak={candidate_streak}/{master_streak_required}"
            )
            graduation_ready = bool(stop_on_master_streak and candidate_streak >= master_streak_required)
            if graduation_ready:
                print("GRADUATION GATE REACHED — stopping marathon before FINAL TEST. FINAL remains LOCKED.")
                break

# Ensure final status exists even if the final block ended between validation intervals.
if last_metrics is None:
    validation_index += 1
    last_metrics = evaluate_model(model, val_frames, rules, int(e["validation_episodes"]), seed=50000)
    last_candidate = passes_candidate_gate(last_metrics, cfg)
    candidate_streak = update_candidate_streak(candidate_streak, last_candidate)
    append_history(last_metrics, last_candidate)
    maybe_archive_best(last_metrics)

write_state_and_status(last_metrics, last_candidate, blocks_completed)

print("=== MARATHON RUN COMPLETE ===")
print(f"Total timesteps: {model.num_timesteps:,}")
print(f"Target: {target_total:,}")
print(f"Candidate streak: {candidate_streak}/{master_streak_required}")
print(f"Elapsed: {(time.time()-run_started)/60:.1f} min")
