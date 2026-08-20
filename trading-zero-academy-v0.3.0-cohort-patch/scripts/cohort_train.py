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
from academy.cohort import student_paths, student_seed, ensure_student_dirs, load_json, write_json
from academy.dataset import (
    load_frames,
    ensure_split_manifest,
    build_academy_partitions,
    partition_summary,
    SPLIT_MANIFEST_NAME,
)
from academy.env import TradingAcademyEnv
from academy.evaluate import evaluate_model, passes_candidate_gate
from academy.marathon import is_better_validation, should_validate, update_candidate_streak

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
learner = cfg["learner"]
base_seed = int(cfg["project"]["seed"])
student_id = int(os.environ["STUDENT_ID"])
paths = student_paths(ROOT, student_id)
ensure_student_dirs(paths)

# Student #1 migration: current Marathon is restored by the workflow into
# migration/student1. This preserves its latest weights, best @ validation, and history.
if not paths["latest"].exists() and student_id == 1:
    legacy = ROOT / "migration" / "student1"
    legacy_models = legacy / "models"
    legacy_history = legacy / "learning_history.csv"
    if not (legacy_models / "latest.zip").exists():
        raise SystemExit(
            "STOP: Student #1 cohort checkpoint not found and legacy Marathon checkpoint was not migrated. "
            "Refusing to create a replacement Student #1 from zero."
        )
    for src in legacy_models.glob("*"):
        if src.is_file():
            shutil.copy2(src, paths["models"] / src.name)
    if legacy_history.exists():
        shutil.copy2(legacy_history, paths["history"])
    print("Migrated Student #1 from legacy Marathon memory into Cohort storage.")

student_base_seed = student_seed(base_seed, student_id)
state = load_json(paths["state"], {}) or {}
cohort_round_index = int(state.get("cohort_round_index", 0)) + 1
run_seed = student_base_seed + cohort_round_index * 100_003

data_dir = ROOT / "data" / "processed"
frames = load_frames(data_dir, m["symbols"], m["bar"])
manifest = ensure_split_manifest(
    frames,
    float(e["train_fraction"]),
    float(e["validation_fraction"]),
    data_dir / SPLIT_MANIFEST_NAME,
)
train_frames, val_frames, frozen_final_frames, live_shadow_frames, validation_source = build_academy_partitions(
    frames,
    manifest,
    live_shadow_days=int(e["live_shadow_days"]),
    train_maturation_days=int(e["train_maturation_days"]),
    min_rolling_validation_bars=int(e["min_rolling_validation_bars"]),
)


def env_fn():
    return TradingAcademyEnv(train_frames, rules, seed=run_seed, random_start=True)


venv = make_vec_env(env_fn, n_envs=4, seed=run_seed)
if paths["latest"].exists():
    model = PPO.load(paths["latest"], env=venv)
    print(f"Continuing Student #{student_id} at {model.num_timesteps:,} total timesteps")
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
        seed=student_base_seed,
    )
    print(f"Created Student #{student_id} from zero with independent seed {student_base_seed}")

block_steps = int(learner.get("timesteps_per_block", 500_000))
max_steps_this_round = int(learner.get("cohort_steps_per_round", 10_000_000))
validation_interval_blocks = int(learner.get("marathon_validation_interval_blocks", 4))
master_streak_required = int(learner.get("marathon_master_candidate_streak", 3))
target_total = int(os.environ.get("COHORT_TARGET_TIMESTEPS") or learner.get("cohort_target_total_timesteps", 500_000_000))

start_total = int(model.num_timesteps)
run_target = min(target_total, start_total + max_steps_this_round)
candidate_streak = int(state.get("candidate_streak", 0))
validation_index = int(state.get("validation_index", 0))
blocks_completed = 0
run_started = time.time()
last_metrics = state.get("last_validation")
last_candidate = bool(state.get("last_candidate", False))

HISTORY_FIELDS = [
    "timestamp_utc", "student_id", "cohort_round_index", "validation_index",
    "total_timesteps", "status", "candidate_streak", "validation_source",
    "episodes", "median_return_pct", "mean_return_pct", "median_max_drawdown_pct",
    "profitable_episode_ratio", "profit_factor", "worst_return_pct", "best_return_pct",
]


def append_history(metrics: dict, candidate: bool):
    import datetime as dt
    exists = paths["history"].exists()
    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "student_id": student_id,
        "cohort_round_index": cohort_round_index,
        "validation_index": validation_index,
        "total_timesteps": int(model.num_timesteps),
        "status": "MASTER_CANDIDATE" if candidate_streak >= master_streak_required else "LEARNING",
        "candidate_streak": candidate_streak,
        "validation_source": validation_source,
        **{k: metrics.get(k) for k in HISTORY_FIELDS if k in metrics},
    }
    with paths["history"].open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in HISTORY_FIELDS})


def save_status(metrics: dict | None, candidate: bool):
    status = "MASTER_CANDIDATE" if paths["master_candidate"].exists() or candidate_streak >= master_streak_required else "LEARNING"
    payload = {
        "student": student_id,
        "student_seed": student_base_seed,
        "status": status,
        "total_timesteps": int(model.num_timesteps),
        "candidate_streak": int(candidate_streak),
        "master_candidate_streak_required": master_streak_required,
        "validation_source": validation_source,
        "validation": metrics or {},
        "final_exam": "LOCKED",
        "cohort": {
            "round_index": cohort_round_index,
            "target_total_timesteps": target_total,
            "round_start_timesteps": start_total,
            "round_target_timesteps": run_target,
        },
        "partitions": {
            "train": partition_summary(train_frames),
            "validation": partition_summary(val_frames),
            "frozen_final": partition_summary(frozen_final_frames),
            "live_shadow": partition_summary(live_shadow_frames),
        },
    }
    write_json(paths["status"], payload)
    write_json(paths["progress"], {
        "student": student_id,
        "current": int(model.num_timesteps),
        "target": target_total,
        "remaining": max(0, target_total - int(model.num_timesteps)),
        "round_index": cohort_round_index,
    })


def run_validation():
    global candidate_streak, validation_index, last_metrics, last_candidate
    validation_index += 1
    metrics = evaluate_model(
        model, val_frames, rules, int(e["validation_episodes"]),
        seed=500_000 + student_id * 10_000,
    )
    candidate = passes_candidate_gate(metrics, cfg)
    candidate_streak = update_candidate_streak(candidate_streak, candidate)
    last_metrics, last_candidate = metrics, candidate

    incumbent = load_json(paths["best_metrics"], None)
    if is_better_validation(metrics, incumbent):
        model.save(paths["best"])
        best_meta = {
            **metrics,
            "student": student_id,
            "total_timesteps": int(model.num_timesteps),
            "validation_source": validation_source,
            "validation_index": validation_index,
        }
        write_json(paths["best_metrics"], best_meta)
        print(f"Student #{student_id}: NEW BEST validation at {model.num_timesteps:,} steps")

    # First pre-specified 3x candidate is frozen for a future one-time FINAL EXAM.
    if candidate_streak >= master_streak_required and not paths["master_candidate"].exists():
        model.save(paths["master_candidate"])
        write_json(paths["master_candidate_metrics"], {
            **metrics,
            "student": student_id,
            "total_timesteps": int(model.num_timesteps),
            "candidate_streak": candidate_streak,
            "validation_source": validation_source,
        })
        print(f"Student #{student_id}: MASTER_CANDIDATE frozen for future manual FINAL EXAM")

    append_history(metrics, candidate)
    save_status(metrics, candidate)
    return metrics, candidate


# If a student already reached the target, do not mutate it; just report.
if int(model.num_timesteps) < target_total:
    while int(model.num_timesteps) < run_target:
        requested = min(block_steps, run_target - int(model.num_timesteps))
        # SB3 may overshoot by a rollout boundary; that is expected.
        model.learn(total_timesteps=max(1, requested), reset_num_timesteps=False)
        model.save(paths["latest"])
        blocks_completed += 1
        at_run_end = int(model.num_timesteps) >= run_target
        if should_validate(blocks_completed, validation_interval_blocks, at_run_end=at_run_end):
            run_validation()
else:
    print(f"Student #{student_id} already reached cohort target {target_total:,}; no training this round.")

# Ensure latest checkpoint exists even for a fresh student/edge case.
model.save(paths["latest"])
if last_metrics is None:
    run_validation()

state.update({
    "cohort_round_index": cohort_round_index,
    "candidate_streak": candidate_streak,
    "validation_index": validation_index,
    "last_validation": last_metrics,
    "last_candidate": last_candidate,
    "student_id": student_id,
    "student_seed": student_base_seed,
})
write_json(paths["state"], state)
save_status(last_metrics, last_candidate)

print("=== COHORT STUDENT ROUND COMPLETE ===")
print(f"Student #{student_id}")
print(f"Total timesteps: {model.num_timesteps:,}")
print(f"Target: {target_total:,}")
print(f"Candidate streak: {candidate_streak}/{master_streak_required}")
print(f"Elapsed: {(time.time()-run_started)/60:.1f} min")
