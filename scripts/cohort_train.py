from __future__ import annotations

import csv
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from academy.config import load_config, ROOT
from academy.cohort import student_paths, student_seed, ensure_student_dirs, load_json, write_json
from academy.splits import load_partition_reference, load_mtf_frames, build_frozen_partitions, partition_summary
from academy.gen2_env import Generation2TradingEnv, build_target_levels
from academy.evaluate import evaluate_model, passes_candidate_gate
from academy.training_utils import is_better_validation, should_validate, update_candidate_streak

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
learner = cfg["learner"]
actions = cfg["action_space"]
base_seed = int(cfg["project"]["seed"])

student_id = int(os.environ["STUDENT_ID"])
student_mode = "GEN2_RAW_MTF_DISCRETE"
paths = student_paths(ROOT, student_id)
ensure_student_dirs(paths)

# Gen2 #001..#007 deliberately reuse the same seeds as Gen1 RAW_MTF #002..#008.
# This strengthens the action-space comparison. Gen2 #008 gets the next unused seed (#009).
seed_reference_id = student_id + 1
student_base_seed = student_seed(base_seed, seed_reference_id)

state = load_json(paths["state"], {}) or {}
cohort_round_index = int(state.get("cohort_round_index", 0)) + 1
run_seed = student_base_seed + cohort_round_index * 100_003

timeframes = list(m["multi_timeframes"])
windows = {str(k): int(v) for k, v in m["multi_timeframe_windows"].items()}
decision_bar = str(m["decision_bar"])
target_levels = build_target_levels(
    int(actions["min_target_leverage"]),
    int(actions["max_target_leverage"]),
    int(actions.get("step", 1)),
)

reference = load_partition_reference(ROOT / m["frozen_partition_reference"])
frames = load_mtf_frames(ROOT / "data" / "mtf", m["symbols"], timeframes)
train_frames, val_frames, frozen_final_frames = build_frozen_partitions(frames, reference, timeframes)

def env_fn():
    return Generation2TradingEnv(
        train_frames, rules,
        timeframes=timeframes,
        windows=windows,
        decision_bar=decision_bar,
        target_levels=target_levels,
        seed=run_seed,
        random_start=True,
    )

def evaluate_current(model):
    return evaluate_model(
        model, val_frames, rules,
        timeframes=timeframes,
        windows=windows,
        decision_bar=decision_bar,
        target_levels=target_levels,
        episodes=int(e["validation_episodes"]),
        # Match Gen1 RAW_MTF #002..#008 evaluation seeds for Gen2 #001..#007.
        seed=500_000 + seed_reference_id * 10_000,
    )

partition_payload = {
    "train": partition_summary(train_frames, decision_bar),
    "validation": partition_summary(val_frames, decision_bar),
    "frozen_final": partition_summary(frozen_final_frames, decision_bar),
}

venv = make_vec_env(env_fn, n_envs=2, seed=run_seed)

run_mode = os.environ.get("GEN2_RUN_MODE", "resume").strip().lower()

if paths["latest"].exists():
    model = PPO.load(paths["latest"], env=venv)
    print(f"Continuing Generation 2 Student #{student_id} at {model.num_timesteps:,} timesteps")
else:
    if run_mode != "start":
        raise SystemExit(
            f"STOP: Gen2 Student #{student_id} checkpoint was not restored in RESUME mode. "
            "Refusing to silently create a replacement student from zero."
        )
    model = PPO(
        str(learner.get("policy", "MlpPolicy")),
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
    print(
        f"Created Generation 2 Student #{student_id} from zero | "
        f"seed={student_base_seed} | discrete actions={list(target_levels)}"
    )

block_steps = int(learner["timesteps_per_block"])
validation_interval_blocks = int(learner["validation_interval_blocks"])
master_streak_required = int(learner["master_candidate_streak"])
steps_this_round = int(learner["steps_per_round"])
target_total = int(os.environ.get("COHORT_TARGET_TIMESTEPS") or learner["target_total_timesteps"])

start_total = int(model.num_timesteps)
run_target = min(target_total, start_total + steps_this_round)
candidate_streak = int(state.get("candidate_streak", 0))
validation_index = int(state.get("validation_index", 0))
blocks_completed = 0
run_started = time.time()
last_metrics = state.get("last_validation")
last_candidate = bool(state.get("last_candidate", False))

HISTORY_FIELDS = [
    "timestamp_utc", "generation", "student_id", "student_mode", "seed_reference_id",
    "cohort_round_index", "validation_index", "total_timesteps", "status",
    "candidate_streak", "validation_source", "episodes", "median_return_pct",
    "mean_return_pct", "median_max_drawdown_pct", "profitable_episode_ratio",
    "profit_factor", "worst_return_pct", "best_return_pct",
]

def append_history(metrics: dict):
    import datetime as dt
    exists = paths["history"].exists()
    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generation": 2,
        "student_id": student_id,
        "student_mode": student_mode,
        "seed_reference_id": seed_reference_id,
        "cohort_round_index": cohort_round_index,
        "validation_index": validation_index,
        "total_timesteps": int(model.num_timesteps),
        "status": "MASTER_CANDIDATE" if candidate_streak >= master_streak_required else "LEARNING",
        "candidate_streak": candidate_streak,
        "validation_source": "GEN1_FROZEN_VALIDATION",
        **{k: metrics.get(k) for k in HISTORY_FIELDS if k in metrics},
    }
    with paths["history"].open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in HISTORY_FIELDS})

def save_status(metrics: dict | None):
    status = "MASTER_CANDIDATE" if paths["master_candidate"].exists() or candidate_streak >= master_streak_required else "LEARNING"
    write_json(paths["status"], {
        "generation": 2,
        "student": student_id,
        "student_mode": student_mode,
        "student_seed": student_base_seed,
        "seed_reference_generation1_student": seed_reference_id,
        "status": status,
        "total_timesteps": int(model.num_timesteps),
        "candidate_streak": int(candidate_streak),
        "master_candidate_streak_required": master_streak_required,
        "validation_source": "GEN1_FROZEN_VALIDATION",
        "validation": metrics or {},
        "final_exam": "LOCKED",
        "experiment": {
            "hypothesis": "Discrete target-position actions remove continuous micro-rebalancing pathology without adding trading rules.",
            "changed_from_generation1_raw_mtf": ["action_space"],
            "action_space": {
                "type": "DISCRETE_TARGET_POSITION",
                "target_levels": list(target_levels),
                "repeat_same_target_means_hold": True,
            },
            "unchanged": [
                "PPO", "MlpPolicy", "raw M1/M5/M15/H1/H4 observation",
                "5m decision clock", "fee", "slippage", "funding",
                "max leverage", "reward", "validation gates",
            ],
        },
        "market_view": {
            "decision_clock": decision_bar,
            "timeframes": timeframes,
            "windows": windows,
            "indicators": [],
        },
        "cohort": {
            "round_index": cohort_round_index,
            "target_total_timesteps": target_total,
            "round_start_timesteps": start_total,
            "round_target_timesteps": run_target,
        },
        "partitions": partition_payload,
    })
    write_json(paths["progress"], {
        "generation": 2,
        "student": student_id,
        "current": int(model.num_timesteps),
        "target": target_total,
        "remaining": max(0, target_total - int(model.num_timesteps)),
        "round_index": cohort_round_index,
    })

def run_validation():
    global candidate_streak, validation_index, last_metrics, last_candidate
    validation_index += 1
    metrics = evaluate_current(model)
    candidate = passes_candidate_gate(metrics, cfg)
    candidate_streak = update_candidate_streak(candidate_streak, candidate)
    last_metrics, last_candidate = metrics, candidate

    incumbent = load_json(paths["best_metrics"], None)
    if is_better_validation(metrics, incumbent):
        model.save(paths["best"])
        write_json(paths["best_metrics"], {
            **metrics,
            "generation": 2,
            "student": student_id,
            "student_mode": student_mode,
            "student_seed": student_base_seed,
            "seed_reference_generation1_student": seed_reference_id,
            "total_timesteps": int(model.num_timesteps),
            "validation_source": "GEN1_FROZEN_VALIDATION",
            "validation_index": validation_index,
            "action_space": "DISCRETE_TARGET_POSITION",
        })
        print(f"Gen2 Student #{student_id}: NEW BEST at {model.num_timesteps:,}")

    if candidate_streak >= master_streak_required and not paths["master_candidate"].exists():
        model.save(paths["master_candidate"])
        write_json(paths["master_candidate_metrics"], {
            **metrics,
            "generation": 2,
            "student": student_id,
            "student_mode": student_mode,
            "total_timesteps": int(model.num_timesteps),
            "candidate_streak": candidate_streak,
            "validation_source": "GEN1_FROZEN_VALIDATION",
            "action_space": "DISCRETE_TARGET_POSITION",
        })
        print(f"Gen2 Student #{student_id}: MASTER_CANDIDATE frozen for manual FINAL EXAM")

    append_history(metrics)
    save_status(metrics)

if int(model.num_timesteps) < target_total:
    while int(model.num_timesteps) < run_target:
        requested = min(block_steps, run_target - int(model.num_timesteps))
        model.learn(total_timesteps=max(1, requested), reset_num_timesteps=False)
        model.save(paths["latest"])
        blocks_completed += 1
        at_run_end = int(model.num_timesteps) >= run_target
        if should_validate(blocks_completed, validation_interval_blocks, at_run_end=at_run_end):
            run_validation()
else:
    print(f"Gen2 Student #{student_id} already reached {target_total:,}; no training.")

model.save(paths["latest"])
if last_metrics is None:
    run_validation()

state.update({
    "generation": 2,
    "cohort_round_index": cohort_round_index,
    "candidate_streak": candidate_streak,
    "validation_index": validation_index,
    "last_validation": last_metrics,
    "last_candidate": last_candidate,
    "student_id": student_id,
    "student_mode": student_mode,
    "student_seed": student_base_seed,
    "seed_reference_id": seed_reference_id,
})
write_json(paths["state"], state)
save_status(last_metrics)

print("=== GENERATION 2 STUDENT ROUND COMPLETE ===")
print(f"Student #{student_id} | {student_mode}")
print(f"Total timesteps: {model.num_timesteps:,} / {target_total:,}")
print(f"Candidate streak: {candidate_streak}/{master_streak_required}")
print(f"Elapsed: {(time.time()-run_started)/60:.1f} min")
