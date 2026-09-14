from __future__ import annotations

import csv
import os
import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from academy.config import load_config, ROOT
from academy.cohort import student_seed
from academy.gen3_paths import gen3_student_paths, ensure_gen3_student_dirs, load_json, write_json
from academy.splits import load_partition_reference, load_mtf_frames, build_frozen_partitions, partition_summary
from academy.gen3_env import Generation3TradingEnv, build_target_levels
from academy.gen3_evaluate import evaluate_recurrent_model, passes_candidate_gate
from academy.training_utils import is_better_validation, update_candidate_streak
from academy.gen3_telemetry import capture_parameter_state, collect_training_telemetry, append_telemetry
from academy.checkpoint_registry import update_validation_registry

class WallClockStopCallback(BaseCallback):
    """Stop RecurrentPPO cleanly from inside model.learn before the GitHub hard timeout."""

    def __init__(self, deadline_monotonic: float, check_freq: int = 256):
        super().__init__(verbose=0)
        self.deadline_monotonic = float(deadline_monotonic)
        self.check_freq = max(1, int(check_freq))
        self.stopped_for_time = False

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        if time.monotonic() >= self.deadline_monotonic:
            self.stopped_for_time = True
            print("Gen3 wall-clock callback: soft runtime reached; stopping model.learn cleanly.")
            return False
        return True


cfg = load_config(ROOT / "config_generation3.yaml")
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
learner = cfg["learner"]
actions = cfg["action_space"]
checkpoint_cfg = cfg.get("checkpointing", {})
base_seed = int(cfg["project"]["seed"])
experiment_id = str(cfg["project"]["experiment_id"])

student_id = int(os.environ["STUDENT_ID"])
student_mode = "GEN3_RAW_MTF_DISCRETE_RECURRENT"
paths = gen3_student_paths(ROOT, student_id)
ensure_gen3_student_dirs(paths)

# Seed-match Gen2: Gen2 Student #001..#008 used seed reference identities #002..#009.
seed_reference_id = student_id + 1
student_base_seed = student_seed(base_seed, seed_reference_id)
state = load_json(paths["state"], {}) or {}
cohort_round_index = int(state.get("cohort_round_index", 0)) + 1
run_seed = student_base_seed + cohort_round_index * 100_003

timeframes = list(m["multi_timeframes"])
windows = {str(k): int(v) for k, v in m["multi_timeframe_windows"].items()}
decision_bar = str(m["decision_bar"])
target_levels = build_target_levels(
    int(actions["min_target_leverage"]), int(actions["max_target_leverage"]), int(actions.get("step", 1))
)
reference = load_partition_reference(ROOT / m["frozen_partition_reference"])
frames = load_mtf_frames(ROOT / "data" / "mtf", m["symbols"], timeframes)
train_frames, val_frames, frozen_final_frames = build_frozen_partitions(frames, reference, timeframes)


def env_fn():
    return Generation3TradingEnv(
        train_frames, rules, timeframes=timeframes, windows=windows,
        decision_bar=decision_bar, target_levels=target_levels,
        seed=run_seed, random_start=True,
    )


def evaluate_current(model):
    return evaluate_recurrent_model(
        model, val_frames, rules, timeframes=timeframes, windows=windows,
        decision_bar=decision_bar, target_levels=target_levels,
        episodes=int(e["validation_episodes"]),
        seed=500_000 + seed_reference_id * 10_000,
    )

partition_payload = {
    "train": partition_summary(train_frames, decision_bar),
    "validation": partition_summary(val_frames, decision_bar),
    "frozen_final": partition_summary(frozen_final_frames, decision_bar),
}
venv = make_vec_env(env_fn, n_envs=2, seed=run_seed)
run_mode = os.environ.get("GEN3_RUN_MODE", "resume").strip().lower()
if run_mode not in {"start", "resume"}:
    raise SystemExit("GEN3_RUN_MODE must be start or resume")

if run_mode == "start":
    if paths["latest"].exists():
        raise SystemExit(f"STOP: Gen3 Student #{student_id} already exists. START refuses to overwrite it.")
    model = RecurrentPPO(
        str(learner.get("policy", "MlpLstmPolicy")),
        venv,
        learning_rate=float(learner["learning_rate"]),
        n_steps=int(learner["n_steps"]),
        batch_size=int(learner["batch_size"]),
        n_epochs=int(learner["n_epochs"]),
        gamma=float(learner["gamma"]),
        gae_lambda=float(learner["gae_lambda"]),
        ent_coef=float(learner["ent_coef"]),
        vf_coef=float(learner["vf_coef"]),
        clip_range=float(learner["clip_range"]),
        max_grad_norm=float(learner["max_grad_norm"]),
        policy_kwargs={
            "net_arch": list(learner["policy_layers"]),
            "lstm_hidden_size": int(learner["lstm_hidden_size"]),
            "n_lstm_layers": int(learner["n_lstm_layers"]),
            "shared_lstm": bool(learner["shared_lstm"]),
            "enable_critic_lstm": bool(learner["enable_critic_lstm"]),
        },
        verbose=1,
        seed=student_base_seed,
    )
    print(f"Created Gen3 Student #{student_id} from zero | seed={student_base_seed} | LSTM sequence memory")
else:
    if not paths["latest"].exists():
        raise SystemExit(
            f"STOP: Gen3 Student #{student_id} checkpoint was not restored in RESUME mode. "
            "Refusing to silently create a replacement student from zero."
        )
    model = RecurrentPPO.load(paths["latest"], env=venv)
    print(f"Continuing Gen3 Student #{student_id} at {model.num_timesteps:,} timesteps")

block_steps = int(learner["timesteps_per_block"])
validation_interval_blocks = int(learner["validation_interval_blocks"])
master_streak_required = int(learner["master_candidate_streak"])
steps_this_round = int(learner["steps_per_round"])
target_total = int(os.environ.get("COHORT_TARGET_TIMESTEPS") or learner["target_total_timesteps"])
if target_total != int(learner["target_total_timesteps"]):
    raise SystemExit("Generation 3 protocol is frozen at exactly 1,000,000,000 target timesteps.")
start_total = int(model.num_timesteps)
run_target = min(target_total, start_total + steps_this_round)
candidate_streak = int(state.get("candidate_streak", 0))
validation_index = int(state.get("validation_index", 0))
blocks_completed = 0
run_started = time.time()
run_started_monotonic = time.monotonic()
soft_runtime_minutes = int(os.environ.get("GEN3_SOFT_RUNTIME_MINUTES", "210"))
soft_runtime_seconds = max(60, soft_runtime_minutes * 60)
soft_deadline_monotonic = run_started_monotonic + soft_runtime_seconds
soft_stopped = False
last_metrics = state.get("last_validation")
last_candidate = bool(state.get("last_candidate", False))

# Preserve the scientific validation cadence across short infrastructure runs.
# Gen3's frozen cadence is still 4 x 500k = ~2M training steps between validations.
validation_every_steps = block_steps * validation_interval_blocks
last_validation_timesteps = int(state.get("last_validation_timesteps", 0) or 0)
if last_validation_timesteps <= 0 and last_metrics is not None:
    last_validation_timesteps = int(state.get("last_saved_timesteps", start_total) or start_total)
next_validation_at = (
    last_validation_timesteps + validation_every_steps
    if last_validation_timesteps > 0
    else validation_every_steps
)
while next_validation_at <= start_total:
    next_validation_at += validation_every_steps

HISTORY_FIELDS = [
    "timestamp_utc", "generation", "experiment_id", "student_id", "student_mode", "seed_reference_id",
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
        "generation": 3,
        "experiment_id": experiment_id,
        "student_id": student_id,
        "student_mode": student_mode,
        "seed_reference_id": seed_reference_id,
        "cohort_round_index": cohort_round_index,
        "validation_index": validation_index,
        "total_timesteps": int(model.num_timesteps),
        "status": "MASTER_CANDIDATE" if candidate_streak >= master_streak_required else "LEARNING",
        "candidate_streak": candidate_streak,
        "validation_source": "GEN1_GEN2_FROZEN_VALIDATION",
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
        "generation": 3,
        "experiment_id": experiment_id,
        "student": student_id,
        "student_mode": student_mode,
        "student_seed": student_base_seed,
        "seed_reference_generation2_student": student_id,
        "status": status,
        "total_timesteps": int(model.num_timesteps),
        "candidate_streak": int(candidate_streak),
        "master_candidate_streak_required": master_streak_required,
        "validation_source": "GEN1_GEN2_FROZEN_VALIDATION",
        "validation": metrics or {},
        "final_exam": "LOCKED",
        "experiment": {
            "hypothesis": "Recurrent sequence memory improves net OOS behavior relative to seed-matched Gen2 MLP policies.",
            "changed_from_generation2": ["policy_architecture: MLP -> LSTM recurrent state", "training_ceiling pre-registered at 1B"],
            "unchanged": [
                "raw M1/M5/M15/H1/H4 observation", "5m decision clock", "discrete target action -10..+10",
                "fee", "slippage", "funding", "max leverage", "reward", "validation gates", "market partitions"
            ],
            "lstm": {
                "hidden_size": int(learner["lstm_hidden_size"]),
                "layers": int(learner["n_lstm_layers"]),
                "shared_lstm": bool(learner["shared_lstm"]),
                "critic_lstm": bool(learner["enable_critic_lstm"]),
            },
        },
        "market_view": {"decision_clock": decision_bar, "timeframes": timeframes, "windows": windows, "indicators": []},
        "cohort": {
            "round_index": cohort_round_index,
            "target_total_timesteps": target_total,
            "round_start_timesteps": start_total,
            "round_target_timesteps": run_target,
        },
        "partitions": partition_payload,
    })
    write_json(paths["progress"], {
        "generation": 3, "experiment_id": experiment_id, "student": student_id,
        "current": int(model.num_timesteps), "target": target_total,
        "remaining": max(0, target_total - int(model.num_timesteps)), "round_index": cohort_round_index,
    })



def persist_state():
    state.update({
        "generation": 3,
        "experiment_id": experiment_id,
        "cohort_round_index": cohort_round_index,
        "candidate_streak": candidate_streak,
        "validation_index": validation_index,
        "last_validation": last_metrics,
        "last_candidate": last_candidate,
        "last_validation_timesteps": int(last_validation_timesteps),
        "student_id": student_id,
        "student_mode": student_mode,
        "student_seed": student_base_seed,
        "seed_reference_id": seed_reference_id,
        "last_saved_timesteps": int(model.num_timesteps),
    })
    write_json(paths["state"], state)


def run_validation():
    global candidate_streak, validation_index, last_metrics, last_candidate, last_validation_timesteps
    validation_index += 1
    metrics = evaluate_current(model)
    candidate = passes_candidate_gate(metrics, cfg)
    candidate_streak = update_candidate_streak(candidate_streak, candidate)
    last_metrics, last_candidate = metrics, candidate
    last_validation_timesteps = int(model.num_timesteps)
    incumbent = load_json(paths["best_metrics"], None)
    metadata = {
        "generation": 3, "experiment_id": experiment_id, "student": student_id,
        "student_mode": student_mode, "student_seed": student_base_seed,
        "seed_reference_generation2_student": student_id,
        "validation_source": "GEN1_GEN2_FROZEN_VALIDATION", "validation_index": validation_index,
        "action_space": "DISCRETE_TARGET_POSITION", "policy": "MlpLstmPolicy",
    }
    update_validation_registry(
        model, metrics, int(model.num_timesteps), paths["registry"], paths["registry_dir"],
        top_k=int(checkpoint_cfg.get("top_validation_checkpoints", 5)), metadata=metadata,
    )
    if is_better_validation(metrics, incumbent):
        model.save(paths["best"])
        write_json(paths["best_metrics"], {**metrics, **metadata, "total_timesteps": int(model.num_timesteps)})
        print(f"Gen3 Student #{student_id}: NEW BEST at {model.num_timesteps:,}")
    if candidate_streak >= master_streak_required and not paths["master_candidate"].exists():
        model.save(paths["master_candidate"])
        write_json(paths["master_candidate_metrics"], {
            **metrics, **metadata, "total_timesteps": int(model.num_timesteps), "candidate_streak": candidate_streak,
        })
        print(f"Gen3 Student #{student_id}: MASTER_CANDIDATE frozen for manual FINAL EXAM")
    append_history(metrics)
    save_status(metrics)
    persist_state()

if int(model.num_timesteps) < target_total:
    while int(model.num_timesteps) < run_target:
        if time.monotonic() >= soft_deadline_monotonic:
            soft_stopped = True
            print(
                f"Gen3 Student #{student_id}: soft runtime ceiling reached "
                f"({soft_runtime_minutes} min). Exiting cleanly so checkpoint/cache can be saved."
            )
            break

        requested = min(block_steps, run_target - int(model.num_timesteps))
        block_start = int(model.num_timesteps)
        before = capture_parameter_state(model)
        block_started = time.time()

        wall_clock_callback = WallClockStopCallback(soft_deadline_monotonic)
        model.learn(
            total_timesteps=max(1, requested),
            reset_num_timesteps=False,
            callback=wall_clock_callback,
        )

        elapsed = time.time() - block_started
        model.save(paths["latest"])
        blocks_completed += 1
        persist_state()
        append_telemetry(paths["telemetry"], collect_training_telemetry(
            model, before, cohort_round_index, blocks_completed, block_start, elapsed
        ))

        # Validation remains tied to cumulative training progress, not workflow run count.
        if int(model.num_timesteps) >= next_validation_at:
            run_validation()
            next_validation_at = last_validation_timesteps + validation_every_steps

        if wall_clock_callback.stopped_for_time:
            soft_stopped = True
            print(
                f"Gen3 Student #{student_id}: model.learn stopped safely at "
                f"{model.num_timesteps:,} timesteps because the soft wall-clock limit was reached."
            )
            break
else:
    print(f"Gen3 Student #{student_id} already reached {target_total:,}; no training.")

# Do NOT force an extra validation merely because an infrastructure run ended.
# This preserves the frozen ~2M-step validation cadence and candidate-streak semantics.
model.save(paths["latest"])
persist_state()
save_status(last_metrics)
print("=== GENERATION 3 STUDENT ROUND COMPLETE ===")
print(f"Student #{student_id} | {student_mode}")
print(f"Total timesteps: {model.num_timesteps:,} / {target_total:,}")
print(f"Candidate streak: {candidate_streak}/{master_streak_required}")
print(f"Elapsed: {(time.time()-run_started)/60:.1f} min")
