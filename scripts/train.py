from pathlib import Path
import json
import time

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

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
learner = cfg["learner"]
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

checkpoint = ROOT / cfg["runtime"]["checkpoint_path"]
checkpoint.parent.mkdir(parents=True, exist_ok=True)
state_path = ROOT / cfg["runtime"]["training_state_path"]
state_path.parent.mkdir(parents=True, exist_ok=True)
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
else:
    state = {"run_index": 0}
run_index = int(state.get("run_index", 0)) + 1

# Use a new deterministic RNG stream every daily run. This prevents every GitHub
# process from replaying the same episode-start sequence while remaining reproducible.
run_seed = base_seed + run_index * 100_003

def env_fn():
    return TradingAcademyEnv(train_frames, rules, seed=run_seed, random_start=True)

venv = make_vec_env(env_fn, n_envs=4, seed=run_seed)
if checkpoint.exists():
    model = PPO.load(checkpoint, env=venv)
    print(f"Continuing Student #1 from {checkpoint}; existing timesteps={model.num_timesteps:,}")
else:
    model = PPO(
        "MlpPolicy", venv,
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

blocks = int(learner.get("blocks_per_run", 1))
block_steps = int(learner.get("timesteps_per_block", learner.get("total_timesteps_per_run", 500000)))
progress_path = ROOT / cfg["runtime"]["progress_path"]
progress_path.parent.mkdir(parents=True, exist_ok=True)
run_started = time.time()

for block in range(1, blocks + 1):
    before = int(model.num_timesteps)
    print(f"=== DAILY TRAINING BLOCK {block}/{blocks}: target +{block_steps:,} steps ===")
    model.learn(total_timesteps=block_steps, reset_num_timesteps=False, progress_bar=False)
    model.save(checkpoint)
    after = int(model.num_timesteps)
    state = {
        "run_index": run_index,
        "run_seed": run_seed,
        "completed_blocks_this_run": block,
        "blocks_per_run": blocks,
        "timesteps_per_block_target": block_steps,
        "total_timesteps": after,
        "last_block_actual_steps": after - before,
        "updated_at_utc": int(time.time()),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Checkpoint saved after block {block}: total_timesteps={after:,}")

metrics = evaluate_model(model, val_frames, rules, int(e["validation_episodes"]), seed=50000 + run_index * 1000)
candidate = passes_candidate_gate(metrics, cfg)

# LIVE SHADOW is observation-only. It never affects training or the candidate gate.
live_metrics = None
min_live_rows = int(rules["observation_window"]) + 20
if all(len(df) >= min_live_rows for df in live_shadow_frames.values()):
    try:
        live_metrics = evaluate_model(
            model,
            live_shadow_frames,
            rules,
            int(e.get("live_shadow_episodes", 10)),
            seed=700000 + run_index * 1000,
        )
    except Exception as exc:
        live_metrics = {"available": False, "error": str(exc)}

status = {
    "timestamp_utc": int(time.time()),
    "status": "MASTER_CANDIDATE" if candidate else "LEARNING",
    "student": 1,
    "run_index": run_index,
    "run_seed": run_seed,
    "total_timesteps": int(model.num_timesteps),
    "training_blocks_completed": blocks,
    "training_elapsed_seconds": round(time.time() - run_started, 2),
    "validation_source": validation_source,
    "validation": metrics,
    "live_shadow": live_metrics,
    "partitions": {
        "train": partition_summary(train_frames),
        "validation": partition_summary(val_frames),
        "live_shadow": partition_summary(live_shadow_frames),
        "frozen_final": partition_summary(frozen_final_frames),
    },
    "data_policy": {
        "live_shadow_days": int(e["live_shadow_days"]),
        "train_maturation_days": int(e["train_maturation_days"]),
        "rule": "Post-launch data: 0-7d LIVE SHADOW, 7-30d rolling validation, >=30d TRAIN. Frozen final never trains.",
    },
    "final_test_locked": True,
    "note": "LIVE SHADOW and FINAL TEST never influence learning. FINAL remains permanently frozen.",
}
status_path = ROOT / cfg["runtime"]["status_path"]
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
