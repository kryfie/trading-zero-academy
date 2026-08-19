from pathlib import Path
import json
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from academy.config import load_config, ROOT
from academy.dataset import load_frames, split_frames, SPLIT_MANIFEST_NAME
from academy.env import TradingAcademyEnv
from academy.evaluate import evaluate_model, passes_candidate_gate

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
seed = int(cfg["project"]["seed"])
frames = load_frames(ROOT / "data" / "processed", m["symbols"], m["bar"])
train_frames, val_frames, _ = split_frames(
    frames,
    float(e["train_fraction"]),
    float(e["validation_fraction"]),
    manifest_path=ROOT / "data" / "processed" / SPLIT_MANIFEST_NAME,
)

checkpoint = ROOT / cfg["runtime"]["checkpoint_path"]
checkpoint.parent.mkdir(parents=True, exist_ok=True)

def env_fn():
    return TradingAcademyEnv(train_frames, rules, seed=seed, random_start=True)

venv = make_vec_env(env_fn, n_envs=4, seed=seed)
learner = cfg["learner"]
if checkpoint.exists():
    model = PPO.load(checkpoint, env=venv)
    print(f"Continuing from {checkpoint}")
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
        seed=seed,
    )

steps = int(learner["total_timesteps_per_run"])
model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
model.save(checkpoint)
metrics = evaluate_model(model, val_frames, rules, int(e["validation_episodes"]), seed=50000)
candidate = passes_candidate_gate(metrics, cfg)
status = {
    "timestamp_utc": int(time.time()),
    "status": "MASTER_CANDIDATE" if candidate else "LEARNING",
    "total_timesteps": int(model.num_timesteps),
    "validation": metrics,
    "final_test_locked": True,
    "note": "Final test is intentionally not used for learning or routine model selection.",
}
status_path = ROOT / cfg["runtime"]["status_path"]
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
print(json.dumps(status, indent=2))
