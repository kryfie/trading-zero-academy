from __future__ import annotations

import json
import os

from stable_baselines3 import PPO

from academy.config import load_config, ROOT
from academy.cohort import student_paths, load_json
from academy.splits import load_partition_reference, load_mtf_frames, build_frozen_partitions
from academy.gen2_env import build_target_levels
from academy.evaluate import evaluate_model

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
learner = cfg["learner"]
actions = cfg["action_space"]

student_id = int(os.environ.get("STUDENT_ID", "1"))
p = student_paths(ROOT, student_id)

if not p["master_candidate"].exists():
    raise SystemExit(f"FINAL TEST LOCKED for Gen2 Student #{student_id}: no frozen MASTER_CANDIDATE.")

master_meta = load_json(p["master_candidate_metrics"], {}) or {}
required = int(learner["master_candidate_streak"])
if int(master_meta.get("candidate_streak", 0)) < required:
    raise SystemExit("FINAL TEST LOCKED: candidate streak requirement not satisfied.")

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
_, _, frozen_final_frames = build_frozen_partitions(frames, reference, timeframes)

model = PPO.load(p["master_candidate"])
episodes = max(50, int(e["validation_episodes"]))
seed_reference_id = student_id + 1
metrics = evaluate_model(
    model, frozen_final_frames, cfg["world_rules"],
    timeframes=timeframes, windows=windows, decision_bar=decision_bar,
    target_levels=target_levels, episodes=episodes,
    seed=900_000 + seed_reference_id * 10_000,
)

out = p["reports"] / "final_exam.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "generation": 2,
    "student": student_id,
    "examined_checkpoint": "master_candidate.zip",
    "candidate_metadata": master_meta,
    "final_metrics": metrics,
    "warning": "FINAL is one-shot evidence. Never use it to tune Gen2.",
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
