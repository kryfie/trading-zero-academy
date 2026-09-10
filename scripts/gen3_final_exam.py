from __future__ import annotations

import json
import os

from sb3_contrib import RecurrentPPO

from academy.config import load_config, ROOT
from academy.gen3_paths import gen3_student_paths, load_json
from academy.splits import load_partition_reference, load_mtf_frames, build_frozen_partitions
from academy.gen3_env import build_target_levels
from academy.gen3_evaluate import evaluate_recurrent_model

cfg = load_config(ROOT / "config_generation3.yaml")
m = cfg["market"]
e = cfg["evaluation"]
learner = cfg["learner"]
actions = cfg["action_space"]
student_id = int(os.environ.get("STUDENT_ID", "1"))
p = gen3_student_paths(ROOT, student_id)
if not p["master_candidate"].exists():
    raise SystemExit(f"FINAL TEST LOCKED for Gen3 Student #{student_id}: no frozen MASTER_CANDIDATE.")
master_meta = load_json(p["master_candidate_metrics"], {}) or {}
required = int(learner["master_candidate_streak"])
if int(master_meta.get("candidate_streak", 0)) < required:
    raise SystemExit("FINAL TEST LOCKED: candidate streak requirement not satisfied.")
timeframes = list(m["multi_timeframes"])
windows = {str(k): int(v) for k, v in m["multi_timeframe_windows"].items()}
decision_bar = str(m["decision_bar"])
target_levels = build_target_levels(int(actions["min_target_leverage"]), int(actions["max_target_leverage"]), int(actions.get("step", 1)))
reference = load_partition_reference(ROOT / m["frozen_partition_reference"])
frames = load_mtf_frames(ROOT / "data" / "mtf", m["symbols"], timeframes)
_, _, frozen_final_frames = build_frozen_partitions(frames, reference, timeframes)
model = RecurrentPPO.load(p["master_candidate"])
episodes = max(50, int(e["validation_episodes"]))
seed_reference_id = student_id + 1
metrics = evaluate_recurrent_model(
    model, frozen_final_frames, cfg["world_rules"], timeframes=timeframes, windows=windows,
    decision_bar=decision_bar, target_levels=target_levels, episodes=episodes,
    seed=900_000 + seed_reference_id * 10_000,
)
out = p["reports"] / "final_exam.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "generation": 3, "experiment_id": cfg["project"]["experiment_id"], "student": student_id,
    "examined_checkpoint": "master_candidate.zip", "candidate_metadata": master_meta,
    "final_metrics": metrics, "warning": "FINAL is one-shot evidence. Never use it to tune Gen3.",
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
