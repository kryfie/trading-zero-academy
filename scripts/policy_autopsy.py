from __future__ import annotations

import csv
import json
import os

from stable_baselines3 import PPO

from academy.config import load_config, ROOT
from academy.cohort import student_paths, ensure_student_dirs, load_json, write_json
from academy.splits import load_partition_reference, load_mtf_frames, build_frozen_partitions
from academy.gen2_env import build_target_levels
from academy.policy_audit import audit_model, compare_audits

cfg = load_config()
m = cfg["market"]
e = cfg["evaluation"]
rules = cfg["world_rules"]
actions = cfg["action_space"]

student_id = int(os.environ["STUDENT_ID"])
paths = student_paths(ROOT, student_id)
ensure_student_dirs(paths)

if not paths["latest"].exists() or not paths["best"].exists():
    print(f"Gen2 Student #{student_id}: autopsy skipped; BEST and/or LATEST missing.")
    raise SystemExit(0)

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
_, val_frames, _ = build_frozen_partitions(frames, reference, timeframes)

episodes = int(e.get("autopsy_episodes", 10))
# Same comparison mapping as training/evaluation.
seed_reference_id = student_id + 1
seed = 770_000 + seed_reference_id * 10_000

best_model = PPO.load(paths["best"])
latest_model = PPO.load(paths["latest"])
kwargs = dict(
    frames=val_frames,
    rules=rules,
    timeframes=timeframes,
    windows=windows,
    decision_bar=decision_bar,
    target_levels=target_levels,
    episodes=episodes,
    seed=seed,
)
best = audit_model(best_model, **kwargs)
latest = audit_model(latest_model, **kwargs)
best_meta = load_json(paths["best_metrics"], {}) or {}

payload = {
    "generation": 2,
    "student": student_id,
    "student_mode": "GEN2_RAW_MTF_DISCRETE",
    "validation_source": "GEN1_FROZEN_VALIDATION",
    "episodes_same_for_both": episodes,
    "best_checkpoint_timesteps": best_meta.get("total_timesteps"),
    "best": best,
    "latest": latest,
    "comparison": compare_audits(best, latest),
    "note": "Validation-only behavioral autopsy. FINAL TEST remains untouched.",
}
write_json(paths["autopsy_json"], payload)

fields = ["checkpoint"] + sorted(set(best) | set(latest))
with paths["autopsy_csv"].open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow({"checkpoint": "BEST", **best})
    w.writerow({"checkpoint": "LATEST", **latest})

print(json.dumps(payload, indent=2))
