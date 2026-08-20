from __future__ import annotations

import csv
import json
import os

from stable_baselines3 import PPO

from academy.config import load_config, ROOT
from academy.cohort import student_paths, ensure_student_dirs, load_json, write_json
from academy.dataset import load_frames, ensure_split_manifest, build_academy_partitions, SPLIT_MANIFEST_NAME
from academy.policy_audit import audit_model, compare_audits

cfg = load_config()
student_id = int(os.environ["STUDENT_ID"])
paths = student_paths(ROOT, student_id)
ensure_student_dirs(paths)

if not paths["latest"].exists() or not paths["best"].exists():
    print(f"Student #{student_id}: autopsy skipped; BEST and/or LATEST checkpoint missing.")
    raise SystemExit(0)

m = cfg["market"]
e = cfg["evaluation"]
data_dir = ROOT / "data" / "processed"
frames = load_frames(data_dir, m["symbols"], m["bar"])
manifest = ensure_split_manifest(frames, float(e["train_fraction"]), float(e["validation_fraction"]), data_dir / SPLIT_MANIFEST_NAME)
_, val_frames, _, _, validation_source = build_academy_partitions(
    frames, manifest,
    live_shadow_days=int(e["live_shadow_days"]),
    train_maturation_days=int(e["train_maturation_days"]),
    min_rolling_validation_bars=int(e["min_rolling_validation_bars"]),
)

episodes = int(e.get("autopsy_episodes", 10))
seed = 770_000 + student_id * 10_000
best_model = PPO.load(paths["best"])
latest_model = PPO.load(paths["latest"])
best = audit_model(best_model, val_frames, cfg["world_rules"], episodes=episodes, seed=seed)
latest = audit_model(latest_model, val_frames, cfg["world_rules"], episodes=episodes, seed=seed)
best_meta = load_json(paths["best_metrics"], {}) or {}

payload = {
    "student": student_id,
    "validation_source": validation_source,
    "episodes_same_for_both": episodes,
    "best_checkpoint_timesteps": best_meta.get("total_timesteps"),
    "best": best,
    "latest": latest,
    "comparison": compare_audits(best, latest),
    "note": "Autopsy uses validation only. FINAL TEST remains untouched. Evidence is descriptive, not causal.",
}
write_json(paths["autopsy_json"], payload)

fields = ["checkpoint"] + sorted(set(best) | set(latest))
with paths["autopsy_csv"].open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow({"checkpoint": "BEST", **best})
    w.writerow({"checkpoint": "LATEST", **latest})

print(json.dumps(payload, indent=2))
