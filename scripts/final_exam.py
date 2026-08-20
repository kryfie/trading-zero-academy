from __future__ import annotations

import json
import os

from stable_baselines3 import PPO

from academy.config import load_config, ROOT
from academy.cohort import student_paths, load_json
from academy.dataset import (
    load_frames,
    ensure_split_manifest,
    build_academy_partitions,
    SPLIT_MANIFEST_NAME,
)
from academy.mtf_dataset import load_mtf_frames, build_mtf_partitions
from academy.evaluate import evaluate_model, evaluate_mtf_model

cfg = load_config()
student_id = int(os.environ.get("STUDENT_ID", "1"))
p = student_paths(ROOT, student_id)

# FINAL is only for a pre-frozen 3x MASTER_CANDIDATE checkpoint.
if not p["master_candidate"].exists():
    raise SystemExit(
        f"FINAL TEST LOCKED for Student #{student_id}: no frozen 3x MASTER_CANDIDATE checkpoint exists."
    )

master_meta = load_json(p["master_candidate_metrics"], {}) or {}
required_streak = int(cfg["learner"].get("marathon_master_candidate_streak", 3))
if int(master_meta.get("candidate_streak", 0)) < required_streak:
    raise SystemExit("FINAL TEST LOCKED: candidate streak requirement not satisfied.")

m = cfg["market"]
e = cfg["evaluation"]
mode = master_meta.get(
    "student_mode",
    "M5_ONLY_BASELINE" if student_id == 1 else "RAW_MTF",
)

legacy_dir = ROOT / "data" / "processed"
legacy_frames = load_frames(legacy_dir, m["symbols"], m["bar"])
manifest = ensure_split_manifest(
    legacy_frames,
    float(e["train_fraction"]),
    float(e["validation_fraction"]),
    legacy_dir / SPLIT_MANIFEST_NAME,
)

model = PPO.load(p["master_candidate"])
episodes = max(50, int(e["validation_episodes"]))
seed = 900_000 + student_id * 10_000

if mode == "RAW_MTF":
    timeframes = list(m["multi_timeframes"])
    windows = {str(k): int(v) for k, v in m["multi_timeframe_windows"].items()}
    decision_bar = str(m["decision_bar"])

    mtf_frames = load_mtf_frames(
        ROOT / "data" / "mtf",
        m["symbols"],
        timeframes,
    )
    _, _, frozen_final_frames, _, _ = build_mtf_partitions(
        mtf_frames,
        manifest,
        decision_bar=decision_bar,
        live_shadow_days=int(e["live_shadow_days"]),
        train_maturation_days=int(e["train_maturation_days"]),
        min_rolling_validation_bars=int(e["min_rolling_validation_bars"]),
    )
    metrics = evaluate_mtf_model(
        model,
        frozen_final_frames,
        cfg["world_rules"],
        timeframes=timeframes,
        windows=windows,
        decision_bar=decision_bar,
        episodes=episodes,
        seed=seed,
    )
else:
    _, _, frozen_final_frames, _, _ = build_academy_partitions(
        legacy_frames,
        manifest,
        live_shadow_days=int(e["live_shadow_days"]),
        train_maturation_days=int(e["train_maturation_days"]),
        min_rolling_validation_bars=int(e["min_rolling_validation_bars"]),
    )
    metrics = evaluate_model(
        model,
        frozen_final_frames,
        cfg["world_rules"],
        episodes,
        seed=seed,
    )

out = p["reports"] / "final_exam.json"
out.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "student": student_id,
    "student_mode": mode,
    "examined_checkpoint": "master_candidate.zip",
    "candidate_metadata": master_meta,
    "final_metrics": metrics,
    "warning": "Do not reuse this frozen FINAL result to tune the student or Academy rules.",
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
