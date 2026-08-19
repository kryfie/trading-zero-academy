import json

from stable_baselines3 import PPO
from academy.config import load_config, ROOT
from academy.dataset import (
    load_frames,
    ensure_split_manifest,
    build_academy_partitions,
    SPLIT_MANIFEST_NAME,
)
from academy.evaluate import evaluate_model

cfg = load_config()
unlock = ROOT / cfg["evaluation"]["final_test_unlock_file"]
if not unlock.exists():
    raise SystemExit(
        "FINAL TEST LOCKED. Create reports/FINAL_TEST_UNLOCK only when you intentionally want to consume the final holdout. "
        "Do not run this repeatedly to tune the learner."
    )

m = cfg["market"]
e = cfg["evaluation"]
data_dir = ROOT / "data" / "processed"
frames = load_frames(data_dir, m["symbols"], m["bar"])
manifest = ensure_split_manifest(
    frames,
    float(e["train_fraction"]),
    float(e["validation_fraction"]),
    data_dir / SPLIT_MANIFEST_NAME,
)
_, _, frozen_final_frames, _, _ = build_academy_partitions(
    frames,
    manifest,
    live_shadow_days=int(e["live_shadow_days"]),
    train_maturation_days=int(e["train_maturation_days"]),
    min_rolling_validation_bars=int(e["min_rolling_validation_bars"]),
)
model = PPO.load(ROOT / cfg["runtime"]["checkpoint_path"])
metrics = evaluate_model(model, frozen_final_frames, cfg["world_rules"], max(50, int(e["validation_episodes"])), seed=900000)
out = ROOT / "reports" / "final_exam.json"
out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
