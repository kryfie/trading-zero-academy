from pathlib import Path
import json

from stable_baselines3 import PPO
from academy.config import load_config, ROOT
from academy.dataset import load_frames, split_frames
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
frames = load_frames(ROOT / "data" / "processed", m["symbols"], m["bar"])
_, _, test_frames = split_frames(frames, float(e["train_fraction"]), float(e["validation_fraction"]))
model = PPO.load(ROOT / cfg["runtime"]["checkpoint_path"])
metrics = evaluate_model(model, test_frames, cfg["world_rules"], max(50, int(e["validation_episodes"])), seed=900000)
out = ROOT / "reports" / "final_exam.json"
out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
