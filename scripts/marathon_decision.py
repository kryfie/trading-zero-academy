from __future__ import annotations

import json
import os
from pathlib import Path

from academy.config import load_config, ROOT

cfg = load_config()
status_path = ROOT / cfg["runtime"]["status_path"]
if not status_path.exists():
    raise SystemExit("status.json missing; refusing to queue another marathon run")

s = json.loads(status_path.read_text(encoding="utf-8"))
m = s.get("marathon", {})
reached = bool(m.get("reached_target", False))
graduation = bool(s.get("graduation_ready", False))
continue_marathon = not reached and not graduation

target = int(m.get("target_total_timesteps", cfg["learner"].get("marathon_target_total_timesteps", 123000000)))
current = int(s.get("total_timesteps", 0))

print(f"current={current:,} target={target:,} reached={reached} graduation_ready={graduation}")
print(f"continue={str(continue_marathon).lower()}")

out = os.environ.get("GITHUB_OUTPUT")
if out:
    with open(out, "a", encoding="utf-8") as f:
        f.write(f"continue={str(continue_marathon).lower()}\n")
        f.write(f"target={target}\n")
        f.write(f"current={current}\n")
        f.write(f"graduation_ready={str(graduation).lower()}\n")
