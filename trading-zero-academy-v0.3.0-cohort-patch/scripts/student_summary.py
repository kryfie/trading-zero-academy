from __future__ import annotations

import json
import os

from academy.config import ROOT
from academy.cohort import student_paths, load_json, write_json

sid = int(os.environ["STUDENT_ID"])
p = student_paths(ROOT, sid)
status = load_json(p["status"], {}) or {}
best = load_json(p["best_metrics"], {}) or {}
autopsy = load_json(p["autopsy_json"], {}) or {}
master = load_json(p["master_candidate_metrics"], None)

out = {
    "student": sid,
    "student_seed": status.get("student_seed"),
    "status": status.get("status", "UNKNOWN"),
    "total_timesteps": status.get("total_timesteps", 0),
    "candidate_streak": status.get("candidate_streak", 0),
    "current_validation": status.get("validation", {}),
    "best_validation": best,
    "master_candidate": master,
    "autopsy_comparison": autopsy.get("comparison", {}),
}
write_json(p["summary"], out)
print(json.dumps(out, indent=2))
