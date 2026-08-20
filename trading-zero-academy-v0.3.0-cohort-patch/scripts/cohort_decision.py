from __future__ import annotations

import json
import os
from pathlib import Path

root = Path("cohort-artifacts")
target = int(os.environ.get("COHORT_TARGET_TIMESTEPS", "500000000"))
student_count = int(os.environ.get("COHORT_STUDENT_COUNT", "8"))
auto_continue = os.environ.get("COHORT_AUTO_CONTINUE", "true").lower() == "true"

summaries = []
for p in root.rglob("student_summary.json"):
    try:
        summaries.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        pass

by_id = {int(x.get("student", 0)): x for x in summaries}
unfinished = []
for sid in range(1, student_count + 1):
    s = by_id.get(sid)
    if s is None or int(s.get("total_timesteps", 0)) < target:
        unfinished.append(sid)

cont = bool(auto_continue and unfinished)

out = Path(os.environ.get("GITHUB_OUTPUT", "/tmp/github_output"))
with out.open("a", encoding="utf-8") as f:
    f.write(f"continue={'true' if cont else 'false'}\n")
    f.write(f"target={target}\n")
    f.write(f"unfinished={','.join(map(str, unfinished))}\n")

print(f"Target per student: {target:,}")
print(f"Expected students: {student_count}")
print(f"Student summaries found: {len(summaries)}")
print(f"Unfinished/missing students: {unfinished}")
print(f"Auto-continue: {cont}")
