from __future__ import annotations

import csv
import json
from pathlib import Path

root = Path("cohort-artifacts")
summaries = []
for p in root.rglob("student_summary.json"):
    try:
        summaries.append(json.loads(p.read_text(encoding="utf-8")))
    except Exception as exc:
        print(f"Skipping {p}: {exc}")


def score(x):
    b = x.get("best_validation") or {}
    return (
        float(b.get("median_return_pct", -1e9)),
        float(b.get("profit_factor", -1e9)),
        -float(b.get("median_max_drawdown_pct", 1e9)),
    )

summaries.sort(key=score, reverse=True)
out_dir = Path("reports/cohort")
out_dir.mkdir(parents=True, exist_ok=True)

leaderboard = {
    "students": len(summaries),
    "master_candidates": [x["student"] for x in summaries if x.get("status") == "MASTER_CANDIDATE"],
    "ranking": summaries,
    "final_exam": "LOCKED",
}
(out_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

fields = [
    "rank", "student", "status", "total_timesteps", "candidate_streak",
    "best_step", "best_median_return_pct", "best_profit_factor", "best_median_max_drawdown_pct",
    "best_profitable_episode_ratio",
]
with (out_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for rank, x in enumerate(summaries, 1):
        b = x.get("best_validation") or {}
        w.writerow({
            "rank": rank,
            "student": x.get("student"),
            "status": x.get("status"),
            "total_timesteps": x.get("total_timesteps", 0),
            "candidate_streak": x.get("candidate_streak", 0),
            "best_step": b.get("total_timesteps"),
            "best_median_return_pct": b.get("median_return_pct"),
            "best_profit_factor": b.get("profit_factor"),
            "best_median_max_drawdown_pct": b.get("median_max_drawdown_pct"),
            "best_profitable_episode_ratio": b.get("profitable_episode_ratio"),
        })

print("TRADING ZERO ACADEMY — COHORT LEADERBOARD")
for rank, x in enumerate(summaries, 1):
    b = x.get("best_validation") or {}
    print(
        f"#{rank:02d} Student {int(x.get('student', 0)):03d} | {x.get('status')} | "
        f"steps={int(x.get('total_timesteps', 0)):,} | "
        f"best@{int(b.get('total_timesteps') or 0):,} | "
        f"median={float(b.get('median_return_pct', 0.0)):.2f}% | "
        f"PF={float(b.get('profit_factor', 0.0)):.3f} | "
        f"DD={float(b.get('median_max_drawdown_pct', 0.0)):.2f}%"
    )
print(f"MASTER_CANDIDATES: {leaderboard['master_candidates'] or 'none'}")
print("FINAL EXAM: LOCKED")
