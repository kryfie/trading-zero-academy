from __future__ import annotations

import json
from pathlib import Path


def student_id_str(student_id: int | str) -> str:
    return f"{int(student_id):03d}"


def student_seed(base_seed: int, student_id: int) -> int:
    """Stable independent seed per student."""
    return int(base_seed) + int(student_id) * 1_000_003


def student_root(root: Path, student_id: int | str) -> Path:
    return Path(root) / "students" / student_id_str(student_id)


def student_paths(root: Path, student_id: int | str) -> dict[str, Path]:
    base = student_root(root, student_id)
    models = base / "models"
    reports = base / "reports"
    return {
        "root": base,
        "models": models,
        "reports": reports,
        "latest": models / "latest.zip",
        "best": models / "best_validation.zip",
        "best_metrics": models / "best_validation.json",
        "master_candidate": models / "master_candidate.zip",
        "master_candidate_metrics": models / "master_candidate.json",
        "state": models / "training_state.json",
        "history": reports / "learning_history.csv",
        "status": reports / "status.json",
        "progress": reports / "progress.json",
        "autopsy_json": reports / "policy_autopsy.json",
        "autopsy_csv": reports / "policy_autopsy.csv",
        "summary": reports / "student_summary.json",
    }


def ensure_student_dirs(paths: dict[str, Path]) -> None:
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default=None):
    if not Path(path).exists():
        return default
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
