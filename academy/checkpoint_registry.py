from __future__ import annotations

import json
from pathlib import Path


def validation_sort_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics.get("median_return_pct", float("-inf"))),
        float(metrics.get("profit_factor", float("-inf"))),
        -float(metrics.get("median_max_drawdown_pct", float("inf"))),
    )


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("checkpoints", []))
    except Exception:
        return []


def update_validation_registry(
    model,
    metrics: dict,
    total_timesteps: int,
    registry_path: str | Path,
    registry_dir: str | Path,
    top_k: int = 5,
    metadata: dict | None = None,
) -> list[dict]:
    registry_path = Path(registry_path)
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    top_k = max(1, int(top_k))
    step = int(total_timesteps)
    entries = [e for e in _load(registry_path) if int(e.get("total_timesteps", -1)) != step]
    entry = {
        **(metadata or {}),
        **metrics,
        "total_timesteps": step,
        "checkpoint": f"validation_{step}.zip",
    }
    ranked = sorted(entries + [entry], key=validation_sort_key, reverse=True)[:top_k]
    retained_steps = {int(e["total_timesteps"]) for e in ranked}
    if step in retained_steps:
        model.save(registry_dir / f"validation_{step}")
        (registry_dir / f"validation_{step}.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    for p in registry_dir.glob("validation_*.zip"):
        try:
            pstep = int(p.stem.split("_")[-1])
        except Exception:
            continue
        if pstep not in retained_steps:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
    payload = {"top_k": top_k, "checkpoints": ranked}
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ranked
