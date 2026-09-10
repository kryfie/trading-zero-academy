import json
from pathlib import Path

from academy.checkpoint_registry import update_validation_registry


class DummyModel:
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")
        path.write_text("dummy", encoding="utf-8")


def m(ret, pf=1.0, dd=20.0):
    return {"median_return_pct": ret, "profit_factor": pf, "median_max_drawdown_pct": dd}


def test_registry_keeps_only_top_k(tmp_path):
    model = DummyModel()
    reg = tmp_path / "registry.json"
    d = tmp_path / "checkpoints"
    update_validation_registry(model, m(1), 100, reg, d, top_k=2)
    update_validation_registry(model, m(3), 200, reg, d, top_k=2)
    update_validation_registry(model, m(2), 300, reg, d, top_k=2)
    payload = json.loads(reg.read_text(encoding="utf-8"))
    assert [x["total_timesteps"] for x in payload["checkpoints"]] == [200, 300]
    assert not (d / "validation_100.zip").exists()
    assert (d / "validation_200.zip").exists()
    assert (d / "validation_300.zip").exists()
