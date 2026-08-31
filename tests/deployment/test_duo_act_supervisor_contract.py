from __future__ import annotations

import json
from pathlib import Path

from deployment.duo_act import supervisor as sup


def _touch_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_supervisor_formal_stage_uses_frozen_causal_run_and_open30(monkeypatch, tmp_path: Path):
    run_root = tmp_path / "duobench-act"
    data_root = run_root / "data_unclipped"
    experiment = run_root / "causal_lag1_prior"
    data_root.mkdir(parents=True)
    _touch_json(data_root / "manifest.json", {"schema": "fixture"})
    _touch_json(run_root / "audit_unclipped.json", {"passed": True})
    _touch_json(run_root / "preflight_state_binary.json", {"passed": True})
    (run_root / "smoke" / "final.pt").parent.mkdir(parents=True)
    (run_root / "smoke" / "final.pt").write_bytes(b"smoke")
    _touch_json(run_root / "smoke" / "validation" / "summary.json", {"status": "complete"})
    # A stale artifact must not satisfy the formal stage.
    (run_root / "formal").mkdir(parents=True)
    (run_root / "formal" / "final.pt").write_bytes(b"stale")

    monkeypatch.setattr(sup, "RUN", run_root)
    monkeypatch.setattr(sup, "DATA", data_root)
    monkeypatch.setattr(sup, "EXPERIMENT", experiment)
    monkeypatch.setattr(sup, "STATUS", run_root / "status.json")
    monkeypatch.setattr(sup, "LOG", run_root / "logs")
    captured: list[tuple[str, list[str]]] = []

    def fake_run(stage: str, command: list[str], retries: int = 1):
        del retries
        captured.append((stage, command))
        if stage == "formal_train":
            experiment.mkdir(parents=True, exist_ok=True)
            (experiment / "final.pt").write_bytes(b"formal")
        elif stage == "validation20":
            _touch_json(experiment / "validation20_open30" / "summary.json", {"status": "complete"})

    monkeypatch.setattr(sup, "run", fake_run)
    sup.main()

    stages = [stage for stage, _ in captured]
    assert stages == ["formal_train", "validation20"]
    formal = captured[0][1]
    assert formal[-2:] == ["--config", str(sup.FROZEN_CONFIG)]
    assert "120000" not in formal
    validation = captured[1][1]
    assert str(data_root) in validation
    assert str(experiment / "final.pt") in validation
    assert validation[-2:] == ["--mode", "open30"]
    status = json.loads((run_root / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["checkpoint"] == str(experiment / "final.pt")
