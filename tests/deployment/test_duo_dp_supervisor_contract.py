from __future__ import annotations

import json
from pathlib import Path

from deployment.duo_dp import supervisor as sup


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "duobench_dp_formal_v1.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_supervisor_launches_formal_and_validation_from_frozen_config(monkeypatch, tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    run_root = tmp_path / "duobench-dp"
    data_root = run_root / "data"
    data_root.mkdir(parents=True)
    _write_json(data_root / "manifest.json", {"schema": "fixture"})
    _write_json(run_root / "preflight.json", {"passed": True})
    _write_json(run_root / "smoke" / "status.json", {"status": "complete", "step": 10})
    (run_root / "smoke" / "final.pt").write_bytes(b"smoke")
    _write_json(run_root / "smoke" / "validation" / "summary.json", {"status": "complete", "total_episodes": 11})

    monkeypatch.setattr(sup, "CONFIG", CONFIG_PATH)
    monkeypatch.setattr(sup, "RUN", run_root)
    monkeypatch.setattr(sup, "DATA", data_root)
    monkeypatch.setattr(sup, "STATUS", run_root / "status.json")
    monkeypatch.setattr(sup, "LOG", run_root / "logs")
    captured: list[tuple[str, list[str]]] = []

    def fake_run(stage: str, command: list[str], retries: int = 3) -> None:
        del retries
        captured.append((stage, command))
        if stage == "formal_train":
            (run_root / "formal").mkdir(parents=True, exist_ok=True)
            (run_root / "formal" / "final.pt").write_bytes(b"formal")
            _write_json(run_root / "formal" / "status.json", {"status": "complete", "step": config["optimization"]["updates"]})
        elif stage == "validation20":
            _write_json(run_root / "formal" / "validation20" / "summary.json", {"status": "complete", "total_episodes": 220})
        elif stage == "finalize":
            _write_json(run_root / "final_report.json", {"status": "complete"})

    monkeypatch.setattr(sup, "run", fake_run)
    sup.main()

    assert [stage for stage, _ in captured] == ["formal_train", "validation20", "finalize"]
    formal = captured[0][1]
    expected_formal = {
        "--steps": config["optimization"]["updates"],
        "--batch-size": config["optimization"]["batch_size"],
        "--workers": config["loader"]["formal_workers"],
        "--save-every": config["checkpointing"]["save_every_updates"],
        "--seed": config["optimization"]["seed"],
        "--learning-rate": config["optimization"]["optimizer"]["learning_rate"],
        "--warmup": config["optimization"]["scheduler"]["warmup_updates"],
        "--transition-fraction": config["optimization"]["transition_fraction"],
        "--gripper-loss-weight": config["optimization"]["gripper_loss_weight"],
    }
    for flag, expected in expected_formal.items():
        assert formal[formal.index(flag) + 1] == str(expected)
    assert "--resume" in formal
    assert "--task-conditioning" not in formal

    validation = captured[1][1]
    expected_validation = {
        "--episodes": config["validation20"]["episodes_per_task"],
        "--workers": config["validation20"]["workers"],
        "--inference-steps": config["validation20"]["inference_steps"],
        "--weights": "ema",
        "--replan-steps": config["validation20"]["replan_interval"],
    }
    for flag, expected in expected_validation.items():
        assert validation[validation.index(flag) + 1] == str(expected)
