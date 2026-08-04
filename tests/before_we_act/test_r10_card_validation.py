from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PARENT = "a" * 40


def _card():
    return {
        "schema_version": 1,
        "round": "R10",
        "candidate_id": "p0",
        "branch": "bwa/r10-p0-calibrated-crossview",
        "parent": {
            "branch": "bwa/r9-core-native",
            "commit": PARENT,
            "checkpoint_sha256": "061b7a4acea8fa10f146779e7a1206822179920dfe573db536d237df81eb541d",
        },
        "allowed_files": ["stereo_core/bwa_perception.py", "configs/before_we_act/r10_perception/p0.yaml"],
        "public_symbols": ["CalibratedUnalignedBridge"],
        "tensor_contracts": ["delta[B,1200,384]"],
        "losses": ["action_imitation"],
        "config_keys": ["bridge.kind"],
        "required_tests": ["test_crossview_not_patch_aligned.py"],
        "papers": [
            {"title": "one", "url": "https://example.test/one", "mechanism": "x", "non_claim": "y"},
            {"title": "two", "url": "https://example.test/two", "mechanism": "x", "non_claim": "y"},
        ],
        "acceptance": ["a", "b", "c", "d", "e"],
    }


def test_valid_card_passes_and_reports_stable_hash(tmp_path):
    path = tmp_path / "implementation_card.yaml"
    path.write_text(yaml.safe_dump(_card(), sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/before_we_act/validate_implementation_card.py"),
            str(path),
            "--expected-parent",
            PARENT,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert len(payload["sha256"]) == 64


def test_card_fails_closed_on_unknown_key_and_parent_drift(tmp_path):
    card = _card()
    card["parent"]["commit"] = "b" * 40
    card["surprise"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(card), encoding="utf-8")
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/before_we_act/validate_implementation_card.py"),
            str(path),
            "--expected-parent",
            PARENT,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert "parent commit differs from frozen round parent" in payload["failures"]
    assert any("unknown keys" in item for item in payload["failures"])
