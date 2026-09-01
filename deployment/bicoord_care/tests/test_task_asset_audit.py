from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from deployment.bicoord_care.task_asset_audit import (
    CONTACT_KEY,
    DEFAULT_TASKS,
    SCHEMA,
    TaskAssetAuditError,
    audit_task_assets,
    main,
)


def _pose(offset: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, offset],
        [0.0, 0.0, 1.0, -offset],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _metadata(scale: float, contacts: list[object]) -> dict[str, object]:
    return {
        "center": [0.0, 0.0, 0.0],
        CONTACT_KEY: contacts,
        "functional_matrix": [_pose(0.1)],
        "scale": [scale, scale, scale],
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    benchmark = tmp_path / "benchmark"
    assets = tmp_path / "assets"
    (benchmark / "envs").mkdir(parents=True)
    source = '''
from .utils import *
class place_plate_and_cup:
    def load_actors(self):
        self.plate = create_actor(self, pose=None, modelname="003_plate")
        self.plate_2 = create_actor(self, pose=None, modelname="003_plate", model_id=0)
        self.cup = create_actor(self, pose=None, modelname="021_cup", model_id=cup_id)
    def play_once(self):
        self.move(self.grasp_actor(self.plate, contact_point_id=2))
        self.move(self.grasp_actor(self.plate_2, contact_point_id=2))
        self.move(self.place_actor(self.plate, functional_point_id=0))
        self.move(self.target.get_functional_point(0))
'''
    (benchmark / "envs" / "place_plate_and_cup.py").write_text(source, encoding="utf-8")
    # A dynamic task demonstrates that unsupported model references are listed,
    # not guessed or turned into false index failures.
    (benchmark / "envs" / "dynamic.py").write_text(
        '''
class dynamic:
    def load_actors(self):
        self.obj = create_actor(self, None, modelname=np.random.choice(names), model_id=np.random.randint(4))
    def play_once(self):
        self.move(self.grasp_actor(self.obj, contact_point_id=np.random.randint(4)))
''',
        encoding="utf-8",
    )
    small = assets / "objects" / "003_plate"
    large = assets / "objects" / "003_plate_large"
    _write_json(small / "model_data0.json", _metadata(0.025, []))
    _write_json(
        large / "model_data0.json",
        _metadata(0.035, [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)]),
    )
    cup = assets / "objects" / "021_cup"
    _write_json(cup / "model_data5.json", _metadata(0.02, [_pose(0.1)]))
    for relative, payload in (("collision/base0.glb", b"mesh"), ("visual/base0.glb", b"visual")):
        for directory in (small, large):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return benchmark, assets


def test_static_audit_extracts_literals_and_proves_plate_overlay(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report = audit_task_assets(benchmark, assets, tasks=["place_plate_and_cup"])

    assert report["schema"] == SCHEMA
    # The fixture intentionally omits the cup's functional metadata contract
    # and therefore this test focuses on the dedicated plate gate; the plate
    # itself must be proven before/after without writing the checkout.
    assert report["overlay_check"]["status"] == "PASSED"
    assert report["overlay_check"]["before"]["status"] == "FAILED"
    assert report["overlay_check"]["after"]["status"] == "PASSED"
    assert report["overlay_check"]["contact_point_id"] == 2
    assert report["overlay_check"]["target_scale_preserved"] is True
    assert report["overlay_check"]["only_contact_field_changed"] is True
    assert report["overlay_check"]["mesh_hashes_equal"] is True
    assert report["benchmark_files_written"] is False
    assert {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_dynamic_references_are_reported_without_guessing(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    report = audit_task_assets(benchmark, assets, tasks=["dynamic"])

    assert report["status"] == "PASSED"
    assert report["dynamic_item_count"] >= 2
    reasons = " ".join(row["reason"] for row in report["dynamic_items"])
    assert "static" in reasons
    assert report["overlay_check"]["status"] == "PASSED"


def test_metadata_mapping_override_is_effective_and_provenanced(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    overlay = _metadata(
        0.025,
        [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)],
    )
    report = audit_task_assets(
        benchmark,
        assets,
        tasks=["place_plate_and_cup"],
        metadata_overrides={("003_plate", 0): overlay},
    )

    assert report["status"] == "PASSED"
    assert report["expected_pristine_defect_count"] == 0
    plate_rows = [
        row
        for row in report["task_reports"][0]["interactions"]
        if row["actor"] in {"plate", "plate_2"} and row["kind"] == "grasp_actor"
    ]
    assert len(plate_rows) == 2
    assert all(row["override_used"] is True for row in plate_rows)
    assert all(row["status"] == "PASSED" for row in plate_rows)
    assert all(row["pristine_metadata_sha256"] for row in plate_rows)
    assert all(row["effective_metadata_sha256"] for row in plate_rows)
    assert all(row["override_key"] == {"modelname": "003_plate", "model_id": 0} for row in plate_rows)
    provenance = report["metadata_overrides"]
    assert len(provenance) == 1
    assert provenance[0]["source_type"] == "mapping"
    assert provenance[0]["source_path"] is None
    assert provenance[0]["source_sha256"] is None
    assert provenance[0]["used_by_interaction_count"] == 3
    assert provenance[0]["contract_status"] == "PASSED"


def test_metadata_file_override_records_source_hash_and_keeps_checkout_read_only(
    tmp_path: Path,
) -> None:
    benchmark, assets = _fixture(tmp_path)
    overlay_path = tmp_path / "run-artifact" / "003_plate_overlay.json"
    overlay = _metadata(
        0.025,
        [_pose(0.1), _pose(0.2), _pose(0.3), _pose(0.4)],
    )
    _write_json(overlay_path, overlay)
    before = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    report = audit_task_assets(
        benchmark,
        assets,
        tasks=["place_plate_and_cup"],
        metadata_overrides={("003_plate", 0): overlay_path},
    )
    assert report["status"] == "PASSED"
    record = report["metadata_overrides"][0]
    assert record["source_type"] == "file"
    assert record["source_path"] == str(overlay_path.resolve())
    assert record["source_sha256"] == hashlib.sha256(overlay_path.read_bytes()).hexdigest()
    assert record["pristine_metadata_sha256"]
    assert record["overlay_metadata_sha256"] == record["override_metadata_sha256"]
    assert {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before


def test_pristine_plate_defect_is_separate_from_unexpected_failures(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    report = audit_task_assets(benchmark, assets, tasks=["place_plate_and_cup"])
    assert report["status"] == "PASSED_WITH_EXPECTED_PRISTINE_DEFECT"
    assert report["expected_pristine_defect_count"] == 2
    assert report["unexpected_violation_count"] == 0
    assert report["task_reports"][0]["expected_pristine_defects"]
    assert report["task_reports"][0]["status"] == "FAILED"


def test_legacy_contact_field_is_dynamic_not_a_false_hard_failure(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    source_path = benchmark / "envs" / "place_plate_and_cup.py"
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "model_id=cup_id", "model_id=5"
        ).replace(
            "self.move(self.grasp_actor(self.plate, contact_point_id=2))",
            "self.move(self.grasp_actor(self.plate, contact_point_id=2))\n"
            "        self.move(self.grasp_actor(self.cup, contact_point_id=0))",
        ),
        encoding="utf-8",
    )
    cup_path = assets / "objects" / "021_cup" / "model_data5.json"
    cup = json.loads(cup_path.read_text(encoding="utf-8"))
    cup["contact_pose"] = cup.pop(CONTACT_KEY)
    _write_json(cup_path, cup)
    report = audit_task_assets(benchmark, assets, tasks=["place_plate_and_cup"])
    assert report["unexpected_violation_count"] == 0
    assert any(
        row["kind"] == "unsupported_metadata_field" and "contact_pose" in row["reason"]
        for row in report["dynamic_items"]
    )
    cup_interactions = [
        row
        for row in report["task_reports"][0]["interactions"]
        if row["actor"] in {"cup", "cup_2"} and row["kind"] == "grasp_actor"
    ]
    assert cup_interactions
    assert all(row["status"] == "UNRESOLVED" for row in cup_interactions)


def test_out_of_range_static_contact_is_a_hard_failure(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    source_path = benchmark / "envs" / "place_plate_and_cup.py"
    source_path.write_text(
        source_path.read_text().replace("contact_point_id=2", "contact_point_id=9"),
        encoding="utf-8",
    )
    report = audit_task_assets(benchmark, assets, tasks=["place_plate_and_cup"])

    assert report["status"] == "FAILED"
    assert any("index 9 out of range" in item for item in report["violations"])


def test_missing_task_is_blocked_and_unknown_task_list_is_rejected(tmp_path: Path) -> None:
    benchmark, assets = _fixture(tmp_path)
    report = audit_task_assets(benchmark, assets, tasks=["does_not_exist"])
    assert report["status"] == "FAILED"
    assert report["task_reports"][0]["status"] == "BLOCKED"
    with pytest.raises(TaskAssetAuditError, match="non-empty and unique"):
        audit_task_assets(benchmark, assets, tasks=[])
    with pytest.raises(TaskAssetAuditError, match="non-empty and unique"):
        audit_task_assets(benchmark, assets, tasks=["dynamic", "dynamic"])


def test_cli_writes_only_requested_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    benchmark, assets = _fixture(tmp_path)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--benchmark-root",
                str(benchmark),
                "--assets-root",
                str(assets),
                "--tasks",
                "dynamic",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["schema"] == SCHEMA
    assert "dynamic" in capsys.readouterr().out


def test_default_task_constant_has_the_frozen_eighteen_tasks() -> None:
    assert len(DEFAULT_TASKS) == 18
    assert len(set(DEFAULT_TASKS)) == 18
