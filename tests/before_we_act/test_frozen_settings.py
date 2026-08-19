from __future__ import annotations

import json
from pathlib import Path

import pytest

from before_we_act.frozen_settings import (
    DEFAULT_SETTINGS_PATH,
    EXPECTED_TASKS,
    load_frozen_settings,
    validate_frozen_settings,
)


def test_repository_reproduction_settings_are_canonical() -> None:
    settings = load_frozen_settings()
    assert tuple(settings["tasks"]) == EXPECTED_TASKS
    assert settings["robofactory"]["sensor"] == {
        "shader_pack": "default",
        "width": 640,
        "height": 480,
    }
    assert settings["care"]["candidate_ids"] == list(range(6))


def test_settings_reject_task_or_runtime_drift(tmp_path: Path) -> None:
    settings = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["tasks"]["place_food"]["max_steps"] = 501
    with pytest.raises(ValueError, match="max_steps"):
        validate_frozen_settings(settings)

    settings = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["robofactory"]["control_mode"] = "ee_pose"
    with pytest.raises(ValueError, match="control_mode"):
        validate_frozen_settings(settings)


def test_settings_file_is_json_and_has_no_result_fields() -> None:
    value = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    serialized = json.dumps(value)
    for forbidden in ("override_rate", "successes", "p_value", "p=", "conclusion"):
        assert forbidden not in serialized.lower()
