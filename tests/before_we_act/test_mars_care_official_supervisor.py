from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _assert_single_shot(config: str) -> None:
    assert "autostart=false" in config
    assert "autorestart=false" in config
    assert "startretries=0" in config
    assert "stopasgroup=true" in config
    assert "killasgroup=true" in config


def test_official_mars_care_pipeline_is_single_shot() -> None:
    config = (
        ROOT
        / "scripts/before_we_act/mars_care_official_pipeline.supervisor.conf"
    ).read_text(encoding="utf-8")

    _assert_single_shot(config)


def test_legacy_mars_care_entrypoint_cannot_autostart_or_retry() -> None:
    config = (
        ROOT / "deployment/mars_care/mars-care-supervisor.conf"
    ).read_text(encoding="utf-8")

    _assert_single_shot(config)
