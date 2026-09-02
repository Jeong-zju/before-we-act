from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_rollout_diagnostic_is_single_shot_and_preregistered() -> None:
    config = (ROOT / "deployment/mars-care-rollout-diagnostic.supervisor.conf").read_text()
    script = (ROOT / "scripts/before_we_act/run_mars_care_rollout_diagnostic.sh").read_text()
    assert "autostart=false" in config
    assert "autorestart=false" in config
    assert "startretries=0" in config
    assert "stopasgroup=true" in config and "killasgroup=true" in config
    assert "automatic_retry\": False" in script
    assert "validation20_used_for_tuning\": False" in script
    assert "20260910 20260911 20260912 20260913" in script
    assert "validation20_seed_range\": [20260827, 20260846]" in script
    assert "refusing to overwrite incomplete rollout evidence" in script
    assert "modes=(selector_off care)" in script
