from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v3_supervisor_is_single_shot_and_uses_all_four_gpus() -> None:
    config = (ROOT / "deployment/mars-care-v3-diagnostic.supervisor.conf").read_text()
    script = (ROOT / "scripts/before_we_act/run_mars_care_v3_diagnostic.sh").read_text()

    assert "autorestart=false" in config
    assert "startretries=0" in config
    assert "autostart=false" in config
    assert "stopasgroup=true" in config
    assert "killasgroup=true" in config
    assert "for gpu in 0 1 2 3" in script
    assert 'CUDA_VISIBLE_DEVICES="${gpu}"' in script
    assert "automatic_retry\": False" in script
    assert "validation20_used_for_tuning\": False" in script


def test_v3_supervisor_freezes_orthogonal_conditions_and_seeds() -> None:
    script = (ROOT / "scripts/before_we_act/run_mars_care_v3_diagnostic.sh").read_text()

    for condition in (
        "v2_control:0:0:0",
        "slot_only:1:0:0",
        "task_only:0:1:0",
        "slot_task_horizon:1:1:1",
    ):
        assert condition in script
    assert "20260904,20260905,20260906" in script
    assert "refusing to overwrite an incomplete diagnostic root" in script
    assert "expected exactly four visible GPUs" in script
