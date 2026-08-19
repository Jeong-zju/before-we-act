import importlib.util
from pathlib import Path

from before_we_act.frozen_settings import load_frozen_settings


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / "before_we_act" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_care_test_protocol_has_twenty_paired_episodes() -> None:
    settings = load_frozen_settings()
    assert settings["closed_loop"]["episodes_per_task"] == 20
    assert settings["closed_loop"]["modes"] == ["selector_off", "care"]
    assert settings["closed_loop"]["paired"] is True


def test_care_test_seeds_are_deterministic_and_task_specific() -> None:
    module = load_script("prepare_care_test_seeds.py")
    first = module.deterministic_seeds("fixed", "lift_barrier", 20)
    second = module.deterministic_seeds("fixed", "lift_barrier", 20)
    other = module.deterministic_seeds("fixed", "pass_shoe", 20)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert first != other
    assert all(0 < value < 2**31 for value in first)


def test_standard_runner_and_summary_expose_no_legacy_experiment_labels() -> None:
    runner = (ROOT / "scripts/before_we_act/run_care_robofactory.sh").read_text()
    summary = (ROOT / "scripts/before_we_act/summarize_care_tests.py").read_text()
    for legacy in ("confirmation50", "validation20", "a6r1", "b-core", "b_core"):
        assert legacy not in runner.lower()
        assert legacy not in summary.lower()
    assert "human_conclusion" not in summary
    assert "gate_" not in summary
