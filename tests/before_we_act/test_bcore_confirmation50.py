import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / "before_we_act" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_confirmation_seeds_are_deterministic_unique_and_excluded():
    module = load_script("prepare_bcore_confirmation50_seeds.py")
    excluded = {1, 2, 3}
    first = module.deterministic_seeds("fixed", "task", 50, excluded)
    second = module.deterministic_seeds("fixed", "task", 50, excluded)
    assert first == second
    assert len(first) == len(set(first)) == 50
    assert not excluded.intersection(first)
    assert all(0 < value < 2**31 for value in first)


def test_exact_mcnemar_is_two_sided():
    module = load_script("summarize_bcore_confirmation50.py")
    assert module.exact_mcnemar_two_sided(6, 0) == 0.03125
    assert module.exact_mcnemar_two_sided(0, 0) == 1.0
    assert module.exact_mcnemar_two_sided(3, 3) == 1.0


def test_paired_bootstrap_interval_tracks_direction():
    module = load_script("summarize_bcore_confirmation50.py")
    differences = np.asarray([1] * 30 + [0] * 270, dtype=np.int8)
    lower, upper = module.paired_bootstrap_ci(differences, samples=5000, seed=17)
    assert 0.0 < lower < upper
