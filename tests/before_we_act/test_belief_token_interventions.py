from pathlib import Path
import importlib.util

import torch


def load_module():
    path = Path(__file__).parents[2] / "scripts/before_we_act/evaluate_belief_token_interventions.py"
    spec = importlib.util.spec_from_file_location("belief_token_interventions", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_interventions_only_change_predefined_token_groups() -> None:
    module = load_module()
    mu = torch.arange(4 * 5 * 3, dtype=torch.float32).view(4, 5, 3)
    permutation = torch.tensor([1, 0, 3, 2])

    teammate = module.intervene_mu(mu, permutation, "teammate_anchor_shuffle", 2)
    assert torch.equal(teammate[:, 0], mu[:, 0])
    assert torch.equal(teammate[:, 1], mu[permutation, 1])
    assert torch.equal(teammate[:, 2:], mu[:, 2:])

    interaction = module.intervene_mu(mu, permutation, "interaction_slots_shuffle", 2)
    assert torch.equal(interaction[:, :2], mu[:, :2])
    assert torch.equal(interaction[:, 2:], mu[permutation, 2:])

    ego = module.intervene_mu(mu, permutation, "ego_anchor_shuffle", 2)
    assert torch.equal(ego[:, 0], mu[permutation, 0])
    assert torch.equal(ego[:, 1:], mu[:, 1:])


def test_mechanism_gate_requires_direction_ci_and_task_breadth() -> None:
    module = load_module()
    passed = {
        "mse_increase": 1e-4,
        "episode_bootstrap_ci95": [1e-6, 2e-4],
        "positive_tasks": 4,
    }
    assert module.mechanism_gate(passed)
    assert not module.mechanism_gate(passed | {"positive_tasks": 3})
    assert not module.mechanism_gate(
        passed | {"episode_bootstrap_ci95": [-1e-6, 2e-4]}
    )
