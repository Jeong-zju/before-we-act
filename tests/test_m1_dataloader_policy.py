from __future__ import annotations

import pytest
import torch

from scripts.train_multimodal_wam import _data_loader_kwargs, _input_pipeline_evidence


@pytest.mark.parametrize(
    ("device", "pin_memory"),
    (("cpu", False), ("cuda:0", True)),
)
def test_zero_worker_loader_policy_omits_worker_only_kwargs(
    device: str,
    pin_memory: bool,
) -> None:
    config = {
        "training": {
            "num_workers": 0,
            "prefetch_factor": 8,
            "persistent_workers": True,
        }
    }

    assert _data_loader_kwargs(config, torch.device(device)) == {
        "num_workers": 0,
        "pin_memory": pin_memory,
    }


@pytest.mark.parametrize(
    ("device", "pin_memory"),
    (("cpu", False), ("cuda:0", True)),
)
def test_multi_worker_loader_policy_uses_configured_prefetch_and_persistence(
    device: str,
    pin_memory: bool,
) -> None:
    config = {
        "training": {
            "num_workers": 6,
            "validation_num_workers": 2,
            "pair_num_workers": 3,
            "prefetch_factor": 3,
            "persistent_workers": True,
        }
    }

    assert _data_loader_kwargs(config, torch.device(device)) == {
        "num_workers": 6,
        "pin_memory": pin_memory,
        "prefetch_factor": 3,
        "persistent_workers": True,
    }


def test_loader_roles_bound_persistent_worker_counts() -> None:
    config = {
        "model": {"vision_encoder_batch_size": 16},
        "training": {
            "batch_size": 64,
            "num_workers": 6,
            "validation_num_workers": 2,
            "pair_num_workers": 3,
            "prefetch_factor": 2,
            "persistent_workers": True,
        },
    }
    device = torch.device("cuda:0")

    assert _data_loader_kwargs(config, device, role="validation")["num_workers"] == 2
    assert _data_loader_kwargs(config, device, role="pair")["num_workers"] == 3
    evidence = _input_pipeline_evidence(config, device)
    assert evidence["vision_encoder_batch_size"] == 16
    assert evidence["train_loader"]["num_workers"] == 6
    assert evidence["validation_loader"]["num_workers"] == 2
    assert evidence["pair_loader"]["num_workers"] == 3
    assert evidence["project_unused_fields"] is True
    with pytest.raises(ValueError, match="unknown M1 DataLoader role"):
        _data_loader_kwargs(config, device, role="unknown")
