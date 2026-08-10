import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from before_we_act.r11_vendor import verify_vendor_checkout


def test_official_qformer_and_flow_fixed_tensor_parity():
    pytest.importorskip("diffusers")
    pytest.importorskip("omegaconf")
    project_root = Path(__file__).resolve().parents[2]
    vendor = Path(
        os.environ.get("R11_LAWAM_VENDOR", "/home/jeong/zeno/wam/r11_upstream/LaWAM")
    )
    verify_vendor_checkout(project_root / "third_party/r11/lawam/SOURCE_RECEIPT.json", vendor)
    sys.path.insert(0, str(vendor))

    # The LAM package __init__ imports its Lightning training wrapper even
    # though this parity uses only the policy modules. Keep the local parity
    # environment small by stubbing that unused framework; remote F0 installs
    # and imports the pinned full dependency set.
    if "lightning" not in sys.modules:
        lightning = ModuleType("lightning")
        lightning.LightningModule = nn.Module
        lightning_pytorch = ModuleType("lightning.pytorch")
        lightning_pytorch.Callback = object
        lightning.pytorch = lightning_pytorch
        sys.modules["lightning"] = lightning
        sys.modules["lightning.pytorch"] = lightning_pytorch
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = ModuleType("wandb")

    from starVLA.model.framework.vlas.flowmatching_expert import (
        ConditionalFlowMatchingConfig,
        ConditionalFlowMatchingHead,
    )
    from starVLA.model.framework.vlas.lawam import VLMToLAMQFormer

    torch.manual_seed(260615768)
    qformer = VLMToLAMQFormer(
        vlm_hidden_dim=32, lam_code_dim=16, num_layers=1, num_heads=4
    )
    context = torch.linspace(-1, 1, 2 * 8 * 32).reshape(2, 8, 32)
    latent = qformer(context)
    assert latent.shape == (2, 1, 16)

    config = ConditionalFlowMatchingConfig(
        action_dim=8,
        hidden_dim=32,
        num_layers=2,
        attention_heads=4,
        vlm_dim=32,
        vision_dim=32,
        num_vision_tokens=4,
        state_dim=9,
        num_embodiments=4,
        horizon_sec=1.0,
        use_alternate_vldit=False,
    )
    flow = ConditionalFlowMatchingHead(config)
    flow.action_horizon = 4
    torch.manual_seed(260615768)
    loss = flow(
        h_t=torch.linspace(-0.5, 0.5, 2 * 4 * 32).reshape(2, 4, 32),
        h_t1_star=torch.linspace(0.5, -0.5, 2 * 4 * 32).reshape(2, 4, 32),
        h_vlm=torch.linspace(-1, 1, 2 * 5 * 32).reshape(2, 5, 32),
        state=torch.ones(2, 9),
        actions=torch.linspace(-1, 1, 2 * 4 * 8).reshape(2, 4, 8),
        action_hz=torch.full((2,), 4.0),
        embodiment_id=torch.tensor([1, 2]),
        state_mask=torch.ones(2, 9, dtype=torch.bool),
        actions_mask=torch.ones(2, 4, 8, dtype=torch.bool),
        attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    fingerprint = torch.stack((latent.float().mean(), latent.float().std(), loss.float()))
    expected = torch.tensor([2.2351742e-08, 1.0159819, 1.2768722])
    torch.testing.assert_close(fingerprint, expected, rtol=1e-5, atol=1e-6)
