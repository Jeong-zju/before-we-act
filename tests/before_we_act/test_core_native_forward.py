from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[2]


class _FakeProcessor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.25, 0.25, 0.25)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()


class _FakeVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8, num_register_tokens=0)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        # Deterministic frozen tokens; spatial variation keeps fusion nontrivial.
        base = torch.linspace(
            -1.0, 1.0, 1201, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, 1201, 1)
        channel = torch.arange(8, device=pixel_values.device, dtype=pixel_values.dtype)
        signal = pixel_values.mean((1, 2, 3)).view(batch, 1, 1)
        return types.SimpleNamespace(last_hidden_state=base + channel + signal)


@pytest.fixture(autouse=True)
def fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.AutoImageProcessor = _FakeProcessor
    module.AutoModel = _FakeVision
    monkeypatch.setitem(sys.modules, "transformers", module)
    torchvision = types.ModuleType("torchvision")
    models = types.ModuleType("torchvision.models")
    models.resnet18 = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("resnet18 is outside this DINO-only test")
    )
    torchvision.models = models
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.models", models)


def _load_upstream_class():
    upstream = ROOT / "vendor/stereo-core/stereo_core"
    sys.path.insert(0, str(upstream))
    try:
        spec = importlib.util.spec_from_file_location(
            "r9_upstream_no_wrist_pair_model", upstream / "no_wrist_pair_model.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NoWristPAIRRoute
    finally:
        sys.path.remove(str(upstream))


def _models():
    from stereo_core.no_wrist_pair_model import NoWristPAIRRoute

    upstream = _load_upstream_class()
    kwargs = dict(
        state_dim=9,
        action_dim=8,
        horizon=3,
        d_model=8,
        enc_layers=1,
        dec_layers=1,
        roles=4,
        role_rank=4,
        dino_model="offline/fake",
    )
    torch.manual_seed(1907)
    reference = upstream(**kwargs)
    torch.manual_seed(1907)
    native = NoWristPAIRRoute(**kwargs)
    native.load_state_dict(reference.state_dict(), strict=True)
    return reference, native


def _inputs(batch=1):
    generator = torch.Generator().manual_seed(77)
    global_rgb = torch.rand(batch, 3, 480, 640, generator=generator)
    local_rgb = torch.rand(batch, 3, 480, 640, generator=generator)
    qpos = torch.rand(batch, 9, generator=generator)
    actions = torch.rand(batch, 3, 8, generator=generator)
    return global_rgb, local_rgb, qpos, actions


def _assert_tuple_exact(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_deployment_forward_route_and_signature_are_bit_exact():
    reference, native = _models()
    reference.eval()
    native.eval()
    global_rgb, local_rgb, qpos, _ = _inputs()
    with torch.no_grad():
        expected = reference(global_rgb, local_rgb, qpos, return_routing=True)
        actual = native(global_rgb, local_rgb, qpos, return_routing=True)
    _assert_tuple_exact(expected, actual)
    torch.testing.assert_close(reference.last_dense_routes, native.last_dense_routes, rtol=0, atol=0)
    torch.testing.assert_close(reference.last_sparse_routes, native.last_sparse_routes, rtol=0, atol=0)


def test_training_forward_and_rng_after_state_are_bit_exact():
    reference, native = _models()
    reference.train()
    native.train()
    global_rgb, local_rgb, qpos, actions = _inputs()
    rng = torch.Generator().manual_seed(9281).get_state()
    torch.set_rng_state(rng)
    expected = reference(
        global_rgb,
        local_rgb,
        qpos,
        actions,
        return_routing=True,
        counterfactual=True,
    )
    expected_after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = native(
        global_rgb,
        local_rgb,
        qpos,
        actions,
        return_routing=True,
        counterfactual=True,
    )
    actual_after = torch.get_rng_state().clone()
    _assert_tuple_exact(expected, actual)
    torch.testing.assert_close(expected_after, actual_after, rtol=0, atol=0)


def test_candidate_bank_is_full_batch_and_candidate_zero_is_native():
    _reference, native = _models()
    native.eval()
    global_rgb, local_rgb, qpos, _ = _inputs(batch=2)
    with torch.no_grad():
        prediction = native(global_rgb, local_rgb, qpos)[0]
        bank = native.propose_core_bank(global_rgb, local_rgb, qpos)
    assert bank.chunks.shape == (2, 5, 3, 8)
    assert bank.routes.shape == (2, 5, 3, 4)
    assert bank.valid_mask.all()
    torch.testing.assert_close(bank.chunks[:, 0], prediction, rtol=0, atol=0)


def test_r9_adds_no_parent_state_dict_entries():
    reference, native = _models()
    assert tuple(reference.state_dict()) == tuple(native.state_dict())
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(expected, native.state_dict()[name], rtol=0, atol=0)


def test_default_deployment_context_is_exact_and_privileged_keys_fail_closed():
    from stereo_core.bwa_contracts import CoreDeploymentContext

    _reference, native = _models()
    native.eval()
    global_rgb, local_rgb, qpos, _ = _inputs()
    with torch.no_grad():
        default = native(global_rgb, local_rgb, qpos)
        explicit = native(
            global_rgb,
            local_rgb,
            qpos,
            deployment_context=CoreDeploymentContext(),
        )
    _assert_tuple_exact(default, explicit)
    with pytest.raises(ValueError, match="privileged"):
        CoreDeploymentContext(fixed_camera_metadata={"simulator_state": object()})
    with pytest.raises(ValueError, match="unknown deployment metadata"):
        CoreDeploymentContext(fixed_camera_metadata={"sim_hint": 1})
    legal = CoreDeploymentContext(
        fixed_camera_metadata={
            "diagnostic_intervention": "normal",
            "calibration_sha256": "fixed",
        }
    )
    assert legal.fixed_camera_metadata["diagnostic_intervention"] == "normal"
