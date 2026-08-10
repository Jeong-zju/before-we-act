import torch
from torch import nn
from types import SimpleNamespace

from before_we_act.r11_lawam_subgoal_flow import (
    ACTION_HORIZON,
    LaWAMRoboFactoryAdapter,
    R11LaWAMSubgoalFlow,
)


class FakeTrainCollator:
    def __init__(self):
        self.features = None

    def __call__(self, features):
        self.features = features
        return {"features": features}


class FakeInferBuilder:
    def __init__(self):
        self.examples = None

    def build_infer_batch(self, examples):
        self.examples = examples
        return {
            "current": torch.tensor(
                [example["primary_image"][0].mean() for example in examples]
            ).float()
        }


class FakeDecoder(nn.Module):
    def forward(self, h_t, latent_action):
        return h_t + latent_action


class FakeQFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, context):
        return context.mean(dim=1, keepdim=True) * self.scale


class FakeLAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = FakeDecoder()


class FakeBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = FakeLAM()
        self.vlm_to_lam = FakeQFormer()
        self.flow_scale = nn.Parameter(torch.tensor(0.5))
        self.step = 0

    def set_flow_train_step(self, step):
        self.step = step

    def _flow_h_t1_pred_prob(self):
        return min(1.0, self.step / 10000)

    def forward(self, batch):
        del batch
        loss = self.flow_scale.square() + self.vlm_to_lam.scale.square()
        return {
            "loss_total": loss,
            "loss_flow": self.flow_scale.square(),
            "loss_perceptual": self.vlm_to_lam.scale.square(),
            "loss_distill": loss * 0,
        }

    def predict_action(self, batch, **_kwargs):
        current = batch["current"].to(self.flow_scale.device)
        context = torch.stack((current, current + 1), dim=1).unsqueeze(-1)
        latent_action = self.vlm_to_lam(context)
        h_t = current[:, None, None].expand(-1, 4, 1)
        h_t1 = self.lam.decoder(h_t, latent_action)
        signal = h_t1.mean(dim=(1, 2)) * self.flow_scale
        actions = signal[:, None, None].expand(-1, ACTION_HORIZON, 8).clone()
        return actions, {"h_t": h_t.cpu(), "h_t1_pred": h_t1.cpu()}

    def _run_shared_encoding_train(self, prepared_batch, **_kwargs):
        values = torch.tensor(
            [feature["primary_videos"][:, 0].float().mean() for feature in prepared_batch["features"]]
        )
        context = torch.stack((values, values + 1), dim=1).unsqueeze(-1)
        latent_action = self.vlm_to_lam(context)
        h_t = values[:, None, None].expand(-1, 4, 1)
        return SimpleNamespace(
            h_t=h_t,
            h_t1_pred=self.lam.decoder(h_t, latent_action),
            h_t1_gt=h_t + 2,
        )


def _batch(batch_size=2):
    torch.manual_seed(22)
    return {
        "current_rgb": torch.randint(0, 256, (batch_size, 2, 3, 20, 24), dtype=torch.uint8),
        "future_rgb": torch.randint(
            0, 256, (batch_size, 4, 2, 3, 20, 24), dtype=torch.uint8
        ),
        "future_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "qpos": torch.randn(batch_size, 9),
        "action": torch.randn(batch_size, 100, 8),
        "action_mask": torch.tensor(
            [[True] * (100 - index) + [False] * index for index in range(batch_size)]
        ),
        "task_text": [f"task-{index}" for index in range(batch_size)],
        "agent": torch.arange(batch_size).remainder(2),
    }


def test_adapter_preserves_two_frame_future_and_tail_action_mask():
    train = FakeTrainCollator()
    infer = FakeInferBuilder()
    adapter = LaWAMRoboFactoryAdapter(train, infer)
    batch = _batch()
    adapter.training_batch(batch, torch.device("cpu"))
    assert len(train.features) == 2
    assert train.features[0]["primary_videos"].shape == (1, 2, 3, 20, 24)
    assert train.features[0]["action"].shape == (100, 8)
    assert train.features[1]["action"].shape == (99, 8)
    assert train.features[0]["embodiment_id"] == 1
    assert train.features[1]["embodiment_id"] == 2
    assert train.features[1]["action_hz"] == 99.0


def test_full_loss_backward_and_scheduled_sampling_probability():
    model = R11LaWAMSubgoalFlow(
        FakeBackend(), LaWAMRoboFactoryAdapter(FakeTrainCollator(), FakeInferBuilder())
    )
    result = model.training_step(_batch(), update=5000)
    result["loss"].backward()
    assert model.policy_backend.flow_scale.grad is not None
    assert model.policy_backend.vlm_to_lam.scale.grad is not None
    assert result["scheduled_prediction_probability"] == 0.5


def test_future_and_latent_action_interventions_change_actions():
    model = R11LaWAMSubgoalFlow(
        FakeBackend(), LaWAMRoboFactoryAdapter(FakeTrainCollator(), FakeInferBuilder())
    ).eval()
    batch = _batch()
    normal = model(batch, mode="normal")
    off = model(batch, mode="prediction_off")
    shuffled = model(batch, mode="prediction_shuffled")
    action_shuffled = model(batch, action_condition_mode="action_shuffled")
    assert normal["action"].shape == (2, 100, 8)
    assert not torch.equal(normal["action"], off["action"])
    assert not torch.equal(normal["action"], shuffled["action"])
    assert not torch.equal(normal["action"], action_shuffled["action"])
    assert batch["task_text"] == ["task-0", "task-1"]


def test_causal_probe_returns_official_lam_prediction_target_and_persistence():
    model = R11LaWAMSubgoalFlow(
        FakeBackend(), LaWAMRoboFactoryAdapter(FakeTrainCollator(), FakeInferBuilder())
    ).eval()
    batch = _batch()
    normal = model.causal_probe(batch, action_condition_mode="normal")
    shuffled = model.causal_probe(batch, action_condition_mode="action_shuffled")
    assert normal["future_prediction"].shape == normal["future_target"].shape
    assert normal["persistence_prediction"].shape == normal["future_target"].shape
    assert not torch.equal(normal["future_prediction"], shuffled["future_prediction"])
