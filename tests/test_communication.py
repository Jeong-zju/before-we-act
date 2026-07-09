import torch

from models.communication import CommunicationConfig, CommunicationTrigger


def test_message_bits_positive():
    cfg = CommunicationConfig(codebook_size=64, residual_dim=64, residual_bits=8)
    trigger = CommunicationTrigger(cfg)
    assert trigger.message_bits() > 0


def test_trigger_when_gain_exceeds_cost():
    cfg = CommunicationConfig(lambda_bits=0.0, lambda_delay=0.0, lambda_redundancy=0.0)
    trigger = CommunicationTrigger(cfg)

    G_no = torch.tensor([2.0, 1.0])
    G_comm = torch.tensor([1.0, 1.2])
    inferred = torch.tensor([1, 2])
    message = torch.tensor([3, 2])

    out = trigger.decide(G_no, G_comm, inferred, message)
    assert out["trigger"].tolist() == [True, False]
    assert torch.isfinite(out["delta_G"]).all()
    assert torch.isfinite(out["C_comm"]).all()


def test_redundancy_increases_cost():
    cfg = CommunicationConfig(lambda_bits=0.0, lambda_delay=0.0, lambda_redundancy=1.0)
    trigger = CommunicationTrigger(cfg)

    G_no = torch.tensor([2.0, 2.0])
    G_comm = torch.tensor([1.5, 1.5])
    inferred = torch.tensor([1, 2])
    message = torch.tensor([1, 3])

    out = trigger.decide(G_no, G_comm, inferred, message)
    assert out["C_comm"][0] > out["C_comm"][1]
