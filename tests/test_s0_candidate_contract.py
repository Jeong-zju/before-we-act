from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_b0_s0_candidate_is_uniform_loss_moe_ensemble():
    config = yaml.safe_load(
        (ROOT / "configs/wam_flow/s0_candidate.yaml").read_text()
    )
    card = yaml.safe_load(
        (ROOT / "experiments/wam_flow/s0/candidate_card.yaml").read_text()
    )

    assert card["candidate_id"] == "B0"
    assert config["model"]["decoder_kind"] == "sparse_moe"
    assert config["inference"]["chunk_aggregation"] == "temporal_ensemble"
    assert config["training"]["seed"] == 101
    assert config["training"]["updates"] == 80000
    assert not any("active" in key for key in config["training"])
