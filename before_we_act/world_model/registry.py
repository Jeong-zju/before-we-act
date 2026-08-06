from __future__ import annotations

from importlib import import_module
from typing import Any


CANDIDATE_SPECS = {
    "p0": {
        "kind": "tdmpc2_latent_dynamics",
        "module": "before_we_act.world_model.tdmpc2_world",
        "official_repo": "https://github.com/nicklashansen/tdmpc2.git",
    },
    "p1": {
        "kind": "lpwm_particle_dynamics",
        "module": "before_we_act.world_model.lpwm_world",
        "official_repo": "https://github.com/taldatech/lpwm.git",
    },
    "p2": {
        "kind": "vjepa2_action_predictor",
        "module": "before_we_act.world_model.vjepa2_ac_world",
        "official_repo": "https://github.com/facebookresearch/vjepa2.git",
    },
    "p3": {
        "kind": "dino_wm_feature_dynamics",
        "module": "before_we_act.world_model.dino_wm_world",
        "official_repo": "https://github.com/gaoyuezhou/dino_wm.git",
    },
}


def build_world_core(candidate_id: str, config: dict[str, Any]):
    try:
        spec = CANDIDATE_SPECS[candidate_id]
    except KeyError as exc:
        raise ValueError(f"unknown R13 candidate {candidate_id!r}") from exc
    if config.get("kind") != spec["kind"]:
        raise ValueError("R13 candidate ID and registered kind differ")
    try:
        module = import_module(spec["module"])
    except ModuleNotFoundError as exc:
        if exc.name == spec["module"]:
            raise RuntimeError(
                f"{candidate_id} implementation is absent from this independent branch"
            ) from exc
        raise
    return module.build_world_core(config)
