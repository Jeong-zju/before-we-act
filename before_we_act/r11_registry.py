"""Fail-closed model registry for the candidate-D branch."""
from pathlib import Path

from before_we_act.r11_lawam_subgoal_flow import (
    MODEL_NAME,
    build_lawam_subgoal_flow,
    load_candidate_config,
)


MODEL_NAMES = (MODEL_NAME,)


def build_r11_model(model_name: str, config_path: str, project_root: str | Path):
    if model_name != MODEL_NAME:
        raise ValueError(f"unsupported model on candidate-D branch: {model_name}")
    return build_lawam_subgoal_flow(load_candidate_config(config_path), project_root)
