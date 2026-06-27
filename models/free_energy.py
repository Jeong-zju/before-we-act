import torch


def simple_goal_cost(pred_object_xy: torch.Tensor, goal_xy: torch.Tensor):
    return ((pred_object_xy - goal_xy) ** 2).sum(dim=-1)


def simple_control_cost(actions: torch.Tensor):
    return (actions ** 2).sum(dim=-1)
