"""Strict local-only LatentToM adaptation for RoboFactory.

This package keeps the official LatentToM diffusion backbone and latent
observation bottleneck, while exposing one shared policy interface:
``local_rgb + local_qpos + task_id -> local 8-D commanded action``.
"""
