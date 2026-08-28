#!/usr/bin/env python3
"""Unix-socket inference worker for decentralized RoboFactory VLA policies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import socket
import struct
import sys
from typing import Any

import numpy as np

TASK_EMBEDS = {
    "camera_alignment": "camera_alignment/lang_embed.pt",
    "lift_barrier": "lift_barrier/lang_embed.pt",
    "long_pipeline_delivery": "long_pipeline_delivery/lang_embed.pt",
    "pass_shoe": "pass_shoe/lang_embed.pt",
    "place_food": "place_food/lang_embed.pt",
    "take_photo": "take_photo/lang_embed.pt",
}


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    blocks = []
    remaining = size
    while remaining:
        block = conn.recv(remaining)
        if not block:
            raise EOFError("peer disconnected")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def _recv(conn: socket.socket) -> Any:
    size = struct.unpack("!Q", _recv_exact(conn, 8))[0]
    if size > 512 * 1024 * 1024:
        raise ValueError(f"RPC payload too large: {size}")
    return pickle.loads(_recv_exact(conn, size))


def _send(conn: socket.socket, value: Any) -> None:
    payload = pickle.dumps(value, protocol=5)
    conn.sendall(struct.pack("!Q", len(payload)) + payload)


def _validate_decentralized_request(observations: Any) -> None:
    """Reject requests that could expose one policy call to peer/global state."""
    if not isinstance(observations, list) or len(observations) != 1:
        raise ValueError("decentralized inference requires exactly one arm-local observation per request")
    observation = observations[0]
    if not isinstance(observation, dict):
        raise TypeError("arm-local observation must be a mapping")
    allowed = {"agent", "task", "prompt", "image", "state"}
    unexpected = set(observation) - allowed
    if unexpected:
        raise ValueError(f"forbidden non-local observation fields: {sorted(unexpected)}")
    image = np.asarray(observation.get("image"))
    state = np.asarray(observation.get("state"))
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"local RGB must be uint8 HxWx3, got {image.shape} {image.dtype}")
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError(f"local qpos must be finite shape (9,), got {state.shape}")


class RDTBackend:
    def __init__(self, checkpoint: str, dataset_root: str):
        import torch
        import yaml
        from PIL import Image

        repo = "/workspace/repos/rdt-1b"
        sys.path.insert(0, repo)
        # before-we-act also has a top-level ``models`` package.  PYTHONPATH is
        # inherited by this worker, so evict any earlier generic-package import
        # before resolving the official RDT modules from the newly prepended repo.
        for module_name in list(sys.modules):
            if module_name in {"models", "configs"} or module_name.startswith(("models.", "configs.")):
                del sys.modules[module_name]
        os.chdir(repo)
        from configs.state_vec import STATE_VEC_IDX_MAPPING
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        from models.rdt_runner import RDTRunner

        self.torch = torch
        self.Image = Image
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        with open("configs/base.yaml", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.vision = SiglipVisionTower("google/siglip-so400m-patch14-384", None)
        self.vision = self.vision.to(device=self.device, dtype=self.dtype).eval()
        self.model = RDTRunner.from_pretrained(checkpoint).to(device=self.device, dtype=self.dtype).eval()
        self.dataset_root = Path(dataset_root)
        self.indices = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
        self.indices.append(STATE_VEC_IDX_MAPPING["right_gripper_open"])
        self.history: dict[int, np.ndarray] = {}
        self.control_frequency = 20
        control_path = Path("configs/dataset_control_freq.json")
        if control_path.is_file():
            import json

            self.control_frequency = int(json.loads(control_path.read_text()).get("robofactory", 20))

    def reset(self) -> None:
        self.history.clear()

    def _image_tensor(self, observations: list[dict]) -> Any:
        torch = self.torch
        processor = self.vision.image_processor
        bg = np.asarray([int(x * 255) for x in processor.image_mean], np.uint8)
        background = np.broadcast_to(bg, (processor.size["height"], processor.size["width"], 3)).copy()
        ordered = []
        for obs in observations:
            image = np.asarray(obs["image"], np.uint8)
            agent = int(obs["agent"])
            previous = self.history.get(agent, image)
            self.history[agent] = image.copy()
            for local in (previous, image):
                ordered.extend((local, background, background))
        tensors = []
        for image in ordered:
            pil = self.Image.fromarray(image)
            width, height = pil.size
            if width != height:
                side = max(width, height)
                square = self.Image.new("RGB", (side, side), tuple(int(x * 255) for x in processor.image_mean))
                square.paste(pil, ((side - width) // 2, (side - height) // 2))
                pil = square
            tensors.append(processor.preprocess(pil, return_tensors="pt")["pixel_values"][0])
        return torch.stack(tensors).to(device=self.device, dtype=self.dtype)

    def infer(self, observations: list[dict]) -> list[np.ndarray]:
        torch = self.torch
        batch_size = len(observations)
        images = self._image_tensor(observations)
        with torch.inference_mode():
            encoded = self.vision(images)
            encoded = encoded.reshape(batch_size, -1, self.vision.hidden_size)
            states = torch.zeros((batch_size, 1, 128), device=self.device, dtype=self.dtype)
            masks = torch.zeros((batch_size, 1, 128), device=self.device, dtype=self.dtype)
            for row, obs in enumerate(observations):
                qpos = np.asarray(obs["state"], np.float32)
                states[row, 0, self.indices[:7]] = torch.as_tensor(qpos[:7], device=self.device, dtype=self.dtype)
                states[row, 0, self.indices[7]] = torch.as_tensor(qpos[7:9].mean() / 0.04, device=self.device, dtype=self.dtype)
                masks[row, 0, self.indices] = 1
            task = observations[0]["task"]
            lang = torch.load(self.dataset_root / TASK_EMBEDS[task], map_location="cpu")
            lang = lang.to(device=self.device, dtype=self.dtype)
            lang = lang.unsqueeze(0).expand(batch_size, -1, -1)
            lang_mask = torch.ones(lang.shape[:2], device=self.device, dtype=torch.bool)
            ctrl = torch.full((batch_size,), self.control_frequency, device=self.device, dtype=torch.long)
            pred = self.model.predict_action(lang, lang_mask, encoded, states, masks, ctrl)
        pred = pred[:, :, self.indices].float().cpu().numpy()
        pred[:, :, 7] = pred[:, :, 7] * 2.0 - 1.0
        return [row.astype(np.float32) for row in pred]


class OpenVLABackend:
    def __init__(self, checkpoint: str, _dataset_root: str):
        from types import SimpleNamespace

        repo = "/workspace/repos/openvla-oft"
        sys.path.insert(0, repo)
        os.chdir(repo)
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action

        self.get_vla_action = get_vla_action
        self.cfg = SimpleNamespace(
            pretrained_checkpoint=checkpoint,
            use_l1_regression=True,
            use_diffusion=False,
            num_diffusion_steps_train=50,
            num_diffusion_steps_inference=50,
            use_film=False,
            num_images_in_input=1,
            use_proprio=True,
            center_crop=False,
            lora_rank=32,
            unnorm_key="robofactory",
            use_relative_actions=False,
            load_in_8bit=False,
            load_in_4bit=False,
        )
        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, 9)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)

    def reset(self) -> None:
        return None

    def infer(self, observations: list[dict]) -> list[np.ndarray]:
        output = []
        for obs in observations:
            values = self.get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                {"full_image": np.asarray(obs["image"], np.uint8), "state": np.asarray(obs["state"], np.float32)},
                obs["prompt"],
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=False,
            )
            output.append(np.asarray(values, np.float32))
        return output


class Pi05Backend:
    def __init__(self, checkpoint: str, _dataset_root: str):
        repo = "/workspace/repos/openpi"
        sys.path.insert(0, repo)
        os.chdir(repo)
        from openpi.policies import policy_config
        from openpi.training import config

        train_config = config.get_config("pi05_robofactory_lora")
        self.policy = policy_config.create_trained_policy(train_config, checkpoint)

    def reset(self) -> None:
        return None

    def infer(self, observations: list[dict]) -> list[np.ndarray]:
        output = []
        for obs in observations:
            result = self.policy.infer(
                {
                    "image": np.asarray(obs["image"], np.uint8),
                    "state": np.asarray(obs["state"], np.float32),
                    "prompt": obs["prompt"],
                }
            )
            output.append(np.asarray(result["actions"], np.float32))
        return output


class GauDPBackend:
    """Strict arm-local GauDP inference with shared weights for every arm."""

    def __init__(self, checkpoint: str, _dataset_root: str):
        from collections import deque
        import torch
        import torch.nn.functional as F

        repo = "/workspace/repos/Policy-Lightning"
        sys.path.insert(0, repo)
        os.chdir(repo)
        from bwa.train_robofactory_gaudp import build_model
        from model.noposplat.encoder import get_encoder

        self.torch, self.F, self.deque = torch, F, deque
        self.device = torch.device("cuda:0")
        self.cfg, self.policy = build_model()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(payload["state_dict"], strict=True)
        self.policy.to(self.device).eval()
        self.gaussian = get_encoder(self.cfg.gau_encoder)
        weights = torch.load(self.cfg.gau_encoder.pretrained_weights, map_location="cpu", weights_only=False)
        state = weights.get("state_dict", weights)
        state = {key[8:]: value for key, value in state.items() if key.startswith("encoder.")}
        self.gaussian.load_state_dict(state, strict=False)
        self.gaussian.to(self.device).eval()
        self.history = {}

    def reset(self) -> None:
        self.history.clear()

    def _local_features(self, images: list[np.ndarray]):
        torch, F = self.torch, self.F
        image_tensor = torch.stack(
            [torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) for image in images]
        ).div_(255.0)
        large = F.interpolate(image_tensor, size=(240, 320), mode="bilinear", align_corners=False)
        small = F.interpolate(image_tensor, size=(120, 160), mode="bilinear", align_corners=False).cpu()
        with torch.inference_mode():
            feature = self.gaussian({"image": large.mul(2.0).sub(1.0).to(self.device)[:, None]})[:, 0].float()
            feature = F.interpolate(feature, size=(120, 160), mode="bilinear", align_corners=False).cpu()
        return list(zip(small, feature, strict=True))

    def infer(self, observations: list[dict]) -> list[np.ndarray]:
        torch = self.torch
        features = self._local_features([np.asarray(obs["image"], np.uint8) for obs in observations])
        for obs, (image, gaussian) in zip(observations, features, strict=True):
            agent = int(obs["agent"])
            qpos = torch.as_tensor(np.asarray(obs["state"], np.float32)[:9]).clone()
            local = (image, gaussian, qpos)
            history = self.history.setdefault(agent, self.deque(maxlen=3))
            if not history:
                history.extend([local, local])
            history.append(local)
        images, gaussians, states = [], [], []
        for obs in observations:
            history = list(self.history[int(obs["agent"])])
            images.append(torch.stack([item[0] for item in history]))
            gaussians.append(torch.stack([item[1] for item in history]))
            states.append(torch.stack([item[2] for item in history]))
        policy_obs = {
            "head_cam_0": torch.stack(images).to(self.device),
            "gaussian_0": torch.stack(gaussians).to(self.device),
            "state": torch.stack(states).to(self.device),
        }
        with torch.inference_mode():
            chunks = self.policy.predict_action(policy_obs)["action"].float().cpu().numpy()
        return [row.astype(np.float32) for row in chunks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("rdt", "openvla", "pi05", "gaudp"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--dataset-root", default="/workspace/datasets/robofactory_multitask")
    args = parser.parse_args()

    backend_type = {"rdt": RDTBackend, "openvla": OpenVLABackend, "pi05": Pi05Backend, "gaudp": GauDPBackend}[args.policy]
    backend = backend_type(args.checkpoint, args.dataset_root)
    socket_path = Path(args.socket)
    socket_path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(4)
    try:
        running = True
        while running:
            conn, _ = listener.accept()
            with conn:
                try:
                    request = _recv(conn)
                    op = request.get("op")
                    if op == "ping":
                        response = {"ok": True, "policy": args.policy}
                    elif op == "reset":
                        backend.reset()
                        response = {"ok": True}
                    elif op == "infer":
                        _validate_decentralized_request(request.get("observations"))
                        chunks = backend.infer(request["observations"])
                        response = {"ok": True, "chunks": chunks}
                    elif op == "shutdown":
                        response = {"ok": True}
                        running = False
                    else:
                        raise ValueError(f"unknown RPC op: {op}")
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                _send(conn, response)
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
