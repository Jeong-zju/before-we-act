from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from models.communication import CommunicationConfig, CommunicationTrigger
from models.free_energy import FreeEnergyConfig, FreeEnergyEvaluator
from models.intention import IntentionConfig, IntentionInferenceModel
from models.plan_tokenizer import PlanTokenizer, PlanTokenizerConfig
from models.slot_encoder import AgentObjectSlotEncoder, SlotEncoderConfig
from models.wam import LatentWorldActionModel, WAMConfig
from train.train_wam import amp_context


@dataclass
class PolicyConfig:
    history: int = 8
    horizon: int = 16
    k_exec: int = 2
    num_candidates: int = 8
    ego_id: int = 0
    active_codes: Tuple[int, ...] = (2, 3, 6, 24, 32, 44, 51)
    residual_noise_std: float = 0.5
    action_clip: float = 1.0
    alpha_goal: float = 1.0
    alpha_safety: float = 2.0
    alpha_collab: float = 1.0
    alpha_unc: float = 0.5
    alpha_ctrl: float = 0.05
    goal_y: float = 3.05
    force_limit: float = 1.0
    lambda_bits: float = 2e-4
    lambda_delay: float = 0.1
    lambda_redundancy: float = 0.2
    delay_steps: float = 1.0
    message_uncertainty_floor: float = 0.10
    require_physical_gain: bool = False
    amp_dtype: str = "bf16"


def load_ckpt(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def load_models(
    wam_ckpt: str,
    slot_ckpt: str,
    plan_ckpt: str,
    intention_ckpt: str,
    device: torch.device,
):
    wam_raw = load_ckpt(wam_ckpt, device)
    wam_cfg = WAMConfig(**wam_raw["config"])
    wam = LatentWorldActionModel(wam_cfg).to(device)
    wam.load_state_dict(wam_raw["model"])
    wam.eval()
    for p in wam.parameters():
        p.requires_grad_(False)

    slot_raw = load_ckpt(slot_ckpt, device)
    slot_cfg = SlotEncoderConfig(**slot_raw["config"])
    slot = AgentObjectSlotEncoder(slot_cfg).to(device)
    slot.load_state_dict(slot_raw["model"])
    slot.eval()
    for p in slot.parameters():
        p.requires_grad_(False)
    slot_norm = {k: v.to(device) for k, v in slot_raw["normalization"].items()}

    plan_raw = load_ckpt(plan_ckpt, device)
    plan_cfg = PlanTokenizerConfig(**plan_raw["config"])
    plan = PlanTokenizer(plan_cfg).to(device)
    plan.load_state_dict(plan_raw["model"])
    plan.eval()
    for p in plan.parameters():
        p.requires_grad_(False)
    plan_norm = {k: v.to(device) for k, v in plan_raw["normalization"].items()}

    intent_raw = load_ckpt(intention_ckpt, device)
    intent_cfg = IntentionConfig(**intent_raw["config"])
    intent = IntentionInferenceModel(intent_cfg).to(device)
    intent.load_state_dict(intent_raw["model"])
    intent.eval()
    for p in intent.parameters():
        p.requires_grad_(False)

    return {
        "wam": wam,
        "wam_cfg": wam_cfg,
        "slot": slot,
        "slot_cfg": slot_cfg,
        "slot_norm": slot_norm,
        "plan": plan,
        "plan_cfg": plan_cfg,
        "plan_norm": plan_norm,
        "intention": intent,
        "intention_cfg": intent_cfg,
    }


def unwrap_reset(ret):
    if isinstance(ret, tuple) and len(ret) == 2:
        return ret[0], ret[1]
    return ret, {}


def unwrap_step(ret):
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, bool(terminated or truncated), info
    if isinstance(ret, tuple) and len(ret) == 4:
        obs, reward, done, info = ret
        return obs, reward, bool(done), info
    raise RuntimeError(f"Unsupported env.step return format: {type(ret)} len={len(ret) if isinstance(ret, tuple) else 'NA'}")


def make_env(seed: int = 0):
    mod = importlib.import_module("envs.two_robot_carry_env")

    def is_env_class(obj):
        if not isinstance(obj, type):
            return False
        name = obj.__name__.lower()
        if "config" in name or name.endswith("cfg") or name.endswith("args"):
            return False
        return callable(getattr(obj, "reset", None)) and callable(getattr(obj, "step", None))

    preferred_names = [
        "TwoRobotCarryEnv",
        "TwoRobotCarryNarrowPassageEnv",
        "CarryEnv",
        "MujocoCarryEnv",
    ]

    env_cls = None
    for name in preferred_names:
        obj = getattr(mod, name, None)
        if is_env_class(obj):
            env_cls = obj
            break

    if env_cls is None:
        candidates = []
        for name in dir(mod):
            obj = getattr(mod, name)
            if is_env_class(obj):
                candidates.append((name, obj))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            env_cls = candidates[0][1]

    if env_cls is None:
        available = []
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                available.append(
                    f"{name}(reset={callable(getattr(obj, 'reset', None))}, "
                    f"step={callable(getattr(obj, 'step', None))})"
                )
        raise RuntimeError(
            "Could not find an environment class with callable reset() and step() "
            f"in envs.two_robot_carry_env. Available classes: {available}"
        )

    cfg_candidates = [
        "TwoRobotCarryConfig",
        "CarryEnvConfig",
        "EnvConfig",
    ]

    cfg = None
    for name in cfg_candidates:
        cfg_cls = getattr(mod, name, None)
        if isinstance(cfg_cls, type):
            try:
                cfg = cfg_cls()
                break
            except Exception:
                cfg = None

    env = None
    constructors = []
    if cfg is not None:
        if hasattr(cfg, "seed"):
            try:
                cfg.seed = seed
            except Exception:
                pass
        constructors.extend([
            lambda: env_cls(cfg),
            lambda: env_cls(config=cfg),
            lambda: env_cls(cfg=cfg),
        ])

    constructors.extend([
        lambda: env_cls(seed=seed),
        lambda: env_cls(),
    ])

    last_err = None
    for ctor in constructors:
        try:
            env = ctor()
            if callable(getattr(env, "reset", None)) and callable(getattr(env, "step", None)):
                break
        except Exception as e:
            last_err = e
            env = None

    if env is None:
        raise RuntimeError(f"Failed to instantiate {env_cls.__name__}; last error: {last_err}")

    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass

    return env


def safe_render(env, width: int = 640, height: int = 480):
    # 1) Try the environment's own rgb_array API first.
    if hasattr(env, "render"):
        for kwargs in [
            {"mode": "rgb_array", "width": width, "height": height},
            {"render_mode": "rgb_array", "width": width, "height": height},
            {"width": width, "height": height},
            {"mode": "rgb_array"},
            {},
        ]:
            try:
                img = env.render(**kwargs)
                if img is not None:
                    arr = np.asarray(img)
                    if arr.ndim == 3 and arr.shape[-1] in [3, 4]:
                        if arr.shape[-1] == 4:
                            arr = arr[..., :3]
                        return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
            except TypeError:
                continue
            except Exception:
                continue

    # 2) Try native MuJoCo Python renderer: env.model + env.data.
    try:
        import mujoco

        model = getattr(env, "model", None)
        data = getattr(env, "data", None)

        if model is None and hasattr(env, "unwrapped"):
            model = getattr(env.unwrapped, "model", None)
            data = getattr(env.unwrapped, "data", None)

        if model is not None and data is not None:
            renderer_key = f"_mujoco_renderer_{width}_{height}"
            renderer = getattr(env, renderer_key, None)

            if renderer is None:
                # Create an offscreen GL context once and cache it on env.
                ctx_key = f"_mujoco_gl_context_{width}_{height}"
                ctx = getattr(env, ctx_key, None)
                if ctx is None:
                    try:
                        ctx = mujoco.GLContext(width, height)
                        ctx.make_current()
                        setattr(env, ctx_key, ctx)
                    except Exception:
                        ctx = None

                renderer = mujoco.Renderer(model, height=height, width=width)
                setattr(env, renderer_key, renderer)

            renderer.update_scene(data)
            img = renderer.render()
            if img is not None:
                arr = np.asarray(img)
                if arr.ndim == 3 and arr.shape[-1] in [3, 4]:
                    if arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    except Exception:
        pass

    # 3) Try mujoco-py style renderer: env.sim.render().
    try:
        sim = getattr(env, "sim", None)
        if sim is None and hasattr(env, "unwrapped"):
            sim = getattr(env.unwrapped, "sim", None)

        if sim is not None and hasattr(sim, "render"):
            img = sim.render(width=width, height=height, camera_name=None)
            if img is not None:
                arr = np.asarray(img)
                if arr.ndim == 3 and arr.shape[-1] in [3, 4]:
                    if arr.shape[-1] == 4:
                        arr = arr[..., :3]
                    # mujoco-py often returns bottom-up images.
                    arr = arr[::-1].copy()
                    return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
    except Exception:
        pass

    return None


def get_action_from_scripted(env, obs=None):
    for name in ["scripted_action", "oracle_action", "get_scripted_action", "expert_action", "policy_action"]:
        if hasattr(env, name):
            fn = getattr(env, name)
            try:
                return np.asarray(fn(obs), dtype=np.float32)
            except TypeError:
                return np.asarray(fn(), dtype=np.float32)

    # Conservative fallback: move both robots forward and keep grippers closed.
    return np.array([0.0, 0.7, 0.0, 1.0, 0.0, 0.7, 0.0, 1.0], dtype=np.float32)


def extract_metric(info: Dict[str, Any], key: str, default=0.0):
    if isinstance(info, dict) and key in info:
        return info[key]
    return default


def extract_any_metric(info: Dict[str, Any], keys, default=0.0):
    if not isinstance(info, dict):
        return default
    for key in keys:
        if key in info:
            return info[key]
    return default


def observation_to_local_history(obs, info, history: int, fallback_action: np.ndarray):
    # Best case: env provides model-ready history in info.
    if isinstance(info, dict):
        for key in ["local_history_agents", "model/local_history_agents"]:
            if key in info:
                arr = np.asarray(info[key], dtype=np.float32)
                if arr.shape[-2:] == (history, 17):
                    return torch.tensor(arr, dtype=torch.float32)

    # Dataset-compatible fallback from observation dict.
    def get_prop(robot_key: str):
        if isinstance(obs, dict):
            candidates = [
                (robot_key, "proprio"),
                (robot_key, "state"),
                (f"obs/{robot_key}/proprio",),
                (robot_key,),
            ]
            for c in candidates:
                try:
                    x = obs
                    for cc in c:
                        x = x[cc]
                    x = np.asarray(x, dtype=np.float32).reshape(-1)
                    if x.size >= 11:
                        return x[:11]
                except Exception:
                    pass
        return np.zeros(11, dtype=np.float32)

    p0 = get_prop("robot_0")
    p1 = get_prop("robot_1")
    a0 = fallback_action[:4].astype(np.float32)
    a1 = fallback_action[4:8].astype(np.float32)
    force = np.array([float(extract_metric(info, "force_proxy", 0.0))], dtype=np.float32)
    contact = np.array([float(extract_metric(info, "contacts", 0.0) > 0)], dtype=np.float32)

    row0 = np.concatenate([p0, a0, force, contact], axis=0)
    row1 = np.concatenate([p1, a1, force, contact], axis=0)
    hist = np.stack(
        [
            np.repeat(row0[None, :], history, axis=0),
            np.repeat(row1[None, :], history, axis=0),
        ],
        axis=0,
    )
    return torch.tensor(hist, dtype=torch.float32)


def extract_pose_context(info, ego_id: int):
    # If env provides exact context, use it.
    if isinstance(info, dict):
        if "rel_target_pose_agents" in info and "object_rel_pose_agents" in info:
            rel = np.asarray(info["rel_target_pose_agents"], dtype=np.float32)
            obj = np.asarray(info["object_rel_pose_agents"], dtype=np.float32)
            return (
                torch.tensor(rel[ego_id], dtype=torch.float32),
                torch.tensor(obj[ego_id], dtype=torch.float32),
            )

    return torch.zeros(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32)


def normalize_local(local_history: torch.Tensor, slot_norm: Dict[str, torch.Tensor], device: torch.device):
    x = local_history.to(device)
    return (x - slot_norm["local_mean"].view(1, 1, -1)) / slot_norm["local_std"].view(1, 1, -1)


def generate_candidates(base_codes, base_residuals, ego_id: int, cfg: PolicyConfig, device):
    B, A = base_codes.shape
    K = cfg.num_candidates
    D = base_residuals.shape[-1]
    codes = base_codes[:, None, :].expand(B, K, A).clone()
    residuals = base_residuals[:, None, :, :].expand(B, K, A, D).clone()

    if K > 1:
        active = torch.tensor(list(cfg.active_codes), dtype=torch.long, device=device)
        idx = torch.randint(0, len(active), (B, K - 1), device=device)
        codes[:, 1:, ego_id] = active[idx]
        noise = torch.randn(B, K - 1, D, device=device, dtype=base_residuals.dtype) * cfg.residual_noise_std
        residuals[:, 1:, ego_id, :] = base_residuals[:, ego_id].unsqueeze(1) + noise
    return codes, residuals


def score_candidates(wam, fe, current_slots, codes, residuals, uncertainty, cfg: PolicyConfig, device):
    B, K, A = codes.shape
    slots_f = current_slots[:, None].expand(B, K, *current_slots.shape[1:]).reshape(B * K, *current_slots.shape[1:])
    codes_f = codes.reshape(B * K, A)
    residuals_f = residuals.reshape(B * K, A, residuals.shape[-1])
    unc_f = uncertainty[:, None].expand(B, K).reshape(B * K)

    amp_enabled = device.type == "cuda" and cfg.amp_dtype != "none"
    with amp_context(device, cfg.amp_dtype, amp_enabled):
        rollout = wam.rollout(slots_f, codes_f, residuals_f)
    score = fe.total_score(rollout, uncertainty=unc_f)
    score_bk = {k: v.reshape(B, K) for k, v in score.items()}
    idx = score_bk["G"].argmin(dim=1)
    action_chunks = rollout["pred_actions"].reshape(B, K, cfg.horizon, -1)
    selected_actions = action_chunks[torch.arange(B, device=device), idx]
    return score_bk, idx, selected_actions


class ClosedLoopPolicy:
    def __init__(
        self,
        model_paths: Dict[str, str],
        cfg: PolicyConfig,
        mode: str,
        device: torch.device,
    ):
        self.cfg = cfg
        self.mode = mode
        self.device = device
        self.models = load_models(
            wam_ckpt=model_paths["wam"],
            slot_ckpt=model_paths["slot"],
            plan_ckpt=model_paths["plan"],
            intention_ckpt=model_paths["intention"],
            device=device,
        )
        self.fe = FreeEnergyEvaluator(
            FreeEnergyConfig(
                goal_y=cfg.goal_y,
                force_limit=cfg.force_limit,
                alpha_goal=cfg.alpha_goal,
                alpha_safety=cfg.alpha_safety,
                alpha_collab=cfg.alpha_collab,
                alpha_unc=cfg.alpha_unc,
                alpha_ctrl=cfg.alpha_ctrl,
            )
        )
        self.comm = CommunicationTrigger(
            CommunicationConfig(
                codebook_size=self.models["plan_cfg"].codebook_size,
                residual_dim=self.models["plan_cfg"].latent_dim,
                lambda_bits=cfg.lambda_bits,
                lambda_delay=cfg.lambda_delay,
                lambda_redundancy=cfg.lambda_redundancy,
                delay_steps=cfg.delay_steps,
            )
        )
        self.last_action = np.zeros(8, dtype=np.float32)
        self.decision_log: List[Dict[str, Any]] = []

    @torch.inference_mode()
    def act(self, obs, info, t: int):
        cfg = self.cfg
        device = self.device
        ego = int(cfg.ego_id)
        teammate = 1 - ego

        if self.mode == "scripted":
            action = get_action_from_scripted(None, obs)
            self.last_action = np.clip(action, -cfg.action_clip, cfg.action_clip).astype(np.float32)
            self.decision_log.append({"t": t, "mode": self.mode, "trigger": 0, "selected_candidate": -1})
            return self.last_action

        local_agents = observation_to_local_history(obs, info, cfg.history, self.last_action).to(device)
        local_agents_norm = normalize_local(local_agents, self.models["slot_norm"], device)

        phase_hist = torch.zeros(2, cfg.history, dtype=torch.long, device=device)
        agent_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
        slot_out = self.models["slot"].encode_slots(local_agents_norm, agent_ids, phase_hist)
        current_slots = slot_out["slots"].reshape(1, 2, -1, slot_out["slots"].shape[-1])

        # Use current ego code as conservative default. Candidate 0 will be close to current belief.
        plan_cfg = self.models["plan_cfg"]
        base_codes = torch.zeros(1, 2, dtype=torch.long, device=device)
        base_residuals = torch.zeros(1, 2, plan_cfg.latent_dim, dtype=torch.float32, device=device)

        ego_slots = current_slots[:, ego]
        rel_pose, obj_pose = extract_pose_context(info, ego)
        rel_pose = rel_pose.view(1, 3).to(device)
        obj_pose = obj_pose.view(1, 3).to(device)

        intent = self.models["intention"].infer_teammate_plan(
            ego_slots=ego_slots,
            ego_plan_codes=base_codes[:, ego],
            ego_plan_residuals=base_residuals[:, ego],
            ego_id=torch.tensor([ego], dtype=torch.long, device=device),
            phase_history=torch.zeros(1, cfg.history, dtype=torch.long, device=device),
            rel_target_pose=rel_pose,
            object_rel_pose=obj_pose,
        )
        inferred_code = intent["target_code"].long()
        inferred_residual = intent["target_residual"]
        inferred_unc = intent["uncertainty"].float()

        own_codes, own_residuals = generate_candidates(base_codes, base_residuals, ego, cfg, device)

        no_codes = own_codes.clone()
        no_res = own_residuals.clone()
        no_codes[:, :, teammate] = inferred_code[:, None].expand(1, cfg.num_candidates)
        no_res[:, :, teammate, :] = inferred_residual[:, None, :].expand(1, cfg.num_candidates, -1)

        comm_codes = own_codes.clone()
        comm_res = own_residuals.clone()

        # In online rollout we do not know teammate true future. For always/selective, message is approximated by inferred belief with uncertainty floor.
        # This keeps the closed-loop controller executable without dataset future labels.
        comm_codes[:, :, teammate] = inferred_code[:, None].expand(1, cfg.num_candidates)
        comm_res[:, :, teammate, :] = inferred_residual[:, None, :].expand(1, cfg.num_candidates, -1)

        no_score, no_idx, no_actions = score_candidates(
            self.models["wam"], self.fe, current_slots, no_codes, no_res, inferred_unc, cfg, device
        )
        comm_unc = torch.full_like(inferred_unc, cfg.message_uncertainty_floor)
        comm_score, comm_idx, comm_actions = score_candidates(
            self.models["wam"], self.fe, current_slots, comm_codes, comm_res, comm_unc, cfg, device
        )

        G_no = no_score["G"].gather(1, no_idx[:, None]).squeeze(1).float()
        G_comm = comm_score["G"].gather(1, comm_idx[:, None]).squeeze(1).float()

        if self.mode == "no_comm":
            trigger = torch.tensor([False], device=device)
            chosen_actions = no_actions
            chosen_idx = no_idx
        elif self.mode == "always_comm":
            trigger = torch.tensor([True], device=device)
            chosen_actions = comm_actions
            chosen_idx = comm_idx
        elif self.mode == "selective_comm":
            dec = self.comm.decide(
                G_no_comm=G_no,
                G_comm=G_comm,
                inferred_code=inferred_code,
                message_code=inferred_code,
            )
            trigger = dec["trigger"]
            chosen_actions = torch.where(trigger.view(1, 1, 1), comm_actions, no_actions)
            chosen_idx = torch.where(trigger, comm_idx, no_idx)
        else:
            raise ValueError(f"Unknown policy mode: {self.mode}")

        action = chosen_actions[0, 0].detach().cpu().float().numpy()
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, -cfg.action_clip, cfg.action_clip).astype(np.float32)

        self.last_action = action
        self.decision_log.append(
            {
                "t": t,
                "mode": self.mode,
                "ego_id": ego,
                "trigger": int(trigger.item()),
                "selected_candidate": int(chosen_idx.item()),
                "G_no_comm": float(G_no.item()),
                "G_comm": float(G_comm.item()),
                "delta_G": float((G_no - G_comm).item()),
                "uncertainty": float(inferred_unc.item()),
                "inferred_code": int(inferred_code.item()),
                **{f"action_{i}": float(action[i]) for i in range(len(action))},
            }
        )
        return action


def rollout_episode(
    env,
    policy: ClosedLoopPolicy,
    max_steps: int = 300,
    render: bool = False,
    width: int = 640,
    height: int = 480,
):
    obs, info = unwrap_reset(env.reset())
    frames = []
    rewards = []
    infos = []
    done = False

    for t in range(max_steps):
        action = policy.act(obs, info, t)
        for _ in range(max(1, policy.cfg.k_exec)):
            obs, reward, done, info = unwrap_step(env.step(action))
            rewards.append(float(reward) if reward is not None else 0.0)
            infos.append(info if isinstance(info, dict) else {})
            if render:
                frame = safe_render(env, width=width, height=height)
                if frame is not None:
                    frames.append(frame)
            if done:
                break
        if done:
            break

    metrics = summarize_episode(infos, rewards, policy.decision_log, max_steps=max_steps)
    return metrics, policy.decision_log, frames


def summarize_episode(infos: List[Dict[str, Any]], rewards: List[float], decisions: List[Dict[str, Any]], max_steps: int):
    last = infos[-1] if infos else {}
    success = bool(extract_metric(last, "success", False))
    failure = bool(extract_metric(last, "failure", False))
    collision = sum(float(extract_any_metric(
        x,
        ["collision", "collisions", "collision_count", "has_collision", "contact_collision"],
        0.0,
    )) for x in infos)

    force_vals = [float(extract_any_metric(
        x,
        ["force_proxy", "contact_force", "max_contact_force", "contact_force_norm", "peak_force", "force"],
        0.0,
    )) for x in infos]

    distance_vals = [float(extract_any_metric(
        x,
        ["robot_distance", "inter_robot_distance", "min_robot_distance", "min_distance", "agent_distance"],
        0.0,
    )) for x in infos]
    triggers = [d.get("trigger", 0) for d in decisions]

    return {
        "success": float(success),
        "failure": float(failure),
        "episode_steps": len(infos),
        "return": float(np.sum(rewards)),
        "collision_count": float(collision),
        "mean_force": float(np.mean(force_vals)) if force_vals else 0.0,
        "max_force": float(np.max(force_vals)) if force_vals else 0.0,
        "mean_robot_distance": float(np.mean(distance_vals)) if distance_vals else 0.0,
        "min_robot_distance": float(np.min(distance_vals)) if distance_vals else 0.0,
        "comm_count": float(np.sum(triggers)),
        "comm_rate": float(np.mean(triggers)) if triggers else 0.0,
        "timeout": float((not success) and (not failure) and len(infos) >= max_steps),
    }
