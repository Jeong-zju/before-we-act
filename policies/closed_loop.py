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
    belief_code_error_prob: float = 0.0
    belief_residual_noise_std: float = 0.0
    belief_uncertainty_boost: float = 0.0
    scripted_mix: float = 0.0
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
    if isinstance(ret, dict) and isinstance(ret.get("metrics"), dict):
        return ret, ret["metrics"]
    return ret, {}


def unwrap_step(ret):
    if isinstance(ret, tuple) and len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        return obs, reward, bool(terminated or truncated), info
    if isinstance(ret, tuple) and len(ret) == 4:
        obs, reward, done, info = ret
        return obs, reward, bool(done), info
    raise RuntimeError(f"Unsupported env.step return format: {type(ret)} len={len(ret) if isinstance(ret, tuple) else 'NA'}")


def make_env(seed: int = 0, scenario: str = "nominal", **kwargs):
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

    cfg_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if scenario is not None:
        cfg_kwargs.setdefault("scenario", scenario)

    env = None
    constructors = []
    if cfg is not None:
        if hasattr(cfg, "seed"):
            try:
                cfg.seed = seed
            except Exception:
                pass
        for key, value in cfg_kwargs.items():
            if hasattr(cfg, key):
                try:
                    setattr(cfg, key, value)
                except Exception:
                    pass
        constructors.extend([
            lambda: env_cls(cfg),
            lambda: env_cls(config=cfg),
            lambda: env_cls(cfg=cfg),
        ])

    constructors.extend([
        lambda: env_cls(seed=seed, **cfg_kwargs),
        lambda: env_cls(**cfg_kwargs),
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

    def set_env_option(key: str, value):
        for container_name in ["cfg", "config"]:
            container = getattr(env, container_name, None)
            if container is not None and hasattr(container, key):
                try:
                    setattr(container, key, value)
                except Exception:
                    pass

        setter = getattr(env, f"set_{key}", None)
        if callable(setter):
            try:
                setter(value)
            except Exception:
                pass

        if hasattr(env, key):
            try:
                setattr(env, key, value)
            except Exception:
                pass

    if "scenario" in cfg_kwargs:
        set_env_option("scenario", cfg_kwargs["scenario"])
        apply_preset = getattr(env, "_apply_scenario_preset", None)
        if callable(apply_preset):
            try:
                apply_preset()
            except Exception:
                pass

    for key, value in cfg_kwargs.items():
        if key != "scenario":
            set_env_option(key, value)

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
    def history_from_model_field(container):
        if not isinstance(container, dict):
            return None
        for key in ["local_history_agents", "model/local_history_agents"]:
            if key not in container:
                continue
            arr = np.asarray(container[key], dtype=np.float32)
            if arr.shape == (2, history, 17):
                return torch.tensor(arr, dtype=torch.float32)
        return None

    def history_from_local_obs_field(container):
        if not isinstance(container, dict):
            return None
        for key in ["local_obs_agents", "model/local_obs_agents"]:
            if key not in container:
                continue
            arr = np.asarray(container[key], dtype=np.float32)
            if arr.shape == (2, 17):
                hist = np.repeat(arr[:, None, :], history, axis=1)
                return torch.tensor(hist, dtype=torch.float32)
        return None

    # Best case: env provides model-ready history in info.
    hist = history_from_model_field(info)
    if hist is not None:
        return hist

    # Common online case: env provides the current per-agent local observation.
    hist = history_from_local_obs_field(info)
    if hist is not None:
        return hist
    hist = history_from_model_field(obs)
    if hist is not None:
        return hist
    hist = history_from_local_obs_field(obs)
    if hist is not None:
        return hist

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

    fallback_action = np.asarray(fallback_action, dtype=np.float32).reshape(-1)
    if fallback_action.size < 8:
        fallback_action = np.pad(fallback_action, (0, 8 - fallback_action.size))
    p0 = get_prop("robot_0")
    p1 = get_prop("robot_1")
    a0 = fallback_action[:4]
    a1 = fallback_action[4:8]
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
        for rel_key, obj_key in [
            ("rel_target_pose_agents", "object_rel_pose_agents"),
            ("model/rel_target_pose_agents", "model/object_rel_pose_agents"),
        ]:
            if rel_key in info and obj_key in info:
                rel = np.asarray(info[rel_key], dtype=np.float32)
                obj = np.asarray(info[obj_key], dtype=np.float32)
                if rel.shape == (2, 3) and obj.shape == (2, 3):
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
        self._plan_context: Optional[Dict[str, Any]] = None
        self._agent_candidate_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._agent_belief_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def _prepare_plan_context(self, obs, info, t: int) -> Dict[str, Any]:
        cfg = self.cfg
        device = self.device

        local_agents = observation_to_local_history(obs, info, cfg.history, self.last_action).to(device)
        local_agents_norm = normalize_local(local_agents, self.models["slot_norm"], device)

        phase_hist = torch.zeros(2, cfg.history, dtype=torch.long, device=device)
        agent_ids = torch.tensor([0, 1], dtype=torch.long, device=device)
        slot_out = self.models["slot"].encode_slots(local_agents_norm, agent_ids, phase_hist)
        current_slots = slot_out["slots"].reshape(1, 2, -1, slot_out["slots"].shape[-1])

        plan_cfg = self.models["plan_cfg"]
        base_codes = torch.zeros(1, 2, dtype=torch.long, device=device)
        base_residuals = torch.zeros(1, 2, plan_cfg.latent_dim, dtype=torch.float32, device=device)

        return {
            "t": t,
            "obs": obs,
            "info": info,
            "current_slots": current_slots,
            "base_codes": base_codes,
            "base_residuals": base_residuals,
        }

    def _ensure_plan_context(self, obs, info, t: int) -> Dict[str, Any]:
        if self._plan_context is None or self._plan_context.get("t") != t:
            self._plan_context = self._prepare_plan_context(obs, info, t)
            self._agent_candidate_cache = {}
            self._agent_belief_cache = {}
        return self._plan_context

    def _infer_teammate_belief(self, agent_id: int, info) -> Dict[str, torch.Tensor]:
        agent_id = int(agent_id)
        if agent_id in self._agent_belief_cache:
            return self._agent_belief_cache[agent_id]

        cfg = self.cfg
        device = self.device
        ctx = self._plan_context
        if ctx is None:
            raise RuntimeError("plan context must be prepared before inferring teammate belief")

        rel_pose, obj_pose = extract_pose_context(info, agent_id)
        intent = self.models["intention"].infer_teammate_plan(
            ego_slots=ctx["current_slots"][:, agent_id],
            ego_plan_codes=ctx["base_codes"][:, agent_id],
            ego_plan_residuals=ctx["base_residuals"][:, agent_id],
            ego_id=torch.tensor([agent_id], dtype=torch.long, device=device),
            phase_history=torch.zeros(1, cfg.history, dtype=torch.long, device=device),
            rel_target_pose=rel_pose.view(1, 3).to(device),
            object_rel_pose=obj_pose.view(1, 3).to(device),
        )
        belief = {
            "code": intent["target_code"].long().reshape(1),
            "residual": intent["target_residual"].float().reshape(1, -1),
            "uncertainty": intent["uncertainty"].float().reshape(1),
        }
        belief = self._perturb_belief(belief)
        self._agent_belief_cache[agent_id] = belief
        return belief

    def _perturb_belief(self, belief: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        code = belief["code"].detach().clone().long().reshape(1)
        residual = belief["residual"].detach().clone().float().reshape(1, -1)
        uncertainty = belief["uncertainty"].detach().clone().float().reshape(1)

        code_error_prob = float(np.clip(cfg.belief_code_error_prob, 0.0, 1.0))
        if code_error_prob > 0.0 and bool(torch.rand((), device=self.device).item() < code_error_prob):
            active_codes = list(cfg.active_codes)
            if active_codes:
                active = torch.tensor(active_codes, dtype=torch.long, device=self.device)
                idx = torch.randint(0, len(active_codes), code.shape, device=self.device)
                code = active[idx].reshape(1)
            else:
                code = torch.randint(0, int(self.models["plan_cfg"].codebook_size), code.shape, device=self.device)

        residual_noise_std = float(max(0.0, cfg.belief_residual_noise_std))
        if residual_noise_std > 0.0:
            residual = residual + torch.randn_like(residual) * residual_noise_std

        uncertainty_boost = float(cfg.belief_uncertainty_boost)
        if uncertainty_boost != 0.0:
            uncertainty = torch.clamp(uncertainty + uncertainty_boost, min=0.0)

        return {
            "code": code,
            "residual": residual,
            "uncertainty": uncertainty,
        }

    def _as_batch_scalar(self, value, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=dtype).reshape(-1)[:1]
        return torch.tensor([value], device=self.device, dtype=dtype)

    def _as_batch_vector(self, value, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=dtype).reshape(1, -1)
        return torch.tensor(value, device=self.device, dtype=dtype).reshape(1, -1)

    def build_message(
        self,
        agent_id,
        selected_plan_code,
        selected_plan_residual,
        uncertainty,
        selected_action_chunk,
    ):
        source_uncertainty = self._as_batch_scalar(uncertainty, torch.float32)
        return {
            "agent_id": int(agent_id),
            "code": self._as_batch_scalar(selected_plan_code, torch.long),
            "residual": self._as_batch_vector(selected_plan_residual, torch.float32),
            "uncertainty": torch.full_like(source_uncertainty, float(self.cfg.message_uncertainty_floor)),
            "source_uncertainty": source_uncertainty,
            "action_chunk": self._as_batch_vector(selected_action_chunk, torch.float32),
        }

    def fuse_teammate_belief(self, inferred_belief, received_message, mode):
        code = inferred_belief["code"].detach().clone()
        residual = inferred_belief["residual"].detach().clone()
        uncertainty = inferred_belief["uncertainty"].detach().clone()

        if mode == "message" and received_message is not None:
            code = self._as_batch_scalar(received_message.get("code", code), torch.long)
            residual = self._as_batch_vector(received_message.get("residual", residual), torch.float32)
            if "uncertainty" in received_message:
                uncertainty = self._as_batch_scalar(received_message["uncertainty"], torch.float32)
            else:
                uncertainty = torch.full_like(uncertainty, float(self.cfg.message_uncertainty_floor))

        return {
            "code": code.long().reshape(1),
            "residual": residual.float().reshape(1, -1),
            "uncertainty": uncertainty.float().reshape(1),
        }

    def plan_for_agent(self, agent_id, obs, info, t, received_message=None, force_comm=False):
        cfg = self.cfg
        device = self.device
        agent_id = int(agent_id)
        teammate_id = 1 - agent_id

        ctx = self._ensure_plan_context(obs, info, t)
        inferred_belief = self._infer_teammate_belief(agent_id, info)
        use_message = received_message is not None and (force_comm or self.mode in {"always_comm", "selective_comm"})
        teammate_belief = self.fuse_teammate_belief(
            inferred_belief,
            received_message,
            "message" if use_message else "none",
        )

        if agent_id not in self._agent_candidate_cache:
            self._agent_candidate_cache[agent_id] = generate_candidates(
                ctx["base_codes"],
                ctx["base_residuals"],
                agent_id,
                cfg,
                device,
            )
        own_codes, own_residuals = self._agent_candidate_cache[agent_id]

        codes = own_codes.clone()
        residuals = own_residuals.clone()
        codes[:, :, teammate_id] = teammate_belief["code"][:, None].expand(1, cfg.num_candidates)
        residuals[:, :, teammate_id, :] = teammate_belief["residual"][:, None, :].expand(1, cfg.num_candidates, -1)

        score, idx, selected_actions = score_candidates(
            self.models["wam"],
            self.fe,
            ctx["current_slots"],
            codes,
            residuals,
            teammate_belief["uncertainty"],
            cfg,
            device,
        )

        selected_G = score["G"].gather(1, idx[:, None]).squeeze(1).float()
        batch_idx = torch.arange(idx.shape[0], device=device)
        selected_codes = codes[batch_idx, idx, agent_id].long()
        selected_residuals = residuals[batch_idx, idx, agent_id, :].float()
        action_slice = slice(agent_id * 4, agent_id * 4 + 4)
        selected_action_chunk = selected_actions[0, 0, action_slice].float()
        message = self.build_message(
            agent_id,
            selected_codes,
            selected_residuals,
            inferred_belief["uncertainty"],
            selected_action_chunk,
        )

        return {
            "agent_id": agent_id,
            "teammate_id": teammate_id,
            "score": score,
            "selected_candidate": idx.long(),
            "G": selected_G,
            "action_chunk": selected_action_chunk,
            "selected_plan_code": selected_codes.reshape(1),
            "selected_plan_residual": selected_residuals.reshape(1, -1),
            "inferred_belief": inferred_belief,
            "teammate_belief": teammate_belief,
            "message": message,
        }

    def _finite_action_chunk(self, chunk) -> np.ndarray:
        arr = chunk.detach().cpu().float().numpy() if torch.is_tensor(chunk) else np.asarray(chunk, dtype=np.float32)
        arr = np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.size < 4:
            arr = np.pad(arr, (0, 4 - arr.size))
        return arr[:4].astype(np.float32)

    def _scripted_log_row(self, t: int, action: np.ndarray) -> Dict[str, Any]:
        row = {
            "t": t,
            "mode": self.mode,
            "ego_id": int(self.cfg.ego_id),
            "trigger": 0,
            "trigger_robot_0": 0,
            "trigger_robot_1": 0,
            "comm_count_step": 0,
            "selected_candidate": -1,
            "selected_candidate_robot_0": -1,
            "selected_candidate_robot_1": -1,
            "G_no_comm_robot_0": 0.0,
            "G_comm_robot_0": 0.0,
            "delta_G_robot_0": 0.0,
            "G_no_comm_robot_1": 0.0,
            "G_comm_robot_1": 0.0,
            "delta_G_robot_1": 0.0,
            "G_no_comm": 0.0,
            "G_comm": 0.0,
            "delta_G": 0.0,
            "message_code_robot_0": -1,
            "message_code_robot_1": -1,
            "uncertainty_robot_0": 0.0,
            "uncertainty_robot_1": 0.0,
            "uncertainty": 0.0,
            "inferred_code": -1,
        }
        row.update({f"action_{i}": float(action[i]) for i in range(8)})
        return row

    def _normalize_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 8:
            action = np.pad(action, (0, 8 - action.size))
        action = np.nan_to_num(action[:8], nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(action, -self.cfg.action_clip, self.cfg.action_clip).astype(np.float32)

    def _update_latest_logged_action(self, action: np.ndarray):
        if not self.decision_log:
            return
        self.decision_log[-1].update({f"action_{i}": float(action[i]) for i in range(8)})

    def _mix_with_scripted_action(self, action: np.ndarray, env=None, obs=None) -> np.ndarray:
        mix = float(np.clip(self.cfg.scripted_mix, 0.0, 1.0))
        if mix <= 0.0:
            return action
        scripted = self._normalize_action(get_action_from_scripted(env, obs))
        mixed = (1.0 - mix) * action + mix * scripted
        return self._normalize_action(mixed)

    def select_joint_action(self, obs, info, t):
        cfg = self.cfg
        if self.mode not in {"no_comm", "always_comm", "selective_comm"}:
            raise ValueError(f"Unknown policy mode: {self.mode}")

        self._plan_context = self._prepare_plan_context(obs, info, t)
        self._agent_candidate_cache = {}
        self._agent_belief_cache = {}

        no_comm = {
            0: self.plan_for_agent(0, obs, info, t, received_message=None, force_comm=False),
            1: self.plan_for_agent(1, obs, info, t, received_message=None, force_comm=False),
        }
        messages = {0: no_comm[0]["message"], 1: no_comm[1]["message"]}

        if self.mode in {"always_comm", "selective_comm"}:
            comm = {
                0: self.plan_for_agent(0, obs, info, t, received_message=messages[1], force_comm=True),
                1: self.plan_for_agent(1, obs, info, t, received_message=messages[0], force_comm=True),
            }
        else:
            comm = no_comm

        triggers: Dict[int, int] = {}
        chosen: Dict[int, Dict[str, Any]] = {}
        for agent_id in [0, 1]:
            if self.mode == "no_comm":
                triggers[agent_id] = 0
                chosen[agent_id] = no_comm[agent_id]
            elif self.mode == "always_comm":
                triggers[agent_id] = 1
                chosen[agent_id] = comm[agent_id]
            else:
                teammate_id = 1 - agent_id
                decision = self.comm.decide(
                    G_no_comm=no_comm[agent_id]["G"],
                    G_comm=comm[agent_id]["G"],
                    inferred_code=no_comm[agent_id]["inferred_belief"]["code"],
                    message_code=messages[teammate_id]["code"],
                )
                trigger = bool(decision["trigger"].item())
                if cfg.require_physical_gain:
                    trigger = trigger and float((no_comm[agent_id]["G"] - comm[agent_id]["G"]).item()) > 0.0
                triggers[agent_id] = int(trigger)
                chosen[agent_id] = comm[agent_id] if trigger else no_comm[agent_id]

        action = np.concatenate(
            [
                self._finite_action_chunk(chosen[0]["action_chunk"]),
                self._finite_action_chunk(chosen[1]["action_chunk"]),
            ],
            axis=0,
        )
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, -cfg.action_clip, cfg.action_clip).astype(np.float32)

        def scalar(value) -> float:
            if torch.is_tensor(value):
                return float(value.detach().cpu().reshape(-1)[0].float())
            return float(value)

        def int_scalar(value) -> int:
            if torch.is_tensor(value):
                return int(value.detach().cpu().reshape(-1)[0])
            return int(value)

        G_no = {i: scalar(no_comm[i]["G"]) for i in [0, 1]}
        G_comm = {i: scalar(comm[i]["G"]) for i in [0, 1]}
        delta = {i: G_no[i] - G_comm[i] for i in [0, 1]}
        comm_count_step = int(triggers[0] + triggers[1])

        row = {
            "t": t,
            "mode": self.mode,
            "ego_id": int(cfg.ego_id),
            "trigger": int(comm_count_step > 0),
            "trigger_robot_0": triggers[0],
            "trigger_robot_1": triggers[1],
            "comm_count_step": comm_count_step,
            "selected_candidate": int_scalar(chosen[0]["selected_candidate"]),
            "selected_candidate_robot_0": int_scalar(chosen[0]["selected_candidate"]),
            "selected_candidate_robot_1": int_scalar(chosen[1]["selected_candidate"]),
            "G_no_comm_robot_0": G_no[0],
            "G_comm_robot_0": G_comm[0],
            "delta_G_robot_0": delta[0],
            "G_no_comm_robot_1": G_no[1],
            "G_comm_robot_1": G_comm[1],
            "delta_G_robot_1": delta[1],
            "G_no_comm": G_no[0],
            "G_comm": G_comm[0],
            "delta_G": delta[0],
            "message_code_robot_0": int_scalar(messages[0]["code"]),
            "message_code_robot_1": int_scalar(messages[1]["code"]),
            "uncertainty_robot_0": scalar(no_comm[0]["inferred_belief"]["uncertainty"]),
            "uncertainty_robot_1": scalar(no_comm[1]["inferred_belief"]["uncertainty"]),
            "uncertainty": scalar(no_comm[0]["inferred_belief"]["uncertainty"]),
            "inferred_code": int_scalar(no_comm[0]["inferred_belief"]["code"]),
        }
        row.update({f"action_{i}": float(action[i]) for i in range(8)})

        self.last_action = action
        self.decision_log.append(row)
        return action

    @torch.inference_mode()
    def act(self, obs, info, t: int, env=None):
        if self.mode == "scripted":
            self.last_action = self._normalize_action(get_action_from_scripted(env, obs))
            self.decision_log.append(self._scripted_log_row(t, self.last_action))
            return self.last_action

        action = self.select_joint_action(obs, info, t)
        action = self._mix_with_scripted_action(action, env=env, obs=obs)
        self.last_action = action
        self._update_latest_logged_action(action)
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
        action = policy.act(obs, info, t, env=env)
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
    success = bool(extract_any_metric(last, ["success", "is_success", "task_success"], False))
    failure = bool(extract_any_metric(last, ["failure", "failed", "is_failure", "task_failure"], False))
    collision = sum(float(extract_any_metric(
        x,
        ["collision_count", "collisions", "contacts", "ncon", "num_contacts", "collision", "has_collision", "contact_collision"],
        0.0,
    )) for x in infos)

    force_vals = [float(extract_any_metric(
        x,
        ["force_proxy", "contact_force", "max_contact_force", "contact_force_norm", "peak_force", "force"],
        0.0,
    )) for x in infos]

    distance_vals = [float(extract_any_metric(
        x,
        [
            "robot_distance",
            "inter_robot_distance",
            "robot_dist",
            "robots_distance",
            "distance_between_robots",
            "min_robot_distance",
            "min_distance",
            "agent_distance",
        ],
        0.0,
    )) for x in infos]

    communication_required_vals = [
        float(bool(extract_any_metric(x, ["communication_required", "comm_required"], False)))
        for x in infos
    ]
    occlusion_vals = [
        float(bool(extract_any_metric(x, ["occlusion_active", "occlusion"], False)))
        for x in infos
    ]
    force_violation_vals = [
        float(bool(extract_any_metric(x, ["force_violation", "force_limit_violation"], False)))
        for x in infos
    ]
    object_goal_distance_vals = [float(extract_any_metric(
        x,
        ["object_goal_distance", "goal_distance", "object_to_goal_distance"],
        0.0,
    )) for x in infos]
    progress_vals = [float(extract_any_metric(
        x,
        ["progress", "task_progress"],
        0.0,
    )) for x in infos]

    has_dual_triggers = any("trigger_robot_0" in d or "trigger_robot_1" in d for d in decisions)
    if has_dual_triggers:
        comm_steps = [
            float(d.get("trigger_robot_0", 0)) + float(d.get("trigger_robot_1", 0))
            for d in decisions
        ]
        comm_count_robot_0 = float(np.sum([float(d.get("trigger_robot_0", 0)) for d in decisions]))
        comm_count_robot_1 = float(np.sum([float(d.get("trigger_robot_1", 0)) for d in decisions]))
        comm_count = float(np.sum(comm_steps))
        comm_rate = float(comm_count / (2.0 * len(decisions))) if decisions else 0.0
    else:
        triggers = [d.get("trigger", 0) for d in decisions]
        comm_count = float(np.sum(triggers))
        comm_rate = float(np.mean(triggers)) if triggers else 0.0
        comm_count_robot_0 = comm_count
        comm_count_robot_1 = 0.0

    comm_rate_robot_0 = float(comm_count_robot_0 / len(decisions)) if decisions else 0.0
    comm_rate_robot_1 = float(comm_count_robot_1 / len(decisions)) if decisions else 0.0

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
        "communication_required_rate": float(np.mean(communication_required_vals)) if communication_required_vals else 0.0,
        "occlusion_rate": float(np.mean(occlusion_vals)) if occlusion_vals else 0.0,
        "force_violation_rate": float(np.mean(force_violation_vals)) if force_violation_vals else 0.0,
        "mean_object_goal_distance": float(np.mean(object_goal_distance_vals)) if object_goal_distance_vals else 0.0,
        "final_object_goal_distance": float(object_goal_distance_vals[-1]) if object_goal_distance_vals else 0.0,
        "progress_mean": float(np.mean(progress_vals)) if progress_vals else 0.0,
        "progress_final": float(progress_vals[-1]) if progress_vals else 0.0,
        "comm_count": comm_count,
        "comm_rate": comm_rate,
        "comm_count_robot_0": comm_count_robot_0,
        "comm_count_robot_1": comm_count_robot_1,
        "comm_rate_robot_0": comm_rate_robot_0,
        "comm_rate_robot_1": comm_rate_robot_1,
        "timeout": float((not success) and (not failure) and len(infos) >= max_steps),
    }
