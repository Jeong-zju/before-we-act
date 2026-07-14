"""Paired closed-loop evaluation for decentralized FE-PC-WAM.

The driver reproduces one external episode recipe for every communication
mode, keeps privileged simulator state outside the runtime, and writes one
atomic record per episode so long evaluations can resume safely.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import h5py
import mujoco
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.local_observation import SensorSimulationConfig  # noqa: E402
from data.schema import (  # noqa: E402
    LEGACY_CONTACT_SEMANTICS,
    LEGACY_FORCE_SEMANTICS,
    LOCAL_FORCE_UNITS,
    SCHEMA_VERSION,
    STRICT_LOCAL_CONTACT_SEMANTICS,
    STRICT_LOCAL_FORCE_SEMANTICS,
    STRICT_LOCAL_SENSOR_PROVENANCE,
)
from data.simulation import SimulationAdapter, ego_action_to_world  # noqa: E402
from envs.two_robot_carry_env import CarryEnvConfig, TwoRobotCarryNarrowPassageEnv  # noqa: E402
from eval.evaluate import (  # noqa: E402
    DEPLOYABLE_COMMUNICATION_MODES,
    compare_communication_modes,
)
from policies.decentralized import DecentralizedPolicyConfig  # noqa: E402
from policies.runtime import DecentralizedRuntime, RuntimeConfig  # noqa: E402
from scripts.audit_contract import run_contract_audit  # noqa: E402
from train.checkpoint import file_sha256, load_checkpoint  # noqa: E402


DEPLOYABLE_MODES = tuple(DEPLOYABLE_COMMUNICATION_MODES)
OFFICIAL_SPLIT_EPISODES = {"val": 160, "test": 80}
POLICY_MODE = {
    "no_comm": "no_comm",
    "always_reply": "always_reply",
    "selective_vpi": "selective",
    "periodic": "periodic",
    "random": "random",
}
UNSAFE_FAILURE_REASONS = frozenset(
    {
        "force_violation",
        "robot_too_far",
        "desync_in_passage",
        "object_yaw_too_large",
        "object_dropped",
        "object_out_of_bounds",
        "robot_out_of_bounds",
    }
)
DECISION_AGGREGATE_KEYS = (
    "request",
    "reply",
    "modeled_protocol_round_trip_bits",
    "communication_delay",
    "expected_latency_cost",
    "incurred_expected_latency_cost",
    "VPI",
    "code_surprise",
    "residual_surprise",
    "plan_surprise",
    "G_before",
    "G_after",
    "G_improvement",
    "replanned",
    "action_change_l2",
)
REQUIRED_VALIDATION_FREEZE_CONDITIONS = frozenset(
    {
        "validation_split",
        "robust_wam_checkpoint",
        "all_five_deployable_modes",
        "complete_split",
        "official_split_size",
        "no_episode_or_scenario_filter",
        "full_episode_horizon",
        "no_truncated_episodes",
        "dataset_randomization_reproduced",
        "cuda_execution",
        "checkpoint_sensor_contract_consistent",
        "strict_local_contact_checkpoint",
        "strict_local_force_checkpoint",
        "strict_local_sensor_dataset",
        "dataset_checkpoint_sensor_contract_compatible",
        "local_force_scale_compatible",
        "trained_message_metadata_distribution_respected",
        "baseline_budget_matching_enabled_and_within_tolerance",
        "artifact_and_candidate_audit_passed",
    }
)


@dataclass(frozen=True)
class EpisodeRecipe:
    source_path: Path
    split: str
    episode_id: str
    episode_index: int
    seed: int
    scenario: str
    object_dropout_prob: float
    source_sha256: str


@dataclass(frozen=True)
class CheckpointSet:
    plan: Path
    belief: Path
    deployment_wam: Path
    deployment_wam_stage: str
    intention: Path
    base_wam: Path | None = None

    @property
    def uses_base_wam(self) -> bool:
        return self.deployment_wam_stage == "wam"

    def runtime_paths(self) -> dict[str, Path]:
        return {
            "plan": self.plan,
            "belief": self.belief,
            self.deployment_wam_stage: self.deployment_wam,
            "intention": self.intention,
        }

    def audit_paths(self) -> list[Path]:
        paths = [
            self.plan,
            self.belief,
            self.base_wam,
            self.intention,
            self.deployment_wam,
        ]
        return list(dict.fromkeys(path for path in paths if path is not None))


class _RolloutVideoRecorder:
    """Stream MuJoCo RGB frames to an atomic MP4 artifact."""

    def __init__(
        self,
        path: Path,
        *,
        model: mujoco.MjModel,
        width: int,
        height: int,
        fps: int,
        label: str,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency is in the runtime env
            raise RuntimeError(
                "video rendering requires opencv-python (import name: cv2)"
            ) from exc

        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_name(
            f"{self.path.stem}.tmp{self.path.suffix}"
        )
        self.temporary_path.unlink(missing_ok=True)
        self.cv2 = cv2
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.label = str(label)
        self.frame_count = 0
        try:
            self.renderer = mujoco.Renderer(
                model, height=self.height, width=self.width
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to create the MuJoCo offscreen renderer; for a headless "
                "machine set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa (CPU)"
            ) from exc

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.temporary_path),
            fourcc,
            float(self.fps),
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            self.renderer.close()
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"failed to open MP4 video writer for {self.path}")

    def capture(self, data: mujoco.MjData) -> None:
        self.renderer.update_scene(data)
        frame = np.asarray(self.renderer.render())
        if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
            raise RuntimeError(f"MuJoCo renderer returned invalid frame shape {frame.shape}")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[:2] != (self.height, self.width):
            frame = self.cv2.resize(frame, (self.width, self.height))
        bgr = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
        overlay = f"{self.label} | frame {self.frame_count:04d}"
        self.cv2.putText(
            bgr,
            overlay,
            (12, 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.writer.write(bgr)
        self.frame_count += 1

    def close(self, *, commit: bool) -> dict[str, Any] | None:
        self.writer.release()
        self.renderer.close()
        if not commit:
            self.temporary_path.unlink(missing_ok=True)
            return None
        if self.frame_count <= 0 or not self.temporary_path.is_file():
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"no frames were written for {self.path}")
        self.temporary_path.replace(self.path)
        return {
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "codec": "mp4v",
        }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _digest_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_checkpoints(
    checkpoint_dir: str | Path,
    *,
    use_base_wam: bool = False,
) -> CheckpointSet:
    root = Path(checkpoint_dir).resolve()
    required = {
        name: root / f"{name}.pt"
        for name in ("plan", "belief", "intention")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    base_wam = root / "wam.pt"
    robust_wam = root / "wam_robust.pt"
    if use_base_wam:
        if not base_wam.is_file():
            raise FileNotFoundError(
                f"--use-base-wam requires the base checkpoint: {base_wam}"
            )
        deployment_wam = base_wam
        deployment_wam_stage = "wam"
    else:
        if not robust_wam.is_file():
            raise FileNotFoundError(
                f"missing checkpoint: {robust_wam}; pass --use-base-wam only for "
                "a diagnostic rollout with the teacher-conditioned base WAM"
            )
        deployment_wam = robust_wam
        deployment_wam_stage = "wam_robust"
    return CheckpointSet(
        plan=required["plan"],
        belief=required["belief"],
        deployment_wam=deployment_wam,
        deployment_wam_stage=deployment_wam_stage,
        intention=required["intention"],
        base_wam=base_wam if base_wam.is_file() else None,
    )


def load_episode_recipes(
    dataset_root: str | Path,
    split: str,
    *,
    max_episodes: int = -1,
    scenarios: Sequence[str] = (),
) -> tuple[list[EpisodeRecipe], dict[str, Any]]:
    root = Path(dataset_root).resolve()
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"dataset manifest must use {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
        )
    split_dir = root / split
    selected_scenarios = {str(value) for value in scenarios}
    manifest_sensor_contract = {
        "local_contact_semantics": manifest.get(
            "local_contact_semantics", LEGACY_CONTACT_SEMANTICS
        ),
        "local_force_semantics": manifest.get(
            "local_force_semantics", LEGACY_FORCE_SEMANTICS
        ),
        "local_force_units": manifest.get("local_force_units"),
        "local_force_scale_newtons": manifest.get("local_force_scale_newtons"),
        "local_sensor_provenance": manifest.get("local_sensor_provenance"),
    }
    strict_manifest = (
        manifest_sensor_contract["local_contact_semantics"]
        == STRICT_LOCAL_CONTACT_SEMANTICS
        and manifest_sensor_contract["local_force_semantics"]
        == STRICT_LOCAL_FORCE_SEMANTICS
    )
    if strict_manifest and (
        manifest_sensor_contract["local_force_units"] != LOCAL_FORCE_UNITS
        or manifest_sensor_contract["local_force_scale_newtons"] is None
        or manifest_sensor_contract["local_sensor_provenance"]
        != STRICT_LOCAL_SENSOR_PROVENANCE
    ):
        raise ValueError("strict dataset manifest lacks local sensor provenance/scale")

    recipes: list[EpisodeRecipe] = []
    seen_indices: set[int] = set()
    for path in sorted(split_dir.glob("episode_*.hdf5")):
        with h5py.File(path, "r") as file:
            if str(file.attrs.get("schema_version", "")) != SCHEMA_VERSION:
                raise ValueError(f"incompatible  episode: {path}")
            metadata = {
                key: _metadata_value(value)
                for key, value in file["metadata"].attrs.items()
            }
            file_sensor_contract = {
                "local_contact_semantics": str(
                    file.attrs.get(
                        "local_contact_semantics", LEGACY_CONTACT_SEMANTICS
                    )
                ),
                "local_force_semantics": str(
                    file.attrs.get("local_force_semantics", LEGACY_FORCE_SEMANTICS)
                ),
                "local_force_units": (
                    None
                    if file.attrs.get("local_force_units") is None
                    else str(file.attrs.get("local_force_units"))
                ),
                "local_force_scale_newtons": (
                    None
                    if file.attrs.get("local_force_scale_newtons") is None
                    else float(file.attrs.get("local_force_scale_newtons"))
                ),
                "local_sensor_provenance": (
                    None
                    if file.attrs.get("local_sensor_provenance") is None
                    else str(file.attrs.get("local_sensor_provenance"))
                ),
            }
        for name, expected in manifest_sensor_contract.items():
            actual = file_sensor_contract[name]
            if name == "local_force_scale_newtons" and expected is not None:
                matches = actual is not None and np.isclose(
                    float(actual), float(expected)
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"{path}: HDF5 and manifest differ for {name}"
                )
        if str(metadata.get("split", "")) != split:
            raise ValueError(f"{path}: metadata split does not match {split!r}")
        try:
            filename_index = int(path.stem.removeprefix("episode_"))
        except ValueError as exc:
            raise ValueError(f"invalid  episode filename: {path.name}") from exc
        episode_index = int(metadata.get("episode_index", -1))
        if episode_index < 0 or episode_index != filename_index:
            raise ValueError(
                f"{path}: episode_index must be non-negative and match filename"
            )
        if episode_index in seen_indices:
            raise ValueError(f"duplicate episode_index={episode_index} in {split_dir}")
        seen_indices.add(episode_index)
        scenario = str(metadata["scenario"])
        if selected_scenarios and scenario not in selected_scenarios:
            continue
        recipes.append(
            EpisodeRecipe(
                source_path=path.resolve(),
                split=split,
                episode_id=f"{split}/{path.stem}",
                episode_index=episode_index,
                seed=int(metadata["seed"]),
                scenario=scenario,
                object_dropout_prob=float(metadata.get("object_dropout_prob", 0.05)),
                source_sha256=file_sha256(path),
            )
        )
        if max_episodes >= 0 and len(recipes) >= max_episodes:
            break
    if not recipes:
        raise ValueError(f"no  episodes selected from {split_dir}")
    manifest = dict(manifest)
    manifest["validated_local_sensor_contract"] = manifest_sensor_contract
    return recipes, manifest


def _external_episode_payload(
    recipe: EpisodeRecipe,
    *,
    checkpoint_hashes: Mapping[str, str],
    max_steps: int,
    randomize: bool,
    environment: Mapping[str, Any],
    sensor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": "fe_pc_wam_paired_external_episode",
        "source_episode": recipe.episode_id,
        "source_sha256": recipe.source_sha256,
        "seed": recipe.seed,
        "scenario": recipe.scenario,
        "randomize": bool(randomize),
        "max_steps": int(max_steps),
        "environment": dict(environment),
        "sensor": dict(sensor),
        "checkpoints": dict(checkpoint_hashes),
    }


def _mode_policy_config(
    mode: str,
    args: argparse.Namespace,
    *,
    matched_request_rate: float | None,
) -> DecentralizedPolicyConfig:
    if mode not in POLICY_MODE:
        raise ValueError(f"unsupported deployable mode {mode!r}")
    cooldown = int(args.cooldown_steps)
    valid_steps = int(args.plan_valid_steps)
    periodic_interval = int(args.periodic_interval)
    periodic_enabled = True
    periodic_request_rate: float | None = None
    random_probability = float(args.random_request_probability)
    if mode == "always_reply":
        # A baseline named "always" must really expose the peer plan at every
        # decision; cached validity and cooldown must not silently thin it.
        cooldown = 0
        valid_steps = 0
    if matched_request_rate is not None and mode in {"periodic", "random"}:
        rate = float(np.clip(matched_request_rate, 0.0, 1.0))
        random_probability = rate
        periodic_request_rate = rate
        periodic_enabled = rate > 0.0
        cooldown = 0
        valid_steps = 0
    return DecentralizedPolicyConfig(
        num_candidates=int(args.num_candidates),
        num_teammate_hypotheses=int(args.num_teammate_hypotheses),
        residual_sigma_points=int(args.residual_sigma_points),
        residual_sigma_scale=float(args.residual_sigma_scale),
        candidate_residual_scale=float(args.candidate_residual_scale),
        action_clip=float(args.action_clip),
        communication_mode=POLICY_MODE[mode],
        cooldown_steps=cooldown,
        plan_valid_steps=valid_steps,
        periodic_interval=periodic_interval,
        periodic_enabled=periodic_enabled,
        periodic_request_rate=periodic_request_rate,
        random_request_probability=random_probability,
        seed=int(args.policy_seed),
        metadata_available_index=(0 if args.use_untrained_message_metadata else -1),
        metadata_age_index=(1 if args.use_untrained_message_metadata else -1),
        metadata_confidence_index=(2 if args.use_untrained_message_metadata else -1),
        metadata_delay_index=(3 if args.use_untrained_message_metadata else -1),
    )


def _runtime_config(
    policy: DecentralizedPolicyConfig, args: argparse.Namespace
) -> RuntimeConfig:
    return RuntimeConfig(
        device=str(args.device),
        policy=policy,
        progress_target=float(args.progress_target),
        force_limit=float(args.force_limit),
        alpha_goal=float(args.alpha_goal),
        alpha_safety=float(args.alpha_safety),
        alpha_collab=float(args.alpha_collab),
        alpha_unc=float(args.alpha_unc),
        alpha_ctrl=float(args.alpha_ctrl),
        lambda_bits=float(args.lambda_bits),
        lambda_delay=float(args.lambda_delay),
        delay_steps=float(args.expected_delay_steps),
        delta_margin=float(args.delta_margin),
        return_scale=float(args.return_scale),
        tail_risk_weight=float(args.tail_risk_weight),
        constraint_risk_weight=float(args.constraint_risk_weight),
        success_risk_weight=float(args.success_risk_weight),
        safety_probability_threshold=float(args.safety_probability_threshold),
        utility_calibration_scale=float(args.utility_calibration_scale),
        utility_calibration_bias=float(args.utility_calibration_bias),
    )


def _resolved_device_name(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return str(device)


def _flatten_decision_steps(
    decision: Any,
    env_step: int,
    expected_delay: float,
    lambda_delay: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local in decision.agents:
        diagnostics = dict(local.diagnostics)
        configured_delay = float(diagnostics.pop("actual_delay_steps", expected_delay))
        requested = bool(diagnostics.get("request_sent", False))
        incurred_delay = configured_delay if requested else 0.0
        modeled_round_trip_bits = int(
            diagnostics.pop("actual_round_trip_bits", 0)
        )
        row = {
            "env_step": int(env_step),
            "agent_id": int(local.agent_id),
            "request": requested,
            "reply": bool(diagnostics.get("reply_received", False)),
            "modeled_protocol_round_trip_bits": modeled_round_trip_bits,
            "bit_accounting_semantics": (
                "modeled request+reply protocol budget with configured residual "
                "quantization; no serializer or wire measurement"
            ),
            # The current in-process router delivers synchronously.  Keep
            # realized transport separate from the delay charged by VPI.
            "communication_delay": 0.0,
            "realized_transport_delay_steps": 0.0,
            # Hypothetical delay charged while VPI decides whether a request
            # would be worthwhile.  It is present on every decision.
            "expected_delay_cost_steps": float(expected_delay),
            "expected_latency_cost": float(lambda_delay) * float(expected_delay),
            # Expected delay cost actually incurred after the mode-specific
            # request gate.  This is not measured transport latency.
            "incurred_expected_delay_steps": incurred_delay,
            "incurred_expected_latency_cost": float(lambda_delay)
            * incurred_delay,
            "plan_code": int(local.plan_code),
            "communicated": bool(local.communicated),
            "routed_messages_pair": int(decision.routed_messages),
        }
        row.update(diagnostics)
        rows.append(row)
    return rows


def _episode_metrics(
    infos: Sequence[Mapping[str, Any]],
    rewards: Sequence[float],
    *,
    done: bool,
    truncated: bool,
) -> dict[str, Any]:
    last = dict(infos[-1]) if infos else {}
    failure_reason = str(last.get("failure_reason", "none"))
    force_flags = [bool(info.get("force_violation", False)) for info in infos]
    force_values = [float(info.get("force_proxy", 0.0)) for info in infos]
    collision_values = [float(info.get("collision_count", 0.0)) for info in infos]
    collision_flags = [value > 0.0 for value in collision_values]
    collision_events = sum(
        int(active and (index == 0 or not collision_flags[index - 1]))
        for index, active in enumerate(collision_flags)
    )
    unsafe = any(force_flags) or failure_reason in UNSAFE_FAILURE_REASONS
    return {
        "success": bool(last.get("success", False)),
        "safe": not unsafe,
        "safety_metric_semantics": (
            "no force violation and no terminal reason in UNSAFE_FAILURE_REASONS; "
            "collision events are reported separately"
        ),
        "failure": bool(last.get("failure", False)),
        "failure_reason": failure_reason,
        "return": float(sum(rewards)),
        # ``collision_count`` is an event count (false->true transitions), not
        # a duration-biased sum of every active MuJoCo contact at every step.
        "collision_count": float(collision_events),
        "collision_event_count": int(collision_events),
        "collision_step_count": int(sum(collision_flags)),
        "collision_contact_instances": float(sum(collision_values)),
        "collision_metric_semantics": (
            "events=rising_edges; steps=active_steps; contact_instances="
            "sum_of_instantaneous_environment_contact_counts"
        ),
        "force_violation_rate": (
            float(np.mean(force_flags)) if force_flags else 0.0
        ),
        "max_force": max(force_values, default=0.0),
        "force_metric_semantics": "environment safety force_proxy (not network/tactile latency)",
        "final_progress": float(last.get("progress", 0.0)),
        "final_object_goal_distance": float(last.get("object_goal_distance", np.nan)),
        "done": bool(done),
        "truncated": bool(truncated),
    }


def run_episode(
    runtime: DecentralizedRuntime,
    recipe: EpisodeRecipe,
    *,
    manifest: Mapping[str, Any],
    mode: str,
    checkpoint_hashes: Mapping[str, str],
    evaluation_config_digest: str,
    args: argparse.Namespace,
    runtime_config: RuntimeConfig | None = None,
    video_path: Path | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    dataset_config = manifest.get("config", {})
    episode_len = int(dataset_config.get("episode_len", 500))
    max_steps = episode_len if int(args.max_steps) < 0 else min(int(args.max_steps), episode_len)
    randomize = (
        bool(dataset_config.get("randomize", True))
        if args.randomize is None
        else bool(args.randomize)
    )
    env = TwoRobotCarryNarrowPassageEnv(
        CarryEnvConfig(
            scenario=recipe.scenario,
            episode_len=episode_len,
            seed=recipe.seed,
        )
    )
    sensor_config = SensorSimulationConfig(
        control_dt=env.cfg.control_dt,
        object_dropout_prob=recipe.object_dropout_prob,
        object_position_std=float(dataset_config.get("object_position_std", 0.025)),
        object_yaw_std=float(dataset_config.get("object_yaw_std", 0.035)),
    )
    adapter = SimulationAdapter(env, sensor_config=sensor_config)
    payload = _external_episode_payload(
        recipe,
        checkpoint_hashes=checkpoint_hashes,
        max_steps=max_steps,
        randomize=randomize,
        environment=asdict(env.cfg),
        sensor=asdict(sensor_config),
    )
    input_digest = _digest_json(payload)

    policy_seed = (
        int(runtime_config.policy.seed)
        if runtime_config is not None
        else int(args.policy_seed)
    )
    expected_delay = (
        float(runtime_config.delay_steps)
        if runtime_config is not None
        else float(args.expected_delay_steps)
    )
    lambda_delay = (
        float(runtime_config.lambda_delay)
        if runtime_config is not None
        else float(args.lambda_delay)
    )
    expected_candidates_per_decision = (
        int(runtime_config.policy.num_candidates)
        if runtime_config is not None
        else int(args.num_candidates)
    )
    runtime.reset(seed=recipe.seed + policy_seed)
    adapter.reset(recipe.seed)
    obs = env.reset(seed=recipe.seed, randomize=randomize)
    info = dict(obs.get("metrics", {}))
    packets = adapter.packets(obs, info, previous_world_action=None)
    rewards: list[float] = []
    infos: list[Mapping[str, Any]] = []
    steps: list[dict[str, Any]] = []
    candidate_code_counts: Counter[int] = Counter()
    done = False
    env_steps = 0
    recorder: _RolloutVideoRecorder | None = None
    step_progress = tqdm(
        total=max_steps,
        desc=f"{mode} ep {recipe.episode_index:06d}",
        unit="step",
        position=1,
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    video_info: dict[str, Any] | None = None
    run_succeeded = False
    try:
        if video_path is not None:
            recorder = _RolloutVideoRecorder(
                video_path,
                model=env.model,
                width=int(args.video_width),
                height=int(args.video_height),
                fps=int(args.video_fps),
                label=(
                    f"{mode} | {recipe.scenario} | "
                    f"episode {recipe.episode_index}"
                ),
            )
        if recorder is not None:
            recorder.capture(env.data)
        while not done and env_steps < max_steps:
            decision = runtime.step(packets)
            ego_action = decision.joint_action.detach().cpu().numpy().astype(np.float32)
            if ego_action.shape != (8,) or not np.isfinite(ego_action).all():
                raise RuntimeError(
                    f"runtime emitted invalid joint action at step {env_steps}: {ego_action}"
                )
            world_action = ego_action_to_world(ego_action, obs)
            if not np.isfinite(world_action).all():
                raise RuntimeError(f"ego-to-world action transform failed at step {env_steps}")
            next_obs, reward, done, next_info = env.step(world_action)
            rows = _flatten_decision_steps(
                decision,
                env_steps,
                expected_delay,
                lambda_delay,
            )
            for row in rows:
                agent_id = int(row.get("agent_id", 0))
                cue_valid = np.asarray(
                    next_info.get("private_event_valid_agents", np.zeros(2)),
                    dtype=np.float32,
                )
                cues = np.asarray(
                    next_info.get("private_event_cue_agents", np.zeros((2, 3))),
                    dtype=np.float32,
                )
                row.update(
                    {
                        "private_event_active": bool(
                            next_info.get("private_event_active", False)
                        ),
                        "private_event_index": int(
                            next_info.get("private_event_index", -1)
                        ),
                        "private_event_type": int(
                            next_info.get("private_event_type", -1)
                        ),
                        "private_event_informed_agent": int(
                            next_info.get("private_event_informed_agent", -1)
                        ),
                        "private_event_maneuver": int(
                            next_info.get("private_event_maneuver", 0)
                        ),
                        "private_event_cue_valid": bool(cue_valid[agent_id] > 0.5),
                        "private_event_cue": cues[agent_id].tolist(),
                        "private_event_necessary": bool(
                            next_info.get("private_event_type", -1) == 0
                        ),
                        "private_event_redundant": bool(
                            next_info.get("private_event_type", -1) == 2
                        ),
                        "private_event_decision_correct": bool(
                            next_info.get("private_event_error_steps", 0) == 0
                        ),
                    }
                )
                candidate_code_counts.update(
                    int(value) for value in row.pop("candidate_codes", ())
                )
                # These vectors are artifact-level invariants or intermediate
                # diagnostics.  Keeping them on every decision makes a formal
                # rollout manifest unnecessarily large.
                row.pop("posterior_active_codes", None)
                row.pop("hypothesis_codes", None)
            steps.extend(rows)
            rewards.append(float(reward))
            infos.append(dict(next_info))
            env_steps += 1
            if recorder is not None:
                recorder.capture(env.data)
            step_progress.update(1)
            step_progress.set_postfix(
                progress=f"{float(next_info.get('progress', 0.0)):.3f}",
                refresh=False,
            )
            obs, info = next_obs, dict(next_info)
            if not done:
                packets = adapter.packets(
                    obs, info, previous_world_action=np.asarray(world_action, dtype=np.float32)
                )
        run_succeeded = True
    finally:
        step_progress.close()
        try:
            if recorder is not None:
                video_info = recorder.close(commit=run_succeeded)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    expected_candidate_observations = len(steps) * expected_candidates_per_decision
    observed_candidate_observations = sum(candidate_code_counts.values())
    if observed_candidate_observations != expected_candidate_observations:
        raise RuntimeError(
            "incomplete candidate-code diagnostics: "
            f"observed={observed_candidate_observations}, "
            f"expected={expected_candidate_observations}"
        )
    record = {
        "record_contract": "fe_pc_wam_closed_loop_episode",
        "mode": mode,
        "seed": recipe.seed,
        "episode_id": recipe.episode_id,
        "episode_index": recipe.episode_index,
        "split": recipe.split,
        "scenario": recipe.scenario,
        "source_episode": str(recipe.source_path),
        "input_digest": input_digest,
        "input_recipe": payload,
        "evaluation_config_digest": evaluation_config_digest,
        "environment_steps": env_steps,
        "decision_count": len(steps),
        "steps": steps,
        "candidate_codes": sorted(candidate_code_counts),
        "candidate_code_counts": {
            str(code): count for code, count in sorted(candidate_code_counts.items())
        },
        "candidate_code_observations": observed_candidate_observations,
        "expected_candidate_code_observations": expected_candidate_observations,
    }
    record.update(
        _episode_metrics(
            infos,
            rewards,
            done=done,
            truncated=not done and env_steps >= max_steps,
        )
    )
    if video_info is not None:
        record["video"] = video_info
    return record


def _video_artifact_valid(record: Mapping[str, Any], video_path: Path) -> bool:
    video = record.get("video")
    return bool(
        isinstance(video, Mapping)
        and video_path.is_file()
        and video.get("sha256") == file_sha256(video_path)
    )


def _video_artifact_path(
    output_dir: Path,
    mode: str,
    recipe: EpisodeRecipe,
    *,
    failure_reason: str | None = None,
) -> Path:
    """Return a stable video name, including the reason for failures."""

    suffix = ""
    if failure_reason is not None:
        slug = re.sub(
            r"[^a-z0-9._-]+",
            "_",
            str(failure_reason).strip().lower(),
        ).strip("._-")
        suffix = f"__{slug or 'unknown_failure'}"
    return (
        output_dir
        / "videos"
        / mode
        / f"episode_{recipe.episode_index:06d}{suffix}.mp4"
    )


def _relocate_valid_video(
    record: Mapping[str, Any],
    *,
    source: Path,
    destination: Path,
) -> bool:
    """Move a valid video to its canonical name without re-rendering it."""

    if source == destination:
        return _video_artifact_valid(record, destination)
    if _video_artifact_valid(record, destination):
        return True
    if not _video_artifact_valid(record, source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    return _video_artifact_valid(record, destination)


def _attach_failure_replay_video(
    original: dict[str, Any],
    replay: Mapping[str, Any],
    *,
    video_path: Path,
) -> None:
    """Attach a deterministic failure replay without replacing original metrics."""

    if bool(original.get("success", False)):
        raise RuntimeError("failure-video replay was requested for a successful episode")
    exact_fields = (
        "input_digest",
        "evaluation_config_digest",
        "success",
        "failure",
        "failure_reason",
        "done",
        "truncated",
        "environment_steps",
    )
    changed = [
        name for name in exact_fields if original.get(name) != replay.get(name)
    ]
    original_return = float(original.get("return", float("nan")))
    replay_return = float(replay.get("return", float("nan")))
    if not np.isclose(original_return, replay_return, rtol=1e-6, atol=1e-6):
        changed.append("return")
    if changed:
        raise RuntimeError(
            "failure-video replay was not deterministic; "
            f"changed={sorted(set(changed))}"
        )
    video = replay.get("video")
    if not isinstance(video, Mapping) or not _video_artifact_valid(replay, video_path):
        raise RuntimeError("failure-video replay did not produce a valid MP4 artifact")
    original["video"] = {
        **dict(video),
        "selection": "failure_replay",
        "replay_verified": True,
    }


def _record_path(output_dir: Path, mode: str, recipe: EpisodeRecipe) -> Path:
    return output_dir / "records" / mode / f"episode_{recipe.episode_index:06d}.json"


def _load_resumable_record(
    path: Path,
    *,
    mode: str,
    config_digest: str,
    recipe: EpisodeRecipe,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    input_recipe = record.get("input_recipe")
    if (
        record.get("mode") != mode
        or record.get("evaluation_config_digest") != config_digest
        or record.get("episode_id") != recipe.episode_id
        or int(record.get("seed", -1)) != recipe.seed
        or record.get("scenario") != recipe.scenario
        or not isinstance(input_recipe, Mapping)
        or input_recipe.get("source_sha256") != recipe.source_sha256
        or record.get("input_digest") != _digest_json(input_recipe)
    ):
        return None
    return record


def _request_rate(records: Sequence[Mapping[str, Any]]) -> float:
    decisions = sum(int(record.get("decision_count", 0)) for record in records)
    requests = 0.0
    for record in records:
        aggregate = record.get("decision_aggregates", {}).get("request")
        if isinstance(aggregate, Mapping):
            requests += float(aggregate.get("sum", 0.0))
            continue
        requests += sum(
            int(bool(step.get("request", False)))
            for step in record.get("steps", ())
            if isinstance(step, Mapping)
        )
    return 0.0 if decisions <= 0 else float(requests / decisions)


def _compact_episode_record(
    record: Mapping[str, Any],
    *,
    record_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a bounded-size manifest row while preserving the full log on disk."""

    raw_steps = record.get("steps", ())
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise RuntimeError("full rollout record has no step sequence")
    steps = [step for step in raw_steps if isinstance(step, Mapping)]
    if len(steps) != int(record.get("decision_count", -1)):
        raise RuntimeError("full rollout record decision_count does not match step rows")

    aggregates: dict[str, dict[str, float | int]] = {}
    for name in DECISION_AGGREGATE_KEYS:
        if name == "G_improvement":
            values = [
                float(step["G_before"]) - float(step["G_after"])
                for step in steps
                if step.get("G_before") is not None and step.get("G_after") is not None
            ]
        else:
            values = [float(step[name]) for step in steps if step.get(name) is not None]
        if values:
            aggregates[name] = {
                "sum": float(sum(values)),
                "count": len(values),
                "mean": float(sum(values) / len(values)),
            }

    selected_codes = Counter(
        int(step["plan_code"]) for step in steps if step.get("plan_code") is not None
    )
    copied_names = (
        "mode",
        "seed",
        "episode_id",
        "episode_index",
        "split",
        "scenario",
        "source_episode",
        "input_digest",
        "evaluation_config_digest",
        "environment_steps",
        "decision_count",
        "candidate_codes",
        "candidate_code_counts",
        "candidate_code_observations",
        "expected_candidate_code_observations",
        "success",
        "safe",
        "safety_metric_semantics",
        "failure",
        "failure_reason",
        "return",
        "collision_count",
        "collision_event_count",
        "collision_step_count",
        "collision_contact_instances",
        "collision_metric_semantics",
        "force_violation_rate",
        "max_force",
        "force_metric_semantics",
        "final_progress",
        "final_object_goal_distance",
        "done",
        "truncated",
        "video",
    )
    compact = {name: record[name] for name in copied_names if name in record}
    compact.update(
        {
            "record_contract": "fe_pc_wam_closed_loop_episode_compact",
            "full_record_path": str(record_path.relative_to(output_dir)),
            "full_record_sha256": file_sha256(record_path),
            "decision_aggregates": aggregates,
            "selected_plan_code_counts": {
                str(code): count for code, count in sorted(selected_codes.items())
            },
        }
    )
    return compact


def _record_set_attestation(
    records_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for mode in sorted(records_by_mode):
        rows: list[dict[str, Any]] = []
        for record in sorted(
            records_by_mode[mode],
            key=lambda value: (int(value.get("seed", -1)), str(value.get("episode_id"))),
        ):
            full_hash = str(record.get("full_record_sha256", ""))
            if len(full_hash) != 64 or any(
                character not in "0123456789abcdef" for character in full_hash.lower()
            ):
                raise RuntimeError("compact rollout record lacks a valid full-record SHA256")
            rows.append(
                {
                    "seed": int(record["seed"]),
                    "episode_id": str(record["episode_id"]),
                    "input_digest": str(record["input_digest"]),
                    "evaluation_config_digest": str(
                        record["evaluation_config_digest"]
                    ),
                    "full_record_sha256": full_hash,
                    "truncated": bool(record.get("truncated", False)),
                }
            )
        evidence[mode] = rows
    return _digest_json(evidence)


def _candidate_evidence(
    records_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    record_count = 0
    expected_total = 0
    observed_total = 0
    for mode, records in records_by_mode.items():
        for record in records:
            record_count += 1
            raw_counts = record.get("candidate_code_counts")
            if not isinstance(raw_counts, Mapping):
                raise RuntimeError(
                    f"{mode}/{record.get('episode_id')}: missing candidate_code_counts"
                )
            parsed = Counter({int(code): int(count) for code, count in raw_counts.items()})
            if any(code < 0 or count < 0 for code, count in parsed.items()):
                raise RuntimeError("candidate code/count values must be non-negative integers")
            observed = int(record.get("candidate_code_observations", -1))
            expected = int(record.get("expected_candidate_code_observations", -1))
            if sum(parsed.values()) != observed or observed != expected:
                raise RuntimeError(
                    f"{mode}/{record.get('episode_id')}: incomplete candidate evidence; "
                    f"counts={sum(parsed.values())}, observed={observed}, expected={expected}"
                )
            counts.update(parsed)
            observed_total += observed
            expected_total += expected
    if record_count == 0 or observed_total == 0:
        raise RuntimeError("paired rollout produced no candidate-code evidence")
    return {
        "candidate_codes": sorted(counts),
        "counts": {str(code): count for code, count in sorted(counts.items())},
        "coverage": {
            "record_count": record_count,
            "observed_candidate_codes": observed_total,
            "expected_candidate_codes": expected_total,
            "complete": observed_total == expected_total,
        },
    }


def _checkpoint_hashes(checkpoints: CheckpointSet) -> dict[str, str]:
    return {
        name: file_sha256(path)
        for name, path in checkpoints.runtime_paths().items()
    }


def _checkpoint_sensor_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    dataset = state.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError(" checkpoint lacks dataset contract metadata")
    return {
        "local_contact_semantics": str(
            dataset.get("local_contact_semantics", LEGACY_CONTACT_SEMANTICS)
        ),
        "local_force_semantics": str(
            dataset.get("local_force_semantics", LEGACY_FORCE_SEMANTICS)
        ),
        "local_force_units": dataset.get("local_force_units"),
        "local_force_scale_newtons": dataset.get("local_force_scale_newtons"),
        "local_sensor_provenance": dataset.get("local_sensor_provenance"),
    }


def _sensor_contracts_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    for name in (
        "local_contact_semantics",
        "local_force_semantics",
        "local_force_units",
        "local_sensor_provenance",
    ):
        if left.get(name) != right.get(name):
            return False
    left_scale = left.get("local_force_scale_newtons")
    right_scale = right.get("local_force_scale_newtons")
    if left_scale is None or right_scale is None:
        return left_scale is None and right_scale is None
    return bool(np.isclose(float(left_scale), float(right_scale)))


def _strict_sensor_contract(contract: Mapping[str, Any]) -> bool:
    scale = contract.get("local_force_scale_newtons")
    return bool(
        contract.get("local_contact_semantics")
        == STRICT_LOCAL_CONTACT_SEMANTICS
        and contract.get("local_force_semantics") == STRICT_LOCAL_FORCE_SEMANTICS
        and contract.get("local_force_units") == LOCAL_FORCE_UNITS
        and contract.get("local_sensor_provenance")
        == STRICT_LOCAL_SENSOR_PROVENANCE
        and scale is not None
        and np.isfinite(float(scale))
        and float(scale) > 0.0
    )


def _load_frozen_runtime_configs(
    path: str | Path,
    *,
    modes: Sequence[str],
    checkpoint_hashes: Mapping[str, str],
    device: str,
    resolved_device: str,
) -> tuple[dict[str, RuntimeConfig], dict[str, Any]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("frozen_config_contract") != "fe_pc_wam_validation_freeze":
        raise ValueError(
            "--frozen-config-from must point to a  frozen_config.json"
        )
    if payload.get("split") != "val":
        raise ValueError("--frozen-config-from must point to a validation freeze")
    if set(payload.get("modes", ())) != set(DEPLOYABLE_MODES):
        raise ValueError("frozen validation manifest must contain all five deployable modes")
    if (
        payload.get("episode_count") != OFFICIAL_SPLIT_EPISODES["val"]
        or payload.get("expected_split_episode_count")
        != OFFICIAL_SPLIT_EPISODES["val"]
    ):
        raise ValueError("frozen validation manifest must attest exactly 160 episodes")
    conditions = payload.get("validation_freeze_conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        REQUIRED_VALIDATION_FREEZE_CONDITIONS
    ):
        raise ValueError(
            "frozen validation manifest has incomplete/unknown freeze conditions"
        )
    failed_conditions = [name for name, passed in conditions.items() if passed is not True]
    if payload.get("validation_freeze_eligible") is not True:
        raise ValueError(
            "validation manifest is not eligible to freeze for test; "
            f"failed={failed_conditions}"
        )
    if failed_conditions:
        raise ValueError(f"frozen validation conditions are not all true: {failed_conditions}")
    if payload.get("paired_inputs_verified") is not True or payload.get(
        "input_digest_verified"
    ) is not True:
        raise ValueError("frozen validation lacks verified paired inputs/digests")
    if payload.get("baseline_budget_match", {}).get("passed") is not True:
        raise ValueError("frozen validation baselines do not match selective budget")
    if payload.get("checkpoint_hashes") != dict(checkpoint_hashes):
        raise ValueError("frozen validation used different checkpoints")

    def verified_reference(name: str) -> tuple[Path, dict[str, Any]]:
        reference = payload.get(name)
        if not isinstance(reference, Mapping):
            raise ValueError(f"frozen validation lacks {name} reference")
        reference_path = Path(str(reference.get("path", ""))).resolve()
        expected_hash = str(reference.get("sha256", ""))
        if not reference_path.is_file() or file_sha256(reference_path) != expected_hash:
            raise ValueError(f"frozen validation {name} reference/hash is invalid")
        value = json.loads(reference_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"frozen validation {name} is not a JSON mapping")
        return reference_path, value

    records_path, records_payload = verified_reference("records_manifest")
    audit_path, audit_payload = verified_reference("artifact_audit")
    snapshot_path, snapshot_payload = verified_reference("experiment_snapshot")
    if audit_payload.get("passed") is not True or payload["artifact_audit"].get(
        "passed"
    ) is not True:
        raise ValueError("frozen validation artifact audit did not pass")
    if audit_payload.get("candidate_code_coverage", {}).get("complete") is not True:
        raise ValueError("frozen validation candidate-code evidence is incomplete")
    validation_records = records_payload.get("modes")
    if not isinstance(validation_records, Mapping):
        raise ValueError("frozen validation records manifest has no compact modes")
    expected_counts = payload.get("validation_record_counts")
    if not isinstance(expected_counts, Mapping):
        raise ValueError("frozen validation lacks per-mode record counts")
    reference_pairs: dict[tuple[int, str], str] | None = None
    for mode in DEPLOYABLE_MODES:
        mode_records = validation_records.get(mode)
        if (
            not isinstance(mode_records, Sequence)
            or isinstance(mode_records, (str, bytes))
            or len(mode_records) != OFFICIAL_SPLIT_EPISODES["val"]
            or expected_counts.get(mode) != OFFICIAL_SPLIT_EPISODES["val"]
        ):
            raise ValueError(f"frozen validation {mode} must contain 160 compact records")
        pairs: dict[tuple[int, str], str] = {}
        for record in mode_records:
            if not isinstance(record, Mapping) or bool(record.get("truncated", False)):
                raise ValueError(f"frozen validation {mode} has invalid/truncated records")
            key = (int(record.get("seed", -1)), str(record.get("episode_id", "")))
            digest = str(record.get("input_digest", ""))
            if key in pairs or key[0] < 0 or not key[1] or not digest:
                raise ValueError(f"frozen validation {mode} has invalid paired keys")
            pairs[key] = digest
        if reference_pairs is None:
            reference_pairs = pairs
        elif pairs != reference_pairs:
            raise ValueError("frozen validation modes do not share paired inputs/digests")
    record_set_hash = _record_set_attestation(validation_records)
    if record_set_hash != payload.get("validation_record_set_sha256"):
        raise ValueError("frozen validation record-set attestation is invalid")
    snapshot_reference = payload["experiment_snapshot"]
    for name in (
        "source_tree_sha256",
        "dataset_manifest_sha256",
        "selected_episode_set_sha256",
        "environment_sha256",
    ):
        if snapshot_reference.get(name) != snapshot_payload.get(name):
            raise ValueError(f"frozen validation snapshot differs for {name}")
    raw_configs = payload.get("effective_mode_configs")
    if not isinstance(raw_configs, Mapping):
        raise ValueError("frozen validation manifest has no effective_mode_configs")
    result: dict[str, RuntimeConfig] = {}
    frozen_source_hashes: set[str] = set()
    frozen_manifest_hashes: set[str] = set()
    frozen_environment_hashes: set[str] = set()
    frozen_resolved_devices: set[str] = set()
    for mode in modes:
        entry = raw_configs.get(mode)
        if not isinstance(entry, Mapping):
            raise ValueError(f"frozen validation manifest has no config for {mode}")
        if entry.get("checkpoint_hashes") != dict(checkpoint_hashes):
            raise ValueError(
                f"frozen {mode} config was evaluated with different checkpoints"
            )
        expected_config_digest = _digest_json(entry)
        if any(
            record.get("evaluation_config_digest") != expected_config_digest
            for record in validation_records[mode]
        ):
            raise ValueError(
                f"frozen {mode} records do not attest the effective runtime config"
            )
        frozen_source_hashes.add(str(entry.get("source_tree_sha256", "")))
        frozen_manifest_hashes.add(str(entry.get("dataset_manifest_sha256", "")))
        frozen_environment_hashes.add(str(entry.get("environment_sha256", "")))
        frozen_resolved_devices.add(str(entry.get("resolved_device", "")))
        runtime_state = dict(entry.get("runtime", {}))
        policy_state = runtime_state.pop("policy", None)
        if not isinstance(policy_state, Mapping):
            raise ValueError(f"frozen {mode} config has no policy mapping")
        runtime_state["device"] = device
        result[mode] = RuntimeConfig(
            policy=DecentralizedPolicyConfig(**dict(policy_state)),
            **runtime_state,
        )
    if len(frozen_source_hashes) != 1 or "" in frozen_source_hashes:
        raise ValueError("frozen validation configs lack one shared source-tree hash")
    if len(frozen_manifest_hashes) != 1 or "" in frozen_manifest_hashes:
        raise ValueError("frozen validation configs lack one shared dataset-manifest hash")
    if len(frozen_environment_hashes) != 1 or "" in frozen_environment_hashes:
        raise ValueError("frozen validation configs lack one shared dependency-environment hash")
    if len(frozen_resolved_devices) != 1 or "" in frozen_resolved_devices:
        raise ValueError("frozen validation configs lack one resolved device")
    frozen_resolved_device = next(iter(frozen_resolved_devices))
    if (
        torch.device(frozen_resolved_device).type != "cuda"
        or torch.device(resolved_device).type != "cuda"
    ):
        raise ValueError("formal validation/test frozen configs require CUDA execution")
    return result, {
        "path": str(source),
        "sha256": file_sha256(source),
        "records_manifest_path": str(records_path),
        "records_manifest_sha256": file_sha256(records_path),
        "validation_record_set_sha256": record_set_hash,
        "artifact_audit_path": str(audit_path),
        "artifact_audit_sha256": file_sha256(audit_path),
        "experiment_snapshot_path": str(snapshot_path),
        "experiment_snapshot_sha256": file_sha256(snapshot_path),
        "selected_episode_set_sha256": snapshot_payload[
            "selected_episode_set_sha256"
        ],
        "validation_observed_selective_request_rate": payload.get(
            "observed_selective_request_rate"
        ),
        "validation_matched_selective_request_rate": payload.get(
            "matched_selective_request_rate"
        ),
        "source_tree_sha256": next(iter(frozen_source_hashes)),
        "dataset_manifest_sha256": next(iter(frozen_manifest_hashes)),
        "environment_sha256": next(iter(frozen_environment_hashes)),
        "resolved_device": frozen_resolved_device,
    }


def _git_output(args: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def build_reproducibility_snapshot(
    *,
    dataset_root: Path,
    checkpoints: CheckpointSet,
    checkpoint_hashes: Mapping[str, str],
    args: argparse.Namespace,
    recipes: Sequence[EpisodeRecipe],
    frozen_config_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_names = _git_output(
        ["ls-files", "--cached", "--others", "--exclude-standard"], PROJECT_ROOT
    )
    source_hashes: dict[str, str] = {}
    if source_names is not None:
        for name in sorted(line for line in source_names.splitlines() if line):
            path = PROJECT_ROOT / name
            if path.is_file():
                source_hashes[name] = file_sha256(path)
    manifest_path = dataset_root / "dataset_manifest.json"
    installed_packages = sorted(
        {
            (
                str(
                    distribution.metadata.get("Name")
                    or distribution.metadata.get("Summary")
                    or "unknown"
                ),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
        }
    )
    environment = {
        "resolved_device": _resolved_device_name(str(args.device)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "mujoco": getattr(mujoco, "__version__", None),
        "installed_packages": [
            {"name": name, "version": version}
            for name, version in installed_packages
        ],
    }
    arguments = vars(args)
    evaluation_arguments = {
        name: value
        for name, value in arguments.items()
        if name not in {"resume", "quiet", "output_dir"}
    }
    selected_episode_files = [
        {
            "episode_id": recipe.episode_id,
            "source_path": str(recipe.source_path),
            "source_sha256": recipe.source_sha256,
        }
        for recipe in recipes
    ]
    return {
        "snapshot_contract": "fe_pc_wam_reproducibility",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_output(["rev-parse", "HEAD"], PROJECT_ROOT),
        "git_status_porcelain": _git_output(["status", "--porcelain=v1"], PROJECT_ROOT),
        "source_tree_sha256": _digest_json(source_hashes),
        "source_file_sha256": source_hashes,
        "dataset_manifest": str(manifest_path.resolve()),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "selected_episode_set_sha256": _digest_json(selected_episode_files),
        "selected_episode_files": selected_episode_files,
        "checkpoints": {
            name: {"path": str(path), "sha256": checkpoint_hashes[name]}
            for name, path in checkpoints.runtime_paths().items()
        },
        "environment": environment,
        "environment_sha256": _digest_json(environment),
        "arguments": arguments,
        "evaluation_arguments_sha256": _digest_json(evaluation_arguments),
        "frozen_validation_config": (
            None if frozen_config_source is None else dict(frozen_config_source)
        ),
    }


def _validate_resume_snapshot(path: Path, current: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"--resume requires the original reproducibility snapshot: {path}"
        )
    previous = json.loads(path.read_text(encoding="utf-8"))
    fields = (
        "source_tree_sha256",
        "dataset_manifest_sha256",
        "checkpoints",
        "environment_sha256",
        "evaluation_arguments_sha256",
        "selected_episode_set_sha256",
        "frozen_validation_config",
    )
    changed = [name for name in fields if previous.get(name) != current.get(name)]
    if changed:
        raise RuntimeError(
            "refusing to resume with a different frozen experiment snapshot; "
            f"changed={changed}. Use a new output directory."
        )
    return previous


def _record_resume_event(output_dir: Path, args: argparse.Namespace) -> None:
    path = output_dir / "resume_history.json"
    history: list[dict[str, Any]] = []
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            history = payload
    history.append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": vars(args),
        }
    )
    _atomic_json(path, history)


def _write_validation_freeze_manifest(
    *,
    output_dir: Path,
    records_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    checkpoint_hashes: Mapping[str, str],
) -> Path:
    records_path = output_dir / "records.json"
    audit_path = output_dir / "artifact_audit.json"
    snapshot_path = output_dir / "experiment_snapshot.json"
    modes = records_manifest["modes"]
    payload = {
        "frozen_config_contract": "fe_pc_wam_validation_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": records_manifest["split"],
        "episode_count": records_manifest["episode_count"],
        "expected_split_episode_count": records_manifest[
            "expected_split_episode_count"
        ],
        "modes": list(modes),
        "validation_record_counts": {
            mode: len(records) for mode, records in modes.items()
        },
        "validation_record_set_sha256": _record_set_attestation(modes),
        "paired_inputs_verified": bool(summary.get("paired_inputs_verified", False)),
        "input_digest_verified": bool(summary.get("input_digest_verified", False)),
        "validation_freeze_eligible": bool(
            records_manifest.get("validation_freeze_eligible", False)
        ),
        "validation_freeze_conditions": records_manifest[
            "validation_freeze_conditions"
        ],
        "baseline_budget_match": records_manifest["baseline_budget_match"],
        "effective_mode_configs": records_manifest["effective_mode_configs"],
        "observed_selective_request_rate": records_manifest[
            "observed_selective_request_rate"
        ],
        "matched_selective_request_rate": records_manifest[
            "matched_selective_request_rate"
        ],
        "checkpoint_hashes": dict(checkpoint_hashes),
        "records_manifest": {
            "path": str(records_path.resolve()),
            "sha256": file_sha256(records_path),
        },
        "artifact_audit": {
            "path": str(audit_path.resolve()),
            "sha256": file_sha256(audit_path) if audit_path.is_file() else None,
            "passed": bool(
                records_manifest["validation_freeze_conditions"].get(
                    "artifact_and_candidate_audit_passed", False
                )
            ),
        },
        "experiment_snapshot": {
            "path": str(snapshot_path.resolve()),
            "sha256": file_sha256(snapshot_path),
            "source_tree_sha256": snapshot["source_tree_sha256"],
            "dataset_manifest_sha256": snapshot["dataset_manifest_sha256"],
            "selected_episode_set_sha256": snapshot[
                "selected_episode_set_sha256"
            ],
            "environment_sha256": snapshot["environment_sha256"],
        },
    }
    path = output_dir / "frozen_config.json"
    _atomic_json(path, payload)
    return path


def run_paired_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.video_episodes) < 0:
        raise ValueError("--video-episodes must be non-negative")
    for name in ("video_fps", "video_width", "video_height"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    unknown = [mode for mode in modes if mode not in DEPLOYABLE_MODES]
    if unknown or not modes:
        raise ValueError(
            f"--modes must contain only {DEPLOYABLE_MODES}; unknown={unknown}"
        )
    if len(set(modes)) != len(modes):
        raise ValueError("--modes contains duplicate entries")
    if args.match_baselines_to_selective and "selective_vpi" not in modes:
        raise ValueError("--match-baselines-to-selective requires selective_vpi")
    if args.split == "test" and args.match_baselines_to_selective:
        raise ValueError(
            "test cannot tune baseline rates; import the frozen validation config"
        )
    if args.split == "test" and not args.frozen_config_from:
        raise ValueError(
            "test evaluation requires --frozen-config-from validation frozen_config.json"
        )
    if args.split == "val" and args.frozen_config_from:
        raise ValueError("--frozen-config-from is only valid for the frozen test run")
    resolved_device = _resolved_device_name(str(args.device))

    checkpoints = resolve_checkpoints(
        args.checkpoint_dir,
        use_base_wam=bool(args.use_base_wam),
    )
    checkpoint_hashes = _checkpoint_hashes(checkpoints)
    checkpoint_sensor_contracts: dict[str, dict[str, Any]] = {}
    for name, path, stage in (
        ("plan", checkpoints.plan, "plan"),
        ("belief", checkpoints.belief, "belief"),
        (
            checkpoints.deployment_wam_stage,
            checkpoints.deployment_wam,
            checkpoints.deployment_wam_stage,
        ),
        ("intention", checkpoints.intention, "intention"),
    ):
        checkpoint_state = load_checkpoint(
            path, expected_stage=stage, map_location="cpu"
        )
        checkpoint_sensor_contracts[name] = _checkpoint_sensor_contract(
            checkpoint_state
        )
        del checkpoint_state
    reference_checkpoint_contract = checkpoint_sensor_contracts["plan"]
    checkpoint_sensor_contract_consistent = all(
        _sensor_contracts_equal(reference_checkpoint_contract, contract)
        for contract in checkpoint_sensor_contracts.values()
    )
    strict_checkpoint_sensor_contract = (
        checkpoint_sensor_contract_consistent
        and all(
            _strict_sensor_contract(contract)
            for contract in checkpoint_sensor_contracts.values()
        )
    )
    if not strict_checkpoint_sensor_contract:
        raise RuntimeError(
            "checkpoints do not satisfy the strict local contact/force contract; "
            "recollect the dataset and retrain"
        )
    frozen_runtime_configs: dict[str, RuntimeConfig] = {}
    frozen_config_source: dict[str, Any] | None = None
    if args.frozen_config_from:
        frozen_runtime_configs, frozen_config_source = _load_frozen_runtime_configs(
            args.frozen_config_from,
            modes=modes,
            checkpoint_hashes=checkpoint_hashes,
            device=str(args.device),
            resolved_device=resolved_device,
        )
    recipes, manifest = load_episode_recipes(
        dataset_root,
        args.split,
        max_episodes=int(args.max_episodes),
        scenarios=args.scenarios,
    )
    dataset_sensor_contract = manifest["validated_local_sensor_contract"]
    strict_dataset_sensor_contract = _strict_sensor_contract(
        dataset_sensor_contract
    )
    dataset_checkpoint_sensor_contract_compatible = (
        checkpoint_sensor_contract_consistent
        and _sensor_contracts_equal(
            dataset_sensor_contract, reference_checkpoint_contract
        )
    )
    if (
        not strict_dataset_sensor_contract
        or not dataset_checkpoint_sensor_contract_compatible
    ):
        raise RuntimeError(
            "validation/test data do not share the strict local sensor contract "
            "of the checkpoints; recollect all splits before evaluation"
        )
    completion_path = output_dir / "COMPLETED.json"
    if args.resume and completion_path.is_file():
        raise RuntimeError(
            f"evaluation directory is already complete: {completion_path}"
        )
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty evaluation directory {output_dir}; "
            "use a new directory or --resume an interrupted run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = build_reproducibility_snapshot(
        dataset_root=dataset_root,
        checkpoints=checkpoints,
        checkpoint_hashes=checkpoint_hashes,
        args=args,
        recipes=recipes,
        frozen_config_source=frozen_config_source,
    )
    if frozen_config_source is not None:
        for name in (
            "source_tree_sha256",
            "dataset_manifest_sha256",
            "environment_sha256",
        ):
            if frozen_config_source[name] != snapshot[name]:
                raise RuntimeError(
                    "frozen validation config does not match the current test snapshot; "
                    f"changed={name}"
                )
    snapshot_path = output_dir / "experiment_snapshot.json"
    if args.resume:
        snapshot = _validate_resume_snapshot(snapshot_path, snapshot)
        _record_resume_event(output_dir, args)
    else:
        _atomic_json(snapshot_path, snapshot)

    dataset_config = manifest.get("config", {})
    effective_randomize = (
        bool(dataset_config.get("randomize", True))
        if args.randomize is None
        else bool(args.randomize)
    )

    execution_order = list(modes)
    if args.match_baselines_to_selective:
        execution_order.remove("selective_vpi")
        execution_order.insert(0, "selective_vpi")
    records_by_mode: dict[str, list[dict[str, Any]]] = {}
    effective_configs: dict[str, Any] = {}
    matched_rate: float | None = None
    observed_selective_rate: float | None = None
    progress = tqdm(
        total=len(execution_order) * len(recipes),
        desc=" paired rollout",
        unit="episode",
        position=0,
        dynamic_ncols=True,
        disable=bool(args.quiet),
    )
    try:
        for mode in execution_order:
            if mode in frozen_runtime_configs:
                runtime_config = frozen_runtime_configs[mode]
            else:
                policy = _mode_policy_config(
                    mode,
                    args,
                    matched_request_rate=(
                        matched_rate
                        if args.match_baselines_to_selective
                        and mode in {"periodic", "random"}
                        else None
                    ),
                )
                runtime_config = _runtime_config(policy, args)
            config_payload = {
                "mode": mode,
                "runtime": asdict(runtime_config),
                "checkpoint_hashes": checkpoint_hashes,
                "source_tree_sha256": snapshot["source_tree_sha256"],
                "dataset_manifest_sha256": snapshot["dataset_manifest_sha256"],
                "environment_sha256": snapshot["environment_sha256"],
                "resolved_device": resolved_device,
                "split": args.split,
                "max_steps": args.max_steps,
                "randomize": effective_randomize,
                "frozen_validation_config": frozen_config_source,
            }
            config_digest = _digest_json(config_payload)
            effective_configs[mode] = config_payload
            progress.set_description(f" rollout [{mode}]")
            runtime = DecentralizedRuntime.from_checkpoints(
                plan_checkpoint=checkpoints.plan,
                belief_checkpoint=checkpoints.belief,
                wam_checkpoint=checkpoints.deployment_wam,
                intention_checkpoint=checkpoints.intention,
                config=runtime_config,
            )
            mode_records: list[dict[str, Any]] = []
            for recipe_position, recipe in enumerate(recipes):
                path = _record_path(output_dir, mode, recipe)
                generic_video_path = _video_artifact_path(
                    output_dir,
                    mode,
                    recipe,
                )
                requested_video = bool(args.render_video) and (
                    recipe_position < int(args.video_episodes)
                )
                initial_video_path = generic_video_path if requested_video else None
                record = (
                    _load_resumable_record(
                        path,
                        mode=mode,
                        config_digest=config_digest,
                        recipe=recipe,
                    )
                    if args.resume
                    else None
                )
                record_changed = False
                if record is not None and requested_video:
                    requested_video_path = _video_artifact_path(
                        output_dir,
                        mode,
                        recipe,
                        failure_reason=(
                            str(record.get("failure_reason", "unknown_failure"))
                            if not bool(record.get("success", False))
                            else None
                        ),
                    )
                    if not _relocate_valid_video(
                        record,
                        source=generic_video_path,
                        destination=requested_video_path,
                    ):
                        record = None
                    elif isinstance(record.get("video"), dict):
                        record["video"]["path"] = str(
                            requested_video_path.relative_to(output_dir)
                        )
                        record_changed = True
                reused = record is not None
                if record is None:
                    record = run_episode(
                        runtime,
                        recipe,
                        manifest=manifest,
                        mode=mode,
                        checkpoint_hashes=checkpoint_hashes,
                        evaluation_config_digest=config_digest,
                        args=args,
                        runtime_config=runtime_config,
                        video_path=initial_video_path,
                        show_progress=not bool(args.quiet),
                    )
                    record_changed = True
                    if initial_video_path is not None and "video" in record:
                        requested_video_path = _video_artifact_path(
                            output_dir,
                            mode,
                            recipe,
                            failure_reason=(
                                str(record.get("failure_reason", "unknown_failure"))
                                if not bool(record.get("success", False))
                                else None
                            ),
                        )
                        if not _relocate_valid_video(
                            record,
                            source=initial_video_path,
                            destination=requested_video_path,
                        ):
                            raise RuntimeError(
                                "requested rollout video could not be assigned its "
                                "canonical artifact name"
                            )
                        record["video"]["selection"] = (
                            "requested_failure"
                            if not bool(record.get("success", False))
                            else "requested"
                        )
                        record["video"]["path"] = str(
                            requested_video_path.relative_to(output_dir)
                        )

                failure_video_required = bool(args.save_failure_videos) and not bool(
                    record.get("success", False)
                )
                artifact_video_path = _video_artifact_path(
                    output_dir,
                    mode,
                    recipe,
                    failure_reason=(
                        str(record.get("failure_reason", "unknown_failure"))
                        if failure_video_required
                        else None
                    ),
                )
                if failure_video_required and _relocate_valid_video(
                    record,
                    source=generic_video_path,
                    destination=artifact_video_path,
                ):
                    if isinstance(record.get("video"), dict):
                        expected_relative_path = str(
                            artifact_video_path.relative_to(output_dir)
                        )
                        if record["video"].get("path") != expected_relative_path:
                            record["video"]["path"] = expected_relative_path
                            record_changed = True
                failure_video_replayed = False
                if failure_video_required and not _video_artifact_valid(
                    record, artifact_video_path
                ):
                    replay = run_episode(
                        runtime,
                        recipe,
                        manifest=manifest,
                        mode=mode,
                        checkpoint_hashes=checkpoint_hashes,
                        evaluation_config_digest=config_digest,
                        args=args,
                        runtime_config=runtime_config,
                        video_path=artifact_video_path,
                        show_progress=not bool(args.quiet),
                    )
                    _attach_failure_replay_video(
                        record,
                        replay,
                        video_path=artifact_video_path,
                    )
                    record["video"]["path"] = str(
                        artifact_video_path.relative_to(output_dir)
                    )
                    failure_video_replayed = True
                    record_changed = True

                if record_changed:
                    _atomic_json(path, record)
                compact_record = _compact_episode_record(
                    record,
                    record_path=path,
                    output_dir=output_dir,
                )
                mode_records.append(compact_record)
                progress.update(1)
                progress.set_postfix(
                    episode=recipe.episode_index,
                    scenario=recipe.scenario,
                    status=(
                        "reused+failure-video"
                        if reused and failure_video_replayed
                        else "reused"
                        if reused
                        else "ran+failure-video"
                        if failure_video_replayed
                        else "ran"
                    ),
                    success=int(bool(compact_record.get("success", False))),
                    steps=int(compact_record.get("environment_steps", 0)),
                    video=int("video" in compact_record),
                    refresh=True,
                )
            records_by_mode[mode] = mode_records
            if mode == "selective_vpi":
                observed_selective_rate = _request_rate(mode_records)
                if args.match_baselines_to_selective:
                    matched_rate = observed_selective_rate
            del runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close()

    ordered_records = {mode: records_by_mode[mode] for mode in modes}
    video_entries = [
        {
            "mode": mode,
            "episode_id": record["episode_id"],
            **dict(record["video"]),
        }
        for mode, records in ordered_records.items()
        for record in records
        if isinstance(record.get("video"), Mapping)
    ]
    videos_manifest_path: Path | None = None
    if args.render_video or args.save_failure_videos:
        videos_manifest_path = output_dir / "videos.json"
        _atomic_json(
            videos_manifest_path,
            {
                "video_manifest_contract": "fe_pc_wam_rollout_videos",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_videos_enabled": bool(args.save_failure_videos),
                "failure_video_method": "deterministic_full_episode_replay",
                "requested_episodes_per_mode": int(args.video_episodes),
                "requested_video_enabled": bool(args.render_video),
                "video_count": len(video_entries),
                "videos": video_entries,
            },
        )
    expected_split_episode_count = int(
        manifest.get("splits", {}).get(args.split, {}).get("episodes", len(recipes))
    )
    contains_truncated = any(
        bool(record.get("truncated", False))
        for records in ordered_records.values()
        for record in records
    )
    expected_delay_by_mode = {
        mode: float(effective_configs[mode]["runtime"]["delay_steps"])
        for mode in modes
    }
    request_rates_by_mode = {
        mode: _request_rate(records) for mode, records in ordered_records.items()
    }
    budget_match_tolerance = 0.02
    selective_rate = request_rates_by_mode.get("selective_vpi")
    baseline_rate_deltas = {
        mode: (
            None
            if selective_rate is None or mode not in request_rates_by_mode
            else request_rates_by_mode[mode] - selective_rate
        )
        for mode in ("periodic", "random")
    }
    baseline_budget_match = {
        "enabled": bool(args.match_baselines_to_selective),
        "target_selective_request_rate": selective_rate,
        "request_rates_by_mode": request_rates_by_mode,
        "absolute_tolerance": budget_match_tolerance,
        "rate_deltas": baseline_rate_deltas,
        "passed": bool(args.match_baselines_to_selective)
        and all(
            delta is not None and abs(delta) <= budget_match_tolerance
            for delta in baseline_rate_deltas.values()
        ),
    }
    formal_conditions = {
        "test_split": args.split == "test",
        "robust_wam_checkpoint": not checkpoints.uses_base_wam,
        "all_five_deployable_modes": set(modes) == set(DEPLOYABLE_MODES),
        "complete_split": len(recipes) == expected_split_episode_count,
        "official_split_size": expected_split_episode_count
        == OFFICIAL_SPLIT_EPISODES[args.split],
        "no_episode_or_scenario_filter": int(args.max_episodes) < 0
        and not args.scenarios,
        "full_episode_horizon": int(args.max_steps) < 0,
        "no_truncated_episodes": not contains_truncated,
        "frozen_validation_config_loaded": frozen_config_source is not None,
        "dataset_randomization_reproduced": effective_randomize
        == bool(dataset_config.get("randomize", True)),
        "cuda_execution": torch.device(resolved_device).type == "cuda"
        and torch.cuda.is_available(),
        "checkpoint_sensor_contract_consistent": checkpoint_sensor_contract_consistent,
        "strict_local_contact_checkpoint": all(
            contract["local_contact_semantics"]
            == STRICT_LOCAL_CONTACT_SEMANTICS
            for contract in checkpoint_sensor_contracts.values()
        ),
        "strict_local_force_checkpoint": strict_checkpoint_sensor_contract,
        "strict_local_sensor_dataset": strict_dataset_sensor_contract,
        "dataset_checkpoint_sensor_contract_compatible": (
            dataset_checkpoint_sensor_contract_compatible
        ),
        "local_force_scale_compatible": reference_checkpoint_contract.get(
            "local_force_scale_newtons"
        )
        is not None
        and np.isclose(
            float(reference_checkpoint_contract["local_force_scale_newtons"]),
            float(CarryEnvConfig().local_force_scale_newtons),
        ),
        "trained_message_metadata_distribution_respected": all(
            int(effective_configs[mode]["runtime"]["policy"][name]) == -1
            for mode in modes
            for name in (
                "metadata_available_index",
                "metadata_age_index",
                "metadata_confidence_index",
                "metadata_delay_index",
            )
        ),
        "validation_budget_matching_frozen": frozen_config_source is not None,
    }
    validation_freeze_conditions = {
        "validation_split": args.split == "val",
        "robust_wam_checkpoint": formal_conditions["robust_wam_checkpoint"],
        "all_five_deployable_modes": formal_conditions[
            "all_five_deployable_modes"
        ],
        "complete_split": formal_conditions["complete_split"],
        "official_split_size": formal_conditions["official_split_size"],
        "no_episode_or_scenario_filter": formal_conditions[
            "no_episode_or_scenario_filter"
        ],
        "full_episode_horizon": formal_conditions["full_episode_horizon"],
        "no_truncated_episodes": formal_conditions["no_truncated_episodes"],
        "dataset_randomization_reproduced": formal_conditions[
            "dataset_randomization_reproduced"
        ],
        "cuda_execution": formal_conditions["cuda_execution"],
        "checkpoint_sensor_contract_consistent": formal_conditions[
            "checkpoint_sensor_contract_consistent"
        ],
        "strict_local_contact_checkpoint": formal_conditions[
            "strict_local_contact_checkpoint"
        ],
        "strict_local_force_checkpoint": formal_conditions[
            "strict_local_force_checkpoint"
        ],
        "strict_local_sensor_dataset": formal_conditions[
            "strict_local_sensor_dataset"
        ],
        "dataset_checkpoint_sensor_contract_compatible": formal_conditions[
            "dataset_checkpoint_sensor_contract_compatible"
        ],
        "local_force_scale_compatible": formal_conditions[
            "local_force_scale_compatible"
        ],
        "trained_message_metadata_distribution_respected": formal_conditions[
            "trained_message_metadata_distribution_respected"
        ],
        "baseline_budget_matching_enabled_and_within_tolerance": baseline_budget_match[
            "passed"
        ],
    }
    records_manifest = {
        "record_manifest_contract": "fe_pc_wam_paired_rollouts",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "episode_count": len(recipes),
        "expected_split_episode_count": expected_split_episode_count,
        "contains_truncated_episodes": contains_truncated,
        "modes": ordered_records,
        "effective_mode_configs": effective_configs,
        "observed_selective_request_rate": observed_selective_rate,
        "baseline_matching_enabled": bool(args.match_baselines_to_selective),
        "matched_selective_request_rate": matched_rate,
        "frozen_validation_config": frozen_config_source,
        "deployment_wam_stage": checkpoints.deployment_wam_stage,
        "diagnostic_base_wam": checkpoints.uses_base_wam,
        "synchronous_transport": True,
        "realized_transport_delay_steps": 0.0,
        "expected_delay_cost_steps_by_mode": expected_delay_by_mode,
        "formal_protocol_conditions": formal_conditions,
        "validation_freeze_conditions": validation_freeze_conditions,
        "baseline_budget_match": baseline_budget_match,
        "local_contact_semantics": {
            "runtime": STRICT_LOCAL_CONTACT_SEMANTICS,
            "dataset": dataset_sensor_contract["local_contact_semantics"],
            "checkpoints": checkpoint_sensor_contracts,
            "compatible": dataset_checkpoint_sensor_contract_compatible
            and strict_checkpoint_sensor_contract,
        },
        "local_force_semantics": {
            "runtime": STRICT_LOCAL_FORCE_SEMANTICS,
            "dataset": dataset_sensor_contract,
            "checkpoints": checkpoint_sensor_contracts,
            "compatible": dataset_checkpoint_sensor_contract_compatible
            and strict_checkpoint_sensor_contract
            and formal_conditions["local_force_scale_compatible"],
            "runtime_scale_newtons": float(
                CarryEnvConfig().local_force_scale_newtons
            ),
            "checkpoint_scale_newtons": reference_checkpoint_contract.get(
                "local_force_scale_newtons"
            ),
        },
    }
    _atomic_json(output_dir / "records.json", records_manifest)

    summary = compare_communication_modes(ordered_records, required_modes=modes)
    summary["split"] = args.split
    summary["deployment_wam_stage"] = checkpoints.deployment_wam_stage
    summary["diagnostic_base_wam"] = checkpoints.uses_base_wam
    summary["synchronous_transport"] = True
    summary["realized_transport_delay_steps"] = 0.0
    summary["expected_delay_cost_steps_by_mode"] = expected_delay_by_mode
    summary["contains_truncated_episodes"] = contains_truncated
    summary["local_contact_semantics"] = records_manifest[
        "local_contact_semantics"
    ]
    summary["local_force_semantics"] = records_manifest[
        "local_force_semantics"
    ]

    candidate_evidence = _candidate_evidence(ordered_records)
    candidates = list(candidate_evidence["candidate_codes"])
    _atomic_json(output_dir / "candidate_codes.json", candidate_evidence)
    audit_passed = False
    if not args.skip_contract_audit:
        audit = run_contract_audit(
            [recipe.source_path for recipe in recipes],
            checkpoints.audit_paths(),
            candidate_codes=candidates,
            candidate_source=str(output_dir / "candidate_codes.json"),
            require_candidate_codes=True,
        )
        audit["candidate_code_coverage"] = candidate_evidence["coverage"]
        _atomic_json(output_dir / "artifact_audit.json", audit)
        audit_passed = bool(audit.get("passed", False))
    formal_conditions["artifact_and_candidate_audit_passed"] = audit_passed
    validation_freeze_conditions[
        "artifact_and_candidate_audit_passed"
    ] = audit_passed
    records_manifest["validation_freeze_eligible"] = all(
        validation_freeze_conditions.values()
    )
    records_manifest["validation_freeze_conditions"] = validation_freeze_conditions
    formal_eligible = all(formal_conditions.values())
    summary["formal_protocol"] = {
        "eligible": formal_eligible,
        "conditions": formal_conditions,
        "failed_conditions": [
            name for name, passed in formal_conditions.items() if not passed
        ],
    }
    records_manifest["formal_protocol_conditions"] = formal_conditions
    _atomic_json(output_dir / "records.json", records_manifest)
    if args.split == "val":
        frozen_path = _write_validation_freeze_manifest(
            output_dir=output_dir,
            records_manifest=records_manifest,
            summary=summary,
            snapshot=snapshot,
            checkpoint_hashes=checkpoint_hashes,
        )
        summary["validation_freeze"] = {
            "eligible": records_manifest["validation_freeze_eligible"],
            "path": str(frozen_path),
            "sha256": file_sha256(frozen_path),
        }
    if not formal_eligible:
        diagnostic_acceptance = summary.get("selective_vpi_acceptance")
        summary["selective_vpi_acceptance_diagnostic"] = diagnostic_acceptance
        summary["selective_vpi_acceptance"] = {
            "status": (
                "not_applicable_validation"
                if args.split == "val"
                else "not_applicable_incomplete_protocol"
            ),
            "passed": None,
            "reason": (
                "Base-WAM rollouts are diagnostic only; formal acceptance requires "
                "wam_robust.pt."
                if checkpoints.uses_base_wam
                else "Formal acceptance requires a frozen, complete 80-episode test protocol."
            ),
            "failed_conditions": summary["formal_protocol"]["failed_conditions"],
        }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(
        completion_path,
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "formal_protocol_eligible": formal_eligible,
            "records_sha256": file_sha256(output_dir / "records.json"),
            "summary_sha256": file_sha256(output_dir / "summary.json"),
            "videos_manifest_sha256": (
                file_sha256(videos_manifest_path)
                if videos_manifest_path is not None
                else None
            ),
            "frozen_config_sha256": (
                file_sha256(output_dir / "frozen_config.json")
                if (output_dir / "frozen_config.json").is_file()
                else None
            ),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired decentralized FE-PC-WAM  closed-loop evaluation"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--use-base-wam",
        action="store_true",
        help=(
            "use wam.pt instead of wam_robust.pt for diagnostic rollouts; "
            "the resulting run is never eligible for formal acceptance"
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", default=",".join(DEPLOYABLE_MODES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--scenarios", nargs="*", default=[])
    parser.add_argument("--randomize", type=int, choices=(0, 1), default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="render MP4 videos for the first N selected episodes of each mode",
    )
    failure_video = parser.add_mutually_exclusive_group()
    failure_video.add_argument(
        "--save-failure-videos",
        dest="save_failure_videos",
        action="store_true",
        default=True,
        help=(
            "save a full MP4 for every unsuccessful episode by deterministic "
            "replay (default)"
        ),
    )
    failure_video.add_argument(
        "--no-save-failure-videos",
        dest="save_failure_videos",
        action="store_false",
        help="disable the default failure-episode MP4 replay",
    )
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--skip-contract-audit", action="store_true")
    parser.add_argument(
        "--frozen-config-from",
        help="validation frozen_config.json whose configs/evidence must be reused for test",
    )

    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--num-teammate-hypotheses", type=int, default=4)
    parser.add_argument("--residual-sigma-points", type=int, choices=(1, 3), default=3)
    parser.add_argument("--residual-sigma-scale", type=float, default=1.0)
    parser.add_argument("--candidate-residual-scale", type=float, default=1.0)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--cooldown-steps", type=int, default=8)
    parser.add_argument("--plan-valid-steps", type=int, default=1)
    parser.add_argument("--periodic-interval", type=int, default=8)
    parser.add_argument("--random-request-probability", type=float, default=0.1)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--use-untrained-message-metadata",
        action="store_true",
        help=(
            "enable cache metadata inputs although the current intention checkpoint "
            "was trained with all-zero metadata"
        ),
    )
    parser.add_argument("--match-baselines-to-selective", action="store_true")

    parser.add_argument("--progress-target", type=float, default=1.0)
    parser.add_argument("--force-limit", type=float, default=1.0)
    parser.add_argument("--alpha-goal", type=float, default=1.0)
    parser.add_argument("--alpha-safety", type=float, default=2.0)
    parser.add_argument("--alpha-collab", type=float, default=1.0)
    parser.add_argument("--alpha-unc", type=float, default=0.5)
    parser.add_argument("--alpha-ctrl", type=float, default=0.05)
    parser.add_argument("--lambda-bits", type=float, default=1e-4)
    parser.add_argument("--lambda-delay", type=float, default=0.05)
    parser.add_argument("--expected-delay-steps", type=float, default=1.0)
    parser.add_argument("--delta-margin", type=float, default=0.0)
    parser.add_argument("--return-scale", type=float, default=100.0)
    parser.add_argument("--tail-risk-weight", type=float, default=0.5)
    parser.add_argument("--constraint-risk-weight", type=float, default=1.0)
    parser.add_argument("--success-risk-weight", type=float, default=0.5)
    parser.add_argument("--safety-probability-threshold", type=float, default=0.5)
    parser.add_argument("--utility-calibration-scale", type=float, default=1.0)
    parser.add_argument("--utility-calibration-bias", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_paired_evaluation(args)
    if args.quiet:
        output_dir = Path(args.output_dir).resolve()
        print(
            json.dumps(
                {
                    "summary": str(output_dir / "summary.json"),
                    "records": str(output_dir / "records.json"),
                    "formal_protocol_eligible": summary.get(
                        "formal_protocol", {}
                    ).get("eligible"),
                    "acceptance_status": summary.get(
                        "selective_vpi_acceptance", {}
                    ).get("status"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
