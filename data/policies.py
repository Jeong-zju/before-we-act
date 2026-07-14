from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.two_robot_carry_env import TwoRobotCarryNarrowPassageEnv


@dataclass
class PolicyOutput:
    action: np.ndarray
    phase: str


class ScriptedPolicy:
    def __init__(self, noise_std: float = 0.0, seed: int = 0, mode: str = "scripted"):
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.mode = mode

    @staticmethod
    def infer_phase(env: TwoRobotCarryNarrowPassageEnv) -> str:
        obj = env._object_pose_xy_yaw()
        y = float(obj[1])

        if env._success():
            return "done"
        if not env.grasped:
            return "grasp"
        if y < -0.20:
            return "carry_to_passage"
        if -0.20 <= y < 2.35:
            return "passage"
        if y < env.cfg.goal_y - env.cfg.goal_tol_xy:
            return "carry_to_goal"
        return "release"

    def __call__(self, env: TwoRobotCarryNarrowPassageEnv) -> PolicyOutput:
        action = env.scripted_action().astype(np.float64)
        phase = self.infer_phase(env)

        if self.mode == "noisy":
            action = self.apply_noisy_disturbance(action, phase)
        elif self.mode == "recovery":
            action = self.apply_recovery_bias(env, action, phase)
        elif self.mode == "exploratory":
            action = self.apply_exploratory_disturbance(action, phase)
        elif self.mode == "near_miss":
            action = self.apply_near_miss_bias(action, phase)
        elif self.mode != "scripted":
            raise ValueError(f"unknown collection policy mode {self.mode!r}")

        return PolicyOutput(action=np.clip(action, -1.0, 1.0), phase=phase)

    def apply_noisy_disturbance(self, action: np.ndarray, phase: str) -> np.ndarray:
        noisy = action.copy()

        # Do not interpret large noise_std as unbounded velocity.
        # It increases event probabilities; action is still bounded by [-1, 1].
        strength = float(np.clip(self.noise_std, 0.0, 2.0))
        event_scale = float(np.clip(self.noise_std / 0.5, 0.0, 1.0))

        if strength > 0:
            noisy[:3] += self.rng.normal(0.0, 0.35 * strength, size=3)
            noisy[4:7] += self.rng.normal(0.0, 0.35 * strength, size=3)

        # One teammate lags or moves backward.
        if phase == "passage" and self.rng.random() < 0.20 + 0.45 * event_scale:
            if self.rng.random() < 0.5:
                noisy[1] = self.rng.uniform(-0.8, 0.1)
            else:
                noisy[5] = self.rng.uniform(-0.8, 0.1)

        # Opposite lateral bias creates twist and disagreement.
        if phase in {"carry_to_passage", "passage"} and self.rng.random() < 0.15 + 0.45 * event_scale:
            bias = self.rng.choice([-0.9, 0.9])
            noisy[0] += bias
            noisy[4] -= bias

        # Opposite yaw commands.
        if phase in {"carry_to_passage", "passage"} and self.rng.random() < 0.10 + 0.35 * event_scale:
            yaw_bias = self.rng.choice([-1.0, 1.0])
            noisy[2] += yaw_bias
            noisy[6] -= yaw_bias

        # Gripper failure.
        if phase in {"grasp", "carry_to_passage", "passage"} and self.rng.random() < 0.05 + 0.25 * event_scale:
            if self.rng.random() < 0.5:
                noisy[3] = 0.0
            else:
                noisy[7] = 0.0

        # Full action dropout for one robot.
        if phase in {"carry_to_passage", "passage"} and self.rng.random() < 0.05 + 0.20 * event_scale:
            if self.rng.random() < 0.5:
                noisy[0:3] = 0.0
            else:
                noisy[4:7] = 0.0

        return noisy

    def apply_recovery_bias(self, env: TwoRobotCarryNarrowPassageEnv, action: np.ndarray, phase: str) -> np.ndarray:
        recovered = action.copy()
        obj = env._object_pose_xy_yaw()
        force_proxy = env._compute_force_proxy()

        if phase == "passage":
            center_bias = -np.clip(obj[0] * 1.8, -0.4, 0.4)
            recovered[0] += center_bias
            recovered[4] += center_bias

        if force_proxy > 0.15:
            recovered[1] *= 0.4
            recovered[5] *= 0.4
            recovered[0] += -np.sign(obj[0]) * 0.3
            recovered[4] += -np.sign(obj[0]) * 0.3

        return recovered

    def apply_exploratory_disturbance(self, action: np.ndarray, phase: str) -> np.ndarray:
        explored = self.apply_noisy_disturbance(action, phase)
        if self.rng.random() < 0.30:
            agent = int(self.rng.integers(0, 2))
            start = 4 * agent
            explored[start : start + 3] = self.rng.uniform(-1.0, 1.0, size=3)
        return explored

    def apply_near_miss_bias(self, action: np.ndarray, phase: str) -> np.ndarray:
        biased = action.copy()
        if phase in {"carry_to_passage", "passage"}:
            side = float(self.rng.choice((-1.0, 1.0)))
            biased[0] += 0.75 * side
            biased[4] -= 0.75 * side
            biased[2] += 0.50 * side
            biased[6] -= 0.50 * side
            if self.rng.random() < 0.15:
                biased[3 if self.rng.random() < 0.5 else 7] = 0.0
        return biased


def robot_distance_from_obs(obs: dict) -> float:
    gs = obs["global_state"]
    ax, ay = gs[0], gs[1]
    bx, by = gs[3], gs[4]
    return float(np.linalg.norm(np.array([ax - bx, ay - by])))
