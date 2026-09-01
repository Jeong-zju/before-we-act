from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .common import atomic_json, sha256_file


ROOT = Path("/workspace/repos/before-we-act")
DP_ROOT = Path("/workspace/repos/RoboFactory/robofactory/policy/Diffusion-Policy")
DATA = Path("/workspace/runs/duobench-dp/data")
BASE = Path("/workspace/runs/duobench-dp/formal/final.pt")
ABLATION = Path("/workspace/runs/duobench-dp/closed_loop_ablation")
RUN = Path("/workspace/runs/duobench-dp/training_probe")
IMPROVED = Path("/workspace/runs/duobench-dp/improved")
PYTHON = "/venv/main/bin/python"
STATUS = RUN / "status.json"
LOG = RUN / "logs"
active: subprocess.Popen | None = None
stopping = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(stage: str, state: str = "running", **extra) -> None:
    atomic_json(
        STATUS,
        {
            "schema": "duobench.dp.training-probe-supervisor.v1",
            "stage": stage,
            "state": state,
            "updated_at": now(),
            "base_checkpoint": str(BASE),
            "base_checkpoint_sha256": sha256_file(BASE),
            "formal_results_are_immutable": True,
            "all_550_demonstrations_no_split": True,
            **extra,
        },
    )


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": f"{ROOT}:{DP_ROOT}:/workspace/repos/duobench/src",
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_GL": "egl",
            "DUOBENCH_PREFIX": "/workspace/datasets/duobench_assets",
            "HF_HOME": "/workspace/.hf_home",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "LD_LIBRARY_PATH": "/venv/main/lib/python3.12/site-packages/mujoco:"
            + env.get("LD_LIBRARY_PATH", ""),
        }
    )
    return env


def stop(_signum, _frame) -> None:
    global stopping
    stopping = True
    if active is not None:
        try:
            os.killpg(active.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def run(stage: str, command: list[str], retries: int = 3) -> None:
    global active
    LOG.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        write_status(stage, attempt=attempt, command=command, log=str(LOG / f"{stage}.log"))
        with (LOG / f"{stage}.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "launch", "time": now(), "command": command}) + "\n")
            stream.flush()
            active = subprocess.Popen(
                command,
                cwd=ROOT,
                env=runtime_env(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = active.wait()
        active = None
        if returncode == 0:
            return
        if stopping:
            raise RuntimeError("training probe supervisor was stopped")
        write_status(stage, "retrying", attempt=attempt, exit_code=returncode)
        time.sleep(min(60, attempt * 10))
    raise RuntimeError(f"{stage} failed after {retries} attempts")


def wait_for_ablation() -> dict:
    while not stopping:
        decision_path = ABLATION / "decision.json"
        if decision_path.is_file():
            try:
                decision = json.loads(decision_path.read_text())
            except json.JSONDecodeError:
                decision = {}
            if decision.get("status") == "complete":
                return decision
        status_path = ABLATION / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("state") == "failed":
                raise RuntimeError(f"closed-loop ablation failed: {status.get('error')}")
        write_status(
            "waiting_for_closed_loop_ablation",
            ablation_status=str(status_path),
            ablation_decision=str(decision_path),
        )
        time.sleep(30)
    raise RuntimeError("stopped while waiting for closed-loop ablation")


def train_command(
    output: Path,
    config: dict,
    steps: int,
    *,
    init: bool,
    smoke: bool = False,
    resume: bool = False,
):
    command = [
        PYTHON,
        "-m",
        "deployment.duo_dp.train",
        "--data",
        str(DATA),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--batch-size",
        "8" if smoke else "64",
        "--workers",
        "0" if smoke else "12",
        "--save-every",
        "10" if smoke else "5000",
        "--seed",
        str(config.get("seed", 20260901)),
        "--transition-fraction",
        str(config["transition_fraction"]),
        "--gripper-loss-weight",
        str(config["gripper_loss_weight"]),
        "--learning-rate",
        str(config.get("learning_rate", 5e-5)),
        "--warmup",
        str(config.get("warmup", 200)),
    ]
    if config["task_conditioning"]:
        command.append("--task-conditioning")
    if init:
        command += ["--init-checkpoint", str(BASE)]
    if resume:
        command.append("--resume")
    if smoke:
        command.append("--smoke")
    return command


def eval_command(checkpoint: Path, output: Path, config: dict, episodes: int, *, smoke=False):
    command = [
        PYTHON,
        "-m",
        "deployment.duo_dp.validation_launcher",
        "--checkpoint",
        str(checkpoint),
        "--data",
        str(DATA),
        "--output",
        str(output),
        "--episodes",
        str(episodes),
        "--workers",
        "1" if smoke else "3",
        "--inference-steps",
        "2" if smoke else str(config["inference_steps"]),
        "--weights",
        "ema",
        "--replan-steps",
        "1" if smoke else str(config["replan_steps"]),
    ]
    if smoke:
        command += ["--max-steps", "2", "--smoke"]
    return command


def training_complete(output: Path, steps: int, config: dict) -> bool:
    path = output / "status.json"
    if not path.is_file() or not (output / "final.pt").is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        value.get("status") == "complete"
        and value.get("step") == steps
        and value.get("task_conditioning") == config["task_conditioning"]
        and value.get("transition_fraction") == config["transition_fraction"]
        and value.get("gripper_loss_weight") == config["gripper_loss_weight"]
    )


def validation_complete(output: Path, episodes: int, checkpoint: Path) -> dict | None:
    path = output / "summary.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if (
        value.get("status") == "complete"
        and value.get("total_episodes") == episodes * 11
        and value.get("checkpoint_sha256") == sha256_file(checkpoint)
    ):
        return value
    return None


def normalized_max(summary: dict) -> float:
    if "normalized_max_stage_progress" in summary:
        return float(summary["normalized_max_stage_progress"])
    return sum(float(row["max_stage_progress"]) for row in summary["rows"]) / len(summary["rows"])


def score(summary: dict) -> float:
    return (
        float(summary["macro_success_rate"])
        + 0.20 * float(summary["normalized_final_stage_progress"])
        + 0.05 * normalized_max(summary)
    )


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        decision = wait_for_ablation()
        best_closed_loop = decision["best_probe"]
        eval_config = {
            "replan_steps": int(best_closed_loop["replan_steps"]),
            "inference_steps": int(best_closed_loop["inference_steps"]),
        }
        # An already healthy full result does not justify changing training.
        full = decision.get("full_validation20")
        if full and float(full["macro_success_rate"]) >= 0.10:
            write_status(
                "complete",
                "complete",
                decision="closed-loop execution fix reached >=10%; retraining skipped",
                ablation_decision=str(ABLATION / "decision.json"),
            )
            return

        smoke_config = {
            "task_conditioning": True,
            "transition_fraction": 0.25,
            "gripper_loss_weight": 4.0,
            "learning_rate": 5e-5,
            "warmup": 2,
            "seed": 20260901,
        }
        smoke_train = RUN / "smoke" / "train"
        if not training_complete(smoke_train, 10, smoke_config):
            run(
                "smoke_train",
                train_command(smoke_train, smoke_config, 10, init=True, smoke=True),
            )
        smoke_validation = RUN / "smoke" / "validation"
        if validation_complete(smoke_validation, 1, smoke_train / "final.pt") is None:
            run(
                "smoke_validation",
                eval_command(
                    smoke_train / "final.pt", smoke_validation, eval_config, 1, smoke=True
                ),
            )

        variants = {
            "transition_only": {
                "task_conditioning": False,
                "transition_fraction": 0.25,
                "gripper_loss_weight": 4.0,
                "learning_rate": 5e-5,
                "warmup": 200,
                "seed": 20260911,
            },
            "task_only": {
                "task_conditioning": True,
                "transition_fraction": 0.0,
                "gripper_loss_weight": 1.0,
                "learning_rate": 5e-5,
                "warmup": 200,
                "seed": 20260921,
            },
            "task_transition": {
                "task_conditioning": True,
                "transition_fraction": 0.25,
                "gripper_loss_weight": 4.0,
                "learning_rate": 5e-5,
                "warmup": 200,
                "seed": 20260931,
            },
        }
        rows = []
        for name, config in variants.items():
            train_output = RUN / name / "train15k"
            if not training_complete(train_output, 15000, config):
                run(
                    f"{name}_train15k",
                    train_command(train_output, config, 15000, init=True),
                )
            checkpoint = train_output / "final.pt"
            validation_output = RUN / name / "validation3"
            summary = validation_complete(validation_output, 3, checkpoint)
            if summary is None:
                run(
                    f"{name}_validation3",
                    eval_command(checkpoint, validation_output, eval_config, 3),
                )
                summary = validation_complete(validation_output, 3, checkpoint)
            if summary is None:
                raise RuntimeError(f"{name} validation did not produce a valid summary")
            rows.append(
                {
                    "name": name,
                    **config,
                    **eval_config,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "successes": summary["successes"],
                    "macro_success_rate": summary["macro_success_rate"],
                    "normalized_final_stage_progress": summary[
                        "normalized_final_stage_progress"
                    ],
                    "normalized_max_stage_progress": normalized_max(summary),
                    "score": score(summary),
                    "summary": str(validation_output / "summary.json"),
                }
            )
        rows.sort(key=lambda row: row["score"], reverse=True)
        best = rows[0]
        baseline_score = float(best_closed_loop["score"])
        improved = (
            best["macro_success_rate"] > float(best_closed_loop["macro_success_rate"])
            or best["score"] >= baseline_score + 0.01
        )
        probe_decision = {
            "schema": "duobench.dp.training-probe-decision.v1",
            "status": "probe_complete",
            "created_at": now(),
            "base_checkpoint": str(BASE),
            "base_checkpoint_sha256": sha256_file(BASE),
            "closed_loop_baseline": best_closed_loop,
            "variants": rows,
            "best_probe": best,
            "meaningful_improvement": improved,
            "selection_rule": "success improves or composite score improves by >=0.01",
            "all_data_no_split": True,
        }
        atomic_json(RUN / "decision.json", probe_decision)
        if not improved:
            probe_decision.update(
                {
                    "status": "complete",
                    "next_stage": "no full retrain; architecture/data intervention is not validated",
                }
            )
            atomic_json(RUN / "decision.json", probe_decision)
            write_status(
                "complete",
                "complete",
                decision=str(RUN / "decision.json"),
                next_stage=probe_decision["next_stage"],
            )
            return

        formal_config = variants[best["name"]].copy()
        formal_config.update({"learning_rate": 1e-4, "warmup": 500, "seed": 20261001})
        formal_train = IMPROVED / best["name"] / "formal60k"
        if not training_complete(formal_train, 60000, formal_config):
            # Once this subprocess starts, the supervisor owns all monitoring,
            # checkpointing, retries, and the subsequent Validation20.
            run(
                "improved_formal_train60k",
                train_command(formal_train, formal_config, 60000, init=False, resume=True),
            )
        formal_checkpoint = formal_train / "final.pt"
        formal_validation = IMPROVED / best["name"] / "validation20"
        full_summary = validation_complete(formal_validation, 20, formal_checkpoint)
        if full_summary is None:
            run(
                "improved_validation20",
                eval_command(formal_checkpoint, formal_validation, eval_config, 20),
            )
            full_summary = validation_complete(formal_validation, 20, formal_checkpoint)
        if full_summary is None:
            raise RuntimeError("improved Validation20 did not produce a valid summary")
        report = {
            "schema": "duobench.dp.improved-final-report.v1",
            "status": "complete",
            "created_at": now(),
            "selected_probe": best,
            "formal_config": formal_config,
            "formal_checkpoint": str(formal_checkpoint),
            "formal_checkpoint_sha256": sha256_file(formal_checkpoint),
            "validation20": {
                "summary": str(formal_validation / "summary.json"),
                "episodes": full_summary["total_episodes"],
                "successes": full_summary["successes"],
                "macro_success_rate": full_summary["macro_success_rate"],
                "normalized_final_stage_progress": full_summary[
                    "normalized_final_stage_progress"
                ],
            },
            "original_formal_results_preserved": True,
        }
        atomic_json(IMPROVED / "final_report.json", report)
        probe_decision.update(
            {
                "status": "complete",
                "next_stage": "improved formal training and Validation20 complete",
                "improved_final_report": str(IMPROVED / "final_report.json"),
            }
        )
        atomic_json(RUN / "decision.json", probe_decision)
        write_status(
            "complete",
            "complete",
            decision=str(RUN / "decision.json"),
            improved_final_report=str(IMPROVED / "final_report.json"),
        )
    except Exception as error:
        write_status(
            "failed",
            "failed",
            error=repr(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
