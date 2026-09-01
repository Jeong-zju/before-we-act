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


ROOT = Path(os.environ.get("DUO_DP_REPO", "/workspace/repos/before-we-act"))
CONFIG = Path(
    os.environ.get("DUO_DP_CONFIG", str(ROOT / "configs" / "duobench_dp_formal_v1.json"))
)
DP_ROOT = Path(os.environ.get("DUO_DP_UPSTREAM", "/workspace/repos/RoboFactory/robofactory/policy/Diffusion-Policy"))
DATASET = Path(os.environ.get("DUO_DP_DATASET", "/workspace/datasets/duobench"))
RUN = Path(os.environ.get("DUO_DP_RUN", "/workspace/runs/duobench-dp"))
DATA = Path(os.environ.get("DUO_DP_DATA", str(RUN / "data")))
PREPARED_SOURCE = Path(
    os.environ.get("DUO_DP_PREPARED_SOURCE", "/workspace/runs/duobench-act/data_unclipped")
)
PYTHON = os.environ.get("DUO_DP_PYTHON", "/venv/main/bin/python")
STATUS = RUN / "status.json"
LOG = RUN / "logs"
active: subprocess.Popen | None = None
stopping = False


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(stage: str, state: str = "running", **extra) -> None:
    atomic_json(
        STATUS,
        {
            "schema": "duobench.dp.supervisor.v1",
            "stage": stage,
            "state": state,
            "updated_at": now(),
            "gpu_schedule": (
                "GPU0 exclusive during preflight/smoke/formal training; after training, "
                "three CPU simulators share batched GPU0 inference"
            ),
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
            "DUO_DP_REPO": str(ROOT),
            "DUO_DP_RUN": str(RUN),
            "DUO_DP_DATA": str(DATA),
        }
    )
    return env


def run(stage: str, command: list[str], retries: int = 3) -> None:
    global active
    LOG.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        status(stage, attempt=attempt, command=command, log=str(LOG / f"{stage}.log"))
        with (LOG / f"{stage}.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "launch", "time": now(), "attempt": attempt, "command": command}) + "\n")
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
        status(stage, "retrying", attempt=attempt, exit_code=returncode)
        if attempt < retries:
            time.sleep(min(60, 10 * attempt))
    raise RuntimeError(f"{stage} exited {returncode} after {retries} attempts")


def stop(_signal, _frame) -> None:
    global stopping
    stopping = True
    if active is not None:
        try:
            os.killpg(active.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def valid_json(path: Path, **expected) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(value.get(key) == target for key, target in expected.items())


def load_config() -> dict:
    """Load the paper-facing frozen contract used by the formal launcher."""
    if not CONFIG.is_file():
        raise FileNotFoundError(f"frozen DuoBench DP config not found: {CONFIG}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("protocol") != "duobench_dp_formal_v1":
        raise RuntimeError("unexpected DuoBench DP config protocol")
    return config


def bootstrap_data() -> None:
    if (DATA / "manifest.json").is_file():
        return
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if PREPARED_SOURCE.is_dir() and (PREPARED_SOURCE / "manifest.json").is_file():
        if DATA.exists() or DATA.is_symlink():
            raise RuntimeError(f"incomplete data target already exists: {DATA}")
        os.symlink(PREPARED_SOURCE, DATA, target_is_directory=True)
        status("data_bootstrap", source=str(PREPARED_SOURCE), mode="audited_prepared_data_symlink")
        return
    run(
        "data_prepare",
        [
            PYTHON,
            "-m",
            "deployment.duo_act.prepare",
            "--dataset",
            str(DATASET),
            "--output",
            str(DATA),
            "--image-size",
            "224",
            "--jobs",
            "6",
        ],
        retries=2,
    )


def main():
    global stopping
    RUN.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        config = load_config()
        train_cfg = config["optimization"]
        loader_cfg = config["loader"]
        validation_cfg = config["validation20"]
        smoke_cfg = {
            "updates": int(train_cfg["smoke_updates"]),
            "batch_size": int(train_cfg["smoke_batch_size"]),
            "workers": int(train_cfg["smoke_workers"]),
            "save_every": int(train_cfg["smoke_save_every"]),
        }
        status("config", config=str(CONFIG), config_sha256=sha256_file(CONFIG))
        bootstrap_data()
        if stopping:
            return
        if not valid_json(RUN / "preflight.json", passed=True):
            run(
                "preflight",
                [
                    PYTHON,
                    "-m",
                    "deployment.duo_dp.preflight",
                    "--data",
                    str(DATA),
                    "--output",
                    str(RUN / "preflight.json"),
                    "--check-model",
                ],
            )
        smoke = RUN / "smoke" / "final.pt"
        if not valid_json(RUN / "smoke" / "status.json", status="complete", step=smoke_cfg["updates"]):
            run(
                "smoke_train",
                [
                    PYTHON,
                    "-m",
                    "deployment.duo_dp.train",
                    "--data",
                    str(DATA),
                    "--output",
                    str(smoke.parent),
                    "--steps",
                    str(smoke_cfg["updates"]),
                    "--batch-size",
                    str(smoke_cfg["batch_size"]),
                    "--workers",
                    str(smoke_cfg["workers"]),
                    "--save-every",
                    str(smoke_cfg["save_every"]),
                    "--smoke",
                ],
            )
        smoke_summary = RUN / "smoke" / "validation" / "summary.json"
        if not valid_json(smoke_summary, status="complete", total_episodes=11):
            run(
                "smoke_validation",
                [
                    PYTHON,
                    "-m",
                    "deployment.duo_dp.validation_launcher",
                    "--checkpoint",
                    str(smoke),
                    "--data",
                    str(DATA),
                    "--output",
                    str(smoke_summary.parent),
                    "--episodes",
                    str(validation_cfg["smoke_episodes_per_task"]),
                    "--max-steps",
                    str(validation_cfg["smoke_max_steps"]),
                    "--workers",
                    "1",
                    "--inference-steps",
                    str(validation_cfg["smoke_inference_steps"]),
                    "--smoke",
                ],
            )
        formal = RUN / "formal" / "final.pt"
        if not valid_json(
            RUN / "formal" / "status.json", status="complete", step=int(train_cfg["updates"])
        ):
            run(
                "formal_train",
                [
                    PYTHON,
                    "-m",
                    "deployment.duo_dp.train",
                    "--data",
                    str(DATA),
                    "--output",
                    str(formal.parent),
                    "--steps",
                    str(train_cfg["updates"]),
                    "--batch-size",
                    str(train_cfg["batch_size"]),
                    "--workers",
                    str(loader_cfg["formal_workers"]),
                    "--save-every",
                    str(config["checkpointing"]["save_every_updates"]),
                    "--seed",
                    str(train_cfg["seed"]),
                    "--learning-rate",
                    str(train_cfg["optimizer"]["learning_rate"]),
                    "--warmup",
                    str(train_cfg["scheduler"]["warmup_updates"]),
                    "--transition-fraction",
                    str(train_cfg["transition_fraction"]),
                    "--gripper-loss-weight",
                    str(train_cfg["gripper_loss_weight"]),
                    "--resume",
                ],
                retries=3,
            )
        validation = RUN / "formal" / "validation20" / "summary.json"
        if not valid_json(validation, status="complete", total_episodes=220):
            run(
                "validation20",
                [
                    PYTHON,
                    "-m",
                    "deployment.duo_dp.validation_launcher",
                    "--checkpoint",
                    str(formal),
                    "--data",
                    str(DATA),
                    "--output",
                    str(validation.parent),
                    "--episodes",
                    str(validation_cfg["episodes_per_task"]),
                    "--workers",
                    str(validation_cfg["workers"]),
                    "--inference-steps",
                    str(validation_cfg["inference_steps"]),
                    "--weights",
                    str(validation_cfg["inference_weights"]).removesuffix("_model"),
                    "--replan-steps",
                    str(validation_cfg["replan_interval"]),
                ],
                retries=3,
            )
        if not valid_json(RUN / "final_report.json", status="complete"):
            run("finalize", [PYTHON, "-m", "deployment.duo_dp.finalize"])
        status(
            "complete",
            "complete",
            checkpoint=str(formal),
            summary=str(validation),
            final_report=str(RUN / "final_report.json"),
        )
    except Exception as error:
        status("failed", "failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
