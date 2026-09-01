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
DP_ROOT = Path(
    os.environ.get(
        "DUO_DP_UPSTREAM",
        "/workspace/repos/RoboFactory/robofactory/policy/Diffusion-Policy",
    )
)
DATA = Path(os.environ.get("DUO_DP_DATA", "/workspace/runs/duobench-dp/data"))
CHECKPOINT = Path(
    os.environ.get("DUO_DP_CHECKPOINT", "/workspace/runs/duobench-dp/formal/final.pt")
)
RUN = Path(
    os.environ.get(
        "DUO_DP_ABLATION_RUN", "/workspace/runs/duobench-dp/closed_loop_ablation"
    )
)
PYTHON = os.environ.get("DUO_DP_PYTHON", "/venv/main/bin/python")
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
            "schema": "duobench.dp.closed-loop-ablation-supervisor.v1",
            "stage": stage,
            "state": state,
            "updated_at": now(),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "formal_results_are_immutable": True,
            "gpu_schedule": "one RTX 5090, three CPU simulators share inference",
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


def load_complete_summary(path: Path, config: dict, episodes: int) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "status": "complete",
        "total_episodes": episodes * 11,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "weights": config["weights"],
        "inference_steps": config["inference_steps"],
        "replan_steps": config["replan_steps"],
    }
    if all(value.get(key) == target for key, target in expected.items()):
        return value
    return None


def run_variant(name: str, config: dict, episodes: int) -> dict:
    global active
    output = RUN / name
    summary_path = output / "summary.json"
    saved = load_complete_summary(summary_path, config, episodes)
    if saved is not None:
        return saved
    LOG.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        "-m",
        "deployment.duo_dp.validation_launcher",
        "--checkpoint",
        str(CHECKPOINT),
        "--data",
        str(DATA),
        "--output",
        str(output),
        "--episodes",
        str(episodes),
        "--workers",
        "3",
        "--inference-steps",
        str(config["inference_steps"]),
        "--weights",
        config["weights"],
        "--replan-steps",
        str(config["replan_steps"]),
    ]
    for attempt in range(1, 4):
        write_status(
            name,
            attempt=attempt,
            episodes_per_task=episodes,
            variant=config,
            output=str(output),
        )
        with (LOG / f"{name}.log").open("a", encoding="utf-8") as stream:
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
            result = load_complete_summary(summary_path, config, episodes)
            if result is None:
                raise RuntimeError(f"{name} exited 0 without a valid summary")
            return result
        if stopping:
            raise RuntimeError("ablation supervisor was stopped")
        write_status(name, "retrying", attempt=attempt, exit_code=returncode)
        time.sleep(min(60, 10 * attempt))
    raise RuntimeError(f"{name} failed after three attempts")


def score(summary: dict) -> float:
    """Prefer success, then dense stage progress; latency is reported, not optimized."""
    return (
        float(summary["macro_success_rate"])
        + 0.20 * float(summary["normalized_final_stage_progress"])
        + 0.05 * normalized_max_progress(summary)
    )


def normalized_max_progress(summary: dict) -> float:
    if "normalized_max_stage_progress" in summary:
        return float(summary["normalized_max_stage_progress"])
    rows = summary.get("rows", [])
    if not rows:
        return float(summary["normalized_final_stage_progress"])
    return sum(float(row["max_stage_progress"]) for row in rows) / len(rows)


def compact(name: str, config: dict, summary: dict) -> dict:
    return {
        "name": name,
        **config,
        "episodes_per_task": summary["episodes_per_task"],
        "successes": summary["successes"],
        "macro_success_rate": summary["macro_success_rate"],
        "normalized_final_stage_progress": summary["normalized_final_stage_progress"],
        "normalized_max_stage_progress": normalized_max_progress(summary),
        "score": score(summary),
        "summary": str(RUN / name / "summary.json"),
    }


def main() -> None:
    global stopping
    RUN.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        smoke = Path("/workspace/runs/duobench-dp/ablation_smoke/summary.json")
        if not smoke.is_file() or json.loads(smoke.read_text()).get("total_episodes") != 11:
            raise RuntimeError("parameterized evaluator smoke did not pass")

        summaries: dict[str, tuple[dict, dict]] = {}
        for replan_steps in (1, 3, 6):
            config = {
                "replan_steps": replan_steps,
                "inference_steps": 20,
                "weights": "ema",
            }
            name = f"probe_replan{replan_steps}_inf20_ema"
            summaries[name] = (config, run_variant(name, config, episodes=5))
            if stopping:
                return

        chunk_best_name = max(summaries, key=lambda name: score(summaries[name][1]))
        chunk_best = summaries[chunk_best_name][0]
        for inference_steps, weights in ((100, "ema"), (20, "online")):
            config = {
                "replan_steps": chunk_best["replan_steps"],
                "inference_steps": inference_steps,
                "weights": weights,
            }
            name = (
                f"probe_replan{config['replan_steps']}_inf{inference_steps}_{weights}"
            )
            if name not in summaries:
                summaries[name] = (config, run_variant(name, config, episodes=5))
            if stopping:
                return

        rows = [compact(name, config, summary) for name, (config, summary) in summaries.items()]
        rows.sort(key=lambda row: row["score"], reverse=True)
        best = rows[0]
        baseline = next(
            row
            for row in rows
            if row["replan_steps"] == 6
            and row["inference_steps"] == 20
            and row["weights"] == "ema"
        )
        improved = (
            best["macro_success_rate"] > baseline["macro_success_rate"]
            or best["normalized_final_stage_progress"]
            >= baseline["normalized_final_stage_progress"] + 0.02
        )
        decision = {
            "schema": "duobench.dp.closed-loop-ablation-decision.v1",
            "status": "probe_complete",
            "created_at": now(),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "probe_episodes_per_task": 5,
            "variants": rows,
            "best_probe": best,
            "baseline_probe": baseline,
            "meaningful_closed_loop_improvement": improved,
            "selection_rule": (
                "macro_success_rate + 0.20*mean_final_stage + 0.05*mean_max_stage; "
                "full Validation20 only if success improves or final stage improves by >=0.02"
            ),
        }
        atomic_json(RUN / "decision.json", decision)
        if improved:
            best_config = {
                key: best[key] for key in ("replan_steps", "inference_steps", "weights")
            }
            full_name = (
                f"validation20_replan{best_config['replan_steps']}"
                f"_inf{best_config['inference_steps']}_{best_config['weights']}"
            )
            full = run_variant(full_name, best_config, episodes=20)
            decision.update(
                {
                    "status": "complete",
                    "full_validation20": compact(full_name, best_config, full),
                    "next_stage": "compare full result; retrain only if improvement survives Validation20",
                }
            )
        else:
            decision.update(
                {
                    "status": "complete",
                    "next_stage": "task-conditioning and transition-aware training probe required",
                }
            )
        atomic_json(RUN / "decision.json", decision)
        write_status(
            "complete",
            "complete",
            decision=str(RUN / "decision.json"),
            next_stage=decision["next_stage"],
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
