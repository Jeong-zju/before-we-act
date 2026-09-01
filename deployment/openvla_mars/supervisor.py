"""Crash-resumable MARS-Control OpenVLA supervisor.

Stages are explicit and artifact-gated.  Training owns all four GPUs; closed
loop workers are assigned one GPU each and receive exactly one arm-local
observation per RPC call.  A failed stage is retried on the next keeper tick.
"""
from __future__ import annotations
import json, os, signal, subprocess, threading, time
from pathlib import Path

RUN = Path(os.environ.get("MARS_OPENVLA_RUN_ROOT", "/workspace/bwa_mars_openvla_runs"))
ROOT = Path("/workspace/repos/before-we-act")
RF = Path("/workspace/repos/RoboFactory-MARS")
PY = "/workspace/venvs/openvla/bin/python"
SIM = os.environ.get("BWA_SIM_PYTHON", "/workspace/venvs/robofactory/bin/python")
active = None
stop_requested = False

def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); os.replace(tmp, path)

def run(name: str, argv: list[str], env: dict[str, str] | None = None) -> None:
    global active
    log = RUN / "supervisor/logs" / f"{name}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy(); merged.update(env or {})
    merged.update({"BWA_MARS_CONTROL": "1", "MARS_OPENVLA_DATA_ROOT": "/workspace/datasets/mars_control", "MARS_ROBOFACTORY_ROOT": str(RF), "PYTHONPATH": f"{ROOT}:{RF}:{merged.get('PYTHONPATH','')}"})
    write(RUN / "supervisor/state.json", {"status": "running", "stage": name, "log": str(log), "updated_at": time.time()})
    with log.open("ab", buffering=0) as stream:
        active = subprocess.Popen(argv, cwd=ROOT, env=merged, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code = active.wait()
    active = None
    if code: write(RUN / "supervisor/state.json", {"status": "failed", "stage": name, "exit_code": code, "log": str(log)}); raise RuntimeError(f"{name} exited {code}")
    write(RUN / "supervisor/receipts" / f"{name}.json", {"status": "complete", "stage": name, "log": str(log), "completed_at": time.time()})

def complete(path: Path) -> bool:
    try: return json.loads(path.read_text()).get("status") == "complete"
    except (FileNotFoundError, json.JSONDecodeError): return False

def checkpoint_complete(status: Path, checkpoint: Path) -> bool:
    """Accept a training stage only when both its receipt and payload exist."""
    return (
        complete(status)
        and (checkpoint / "action_head--latest_checkpoint.pt").is_file()
        and (checkpoint / "lora_adapter/adapter_model.safetensors").is_file()
    )

def stop(_sig, _frame):
    global stop_requested
    stop_requested = True
    write(RUN / "supervisor/state.json", {"status": "stopping", "stage": "signal", "updated_at": time.time()})
    if active is not None:
        try: os.killpg(active.pid, signal.SIGTERM)
        except ProcessLookupError: pass

def main() -> None:
    global stop_requested
    RUN.mkdir(parents=True, exist_ok=True); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    def heartbeat():
        while not stop_requested:
            try:
                state = json.loads((RUN / "supervisor/state.json").read_text())
                state["heartbeat_at"] = time.time(); write(RUN / "supervisor/state.json", state)
            except Exception:
                pass
            time.sleep(30)
    threading.Thread(target=heartbeat, daemon=True).start()
    assets_receipt = RUN / "supervisor/receipts/assets.json"
    if not complete(assets_receipt): run("assets", [PY, "-m", "deployment.openvla_mars.ensure_assets"])
    receipts = [Path(f"/workspace/datasets/mars_control/{task}/download_receipt.json") for task in ("place_cube_in_cup", "strike_cube_hard", "three_robots_place_shoes", "four_robots_stack_cube")]
    if not all(complete(path) for path in receipts): run("download", [PY, "-m", "deployment.openvla_mars.download"])
    preflight = RUN / "audit/preflight.json"
    if not complete(preflight): run("preflight", [PY, "-m", "deployment.openvla_mars.preflight"])
    contract = RUN / "audit/contract.json"
    if not complete(contract): run("audit_contract", [PY, "-m", "deployment.openvla_mars.audit_contract"])
    smoke_status = RUN / "smoke/openvla_oft/status.json"
    smoke_ckpt = RUN / "smoke/openvla_oft/final"
    if not checkpoint_complete(smoke_status, smoke_ckpt): run("smoke_train", ["/bin/bash", str(ROOT / "deployment/openvla_mars/run_smoke.sh")], {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    smoke_val = RUN / "smoke/validation20/summary.json"
    if not complete(smoke_val):
        run("smoke_closed_loop", [SIM, str(ROOT / "deployment/vla_baselines/validation_launcher.py"), "--policy", "openvla", "--checkpoint", str(smoke_ckpt), "--output-root", str(RUN / "smoke/validation20"), "--episodes", "1", "--seed", "990000", "--smoke", "--max-steps-override", "2"], {"BWA_VALIDATION_PARALLEL": "1", "BWA_GPU_COUNT": "4"})
    smoke = json.loads(smoke_val.read_text())
    if smoke.get("status") != "complete" or smoke.get("total_episodes") != 4: raise RuntimeError("closed-loop smoke did not complete all four MARS tasks")
    formal_status = RUN / "formal/openvla_oft/status.json"
    final = RUN / "formal/openvla_oft/final"
    if not checkpoint_complete(formal_status, final):
        run("formal_train", ["/bin/bash", str(ROOT / "deployment/openvla_mars/run_train.sh")], {"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
    validation = RUN / "formal/validation20/summary.json"
    if not complete(validation):
        run("validation20", [SIM, str(ROOT / "deployment/vla_baselines/validation_launcher.py"), "--policy", "openvla", "--checkpoint", str(final), "--output-root", str(RUN / "formal/validation20"), "--episodes", "20", "--seed", "20260820"], {"BWA_VALIDATION_PARALLEL": "1", "BWA_GPU_COUNT": "4"})
    if not complete(validation): raise RuntimeError("Validation20 artifact incomplete")
    report = {"schema": "mars-control.openvla.final-report.v1", "status": "complete", "baseline": "OpenVLA-OFT", "benchmark": "MARS-Control", "episodes": 600, "validation20": json.loads(validation.read_text()), "checkpoint": str(final), "policy_contract": "shared_weights_decentralized_local_rgb_qpos9_to_local_action8", "normalization": "full clipped corpus min/max; backbone-specific OpenVLA image processor", "completed_at": time.time()}
    write(RUN / "final_report.json", report); write(RUN / "supervisor/state.json", {"status": "complete", "stage": "complete", "updated_at": time.time()})

if __name__ == "__main__": main()
