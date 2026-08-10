from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_launcher_has_exact_independent_branch_gpu_tmux_mapping():
    launcher = source("scripts/before_we_act/launch_r11_4gpu_tmux.sh")
    for branch in (
        "feat/r11-vjepa21-ac-refine",
        "feat/r11-dreamzero-wan22-wam",
        "feat/r11-cosmos-policy-latent",
        "feat/r11-lawam-latent-subgoal",
    ):
        assert branch in launcher
    for session in (
        "bwa-r11-a-vjepa",
        "bwa-r11-b-dreamzero",
        "bwa-r11-c-cosmos",
        "bwa-r11-d-lawam",
    ):
        assert session in launcher
    assert "CUDA_VISIBLE_DEVICES='$gpu'" in launcher
    assert "merge --ff-only" in launcher
    assert "record_r11_deployment.py" in launcher


def test_pipeline_preserves_exact_gate_order_and_no_w10_action_fallback():
    runner = source("scripts/before_we_act/run_r11_candidate.sh")
    ordered = [
        runner.index("F1_FRESH="),
        runner.index("DISCOVERY="),
        runner.index("run_suite discovery normal"),
        runner.index("SELECTION="),
        runner.index("FORMAL="),
        runner.index("run_suite formal normal"),
        runner.index("formal-acceptance"),
    ]
    assert ordered == sorted(ordered)
    assert "--updates \"$target\"" in runner
    assert "evaluate_no_wrist_pair.py" not in runner
    assert "fallback" not in source("before_we_act/evaluate_r11_candidate.py").lower() or \
        '"fallback_calls": 0' in source("before_we_act/evaluate_r11_candidate.py")


def test_graceful_stop_uses_only_exact_pid_start_identity_and_no_pattern_kill():
    stop = source("scripts/before_we_act/stop_r11_4gpu_tmux.sh")
    assert "pid_start_time_ticks" in stop
    assert "/proc/$pid/stat" in stop
    assert "kill -USR1" in stop
    assert "pkill" not in stop
    assert "killall" not in stop


def test_acceptance_contains_all_frozen_thresholds_and_score_weights():
    acceptance = source("scripts/before_we_act/accept_r11_candidate.py")
    for expression in (
        "total >= 80",
        "protected_total >= 72",
        "min(protected_values) >= 16",
        "camera >= 6",
        "camera + food >= 8",
        "60 * total / 120",
        "10 * protected_total / 80",
        "10 * (camera + food) / 40",
        "8 * min(max(macro_gain / 0.20",
        "7 * causal_fraction",
        "5 * min(w10_latency / candidate_latency",
    ):
        assert expression in acceptance


def test_monitor_lists_all_required_states_and_fields():
    runtime = source("scripts/before_we_act/r11_runtime.py")
    for state in (
        "NOT_STARTED", "PREPARING", "DOWNLOADING", "PREFLIGHT", "TRAINING",
        "VALIDATING", "ACCEPTING", "PASSED", "FAILED", "FAILED_FIT", "STOPPED",
        "STALE", "UNKNOWN",
    ):
        assert f'"{state}"' in runtime
    for field in (
        "action_loss", "world_loss", "value_loss", "pred_gain", "heartbeat_age",
        "checkpoint=", "acceptance=", "upstream=", "power=", "queue=",
    ):
        assert field in runtime


def test_runner_routes_parity_to_each_remote_read_only_vendor():
    runner = source("scripts/before_we_act/run_r11_candidate.sh")
    expected = {
        "R11_VJEPA2_VENDOR": "/workspace/artifacts/r11_upstream/vjepa2",
        "R11_DREAMZERO_VENDOR": "/workspace/artifacts/r11_upstream/dreamzero",
        "R11_COSMOS_VENDOR": "/workspace/artifacts/r11_upstream/cosmos-predict2.5",
        "R11_LAWAM_VENDOR": "/workspace/artifacts/r11_upstream/LaWAM",
    }
    for variable, path in expected.items():
        assert variable in runner
        assert path in runner
    assert '"${VENDOR_ENV_NAMES[$CANDIDATE]}=${VENDOR_PATHS[$CANDIDATE]}"' in runner
