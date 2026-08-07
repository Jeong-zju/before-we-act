import argparse
import json

from scripts.before_we_act import r15_runtime


def status_args(run_root, candidate, state="VALIDATING"):
    return argparse.Namespace(
        run_root=run_root,
        candidate=candidate,
        state=state,
        stage="closed_loop",
        program="eval",
        detail="test",
        pid=123,
        child_pid=456,
        log="candidate.log",
        exit_code=None,
    )


def register_args(tmp_path, candidate, reference=False):
    artifact = tmp_path / f"{candidate}.pt"
    artifact.write_bytes(b"checkpoint")
    config = tmp_path / f"{candidate}.yaml"
    config.write_text("candidate: test\n")
    return argparse.Namespace(
        run_root=tmp_path / "run",
        run_id="unit",
        split="discovery20",
        seed_file=str(tmp_path / "seeds.json"),
        seed_file_sha256="a" * 64,
        candidate=candidate,
        label=f"label-{candidate}",
        gpu=int(candidate[1]),
        worktree=str(tmp_path),
        branch=f"branch-{candidate}",
        commit=candidate * 20,
        config=str(config),
        checkpoint=str(artifact),
        reference=reference,
    )


def write_result(run_root, candidate, successes):
    path = run_root / "candidates" / candidate / "validation" / "discovery20.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"seed": seed, "success": seed < successes, "steps": 800}
        for seed in range(20)
    ]
    path.write_text(json.dumps({"rows": rows, "route": candidate}) + "\n")


def test_paired_screen_requires_strict_gain(tmp_path):
    (tmp_path / "seeds.json").write_text(json.dumps({"seeds": list(range(20))}))
    for candidate, reference in (("p0", True), ("p1", False), ("p2", False)):
        r15_runtime.register(register_args(tmp_path, candidate, reference))
        r15_runtime.update_status(status_args(tmp_path / "run", candidate))
    write_result(tmp_path / "run", "p0", 2)
    write_result(tmp_path / "run", "p1", 3)
    write_result(tmp_path / "run", "p2", 2)

    assert r15_runtime.accept(argparse.Namespace(run_root=tmp_path / "run", candidate="p0")) == 0
    assert r15_runtime.accept(argparse.Namespace(run_root=tmp_path / "run", candidate="p1")) == 0
    assert r15_runtime.accept(argparse.Namespace(run_root=tmp_path / "run", candidate="p2")) == 10
    p1 = json.loads((tmp_path / "run/candidates/p1/acceptance.json").read_text())
    assert (p1["delta_successes"], p1["paired_wins"], p1["paired_losses"]) == (1, 1, 0)


def test_status_preserves_created_at(tmp_path):
    run_root = tmp_path / "run"
    path = run_root / "candidates/p0/status.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"created_at": "2026-01-01T00:00:00Z"}))
    r15_runtime.update_status(status_args(run_root, "p0"))
    assert json.loads(path.read_text())["created_at"] == "2026-01-01T00:00:00Z"


def test_logged_rows_deduplicates_resume_attempts(tmp_path):
    path = tmp_path / "eval.log"
    path.write_text(
        "warning\n"
        + json.dumps({"seed": 7, "success": False, "steps": 800})
        + "\n"
        + json.dumps({"seed": 7, "success": True, "steps": 401})
        + "\n"
        + json.dumps({"seed": 8, "success": False, "steps": 800})
        + "\n"
    )
    rows = r15_runtime.logged_rows(path)
    assert set(rows) == {7, 8}
    assert rows[7]["success"] is True


def test_monitor_reads_stack_expert_finetune_progress(tmp_path, monkeypatch):
    (tmp_path / "seeds.json").write_text(json.dumps({"seeds": list(range(20))}))
    r15_runtime.register(register_args(tmp_path, "p1"))
    progress = tmp_path / "run/candidates/p1/train/stack_expert/progress.jsonl"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps(
            {
                "fine_tune_update": 725,
                "loss": 0.03125,
                "eta_hours": 0.5,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(r15_runtime, "gpu_rows", lambda: {})
    rendered = r15_runtime.render(tmp_path / "run", ("p1",))
    assert "train_update=725 loss=0.03125 eta=0.5h" in rendered


def test_monitor_summarizes_stack_stages_and_planner(tmp_path, monkeypatch):
    (tmp_path / "seeds.json").write_text(json.dumps({"seeds": list(range(20))}))
    r15_runtime.register(register_args(tmp_path, "p1"))
    result = tmp_path / "run/candidates/p1/validation/discovery20.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "seed": 7,
                        "success": False,
                        "steps": 800,
                        "terminal_info": {
                            "cubeB_placed": True,
                            "is_cubeA_on_cubeB": True,
                            "is_cubeC_on_cubeA": False,
                        },
                    }
                ]
            }
        )
        + "\n"
    )
    progress = result.parent / "closed_loop_progress.json"
    progress.write_text(
        json.dumps({"interventions": 4, "fallbacks": 5, "planner_timeouts": 0})
    )
    monkeypatch.setattr(r15_runtime, "gpu_rows", lambda: {})
    rendered = r15_runtime.render(tmp_path / "run", ("p1",))
    assert "'cubeB_placed': 1, 'A_on_B': 1, 'C_on_A': 0" in rendered
    assert "planner=interventions:4,fallbacks:5,timeouts:0,exceptions:-" in rendered
