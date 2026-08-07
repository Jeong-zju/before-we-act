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
