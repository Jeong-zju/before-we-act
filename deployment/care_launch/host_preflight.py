"""Host checks that must pass before a CARE run touches a GPU.

The three CARE pipelines each grew their own preflight, and each is missing
something another one has. The costly gaps on a rented host are:

* no disk-headroom check outside the VLA baselines, so a multi-hundred-gigabyte
  corpus can fill the volume hours into a run;
* no single place that reports *every* problem, so a run fails, gets fixed, and
  fails again on the next missing prerequisite.

This module collects failures instead of raising on the first one, so one
preflight pass lists everything that needs fixing. Each check is independent and
returns a structured row, so the report is machine-readable and a stage gate can
assert on it.

Nothing here imports a simulator or allocates GPU memory; it is safe to run on a
login shell before scheduling work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence


REPORT_VERSION = "before-we-act.care-host-preflight/1"
GIB = 1024**3


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(argv: Sequence[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        finished = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return 127, f"{argv[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, f"{argv[0]} timed out"
    return finished.returncode, (finished.stdout or finished.stderr).strip()


def check_disk_headroom(
    path: Path, *, required_bytes: int, label: str = "workspace"
) -> CheckResult:
    """Refuse to start when the volume cannot hold the run's own output.

    The requirement is the corpus size plus a floor, because a volume that ends
    a run at 100% leaves no room for the receipts that prove it finished.
    """

    target = Path(path)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        return CheckResult(
            f"disk_headroom:{label}", False, f"{target} has no existing parent"
        )
    usage = shutil.disk_usage(probe)
    ok = usage.free >= required_bytes
    return CheckResult(
        f"disk_headroom:{label}",
        ok,
        (
            f"{usage.free / GIB:.1f} GiB free at {probe}, "
            f"need {required_bytes / GIB:.1f} GiB"
        ),
        {
            "path": str(probe),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
            "required_bytes": int(required_bytes),
        },
    )


def check_gpu_inventory(
    *, expected_count: int, allowed_models: Sequence[str] | None = None
) -> CheckResult:
    """Confirm the host has the GPUs the protocol was budgeted for.

    Model names are matched as substrings against an allow-list rather than a
    single SKU: the same protocol runs on more than one Blackwell part, and
    encoding a vendor SKU into a scientific contract is not the check we want.
    """

    code, output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    if code != 0:
        return CheckResult("gpu_inventory", False, f"nvidia-smi failed: {output}")
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    names = [row.split(",")[0].strip() for row in rows]
    if len(names) != expected_count:
        return CheckResult(
            "gpu_inventory",
            False,
            f"expected {expected_count} GPUs, found {len(names)}: {names}",
            {"names": names},
        )
    if allowed_models:
        unmatched = [
            name
            for name in names
            if not any(model in name for model in allowed_models)
        ]
        if unmatched:
            return CheckResult(
                "gpu_inventory",
                False,
                f"unsupported GPU model(s) {unmatched}; allowed {list(allowed_models)}",
                {"names": names},
            )
    return CheckResult("gpu_inventory", True, f"{len(names)} GPUs: {names}", {"names": names})


def check_no_foreign_gpu_processes() -> CheckResult:
    """A busy GPU turns a scheduled run into an out-of-memory crash later."""

    code, output = _run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]
    )
    if code != 0:
        return CheckResult(
            "gpu_processes", False, f"nvidia-smi failed: {output}"
        )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    return CheckResult(
        "gpu_processes",
        not rows,
        "no compute processes" if not rows else f"GPUs already busy: {rows}",
        {"processes": rows},
    )


def check_token_file(path: Path) -> CheckResult:
    """A readable token is required, and it must not be world-readable."""

    target = Path(path)
    if not target.is_file():
        return CheckResult("hf_token", False, f"missing token file {target}")
    if not target.read_text(encoding="utf-8").strip():
        return CheckResult("hf_token", False, f"empty token file {target}")
    mode = target.stat().st_mode & 0o077
    if mode:
        return CheckResult(
            "hf_token",
            False,
            f"{target} is group/world accessible (mode bits {oct(mode)}); chmod 600 it",
        )
    return CheckResult("hf_token", True, f"{target} present and private")


def check_git_revision(repo: Path, expected: str, *, label: str) -> CheckResult:
    """Pinned third-party checkouts must be at the revision the protocol names."""

    target = Path(repo)
    if not (target / ".git").exists():
        return CheckResult(f"git_revision:{label}", False, f"{target} is not a checkout")
    code, output = _run(["git", "-C", str(target), "rev-parse", "HEAD"])
    if code != 0:
        return CheckResult(f"git_revision:{label}", False, f"rev-parse failed: {output}")
    head = output.strip()
    ok = head == expected
    return CheckResult(
        f"git_revision:{label}",
        ok,
        f"{target} at {head[:12]}" + ("" if ok else f", expected {expected[:12]}"),
        {"head": head, "expected": expected},
    )


def check_python_imports(modules: Sequence[str], *, python: str | None = None) -> CheckResult:
    """Probe imports in a subprocess so an ABI crash cannot take the run with it.

    SAPIEN and Warp abort the process on an ABI mismatch rather than raising, so
    importing them inline would kill the preflight instead of reporting.
    """

    interpreter = python or os.environ.get("CARE_PYTHON") or "python"
    script = "import " + ", ".join(modules)
    code, output = _run([interpreter, "-c", script], timeout=180.0)
    return CheckResult(
        "python_imports",
        code == 0,
        "ok" if code == 0 else f"import failed: {output.splitlines()[-1] if output else code}",
        {"modules": list(modules), "interpreter": interpreter},
    )


def check_pinned_distributions(
    pins: Mapping[str, str], *, python: str | None = None
) -> CheckResult:
    """Exact pins matter where a newer release is not a valid substitute.

    SAPIEN's Warp and NumPy pins are ABI constraints, not preferences.
    """

    interpreter = python or os.environ.get("CARE_PYTHON") or "python"
    script = (
        "import json,importlib.metadata as m;"
        "print(json.dumps({n: (m.version(n) if m.distributions else None) "
        "for n in " + json.dumps(list(pins)) + "}))"
    )
    code, output = _run([interpreter, "-c", script], timeout=120.0)
    if code != 0:
        return CheckResult("pinned_distributions", False, f"probe failed: {output}")
    try:
        observed = json.loads(output.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return CheckResult("pinned_distributions", False, f"unreadable probe: {output}")
    drift = {
        name: {"observed": observed.get(name), "expected": expected}
        for name, expected in pins.items()
        if observed.get(name) != expected
    }
    return CheckResult(
        "pinned_distributions",
        not drift,
        "pins satisfied" if not drift else f"version drift: {drift}",
        {"observed": observed, "expected": dict(pins)},
    )


def check_offscreen_render(*, python: str | None = None) -> CheckResult:
    """Prove the Vulkan renderer initializes before a rollout stage needs it.

    SAPIEN's renderer is effectively process-global here, and a missing ICD or
    stale driver library surfaces as a device-lost crash mid-rollout. Failing in
    preflight costs seconds; failing mid-run costs the stage.
    """

    interpreter = python or os.environ.get("CARE_PYTHON") or "python"
    script = (
        "import sapien;"
        "sapien.render.RenderSystem(sapien.Device('cuda:0'));"
        "print('render ok')"
    )
    code, output = _run([interpreter, "-c", script], timeout=180.0)
    return CheckResult(
        "offscreen_render",
        code == 0,
        "renderer initialized" if code == 0 else f"renderer failed: {output[-400:]}",
    )


def check_paths_exist(paths: Mapping[str, Path]) -> list[CheckResult]:
    return [
        CheckResult(
            f"path:{label}",
            Path(path).exists(),
            f"{path}" + ("" if Path(path).exists() else " is missing"),
        )
        for label, path in paths.items()
    ]


def run_checks(checks: Sequence[Callable[[], CheckResult]]) -> dict[str, Any]:
    """Run every check, collecting failures rather than stopping at the first."""

    rows: list[CheckResult] = []
    for check in checks:
        try:
            rows.append(check())
        except Exception as error:  # noqa: BLE001 - a broken check is a failure
            rows.append(
                CheckResult(
                    getattr(check, "__name__", "check"),
                    False,
                    f"check raised {type(error).__name__}: {error}",
                )
            )
    failures = [row for row in rows if not row.passed]
    return {
        "report_version": REPORT_VERSION,
        "status": "PASSED" if not failures else "FAILED",
        "checks": [row.to_dict() for row in rows],
        "failures": [row.name for row in failures],
    }


def write_report(report: Mapping[str, Any], output: Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


__all__ = [
    "GIB",
    "REPORT_VERSION",
    "CheckResult",
    "check_disk_headroom",
    "check_git_revision",
    "check_gpu_inventory",
    "check_no_foreign_gpu_processes",
    "check_offscreen_render",
    "check_paths_exist",
    "check_pinned_distributions",
    "check_python_imports",
    "check_token_file",
    "run_checks",
    "write_report",
]
