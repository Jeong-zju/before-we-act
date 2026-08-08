from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/before_we_act/monitor_r15_portfolio.sh"


def test_portfolio_monitor_requires_explicit_safe_targets():
    completed = subprocess.run(
        [str(SCRIPT), "--once"], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 2
    assert "at least one screen" in completed.stderr


def test_portfolio_monitor_supports_all_evolution_target_types():
    source = SCRIPT.read_text()
    assert "--screen" in source
    assert "--expert-collection" in source
    assert "--expert-cache" in source
    assert "--once" in source
    assert "--interval" in source
