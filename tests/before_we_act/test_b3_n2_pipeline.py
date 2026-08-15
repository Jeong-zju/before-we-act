from pathlib import Path


def test_pipeline_freezes_a_checked_safe_directory_source_commit() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/before_we_act/run_b3_n2_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert 'SOURCE_COMMIT="$(git -c safe.directory="${ROOT}" -C "${ROOT}" rev-parse HEAD)"' in launcher
    assert '[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]' in launcher
    assert '--source-commit "${SOURCE_COMMIT}"' in launcher


def test_pipeline_requires_frozen_owner_authorization_and_gates_validation20() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/before_we_act/run_b3_n2_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert "AUTHORIZED_OWNER_N2_FULL_BUDGET_CLOSED_LOOP_DIAGNOSTIC_20260815" in launcher
    assert "BWA_N2_AUTHORIZE_VALIDATION20_DIAGNOSTIC" in launcher
    assert '"${final_status}" == POSITIVE_SIGNAL' in launcher
    assert "run_b3_n2_validation20.sh" in launcher
