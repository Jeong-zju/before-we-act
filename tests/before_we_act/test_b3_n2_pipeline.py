from pathlib import Path


def test_pipeline_freezes_a_checked_safe_directory_source_commit() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/before_we_act/run_b3_n2_pipeline.sh").read_text(
        encoding="utf-8"
    )

    assert 'SOURCE_COMMIT="$(git -c safe.directory="${ROOT}" -C "${ROOT}" rev-parse HEAD)"' in launcher
    assert '[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]' in launcher
    assert '--source-commit "${SOURCE_COMMIT}"' in launcher
