from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import httpx
import huggingface_hub
from huggingface_hub.utils import GatedRepoError
import pytest

from models.wam_multimodal import (
    DEFAULT_DINOV3_CONFIG_SHA256,
    DEFAULT_DINOV3_ENCODER,
    DEFAULT_DINOV3_MODEL_ID,
    DEFAULT_DINOV3_REVISION,
    DEFAULT_DINOV3_WEIGHTS_SHA256,
    canonical_json_sha256,
)
from scripts import prepare_dinov3_encoder as prepare


def test_prepare_default_identity_is_fully_pinned() -> None:
    args = prepare._parser().parse_args([])

    spec, revision, weights_sha256, config_sha256 = prepare._resolved_identity(args)

    assert args.encoder == DEFAULT_DINOV3_ENCODER
    assert spec.name == DEFAULT_DINOV3_ENCODER
    assert spec.model_id == DEFAULT_DINOV3_MODEL_ID
    assert revision == DEFAULT_DINOV3_REVISION
    assert weights_sha256 == DEFAULT_DINOV3_WEIGHTS_SHA256
    assert config_sha256 == DEFAULT_DINOV3_CONFIG_SHA256
    assert len(revision) == 40
    assert len(weights_sha256) == len(config_sha256) == 64


def test_prepare_nondefault_alias_requires_all_explicit_pins() -> None:
    alias = "dinov3_vitb16_lvd"
    args = prepare._parser().parse_args(["--encoder", alias])

    with pytest.raises(
        ValueError,
        match=(
            r"non-default encoder .* requires explicit --revision, "
            r"--expected-weights-sha256, --expected-config-sha256"
        ),
    ):
        prepare._resolved_identity(args)

    pinned = prepare._parser().parse_args(
        [
            "--encoder",
            alias,
            "--revision",
            "1" * 40,
            "--expected-weights-sha256",
            "2" * 64,
            "--expected-config-sha256",
            "3" * 64,
        ]
    )
    spec, revision, weights_sha256, config_sha256 = prepare._resolved_identity(pinned)
    assert spec.name == alias
    assert revision == "1" * 40
    assert weights_sha256 == "2" * 64
    assert config_sha256 == "3" * 64


@pytest.fixture()
def verified_default_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    output_dir = tmp_path / "dinov3"
    output_dir.mkdir()
    config = {
        "architectures": ["DINOv3ViTModel"],
        "hidden_size": 1024,
        "model_type": "dinov3_vit",
        "num_channels": 3,
        "num_register_tokens": 4,
        "patch_size": 16,
    }
    config_path = output_dir / prepare.CONFIG_FILENAME
    config_path.write_text(json.dumps(config), encoding="utf-8")
    weights = b"offline-test-dinov3-weights"
    (output_dir / prepare.WEIGHTS_FILENAME).write_bytes(weights)
    monkeypatch.setattr(
        prepare,
        "DEFAULT_DINOV3_CONFIG_SHA256",
        canonical_json_sha256(config),
    )
    monkeypatch.setattr(
        prepare,
        "DEFAULT_DINOV3_WEIGHTS_SHA256",
        hashlib.sha256(weights).hexdigest(),
    )
    return output_dir


def test_prepare_existing_verified_artifact_is_idempotent_and_offline(
    verified_default_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    download_calls = 0

    def fail_if_downloaded(*_args: object, **_kwargs: object) -> None:
        nonlocal download_calls
        download_calls += 1
        raise AssertionError("verified artifact must not call the Hub")

    monkeypatch.setattr(prepare, "_download_snapshot", fail_if_downloaded)

    assert prepare.main(["--output-dir", str(verified_default_artifact)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert prepare.main(["--output-dir", str(verified_default_artifact)]) == 0
    second = json.loads(capsys.readouterr().out)

    assert download_calls == 0
    assert first == second
    assert first["status"] == "already_verified"
    assert first["encoder"] == DEFAULT_DINOV3_ENCODER
    assert first["revision"] == DEFAULT_DINOV3_REVISION
    assert sorted(path.name for path in verified_default_artifact.iterdir()) == [
        prepare.CONFIG_FILENAME,
        prepare.WEIGHTS_FILENAME,
    ]


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(
            lambda directory: (directory / prepare.WEIGHTS_FILENAME).write_bytes(
                b"tampered"
            ),
            id="tampered-weights",
        ),
        pytest.param(
            lambda directory: (directory / "unexpected.txt").write_text(
                "unexpected", encoding="utf-8"
            ),
            id="unexpected-entry",
        ),
    ],
)
def test_prepare_existing_tampered_or_extra_artifact_fails_closed_without_network(
    verified_default_artifact: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt: Callable[[Path], object],
) -> None:
    corrupt(verified_default_artifact)
    download_calls = 0

    def fail_if_downloaded(*_args: object, **_kwargs: object) -> None:
        nonlocal download_calls
        download_calls += 1
        raise AssertionError("invalid existing artifact must fail before the Hub")

    monkeypatch.setattr(prepare, "_download_snapshot", fail_if_downloaded)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        prepare.main(["--output-dir", str(verified_default_artifact)])

    assert download_calls == 0


def test_download_preflight_reports_gated_approval_before_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        403,
        request=httpx.Request(
            "HEAD",
            f"https://huggingface.co/{DEFAULT_DINOV3_MODEL_ID}/resolve/"
            f"{DEFAULT_DINOV3_REVISION}/config.json",
        ),
    )
    gated = GatedRepoError("Access denied", response=response)
    transfer_calls = 0

    def deny_metadata(*_args: object, **_kwargs: object) -> None:
        raise gated

    def fail_if_transferred(*_args: object, **_kwargs: object) -> None:
        nonlocal transfer_calls
        transfer_calls += 1
        raise AssertionError("gated preflight must stop before snapshot transfer")

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", deny_metadata)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_if_transferred)

    with pytest.raises(RuntimeError) as caught:
        prepare._download_snapshot(
            tmp_path,
            model_id=DEFAULT_DINOV3_MODEL_ID,
            revision=DEFAULT_DINOV3_REVISION,
            token="hf_unit_test_secret",
        )

    message = str(caught.value)
    assert "access to gated repository" in message
    assert "has not been granted" in message
    assert f"https://huggingface.co/{DEFAULT_DINOV3_MODEL_ID}" in message
    assert "public gated repos" in message
    assert "--dry-run" in message
    assert transfer_calls == 0


def test_download_preflight_heads_both_files_before_pinned_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "hf_unit_test_secret"
    metadata_calls: list[tuple[str, dict[str, object]]] = []
    transfer_calls: list[dict[str, object]] = []

    def metadata(url: str, **kwargs: object) -> object:
        metadata_calls.append((url, kwargs))
        return object()

    def transfer(**kwargs: object) -> None:
        transfer_calls.append(kwargs)

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", metadata)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", transfer)

    prepare._download_snapshot(
        tmp_path,
        model_id=DEFAULT_DINOV3_MODEL_ID,
        revision=DEFAULT_DINOV3_REVISION,
        token=token,
    )

    assert len(metadata_calls) == 2
    assert [url.rsplit("/", 1)[-1] for url, _ in metadata_calls] == [
        prepare.CONFIG_FILENAME,
        prepare.WEIGHTS_FILENAME,
    ]
    assert all(call[1]["token"] == token for call in metadata_calls)
    assert all(call[1]["timeout"] == 30.0 for call in metadata_calls)
    assert transfer_calls == [
        {
            "repo_id": DEFAULT_DINOV3_MODEL_ID,
            "repo_type": "model",
            "revision": DEFAULT_DINOV3_REVISION,
            "allow_patterns": list(prepare.DOWNLOAD_PATTERNS),
            "local_dir": tmp_path,
            "token": token,
        }
    ]


def test_required_hf_token_uses_only_the_explicit_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        prepare._required_hf_token()

    token = "hf_unit_test_secret"
    monkeypatch.setenv("HF_TOKEN", token)
    assert prepare._required_hf_token() == token

    monkeypatch.setenv("HF_TOKEN", "hf_invalid token")
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        prepare._required_hf_token()
