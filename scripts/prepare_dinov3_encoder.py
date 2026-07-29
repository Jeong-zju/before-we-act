"""Direct-download and verify one pinned DINOv3 visual-teacher artifact.

Training and evaluation deliberately never call Hugging Face Hub. This
preparation boundary disables Xet and invokes ``hf download`` with one worker
against the final local directory, so an interrupted transfer resumes in place
without a second snapshot or temporary install copy.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.wam_multimodal import (  # noqa: E402
    DEFAULT_DINOV3_CONFIG_SHA256,
    DEFAULT_DINOV3_ENCODER,
    DEFAULT_DINOV3_REVISION,
    DEFAULT_DINOV3_WEIGHTS_SHA256,
    DINOV3_ENCODER_SPECS,
    DINOv3EncoderSpec,
    canonical_json_sha256,
    sha256_file,
)


CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.safetensors"
DOWNLOAD_PATTERNS = (CONFIG_FILENAME, WEIGHTS_FILENAME)
DINOV3_MODEL_PAGE = "https://huggingface.co/{model_id}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a pinned official DINOv3 encoder and verify it before "
            "making it visible to Phase M1. Authenticate first with "
            "`hf auth login` and accept the upstream DINOv3 license."
        )
    )
    parser.add_argument(
        "--encoder",
        choices=tuple(sorted(DINOV3_ENCODER_SPECS)),
        default=DEFAULT_DINOV3_ENCODER,
        help=f"project encoder alias (default: {DEFAULT_DINOV3_ENCODER})",
    )
    parser.add_argument(
        "--revision",
        help=(
            "full 40-character lowercase Hugging Face commit SHA; the pinned "
            "default is supplied only for the default encoder"
        ),
    )
    parser.add_argument(
        "--expected-weights-sha256",
        help=(
            "expected byte SHA-256 of model.safetensors; the pinned default is "
            "supplied only for the default encoder"
        ),
    )
    parser.add_argument(
        "--expected-config-sha256",
        help=(
            "expected canonical semantic SHA-256 of config.json; the pinned "
            "default is supplied only for the default encoder"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "installation directory (default: "
            "artifacts/vision/<encoder> relative to the repository root)"
        ),
    )
    return parser


def _full_lower_hex(value: str, *, length: int, label: str) -> str:
    normalized = str(value)
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(
            f"{label} must be exactly {length} lowercase hexadecimal characters"
        )
    return normalized


def _resolved_identity(
    args: argparse.Namespace,
) -> tuple[DINOv3EncoderSpec, str, str, str]:
    spec = DINOV3_ENCODER_SPECS[args.encoder]
    if args.encoder == DEFAULT_DINOV3_ENCODER:
        revision = args.revision or DEFAULT_DINOV3_REVISION
        weights_sha256 = args.expected_weights_sha256 or DEFAULT_DINOV3_WEIGHTS_SHA256
        config_sha256 = args.expected_config_sha256 or DEFAULT_DINOV3_CONFIG_SHA256
    else:
        missing = [
            flag
            for flag, value in (
                ("--revision", args.revision),
                ("--expected-weights-sha256", args.expected_weights_sha256),
                ("--expected-config-sha256", args.expected_config_sha256),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"non-default encoder {args.encoder!r} requires explicit "
                + ", ".join(missing)
            )
        revision = args.revision
        weights_sha256 = args.expected_weights_sha256
        config_sha256 = args.expected_config_sha256

    assert revision is not None
    assert weights_sha256 is not None
    assert config_sha256 is not None
    return (
        spec,
        _full_lower_hex(revision, length=40, label="revision"),
        _full_lower_hex(
            weights_sha256,
            length=64,
            label="expected weights SHA-256",
        ),
        _full_lower_hex(
            config_sha256,
            length=64,
            label="expected config SHA-256",
        ),
    )


def _read_config(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"DINOv3 config must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"DINOv3 config is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("DINOv3 config.json root must be an object")
    return payload


def _validate_architecture(
    payload: Mapping[str, Any],
    spec: DINOv3EncoderSpec,
) -> None:
    expected: dict[str, Any] = {
        "model_type": "dinov3_vit",
        "hidden_size": spec.output_dim,
        "patch_size": spec.patch_size,
        "num_register_tokens": spec.register_tokens,
        "num_channels": 3,
        "architectures": ["DINOv3ViTModel"],
    }
    mismatched = {
        key: {"expected": expected_value, "observed": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatched:
        raise ValueError(f"DINOv3 architecture identity mismatch: {mismatched}")


def _validate_artifact_directory(
    directory: Path,
    *,
    spec: DINOv3EncoderSpec,
    expected_weights_sha256: str,
    expected_config_sha256: str,
) -> dict[str, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(
            f"DINOv3 artifact path must be a regular directory: {directory}"
        )
    config_path = directory / CONFIG_FILENAME
    weights_path = directory / WEIGHTS_FILENAME
    if weights_path.is_symlink() or not weights_path.is_file():
        raise ValueError(f"DINOv3 weights must be a regular file: {weights_path}")

    actual_weights_sha256 = sha256_file(weights_path)
    if actual_weights_sha256 != expected_weights_sha256:
        raise ValueError(
            "DINOv3 weights SHA-256 mismatch: "
            f"expected {expected_weights_sha256}, got {actual_weights_sha256}"
        )
    raw_config = _read_config(config_path)
    actual_config_sha256 = canonical_json_sha256(raw_config)
    if actual_config_sha256 != expected_config_sha256:
        raise ValueError(
            "DINOv3 config semantic SHA-256 mismatch: "
            f"expected {expected_config_sha256}, got {actual_config_sha256}"
        )
    _validate_architecture(raw_config, spec)
    return {
        "weights_sha256": actual_weights_sha256,
        "config_sha256": actual_config_sha256,
    }


def _existing_install_is_valid(
    output_dir: Path,
    *,
    spec: DINOv3EncoderSpec,
    expected_weights_sha256: str,
    expected_config_sha256: str,
) -> dict[str, str] | None:
    if not output_dir.exists():
        return None
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"refusing to overwrite non-directory path: {output_dir}")
    entries = tuple(output_dir.iterdir())
    if not entries:
        return None
    unexpected = sorted(
        entry.name
        for entry in entries
        if entry.name not in {*DOWNLOAD_PATTERNS, ".cache"}
    )
    if unexpected:
        raise ValueError(
            f"refusing to overwrite non-empty DINOv3 directory with unexpected "
            f"entries: {unexpected}"
        )
    if not all((output_dir / name).is_file() for name in DOWNLOAD_PATTERNS):
        return None
    try:
        return _validate_artifact_directory(
            output_dir,
            spec=spec,
            expected_weights_sha256=expected_weights_sha256,
            expected_config_sha256=expected_config_sha256,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"refusing to overwrite mismatched DINOv3 artifact at {output_dir}: {exc}"
        ) from exc


def _direct_download(
    destination: Path,
    *,
    model_id: str,
    revision: str,
    token: str,
) -> None:
    try:
        from huggingface_hub import (
            get_hf_file_metadata,
            hf_hub_url,
        )
        from huggingface_hub.utils import (
            EntryNotFoundError,
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as exc:  # pragma: no cover - dependency error surface
        raise RuntimeError(
            "prepare_dinov3_encoder.py requires huggingface-hub==1.21.0"
        ) from exc

    model_page = DINOV3_MODEL_PAGE.format(model_id=model_id)
    try:
        # HEAD both pinned files before creating a multi-GB transfer. This turns
        # the Hub's otherwise opaque LocalEntryNotFoundError into an actionable
        # gated/revision/file/network error without downloading file contents.
        for filename in DOWNLOAD_PATTERNS:
            get_hf_file_metadata(
                hf_hub_url(
                    model_id,
                    filename,
                    repo_type="model",
                    revision=revision,
                ),
                token=token,
                timeout=30.0,
                retry_on_errors=False,
            )
    except GatedRepoError as exc:
        raise RuntimeError(
            f"access to gated repository {model_id!r} has not been granted to "
            "the active Hugging Face token. Open "
            f"{model_page} in the same account, submit/accept the access "
            "request, wait for approval, and ensure a fine-grained token has "
            "'Read access to contents of all public gated repos you can "
            "access'. Verify with `hf download "
            f"{model_id} config.json --revision {revision} --dry-run` before "
            "retrying."
        ) from exc
    except RevisionNotFoundError as exc:
        raise RuntimeError(
            f"pinned DINOv3 revision does not exist: {model_id}@{revision}"
        ) from exc
    except EntryNotFoundError as exc:
        raise RuntimeError(
            f"pinned DINOv3 revision is missing config.json or model.safetensors: "
            f"{model_id}@{revision}"
        ) from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(
            f"DINOv3 repository {model_id!r} is unavailable to the active "
            "Hugging Face token; check `hf auth whoami` and the repository URL"
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise RuntimeError(
            f"Hugging Face rejected the DINOv3 metadata request"
            f"{'' if status is None else f' with HTTP {status}'}: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - normalize network/proxy failures
        raise RuntimeError(
            "cannot reach Hugging Face to verify the pinned DINOv3 files; "
            f"check proxy/TLS connectivity and retry (`{type(exc).__name__}: {exc}`)"
        ) from exc

    hf = shutil.which("hf")
    if hf is None:
        raise RuntimeError("the `hf` CLI is required for direct DINOv3 download")
    destination.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "HF_TOKEN": token,
        "HF_HUB_DISABLE_XET": "1",
        "HF_XET_HIGH_PERFORMANCE": "0",
    }
    command = [
        hf,
        "download",
        model_id,
        *DOWNLOAD_PATTERNS,
        "--revision",
        revision,
        "--local-dir",
        str(destination),
        "--max-workers",
        "1",
    ]
    try:
        subprocess.run(command, check=True, env=environment)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"metadata access succeeded, but the direct DINOv3 transfer failed "
            f"for {model_id}@{revision} (`{type(exc).__name__}: {exc}`). "
            "Rerun the same command to resume in the same local directory."
        ) from exc


def _required_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token.startswith("hf_") or any(character.isspace() for character in token):
        raise RuntimeError(
            "HF_TOKEN must contain the hidden interactive Hugging Face user "
            "access token before downloading DINOv3"
        )
    return token


def _result(
    *,
    status: str,
    output_dir: Path,
    spec: DINOv3EncoderSpec,
    revision: str,
    hashes: Mapping[str, str],
) -> dict[str, str]:
    return {
        "status": status,
        "encoder": spec.name,
        "model_id": spec.model_id,
        "revision": revision,
        "output_dir": str(output_dir),
        "config_path": str(output_dir / CONFIG_FILENAME),
        "weights_path": str(output_dir / WEIGHTS_FILENAME),
        "config_sha256": hashes["config_sha256"],
        "weights_sha256": hashes["weights_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec, revision, weights_sha256, config_sha256 = _resolved_identity(args)
        output_dir = (
            (
                args.output_dir
                if args.output_dir is not None
                else ROOT / "artifacts" / "vision" / args.encoder
            )
            .expanduser()
            .resolve()
        )
        existing = _existing_install_is_valid(
            output_dir,
            spec=spec,
            expected_weights_sha256=weights_sha256,
            expected_config_sha256=config_sha256,
        )
        if existing is not None:
            print(
                json.dumps(
                    _result(
                        status="already_verified",
                        output_dir=output_dir,
                        spec=spec,
                        revision=revision,
                        hashes=existing,
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        token = _required_hf_token()
        _direct_download(
            output_dir,
            model_id=spec.model_id,
            revision=revision,
            token=token,
        )
        installed = _validate_artifact_directory(
            output_dir,
            spec=spec,
            expected_weights_sha256=weights_sha256,
            expected_config_sha256=config_sha256,
        )
        print(
            json.dumps(
                _result(
                    status="downloaded_and_verified",
                    output_dir=output_dir,
                    spec=spec,
                    revision=revision,
                    hashes=installed,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
