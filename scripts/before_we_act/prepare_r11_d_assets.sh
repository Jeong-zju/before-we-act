#!/usr/bin/env bash
set -euo pipefail

HF_BIN="${HF_BIN:-/venv/robofactory-act/bin/hf}"
PYTHON_BIN="${PYTHON_BIN:-/venv/robofactory-act/bin/python}"
TOKEN_FILE="${R11_HF_TOKEN_FILE:-/workspace/bwa_runs/r11-four-way-v1/secrets/hf_token}"
ASSET_ROOT="${R11_D_ASSET_ROOT:-/workspace/artifacts/r11_upstream/lawam_assets}"
HF_CACHE_ROOT="${HF_HOME:-/workspace/.cache/huggingface}"
RECEIPT="${ASSET_ROOT}/ASSET_BUNDLE_RECEIPT.json"

QWEN_REPO="Qwen/Qwen3-VL-2B-Instruct"
QWEN_REV="89644892e4d85e24eaac8bacfd4f463576704203"
DINO_REPO="facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REV="5931719e67bbdb9737e363e781fb0c67687896bc"
LAM_REPO="jialei02/lawam_lam"
LAM_REV="bd993da2a0861afaac5a95ac86d2555b1313ab8c"
PRETRAIN_REPO="jialei02/lawam_pretrain"
PRETRAIN_REV="62b14a8e8990050ec8aeb1e1b8c8694d2bf60e84"

[[ -x "${HF_BIN}" ]] || { echo "missing hf CLI: ${HF_BIN}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${TOKEN_FILE}" ]] || { echo "missing mode-0600 token file" >&2; exit 2; }
[[ "$(stat -c %a "${TOKEN_FILE}")" == "600" ]] || {
  echo "HF token file must have mode 0600" >&2
  exit 2
}

mkdir -p "${ASSET_ROOT}"
if [[ -f "${RECEIPT}" && "$(stat -c %a "${RECEIPT}")" == "444" ]]; then
  ASSET_RECEIPT="${RECEIPT}" "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

receipt = json.loads(Path(os.environ["ASSET_RECEIPT"]).read_text(encoding="utf-8"))
if receipt.get("status") != "PASSED":
    raise SystemExit("immutable D asset receipt is not PASSED")
for item in receipt["files"]:
    path = Path(item["local_path"])
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != item["sha256"]:
        raise SystemExit(f"D asset hash mismatch: {path}")
print("R11 D asset receipt already PASSED")
PY
  exit 0
fi

IFS= read -r HF_TOKEN < "${TOKEN_FILE}" || [[ -n "${HF_TOKEN:-}" ]]
export HF_TOKEN
export HF_HOME="${HF_CACHE_ROOT}"
export HF_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

download_repo() {
  local repo="$1"
  local revision="$2"
  local destination="$3"
  mkdir -p "${destination}"
  "${HF_BIN}" download "${repo}" --revision "${revision}" --local-dir "${destination}"
}

download_repo "${QWEN_REPO}" "${QWEN_REV}" "${ASSET_ROOT}/qwen3-vl-2b"
download_repo "${DINO_REPO}" "${DINO_REV}" "${ASSET_ROOT}/dinov3-vitb16"
download_repo "${LAM_REPO}" "${LAM_REV}" "${ASSET_ROOT}/lawam_lam"
download_repo "${PRETRAIN_REPO}" "${PRETRAIN_REV}" "${ASSET_ROOT}/lawam_pretrain"
unset HF_TOKEN

export ASSET_ROOT RECEIPT QWEN_REPO QWEN_REV DINO_REPO DINO_REV
export LAM_REPO LAM_REV PRETRAIN_REPO PRETRAIN_REV
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import time

import yaml

root = Path(os.environ["ASSET_ROOT"]).resolve()
receipt_path = Path(os.environ["RECEIPT"])
qwen = root / "qwen3-vl-2b"
dino = root / "dinov3-vitb16"
lam = root / "lawam_lam"
pretrain = root / "lawam_pretrain"

required = [
    qwen / "model.safetensors",
    qwen / "README.md",
    dino / "model.safetensors",
    dino / "LICENSE.md",
    lam / "checkpoints/pytorch_model.pt",
    lam / "dino_large_vae.yaml",
    pretrain / "final_model/pytorch_model.pt",
    pretrain / "config.yaml",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing D foundation file: {missing[0]}")

source_yaml = yaml.safe_load((lam / "dino_large_vae.yaml").read_text(encoding="utf-8"))
if not isinstance(source_yaml, dict) or not isinstance(source_yaml.get("model"), dict):
    raise SystemExit("invalid upstream LAM YAML")
source_yaml["model"]["vision_model_id"] = str(dino)
patched_yaml = root / "dino_large_vae.r11.yaml"
patched_yaml.write_text(yaml.safe_dump(source_yaml, sort_keys=False), encoding="utf-8")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or ".cache" in path.parts or path == receipt_path:
        continue
    files.append(
        {
            "local_path": str(path.resolve()),
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )

receipt = {
    "format_version": "before-we-act.r11.foundation_bundle/1",
    "status": "PASSED",
    "candidate": "D",
    "created_unix": time.time(),
    "repositories": {
        os.environ["QWEN_REPO"]: os.environ["QWEN_REV"],
        os.environ["DINO_REPO"]: os.environ["DINO_REV"],
        os.environ["LAM_REPO"]: os.environ["LAM_REV"],
        os.environ["PRETRAIN_REPO"]: os.environ["PRETRAIN_REV"],
    },
    "licenses": {
        os.environ["QWEN_REPO"]: "Apache-2.0",
        os.environ["DINO_REPO"]: "DINOv3 License",
        os.environ["LAM_REPO"]: "MIT",
        os.environ["PRETRAIN_REPO"]: "MIT",
    },
    "license_acceptance": {os.environ["DINO_REPO"]: "verified_noninteractive"},
    "task_sft_checkpoint": "none",
    "artifacts": {
        "qwen_directory": str(qwen),
        "dino_directory": str(dino),
        "lam_checkpoint": str(lam / "checkpoints/pytorch_model.pt"),
        "lam_yaml": str(patched_yaml),
        "pretrain_checkpoint": str(pretrain / "final_model/pytorch_model.pt"),
    },
    "files": files,
}
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
receipt_path.chmod(0o444)
print(f"R11 D asset receipt PASSED: files={len(files)} bytes={sum(x['bytes'] for x in files)}")
PY
