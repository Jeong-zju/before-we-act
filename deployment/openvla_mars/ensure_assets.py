"""Resume the OpenVLA base-model and RoboFactory simulator asset snapshots."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "/workspace/.secrets/hf_token"))

def main() -> None:
    token = TOKEN_FILE.read_text().strip()
    if not token:
        raise RuntimeError("empty Hugging Face token")
    snapshot_download("sparklexfantasy/RoboFactory_asset", repo_type="dataset",
                      local_dir="/workspace/repos/RoboFactory-MARS/assets", token=token, max_workers=8)
    asset_files = sum(path.is_file() for path in Path("/workspace/repos/RoboFactory-MARS/assets").rglob("*"))
    if asset_files < 1350:
        raise RuntimeError(f"RoboFactory asset snapshot incomplete: {asset_files} files")
    snapshot_download("openvla/openvla-7b", local_dir="/workspace/models/openvla-7b",
                      cache_dir="/workspace/.hf_home", token=token, max_workers=4)
    model = Path("/workspace/models/openvla-7b")
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise RuntimeError("OpenVLA-7B snapshot incomplete")
    print({"assets": asset_files, "model": str(model), "status": "complete"}, flush=True)

if __name__ == "__main__": main()
