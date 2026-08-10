import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from before_we_act.r11_vendor import verify_vendor_checkout


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_vendor_checkout_commit_origin_and_hash_are_fail_closed(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init")
    _git(upstream, "config", "user.name", "R11 Test")
    _git(upstream, "config", "user.email", "r11@example.invalid")
    source = upstream / "module.py"
    source.write_text("VALUE = 7\n", encoding="utf-8")
    _git(upstream, "add", "module.py")
    _git(upstream, "commit", "-m", "fixture")
    commit = _git(upstream, "rev-parse", "HEAD")
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", str(upstream), str(checkout)], check=True, capture_output=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = {
        "format_version": "before-we-act.r11.source_receipt/1",
        "upstream_url": str(upstream),
        "upstream_commit": commit,
        "files": [{"upstream_path": "module.py", "sha256": digest}],
    }
    receipt_path = tmp_path / "SOURCE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_vendor_checkout(receipt_path, checkout)["status"] == "PASSED"
    (checkout / "module.py").write_text("VALUE = 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked changes"):
        verify_vendor_checkout(receipt_path, checkout)
    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_vendor_checkout(receipt_path, checkout, require_clean=False)
