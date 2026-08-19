#!/usr/bin/env python3
"""Serve a small read-only dashboard backed by SSH status files."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import urlparse

REMOTE = os.environ.get("BWA_REMOTE_HOST", "root@69.176.92.104")
PORT = os.environ.get("BWA_REMOTE_PORT", "10328")
REMOTE_ROOT = os.environ.get("BWA_REMOTE_ROOT", "/workspace/bwa-baselines")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-p", PORT, REMOTE]
SAFE_PATH = re.compile(r"^/[A-Za-z0-9_./-]+$")
HTML = """<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Before We Act Baselines</title><style>body{margin:0;background:#101316;color:#e8edf2;font:14px system-ui;padding:24px}h1{font-size:18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.card{background:#191e24;border:1px solid #303943;border-radius:6px;padding:14px}.muted{color:#99a5b1}pre{white-space:pre-wrap;color:#c9d3dc}</style><h1>Before We Act / RoboFactory Baselines</h1><p id='meta' class='muted'>loading...</p><div id='cards' class='grid'></div><pre id='error'></pre><script>async function load(){let d=await fetch('/api/status').then(r=>r.json());meta.textContent=`${d.remote.host} · ${d.remote.root} · ${new Date().toLocaleTimeString()}`;cards.innerHTML=(d.statuses||[]).map(x=>`<article class=card><b>${x.display_name||x.baseline}</b><p>状态: ${x.status}</p><p>进度: ${x.step??'—'} / ${x.total_steps??'—'}</p><p>loss: ${x.loss==null?'—':Number(x.loss).toFixed(5)}</p><p>设备: ${x.device||'—'}</p></article>`).join('')||'<p class=muted>尚未发现 status.json</p>';error.textContent=(d.errors||[]).join('\\n')}load();setInterval(load,5000)</script>"""


def status() -> dict:
    try:
        command = f"find {shlex.quote(REMOTE_ROOT)} -name status.json -type f -print"
        raw = subprocess.run([*SSH, command], text=True, capture_output=True, timeout=15, check=True).stdout
        values = []
        errors = []
        for path in raw.splitlines():
            if not SAFE_PATH.fullmatch(path):
                errors.append(f"ignored unsafe path: {path}")
                continue
            try:
                text = subprocess.run([*SSH, f"cat {shlex.quote(path)}"], text=True, capture_output=True, timeout=8, check=True).stdout
                values.append(json.loads(text))
            except (json.JSONDecodeError, subprocess.SubprocessError) as exc:
                errors.append(f"{path}: {exc}")
        return {"remote": {"host": REMOTE, "root": REMOTE_ROOT}, "statuses": values, "errors": errors}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"remote": {"host": REMOTE, "root": REMOTE_ROOT}, "statuses": [], "errors": [str(exc)]}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body, content_type = HTML.encode(), "text/html; charset=utf-8"
        elif path == "/api/status":
            body, content_type = json.dumps(status(), ensure_ascii=False).encode(), "application/json"
        elif path == "/api/health":
            body, content_type = b'{"ok":true}', "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        pass


def main() -> None:
    host = os.environ.get("BWA_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("BWA_WEB_PORT", "8088"))
    print(f"dashboard: http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
