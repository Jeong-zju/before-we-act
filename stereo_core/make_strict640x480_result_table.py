"""Render the final five-task, four-method frozen-100-seed comparison table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = [
    ("LiftBarrier", 2, "lift_barrier"),
    ("CameraAlignment", 3, "camera_alignment"),
    ("ThreeRobotsStackCube", 3, "three_robots_stack_cube"),
    ("LongPipelineDelivery", 4, "long_pipeline_delivery"),
    ("TakePhoto", 4, "take_photo"),
]
METHODS = [
    ("冻结 DINOv3-ACT", "frozen_dinov3_act_all5_80k"),
    ("Stereo-ACT-cross_relbias", "stereo_cross_relbias_all5_80k"),
    ("Stereo-FFN-MoE", "stereo_ffn_moe_all5_80k"),
    ("Local-ARCA", "local_arca_all5_80k"),
]


def read_result(root: Path, method: str, task: str, seed_root: Path):
    # The watcher writes one audited JSON per task beside the exact final
    # checkpoint.  Keeping the path tied to the training run prevents a stale
    # result from a historical run from entering the formal All-5 table.
    path = root / method / "formal_heldout_100" / f"eval_{task}.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("episodes") != 100:
        raise ValueError(f"not formal 100-seed output: {path}")
    expected = hashlib.sha256((seed_root / f"{task}.json").read_bytes()).hexdigest()
    protocol = raw.get("seed_protocol", {})
    if protocol.get("sha256") != expected or protocol.get("training_seed_overlap") != 0:
        raise ValueError(f"seed audit failed: {path}")
    return int(raw["successes"]), int(raw["episodes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/RoboFactory/runs/strict640x480_v2/results")
    parser.add_argument("--output", default="/workspace/RoboFactory/runs/strict640x480_v2/final_report")
    args = parser.parse_args(); root, out = Path(args.root), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    seed_root = root.parent / "heldout_seeds"
    values = [[read_result(root, method_dir, task_id, seed_root) for _, _, task_id in TASKS] for _, method_dir in METHODS]
    lines = ["| Training corpus | Test task (robots) | " + " | ".join(name for name, _ in METHODS) + " |",
             "|---|---|" + "---|" * len(METHODS)]
    table = []
    for task_index, (task_name, robots, _) in enumerate(TASKS):
        row = [column[task_index] for column in values]
        best = max((item[0] / item[1] for item in row if item), default=None)
        cells = []
        for item in row:
            if not item: cells.append("—"); continue
            text = f"{item[0]}/{item[1]} ({100 * item[0] / item[1]:.1f}%)"
            cells.append(f"**{text}**" if item[0] / item[1] == best else text)
        lines.append("| All-5 strict640x480-v2 | " + f"{task_name} ({robots}) | " + " | ".join(cells) + " |")
        table.append(cells)
    (out / "performance_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "formal_seed_audit.json").write_text(json.dumps({
        "status": "PASS", "episodes_per_cell": 100,
        "tasks": [task_id for _, _, task_id in TASKS],
        "methods": [method_id for _, method_id in METHODS],
        "condition": "each result uses the matching frozen manifest with zero training-seed overlap",
    }, indent=2) + "\n", encoding="utf-8")
    fig, axis = plt.subplots(figsize=(16, 4.8)); axis.axis("off")
    rendered = axis.table(cellText=table, colLabels=[name for name, _ in METHODS],
                          rowLabels=[f"{name} ({robots})" for name, robots, _ in TASKS],
                          cellLoc="center", loc="center")
    rendered.auto_set_font_size(False); rendered.set_fontsize(10); rendered.scale(1, 2.0)
    axis.set_title("RoboFactory — strict wrist-only RGB-D, All-5, frozen unseen 100-seed evaluation", pad=20, fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(out / "performance_table.png", dpi=220, bbox_inches="tight")
    print(out / "performance_table.png")


if __name__ == "__main__":
    main()
