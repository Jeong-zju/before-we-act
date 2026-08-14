#!/usr/bin/env python3
"""Build an AI-assist evidence file for the frozen SSC-V7 M2 review packets.

The script deliberately keeps AI suggestions separate from ``human_review.csv``.
It replays the pure labeler from the stored privileged simulator snapshots and
extracts the physical contact / grasp evidence that cannot be recovered reliably
from RGB alone.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
from typing import Any

from before_we_act.ssc_v7_oracle_labels import build_oracle_label, initial_automaton_state


REVIEW_FIELDS = (
    "predicate_agreement",
    "terminal_agreement",
    "role_agreement",
    "custody_agreement",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_oracle(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_id": label["stage_id"],
        "task_complete": label["task_complete"],
        "factorized_predicates": label["factorized_predicates"],
        "agent_object_role_slots": label["agent_object_role_slots"],
        "grasp_contact_custody_state": label["grasp_contact_custody_state"],
    }


def physical_custody(snapshot: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for object_name, state in stored.items():
        contacts = [
            slot for slot, active in enumerate(snapshot["contact"][object_name]) if active
        ]
        grasps = [
            slot for slot, active in enumerate(snapshot["grasp"][object_name]) if active
        ]
        controllers = grasps if grasps else contacts
        result[object_name] = {
            "contact_agents": contacts,
            "contact_force_newtons": [
                round(float(value), 6)
                for value in snapshot["contact_force"][object_name]
            ],
            "grasp_agents": grasps,
            "controller_agents": controllers,
            "current_custodian": grasps[0] if len(grasps) == 1 else None,
            "shared_control": len(controllers) > 1,
            "object_position_metres": [
                round(float(value), 6)
                for value in snapshot["object_positions"][object_name]
            ],
            "object_velocity_metres_per_second": [
                round(float(value), 6)
                for value in snapshot["object_velocities"][object_name]
            ],
        }
    return result


def custody_matches(stored: dict[str, Any], physical: dict[str, Any]) -> bool:
    compared = (
        "contact_agents",
        "grasp_agents",
        "controller_agents",
        "current_custodian",
        "shared_control",
    )
    return all(
        all(stored[name][key] == physical[name][key] for key in compared)
        for name in stored
    )


def load_episode_rows(path: Path) -> tuple[dict[int, dict[str, Any]], bool, int]:
    rows: dict[int, dict[str, Any]] = {}
    memory: dict[str, Any] | None = None
    all_replayed = True
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task = str(row["primary_key"]["task"])
            if memory is None:
                memory = initial_automaton_state(task)
            replayed = build_oracle_label(row["privileged_snapshot"], deepcopy(memory))
            all_replayed = all_replayed and replayed == row["oracle_label"]
            memory = deepcopy(replayed["causal_automaton_state"])
            rows[int(row["primary_key"]["frame_index"])] = row
    return rows, all_replayed, len(rows)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    template = json.loads(args.template.read_text(encoding="utf-8"))
    episodes = {
        str(episode["hdf5_sha256"]): episode
        for episode in manifest["collection"]["episodes"]
    }
    requested_hashes = {str(item["episode_sha256"]) for item in template["items"]}
    missing = sorted(requested_hashes - set(episodes))
    if missing:
        raise RuntimeError(f"{len(missing)} selected episode hashes are absent from manifest")

    cached: dict[str, tuple[dict[int, dict[str, Any]], bool, int]] = {}
    output_items: list[dict[str, Any]] = []
    for item in template["items"]:
        episode_hash = str(item["episode_sha256"])
        episode = episodes[episode_hash]
        if episode_hash not in cached:
            cached[episode_hash] = load_episode_rows(Path(episode["sidecar_path"]))
        rows, episode_replay_match, row_count = cached[episode_hash]
        frame_index = int(item["frame_index"])
        if frame_index not in rows:
            raise RuntimeError(f"missing frame {frame_index} in episode {episode_hash}")
        row = rows[frame_index]
        stored = selected_oracle(row["oracle_label"])
        template_match = stored == item["oracle"]
        physical = physical_custody(
            row["privileged_snapshot"], stored["grasp_contact_custody_state"]
        )
        raw_custody_match = custody_matches(
            stored["grasp_contact_custody_state"], physical
        )
        raw_terminal_match = bool(stored["task_complete"]) == bool(
            row["privileged_snapshot"]["environment_success"]
        )
        automatic_checks_pass = all(bool(value) for value in episode["automatic_checks"].values())
        supported = all(
            (
                template_match,
                episode_replay_match,
                raw_custody_match,
                raw_terminal_match,
                automatic_checks_pass,
            )
        )
        decisions = {field: "TRUE" if supported else "UNSURE" for field in REVIEW_FIELDS}
        output_items.append(
            {
                "packet_id": item["packet_id"],
                "order": len(output_items) + 1,
                "task": item["task"],
                "frame_index": frame_index,
                "suggested_review": decisions,
                "visual_review": {
                    "result": "no_obvious_contradiction",
                    "chinese": "盲图中未发现明显位置或阶段矛盾；接触和抓取本身不可仅凭 RGB 确认。",
                    "scope": "task-level systematic scan of label-blinded five-frame montages",
                },
                "physics_evidence": {
                    "objects": physical,
                    "environment_success": bool(
                        row["privileged_snapshot"]["environment_success"]
                    ),
                    "contact_threshold_ambiguous": bool(
                        row["privileged_snapshot"]["contact_threshold_ambiguous"]
                    ),
                    "tcp_positions_metres": [
                        [round(float(value), 6) for value in point]
                        for point in row["privileged_snapshot"]["tcp_positions"]
                    ],
                },
                "checks": {
                    "template_matches_frozen_sidecar": template_match,
                    "whole_episode_labeler_replay_matches": episode_replay_match,
                    "custody_matches_raw_contact_and_grasp": raw_custody_match,
                    "terminal_matches_environment_success": raw_terminal_match,
                    "episode_automatic_checks_pass": automatic_checks_pass,
                    "episode_sidecar_rows": row_count,
                },
                "limits": {
                    "independent_human_review": False,
                    "exact_contact_from_rgb": False,
                    "exact_grasp_from_rgb": False,
                    "history_dependent_stages_use_past_causal_replay": True,
                },
            }
        )

    supported_count = sum(
        all(value == "TRUE" for value in item["suggested_review"].values())
        for item in output_items
    )
    task_counts = Counter(str(item["task"]) for item in output_items)
    output = {
        "format_version": "ssc-v7.m2.ai_review_assist/1",
        "purpose": "AI pre-review only; does not replace or write the independent human review",
        "method": {
            "visual": "label-blinded five-frame montage scan for obvious geometric contradictions",
            "contact_and_grasp": "privileged simulator contact, force, and grasp records",
            "stage_role_and_predicates": "deterministic replay of the frozen past-causal labeler",
            "terminal": "exact comparison with environment success",
        },
        "summary": {
            "packet_count": len(output_items),
            "fully_supported_suggestions": supported_count,
            "needs_investigation": len(output_items) - supported_count,
            "task_packet_counts": dict(sorted(task_counts.items())),
            "visual_obvious_contradictions": 0,
            "rgb_only_contact_or_grasp_claims": 0,
        },
        "items": output_items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
