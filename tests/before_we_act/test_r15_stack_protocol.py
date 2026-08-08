import hashlib
import json

import pytest

from scripts.before_we_act.prepare_r15_stack_protocol import prepare


TASK = "three_robots_stack_cube"


def write_inputs(tmp_path, *, gate_offset=0):
    rows = [
        {"seed": 1_000_000_000 + index, "success": index % 7 == 0, "steps": 800}
        for index in range(100)
    ]
    frozen = tmp_path / "frozen100.json"
    frozen.write_text(json.dumps({"task": TASK, "rows": rows}) + "\n")
    gate = tmp_path / "gate20.json"
    gate.write_text(
        json.dumps(
            {
                "task": TASK,
                "seeds": [row["seed"] for row in rows[gate_offset : gate_offset + 20]],
            }
        )
        + "\n"
    )
    return frozen, gate, rows


def test_prepare_freezes_disjoint_ordered_splits(tmp_path):
    frozen, gate, rows = write_inputs(tmp_path)
    output = tmp_path / "protocol"
    manifest = prepare(frozen, gate, output)

    seen = []
    for index, name in enumerate(
        ("gate20", "discovery20", "validation20", "reserve20", "final20")
    ):
        payload = json.loads((output / f"{name}.json").read_text())
        expected = [row["seed"] for row in rows[index * 20 : (index + 1) * 20]]
        assert payload["seeds"] == expected
        assert payload["source_baseline_successes"] == sum(
            row["success"] for row in rows[index * 20 : (index + 1) * 20]
        )
        seen.extend(payload["seeds"])
        assert manifest["splits"][name]["sha256"] == hashlib.sha256(
            (output / f"{name}.json").read_bytes()
        ).hexdigest()
    assert len(seen) == len(set(seen)) == 100


def test_prepare_rejects_gate20_drift(tmp_path):
    frozen, gate, _ = write_inputs(tmp_path, gate_offset=1)
    with pytest.raises(ValueError, match="rows 0:20"):
        prepare(frozen, gate, tmp_path / "protocol")
