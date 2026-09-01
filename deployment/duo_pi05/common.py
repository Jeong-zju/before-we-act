from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

TASKS = (
    "ball_maze", "bin_sort", "block_balance", "carry_pot", "hinge_chest",
    "join_blocks", "pour_marbles", "spring_door", "transfer_cube",
    "transfer_gate", "transfer_reorient",
)
PROMPTS = {
    "ball_maze": "pick up the board and tilt it so the ball roles onto the red square",
    "bin_sort": "use the left arm to place the white cube in the white bowl; use the right arm to place the black cube in the black bowl",
    "block_balance": "place the beam on the cube and then place the other blocks on the beam simultaneously using one arm for each cube",
    "carry_pot": "use two arms to carry the pot at the handle on the stove",
    "hinge_chest": "open the box with the right arm and place the cube inside the box with the left arm",
    "join_blocks": "join the two blocks using the peg on the left block and join the free socket of the right block with the peg on the wall",
    "pour_marbles": "grasp and lift both cups, then pour the marbles from one cup into the other and place the cups back to their original location inside the green square",
    "spring_door": "use the left arm to open the microwave door, then use the right arm to place the box inside the microwave, and close the door again",
    "transfer_cube": "grasp the white cube with the right arm, hand it over to the left arm and place it in the white bowl with the left arm",
    "transfer_gate": "use the right arm to pick up the white box, and hand it over to the left arm through the hoop, then place it on the green mat with the left arm",
    "transfer_reorient": "grasp the block with the right arm, hand it over to the left arm such that the left arm can easily insert the piece later, then insert the block into the socket with the left arm",
}
MAX_STEPS = {
    "ball_maze": 526, "bin_sort": 365, "block_balance": 1091,
    "carry_pot": 840, "hinge_chest": 610, "join_blocks": 1314,
    "pour_marbles": 549, "spring_door": 1070, "transfer_cube": 605,
    "transfer_gate": 630, "transfer_reorient": 883,
}
DATASET_REVISION = "b741bc915d942ecadaefb4e3de6bbd716c1b8b1b"
DUOBENCH_REVISION = "082a57cdafea9db115029e6fe9e03691e755f93f"
RCS_REVISION = "4f78aeffae3bc4d0c02e7beab993e5406261dcf6"
OPENPI_REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
ACTION_LOW = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, 0.0)
ACTION_HIGH = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 1.0)
POLICY_CONTRACT = "shared_weights_decentralized_head_rgb_local_wrist_rgb_own_state8_to_own_action8"
TEMPORAL_CONTRACT = "post_action_observation_row_i_to_action_row_i_plus_1_horizon16_episode_safe"
EVALUATOR_REVISION = "duobench-pi05-lora-decentralized-lag1-v1"


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if root.is_file(): return sha256_file(root)
    if not root.is_dir(): raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8")); digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def checkpoint_identity(path: str | Path) -> str:
    """Fast, stable-on-this-instance identity for retry receipt matching."""
    root = Path(path); digest = hashlib.sha256()
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        stat = item.stat(); digest.update(item.relative_to(root).as_posix().encode()); digest.update(b"\0")
        digest.update(str(stat.st_size).encode()); digest.update(b"\0"); digest.update(str(stat.st_mtime_ns).encode()); digest.update(b"\n")
    return digest.hexdigest()
