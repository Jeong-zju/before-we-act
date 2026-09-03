"""Precompute one frozen T5 embedding per DuoBench task for upstream RDT."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from models.multimodal_encoder.t5_encoder import T5Embedder
from .protocol import TASKS, T5_MODEL

TASK_TEXT = {
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

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); p.add_argument("--device", default="cuda:0"); p.add_argument("--model", default=T5_MODEL)
    a = p.parse_args(); cfg = json.loads((a.data / "manifest.json").read_text()); del cfg
    embedder = T5Embedder(from_pretrained=a.model, model_max_length=1024, device=torch.device(a.device))
    for task in TASKS:
        out = a.data / task / "lang_embed.pt"; out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_file(): continue
        embeds, _ = embedder.get_text_embeddings([TASK_TEXT[task]])
        torch.save(embeds[0].cpu(), out); print(json.dumps({"task":task,"path":str(out)}), flush=True)
    empty = Path("data/empty_lang_embed.pt"); empty.parent.mkdir(exist_ok=True)
    if not empty.is_file(): torch.save(torch.zeros((1, embedder.model.config.d_model), dtype=torch.bfloat16), empty)
if __name__ == "__main__": main()
