# DuoBench Diffusion Policy reproducibility bundle

This directory freezes the auditable evidence for the decentralized DuoBench
Diffusion Policy result. The executable implementation is in
`deployment/duo_dp/`; the complete paper-facing configuration is
`configs/duobench_dp_formal_v1.json`.

## Formal result

- Checkpoint: `/workspace/runs/duobench-dp/formal/final.pt`
- Checkpoint SHA-256: `7d71c2877fa90b53ca4cf2c48d9f22d14875f995c99649b374ef2219ba2f341e`
- Training: 60,000 updates, batch 64, all 550 demonstrations, no split
- Validation20: 1/220 successes (0.45%); normalized final-stage progress 0.0784091
- Successful task: Transfer-Gate, 1/20
- Evaluator: `duobench-dp-obs3-lag1-h8-exec6-direct-v1`

## Reproduction commands

Install the upstream RoboFactory Diffusion Policy implementation and DuoBench,
prepare the 550 demonstrations with the pinned revisions in the JSON contract,
then launch the managed pipeline:

```bash
export DUO_DP_REPO=/workspace/repos/before-we-act
export DUO_DP_UPSTREAM=/workspace/repos/RoboFactory/robofactory/policy/Diffusion-Policy
export DUO_DP_DATASET=/workspace/datasets/duobench
export DUO_DP_RUN=/workspace/runs/duobench-dp
export DUO_DP_CONFIG=$DUO_DP_REPO/configs/duobench_dp_formal_v1.json
/venv/main/bin/python -m deployment.duo_dp.supervisor
```

The supervisor performs data bootstrap, preflight, training smoke, closed-loop
smoke, formal training, Validation20, and final report generation. It resumes
completed stages safely and binds its launch values to the frozen JSON contract.

The JSON retains separate source hashes for the exact formal-run freeze and the
current synchronized implementation. The latter includes later diagnostic probe
support plus a local supervisor hardening patch that binds every launch value to
the frozen configuration; it must not be misrepresented as the byte-identical
source snapshot of the already-completed formal training run.

## Important fixes and diagnostics

- The data adapter respects the post-action recording convention: observation
  row `i` predicts action row `i+1`, never crossing episode boundaries.
- State and action are controller-equivalent absolute joint7 plus binary
  gripper1, normalized once with the frozen corpus min/max vectors and clipped
  once to the pinned FR3 actuator ranges after decoding.
- Each arm receives shared head RGB, its own wrist RGB and its own qpos only;
  peer proprioception, peer wrist images and simulator state remain inaccessible.
- Validation supports explicit replan interval, diffusion inference steps and
  EMA/online weights, with variant-specific evaluator revisions so cached rows
  cannot be mixed across protocols.
- Evaluation records emitted joint ranges, maximum joint deltas and gripper
  transitions, making action-semantic failures visible in each rollout record.
- Optional task conditioning and transition-aware gripper sampling/loss were
  implemented as isolated probes. Their defaults preserve the formal policy
  (`task_conditioning=false`, `transition_fraction=0`, gripper weight `1`).
- The archived probe decisions show that these changes did not yield a
  meaningful improvement; the formal checkpoint and Validation20 result remain
  unchanged.

`interface_audit.json` is the final diagnosis. The official-stage replay and
all preprocessing/normalization checks found no material interface fault; the
remaining failures are attributed to policy accuracy and closed-loop robustness
under the required decentralized contract.
