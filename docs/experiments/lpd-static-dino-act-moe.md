# LPD static-RGB DINO+ACT+MoE experiment

This branch absorbs the parts of the reported 90%+ stack that do not require a
new dataset:

- the existing per-agent 640×480 **world-fixed** RGB stream is used unchanged;
- the already pinned DINOv3 artifact supplies the full 30×40 patch grid;
- one shared policy maps a normalized 18D Panda state to an 8D local action;
- ACT predicts a 100-step action chunk with a 4/7 CVAE;
- every action-decoder FFN is replaced by four experts with top-2 routing;
- inference replans every step and exponentially ensembles overlapping chunks.

It deliberately has no depth input and never imports a wrist-camera patch. It
reads the same converted LiftBarrier and LongPipelineDelivery manifests as the
current WAM, so no collection, conversion or upload is required.

Run `./scripts/run_lpd_single_5090.sh full`.

## Preliminary closed-loop result

The operator reported the following result from commit `34cd6e7` using the
branch's default fixed-seed gate (`20` episodes per task, seed start `900`):

| Task | Successes | Success rate |
|---|---:|---:|
| LiftBarrier | 17/20 | 85% |
| LongPipelineDelivery | 15/20 | 75% |

This is a large improvement over the failed joint M2 checkpoint at the branch
point and is sufficient to promote this implementation to the candidate
baseline on `feat/model-improvements`.

The result establishes the complete static-RGB DINO+ACT+MoE stack as the
winning candidate. It does not by itself isolate the effects of the per-agent
representation, ACT chunking, sparse MoE, or temporal ensembling. Those claims
require matched branch comparisons and component ablations.

## Evidence boundary and reproduction

The original checkpoint and gate JSON lived in a temporary worktree and were
removed when that worktree was cleaned. Therefore the numbers above are an
operator-reported preliminary result, not an immutable accepted checkpoint.
Before release acceptance:

1. retrain from the pinned config and data revisions;
2. retain the checkpoint SHA-256 and gate JSON outside temporary storage;
3. run the formal `100`-episode-per-task gate;
4. compare the factorized-M2 and temporal-ensemble branches on the same seeds;
5. run matched ablations before assigning improvement to an individual
   component.

## Candidate baseline integration

The experiment history was merged into `feat/model-improvements` by merge
commit `859cecd`. The merged implementation is the new candidate baseline; it
does not replace the formal acceptance gate.

The gate now emits `wam.robofactory.lpd_fixed_seed_gate/2`, which binds:

- source commit, config SHA-256, checkpoint SHA-256 and checkpoint format;
- the complete inference client identity;
- every evaluation seed and episode success;
- per-task Wilson intervals and the fixed-seed protocol.

`scripts/summarize_lpd_experiment_matrix.py` rejects unpaired seed schedules
and reports paired deltas plus exact McNemar counts. This prevents aggregate
rates from different seed sets being treated as a controlled comparison.

## Minimal causal matrix

| Candidate | Representation/action change | Config or branch | Checkpoint |
|---|---|---|---|
| full candidate | per-agent DINO + ACT + top-2 MoE + temporal ensemble | `configs/static_act/lpd_static_dino_act_moe.yaml` | full candidate |
| dense decoder | replace top-2 MoE by a 3072-wide compute-matched dense FFN | `configs/static_act/lpd_static_dino_act_dense_compute_matched.yaml` | separately trained |
| latest chunk | remove overlapping-chunk temporal aggregation only | `configs/static_act/lpd_static_dino_act_moe_latest_chunk.yaml` | reuse full candidate |
| factorized WAM | factorize WAM action I/O without importing this ACT stack | `exp/lpd-agent-factorized-m2` | branch-specific |
| temporal WAM | add temporal ensembling to the branch-point WAM | `exp/lpd-temporal-ensemble` | branch-specific |

The two inference configs for the full and latest-chunk variants have identical
data, vision, model, training and checkpoint sections. The dense config changes
only the decoder kind, neutralizes the router loss, and uses a separate
checkpoint path.

After gate artifacts exist, create the paired report without rerunning any
model:

```bash
uv run --frozen python scripts/summarize_lpd_experiment_matrix.py \
  --candidate branch_point=/path/to/branch_point/gate_summary.json \
  --candidate full=/path/to/full/gate_summary.json \
  --candidate factorized=/path/to/factorized/gate_summary.json \
  --candidate temporal=/path/to/temporal/gate_summary.json \
  --candidate dense=/path/to/dense/gate_summary.json \
  --candidate latest=/path/to/latest/gate_summary.json \
  --baseline branch_point \
  --output-json outputs/lpd_comparison/comparison.json \
  --output-markdown outputs/lpd_comparison/comparison.md
```

An attempted local reproduction was stopped at update 100 before any
checkpoint or resume file was written, and no result from that attempt is used
as evidence. Per operator instruction, the ablation infrastructure itself is
validated with code tests only; it does not start training or rollout.
