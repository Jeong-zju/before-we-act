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

## Result snapshot

| Commit | LongPipelineDelivery (LPD) | LiftBarrier |
|---|---:|---:|
| `4dae9e79bbfdd2d36a0381da2179a97390766d83` | 全部失败 | 仅少数成功 |
| `c79ff1ed515a8284cf6dd7d6979f43b245c90487` | 15/20 | 17/20 |

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

## S0 Round 1 decision (2026-07-28)

The operator reported the following Gate20 aggregates from the `s0-round1`
monitor captured at `2026-07-28T14:10:34.935096+00:00`. B0 and B1 each
completed `80,000/80,000` optimizer updates, with monitor-rounded terminal loss
`0.002`. B3 correctly shows no training because the frozen protocol makes it
reuse B0's checkpoint and change only chunk aggregation. B2 was at
`4,684/80,000` updates (5.9%) in the captured monitor. Because its training
time was too long for the fast-track schedule, the operator requested an early
stop before Gate20 and selected B0 as the parent coordinate for the next stage.

| Candidate | Controlled coordinate | LiftBarrier | Wilson 95% | LongPipelineDelivery | Wilson 95% | Gate20 |
|---|---|---:|---:|---:|---:|---|
| B0 | sparse top-2 MoE + temporal ensemble | 17/20 (85%) | [64.0%, 94.8%] | 19/20 (95%) | [76.4%, 99.1%] | pass |
| B1 | compute-matched dense + temporal ensemble | 11/20 (55%) | [34.2%, 74.2%] | 0/20 (0%) | [0.0%, 16.1%] | done (failed) |
| B2 | Rectified Flow reference | — | — | — | — | operator stop requested before Gate20 |
| B3 | B0 checkpoint + latest chunk | 6/20 (30%) | [14.5%, 51.9%] | 0/20 (0%) | [0.0%, 16.1%] | done (failed) |

The Wilson intervals use the aggregate counts shown by the monitor. In this
monitor, `gate=pass` means `gate_summary.passed=true`, while `gate=done` means
the gate completed with `passed=false`; it does not mean validation is still
running. The local repository does not contain this remote run's gate JSON, so
checkpoint hashes and episode-level paired success records remain pending
artifact synchronization.

### S0 decision

1. **Use B0 as the next-stage engineering parent.** It is the only completed
   coordinate that succeeds on both tasks. Relative to B0, B1 loses 30
   percentage points on LiftBarrier and 95 points on LongPipelineDelivery;
   B3 loses 55 and 95 points respectively. The fast-track therefore closes S0
   with B0 as the parent coordinate for R1. This is an engineering scheduling
   decision from the available success-rate evidence, not formal acceptance or
   proof that B0 is universally better than Rectified Flow.
2. **Do not draw a model-quality conclusion from B2.** B2 is stopped because
   of training wall time, before completing the matched update budget or any
   Gate20 rollout. It is recorded as `operator_stop`, not as a zero-success
   candidate or evidence against Rectified Flow. Resuming B2 is not required
   for the next stage.
3. **Reject latest-chunk inference for this checkpoint.** The frozen manifest
   and launcher require B3 to reuse B0's checkpoint and paired Gate20 seeds, and
   B3's `not started` training state is consistent with that protocol. At the
   aggregate-result level this is the cleanest controlled comparison in the
   round: replacing temporal ensembling with latest-chunk inference collapses
   LongPipelineDelivery from 19/20 to 0/20 and LiftBarrier from 17/20 to 6/20.
   Temporal ensembling is therefore part of the effective deployed policy, not
   an optional smoothing detail. The remote checkpoint SHA-256 must still be
   synchronized before calling the comparison artifact-complete.
4. **Keep sparse MoE over the tested compute-matched dense decoder.** With the
   same temporal ensemble and training protocol, B1 does not pass even the
   weak Gate20 because LongPipelineDelivery is 0/20. This is strong evidence
   against replacing the current decoder with this dense alternative. Because
   B0 and B1 were trained independently, a publication-level claim that MoE
   itself causes the full gap still requires the retained gate artifacts and
   at least one training-seed replication.
5. **Use LongPipelineDelivery as the discriminating regression task.** B1 and
   B3 retain 55% and 30% LiftBarrier success while both score 0% on the longer
   task. LiftBarrier alone would therefore hide failures in long-horizon
   temporal consistency and coordinated execution.
6. **Do not select candidates by terminal training loss.** B0 and B1 both show
   a monitor-rounded terminal loss of `0.002`, yet differ by 30 percentage
   points on LiftBarrier and 95 points on LongPipelineDelivery. Closed-loop
   validation, especially the longer task, remains the selection signal.

For Gate20, `pass` means only that both tasks completed with at least one
success. It is a screening gate, not formal acceptance. The formal protocol is
100 episodes per task with at least 90% success on each task; the 20-episode
B0 result must not be relabeled as formal acceptance. Its LiftBarrier point
estimate is also 85%, below the formal 90% threshold.

## Evidence boundary and reproduction

The pre-S0 `34cd6e7` checkpoint and gate JSON lived in a temporary worktree and
were removed when that worktree was cleaned. Its historical 17/20 and 15/20
numbers are therefore operator-reported preliminary evidence, not an immutable
accepted checkpoint. S0 Round 1 retrained the controlled coordinates and the
monitor confirms paired validation artifacts exist remotely, but those
artifacts have not yet been synchronized into this local worktree. Before
release acceptance:

1. synchronize and retain the S0 checkpoint SHA-256, gate JSON, and per-episode
   success records outside temporary storage;
2. verify the candidate/config/checkpoint identities against the frozen S0
   manifest;
3. run the formal `100`-episode-per-task gate;
4. replicate the independently trained B0/B1 decoder comparison with another
   training seed before assigning the complete gap to decoder family.

B2 completion is no longer a prerequisite for R1. Any later B2 reproduction
must use a new run ID and is separate from this closed S0 decision.

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
