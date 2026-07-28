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
