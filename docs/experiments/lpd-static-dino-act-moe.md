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
