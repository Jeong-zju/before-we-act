# No-wrist deployment

This deployment targets the user's original RoboFactory cameras. It does not
install `wrist_camera_patch` and does not consume depth.

## Policy contract

- Current fixed global RGB at 640x480.
- Current matching fixed `head_camera_agentN` RGB at 640x480.
- Matching robot qpos only.
- No wrist RGB/depth, task or robot ID, language, peer state, peer action, or
  future observation.

The adaptation retains the released frozen DINOv3-B/16 visual encoder,
30x40 relative-bias cross-view fusion, ACT 4-layer posterior/7-layer decoder,
four role adapters, and capability-based PAIR routing. The depth branch is
replaced with the fixed global RGB context because the target environment has
no wrist depth.

## Formal run

- Dataset root: `/workspace/datasets/robofactory_multitask`
- Training split: 120 demonstrations per task, 600 total.
- Batch: exactly 40, eight samples from each of five tasks per update.
- Budget: 120,000 updates / 4.8 million local action chunks.
- Run root: `/workspace/runs/no_wrist_stereo_core_120k`
- Frozen test protocol: the released `protocol/frozen100` manifests, 100 seeds
  per task, SR@1, at most 1500 steps.

`deployment/evaluate_after_training.sh` watches the formal checkpoint,
recovers the trainer from `checkpoint_latest.pt` if necessary, and then runs
all five frozen 100-seed evaluations with per-seed log recovery.
