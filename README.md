---
license: mit
datasets:
- B111ue/RoboFactory-5Task-RGBD-Decentralized
tags:
- robotics
- imitation-learning
- multi-robot
- decentralized
- rgb-d
---

# Stereo-CoRE

Official reproducibility package for **Stereo-CoRE**, a strictly decentralized shared policy for
multi-task, multi-robot manipulation. Each robot receives only its own `panda_hand` wrist RGB-D
observation and qpos. Deployment uses no task/agent ID, language, communication, global camera,
peer observation, right camera, or FastFS.

## Observation and policy contract

- wrist RGB-D: 640x480
- native metric depth decoded from millimetres
- frozen DINOv3-B/16 RGB and DeFM-S/14 depth encoders
- aligned 30x40 RGB/depth patch grids with learned 2-D relative-bias RGB-to-depth attention
- ACT: 4-layer latent encoder, 7-layer decoder, chunk length 100
- shared policy across LiftBarrier (2), CameraAlignment (3), ThreeRobotsStackCube (3),
  LongPipelineDelivery (4), and TakePhoto (4)

## Main method

Stereo-CoRE couples the local action-query router to counterfactual expert capability. At a
scheduled update, every expert predicts the same ground-truth action chunk; its true action error
defines a soft capability target, and `KL(q_capability || p_router)` trains the router to select
experts that are actually competent for the current local action role. The released main run uses
`capability_weight=0.05` and disables relation, specialization, and anchor auxiliaries.

## Repositories

- Code: https://github.com/YananZHOU5555/Stereo-CoRE
- Dataset: https://huggingface.co/datasets/B111ue/RoboFactory-5Task-RGBD-Decentralized
- Models: https://huggingface.co/B111ue/Stereo-CoRE
- Upstream RoboFactory commit: `5868242322414a91454e22f1dd9641f613ba1bcf`

See `docs/REPRODUCE.md`, `docs/METHOD.md`, and `docs/RESULTS.md`.

`MODEL_REGISTRY.json` binds every All-5 paper row to one public checkpoint, its SHA-256, embedded
normalization statistics, exact config, frozen-seed results and evaluation protocol.
