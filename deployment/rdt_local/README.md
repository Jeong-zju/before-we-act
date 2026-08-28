# RDT-1B six-task deployment

This deployment adapts the official RDT-1B trainer to the six pinned
RoboFactory datasets.  Every `(task, episode, agent)` becomes an arm-local
training stream.  A policy call receives only that arm's RGB and qpos and
returns only its 8-D action.  All arms use the same trained weights.

The formal pipeline is owned by system supervisor and advances through:

1. pinned, resumable download and integrity verification of all 900 episodes;
2. decentralized data-contract audit and RDT statistics/language preparation;
3. four-GPU full-parameter training smoke and checkpoint reload;
4. six-task closed-loop smoke;
5. four-GPU 300k-step formal fine-tuning;
6. six-task Validation20 (120 closed-loop episodes).

No train/test split is applied.  Dataset manifest split labels are deliberately
ignored by the RDT adapter.
