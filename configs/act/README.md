# ACT configuration artifacts

`mars_control_full_data_v1.json` is the frozen configuration for the reported
ACT/MARS-Control run. It is benchmark-specific and must not be silently
substituted for the six-task RoboFactory ACT protocol in
`../robofactory_act_formal_v1.json`.

The artifact records exact values that are often implicit framework defaults,
including AdamW betas/epsilon, sampler replacement, loader flags, transformer
dropout, precision, seeds, checkpoint cadence, and the decentralized input
contract. All 600 demonstrations are used without a train/validation split.
Every arm loads the same weights, consumes only its own RGB/qpos, and emits only
its own action. If a future run changes a value, create a new versioned artifact
instead of editing this one in place.
