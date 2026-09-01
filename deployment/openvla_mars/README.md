# OpenVLA-OFT on MARS-Control

This directory is the reproducible, decentralized OpenVLA-OFT baseline for the
four MARS-Control tasks. The formal run uses all 600 successful demonstrations
(1,650 arm-local streams) and four GPUs; it does not create a train/test split.
The machine-readable source of truth is
`configs/openvla_oft_mars_control_lora_r32_formal_v1.json`.

## Reproduce from a clean checkout

The source patch was captured from OpenVLA-OFT commit
`e4287e94541f459edc4feabc4e181f537cd569a8`, and the benchmark checkout was
MARS-EAI/RoboFactory commit `2d34fb38c80cb06550a5dbf99abac2c89f4336ed`:

```bash
git clone https://github.com/moojink/openvla-oft.git /workspace/repos/openvla-oft
git -C /workspace/repos/openvla-oft checkout e4287e94541f459edc4feabc4e181f537cd569a8
git -C /workspace/repos/openvla-oft apply \
  /workspace/repos/before-we-act/patches/openvla_oft_mars_control_formal.patch
```

Place a Hugging Face token (mode 600) at `/workspace/.secrets/hf_token`, then
run the supervisor wrapper:

```bash
chmod 600 /workspace/.secrets/hf_token
/workspace/repos/before-we-act/deployment/openvla_mars/mars-openvla-supervisor.sh
```

The supervisor is crash-resumable and enforces these stage gates:

1. assets/model and four dataset receipts;
2. host, GPU, dataset and action-contract audits;
3. one-step/four-GPU training and one episode per task smoke validation;
4. formal 150,000-step LoRA-r32 training;
5. 20 episodes per task closed-loop Validation20 and a final report.

Training owns GPUs 0--3. Validation launches one policy worker per GPU and sends
each worker exactly one arm-local RGB image and qpos vector. No peer observation,
task ID, or global state is passed to the policy. The policy predicts eight
values: seven joint residuals referenced to qpos at chunk start and an absolute
gripper command. Residual chunks are decoded to absolute Panda commands before
temporal ensembling, then clipped to the environment bounds. Terminal chunks
repeat the final valid absolute action. Images are uint8 HWC 320x240 and use the
OpenVLA processor's backbone-specific normalization.

The default run root is `/workspace/bwa_mars_openvla_runs`; set
`MARS_OPENVLA_RUN_ROOT` to relocate it. The historical server artifact was under
`.../repaired`, but that path is deliberately not required for a fresh run.
Do not commit datasets, checkpoints, logs, or tokens. Hashes for the completed
server artifacts are recorded in the frozen JSON and final report.
