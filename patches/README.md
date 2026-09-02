# Reproducibility patches

## RDT-1B / MARS-Control

`rdt1b_mars_control_formal.patch` is the exact upstream RDT-1B checkout patch
used by the formal MARS-Control run. It applies to RDT commit
`cd79363a1387e8f81c7724d070ef7e45fd23150f` and captures the benchmark-specific
changes that are otherwise easy to miss: the 20 Hz dataset registration,
full-corpus normalization statistics, MARS dataset/weight registration, the
generated empty language embedding, and the strict local-arm HDF5 adapter
(including aligned state/action sampling and terminal-chunk padding). It also
contains the binary embedding artifact used by the run.

To reproduce the patched upstream checkout:

```bash
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git /workspace/repos/rdt-1b
git -C /workspace/repos/rdt-1b checkout cd79363a1387e8f81c7724d070ef7e45fd23150f
git -C /workspace/repos/rdt-1b apply \
  /workspace/repos/before-we-act/patches/rdt1b_mars_control_formal.patch
```

Patch SHA-256: `feadc05c9e081d82dc6421635d7a53a9e4d21055788a06b23a6bfafd22309b7b`.
The higher-level downloader, configuration generator, four-GPU launcher,
checkpoint audit, supervisor, and strict Validation20 evaluator are in
`deployment/rdt_mars/`; the complete resolved parameter contract is in
`configs/rdt/mars_control_rdt1b_full_data_v1.json`. The launcher regenerates
the language embeddings and dataset statistics when needed, so the patch is a
provenance artifact rather than a secret-bearing snapshot.

## OpenVLA-OFT / MARS-Control

`openvla_oft_mars_control_formal.patch` is the exact five-file source patch
used for the MARS-Control OpenVLA-OFT run. It applies to OpenVLA-OFT commit
`e4287e94541f459edc4feabc4e181f537cd569a8` and includes the MARS HDF5 adapter,
task-balanced all-data sampler, residual action contract, dataset statistics,
DDP/checkpoint fixes, and rank-0 LoRA merge:

```bash
git -C /workspace/repos/openvla-oft apply \
  /workspace/repos/before-we-act/patches/openvla_oft_mars_control_formal.patch
```

The corresponding four-GPU supervisor, smoke/formal launchers and frozen
parameter record are in `deployment/openvla_mars/` and
`configs/openvla_oft_mars_control_lora_r32_formal_v1.json`.

## OpenVLA-OFT / RoboFactory-MA

`openvla_oft_robofactory_formal.patch` is the exact source patch applied to
the OpenVLA-OFT checkout used for the formal run. It applies to
OpenVLA-OFT commit `e4287e94541f459edc4feabc4e181f537cd569a8`:

```bash
git clone https://github.com/openvla/openvla-oft.git
cd openvla-oft
git checkout e4287e94541f459edc4feabc4e181f537cd569a8
git apply /path/to/before-we-act/patches/openvla_oft_robofactory_formal.patch
```

The patch contains the RoboFactory constants and normalization policy, the
strict all-900-episode local-arm HDF5 adapter and deterministic distributed
sampler, plus DDP gradient-accumulation, checkpoint resume, optimizer-state
resume, rank-0 LoRA merge, and offline checkpoint compatibility fixes.

The launcher and supervisor contract are in
`deployment/vla_baselines/pipeline.openvla.json`,
`run_openvla_oft.sh`, and `run_openvla_formal.sh`. The complete frozen
parameter set is in
`configs/openvla_oft_robofactory_lora_formal_v1.json`.

The formal run used RoboFactory commit
`5868242322414a91454e22f1dd9641f613ba1bcf`, four A100 GPUs, all six task
manifests (150 episodes each), and 150,000 optimizer steps. The upstream
training entry point did not configure a random seed; the reproducibility
record intentionally leaves that field unset.
