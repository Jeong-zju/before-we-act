# RDT-1B configuration artifacts

`robofactory_rdt1b_full_data_v1.json` is the frozen configuration for the
reported RDT-1B/RoboFactory run. It records the resolved launcher arguments,
official RDT base configuration, decentralized HDF5 data contract, model and
diffusion settings, optimizer and scheduler, DeepSpeed runtime, checkpointing,
software revisions, and Validation20 contract.

`mars_control_rdt1b_full_data_v1.json` is the corresponding frozen contract for
the completed MARS-Control full-parameter adaptation. It records all 48
arguments accepted by the pinned RDT trainer, including explicit `None`/disabled
values, plus the effective base-model, HDF5 sampling, local-observation,
normalization, DeepSpeed, checkpoint-audit, and four-task Validation20 fields.
The companion `deployment/rdt_mars/audit_frozen_config.py` checks the schema,
argument coverage, parameter count, training budget, and result identity.

`duobench_rdt1b_full_data_v1.json` freezes the completed DuoBench RDT-1B
full-parameter run. It records every one of the pinned upstream trainer's 48
arguments (including explicit `null` and disabled values), the upstream base
model and diffusion settings, all-data decentralized adapter contract,
normalization/statistics provenance, optimizer and ZeRO-2 runtime, external
checkpoint retention mechanism, software revisions, artifact hashes, and the
11-task Validation20 contract. Run
`python deployment/rdt_duo/audit_frozen_config.py` to verify its schema,
argument coverage, training budget, parameter count, corpus identity,
checkpoint semantics, and reported result.

The upstream checkout also has a replayable overlay at
`deployment/rdt_duo/upstream_rdt1b_duobench.patch.gz`. Apply it with
`deployment/rdt_duo/apply_upstream_patch.sh /path/to/rdt-1b` after checking out
RDT commit `cd79363a1387e8f81c7724d070ef7e45fd23150f`; this restores the exact
DuoBench dataset adapter, dataset statistics, control-frequency map, dataset
selection, and sampling weights used by the run. The normal supervisor's
`configure` stage remains self-contained and writes the same adapter/config
files directly.

The artifact is immutable by convention. If any policy-defining value changes,
create a new versioned artifact rather than editing the existing version in
place. The completed-run files are retrospective records, not mutable launcher
defaults. Secret
tokens and private credentials are intentionally not recorded.
