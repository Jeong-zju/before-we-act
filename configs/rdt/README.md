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

The artifact is immutable by convention. If any policy-defining value changes,
create a new versioned artifact rather than editing this file in place. Secret
tokens and private credentials are intentionally not recorded.
