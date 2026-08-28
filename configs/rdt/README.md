# RDT-1B configuration artifacts

`robofactory_rdt1b_full_data_v1.json` is the frozen configuration for the
reported RDT-1B/RoboFactory run. It records the resolved launcher arguments,
official RDT base configuration, decentralized HDF5 data contract, model and
diffusion settings, optimizer and scheduler, DeepSpeed runtime, checkpointing,
software revisions, and Validation20 contract.

The artifact is immutable by convention. If any policy-defining value changes,
create a new versioned artifact rather than editing this file in place. Secret
tokens and private credentials are intentionally not recorded.
