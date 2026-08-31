# π0.5 RoboFactory and MARS-Control OpenPI patch

This directory contains the application-specific OpenPI changes used by the
formal RoboFactory-MA and MARS-Control runs. It is intentionally kept separate
from the upstream OpenPI checkout so that the upstream source remains
attributable and the patch can be reviewed before release.

## Pinned source

- OpenPI base commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- RoboFactory-MA source commit: `5868242322414a91454e22f1dd9641f613ba1bcf`
- MARS-Control source commit: `2d34fb38c80cb06550a5dbf99abac2c89f4336ed`
- Formal checkpoint: `/workspace/bwa_pi05_runs/formal/pi05/checkpoints/pi05_robofactory_lora/h200_all900/119999`
- Formal contract: `configs/pi05_robofactory_lora_formal_v1.json`
- MARS-Control final checkpoint: `/workspace/runs/pi05_mars/checkpoints/pi05_mars_control_lora/all600_4gpu_dp_b128/29999`
- MARS-Control contract: `configs/pi05_mars_control_lora_v1.json`

The patch registers `pi05_robofactory_lora` and `pi05_mars_control_lora` in
OpenPI, adds their HDF5 dataset loaders, and implements the local-only
observation/action transforms. Both loaders use every demonstration and fail
closed on corpus cardinality or schema drift. The MARS loader samples four
equal-probability virtual task lanes so the longer four-arm task does not
dominate merely because it has more indexed arm-local timesteps.

## Apply

From an OpenPI checkout at the pinned commit, run:

```bash
./apply.sh /path/to/openpi
```

### Blackwell multi-GPU runtime

On RTX PRO 6000/Blackwell hosts, the upstream OpenPI lock (`jax==0.5.3`
with CUDA 12.6 wheels) can deadlock while initializing a multi-device
collective.  Install the CUDA 12.9 userspace wheels before launching a 4-GPU
run:

```bash
uv pip install --python /path/to/openpi-venv \
  -r requirements-blackwell-cu129.txt
```

The upgrade is required as a set: in particular `nvidia-nccl-cu12>=2.31` must
be selected alongside the JAX CUDA plugin.  Verify with
`jax.pmap(lambda x: jax.lax.psum(x, "i"), axis_name="i")` on all four devices
before starting training.

The script verifies the base commit, verifies this bundle's checksums, applies
`openpi_tracked.patch`, and copies the five new OpenPI modules into their
package locations. It does not download data, alter checkpoints, or modify
unrelated OpenPI files.

## Integrity

The vendored files and tracked patch are hash-bound in `SHA256SUMS`.
