# π0.5 RoboFactory OpenPI patch

This directory contains the application-specific OpenPI changes used by the
formal RoboFactory-MA run. It is intentionally kept separate from the upstream
OpenPI checkout so that the upstream source remains attributable and the patch
can be reviewed before release.

## Pinned source

- OpenPI base commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- RoboFactory source commit: `5868242322414a91454e22f1dd9641f613ba1bcf`
- Formal checkpoint: `/workspace/bwa_pi05_runs/formal/pi05/checkpoints/pi05_robofactory_lora/h200_all900/119999`
- Formal contract: `configs/pi05_robofactory_lora_formal_v1.json`

The patch registers `pi05_robofactory_lora` in OpenPI, adds the HDF5 dataset
loader, and implements the local-only observation/action transforms. The
loader intentionally ignores manifest split labels and fails closed unless all
six tasks and all 900 episodes are present.

## Apply

From an OpenPI checkout at the pinned commit, run:

```bash
./apply.sh /path/to/openpi
```

The script verifies the base commit, applies `openpi_tracked.patch`, and copies
the three new OpenPI modules into their package locations. It does not download
data, alter checkpoints, or modify unrelated OpenPI files.

## Integrity

The vendored files and tracked patch are hash-bound in `SHA256SUMS`.
