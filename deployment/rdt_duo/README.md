# DuoBench RDT-1B deployment

This adapter preserves the upstream RDT-1B model and trainer.  It presents
each `(task, episode, arm)` as an independent local stream, with only the
shared head RGB, that arm's wrist RGB, and its own 7-joint + binary-gripper
state.  Both arms use the same weights; no peer observation, arm id, or global
action enters training or validation.

The prepared dataset uses all 550 successful demonstrations (11 tasks × 50),
with the pinned causal contract `observation[i] -> action[i+1]`, controller
ctrlrange canonicalization, and official 224-square RGB streams.  RDT retains
its 128-D unified vector, 64-action horizon, 2-frame/3-camera condition layout,
SigLIP/T5 encoders, DDPM schedule, and full `rdt.parameters()` optimizer.

The supervisor runs dependency/repository/data stages, audits all contracts,
then four-GPU 2-step smoke training + checkpoint reload + 11-task smoke
validation.  Only after all gates pass does it launch the resumable 215,000
step full-parameter run (`global batch=16`, `bf16`, ZeRO-2, 4 GPUs), followed
by task-specific Validation20 (20 episodes × 11 tasks).  Install the supplied
`.conf` into `/etc/supervisor/conf.d` only when you want the platform
supervisor to own the run.

## Replaying the formal upstream modifications

The deployment does not require a private fork of RDT. Check out upstream
commit `cd79363a1387e8f81c7724d070ef7e45fd23150f`, then either run the normal
supervisor or apply the exact completed-run overlay directly:

```bash
deployment/rdt_duo/apply_upstream_patch.sh /path/to/RoboticsDiffusionTransformer
```

`upstream_rdt1b_duobench.patch.gz` contains the arm-local DuoBench dataset
adapter, the computed 128-D dataset statistics, the 30-Hz control-frequency
entry, and the single-dataset sampling configuration. The wrapper rejects a
different upstream commit and adds explicit package markers that prevent a
sibling repository's `models` namespace from shadowing RDT during distributed
launch. The `configure` stage performs the same runtime wiring and is audited
before smoke training.
