# DuoBench CARE DINO B0-H reference

This adapter ports the current CARE/B0-H RoboFactory reference backbone without
changing its method family: frozen DINOv3 ViT-B/16 plus the project-owned
`TemporalHistoryPolicy(hidden_residual)` Transformer.  This is distinct from
the historical W10 `NoWristPAIRRoute(ACT)` checkpoint: that ACT-derived model
is retained only as a separately labelled legacy baseline and is never loaded
by this formal path.  The two arms share weights, but each
sample/runtime row receives only shared head RGB, its own wrist RGB, local
qpos8, and its own executed-action history.  It predicts H=100 absolute local
action8 chunks; the gripper is binary.

Formal sampling uses the matched RoboFactory effective batch of 48: four rows
per task plus four rotating extras.  The rotation is exactly equal over eleven
updates, and four-GPU DDP receives twelve rows per rank.

The prepared numeric/image corpus must use
`absolute_joint7_binary_gripper1`.  Build it with the existing lossless Duo
action-data converter (`deployment.duo_act.prepare`; the converter name does
not imply that the ACT policy is used) rather than reusing the residual-action
`duo_care` folder:

```bash
python -m deployment.duo_act.prepare \
  --dataset /workspace/datasets/duobench \
  --output /workspace/runs/duobench-care-dino/dino_prepared \
  --image-size 224

torchrun --standalone --nproc_per_node=4 \
  -m deployment.duo_dino_reference.cache_dino \
  --prepared-data /workspace/runs/duobench-care-dino/dino_prepared \
  --dino-model /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
  --output /workspace/runs/duobench-care-dino/dino_cache

python -m deployment.duo_dino_reference.smoke \
  --prepared-data /workspace/runs/duobench-care-dino/dino_prepared \
  --visual-cache /workspace/runs/duobench-care-dino/dino_cache \
  --output /workspace/runs/duobench-care-dino/data_smoke.json

torchrun --standalone --nproc_per_node=4 \
  -m deployment.duo_dino_reference.train_b0h \
  --prepared-data /workspace/runs/duobench-care-dino/dino_prepared \
  --visual-cache /workspace/runs/duobench-care-dino/dino_cache \
  --dino-model /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
  --output /workspace/runs/duobench-care-dino/b0h_smoke \
  --stage smoke --updates 2

python -m deployment.duo_dino_reference.smoke \
  --prepared-data /workspace/runs/duobench-care-dino/dino_prepared \
  --visual-cache /workspace/runs/duobench-care-dino/dino_cache \
  --checkpoint /workspace/runs/duobench-care-dino/b0h_smoke/final.pt \
  --dino-model /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
  --env-task ball_maze \
  --duobench-root /workspace/repos/duobench/src \
  --output /workspace/runs/duobench-care-dino/closed_loop_smoke.json

torchrun --standalone --nproc_per_node=4 \
  -m deployment.duo_dino_reference.train_b0h \
  --prepared-data /workspace/runs/duobench-care-dino/dino_prepared \
  --visual-cache /workspace/runs/duobench-care-dino/dino_cache \
  --dino-model /workspace/artifacts/dinov3-vitb16-pretrain-lvd1689m \
  --output /workspace/runs/duobench-care-dino/b0h_formal \
  --stage formal --updates 120000
```

Validation is resumable per task.  Run twenty episodes for each of the eleven
task IDs (normally one process/GPU at a time under the supervisor):

```bash
python -m deployment.duo_dino_reference.evaluate \
  --checkpoint /workspace/runs/duobench-care-dino/b0h_formal/final.pt \
  --prepared-data /workspace/runs/duobench-care-dino/dino_prepared \
  --task ball_maze --episodes 20 \
  --duobench-root /workspace/repos/duobench/src \
  --output /workspace/runs/duobench-care-dino/validation20/ball_maze.json
```

Formal gates reject ACT checkpoints, residual-action manifests, smoke DINO
caches, incomplete 550-episode corpora, and non-120k formal budgets.
