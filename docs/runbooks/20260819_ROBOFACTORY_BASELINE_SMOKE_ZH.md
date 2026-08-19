# RoboFactory 七基线烟测回执

远程节点：`root@69.176.92.104:10328`，GPU 为 4×RTX 5090。数据根目录为
`/workspace/datasets/robofactory_multitask`，六个任务各有 150 个 episode，
training manifest 和 normalization 均通过协议检查。

## 已完成

`act`、`dp`、`latent_tom`、`gaudp`、`maniflow`、`rdt_1b`、`openvla_oft` 的
统一 adapter smoke 均在 `cuda:0` 完成 8 个 optimizer step，并保存
`/workspace/bwa-baselines-runs/smoke/<baseline>/smoke.pt` 与 `status.json`。

这些结果只证明统一数据/训练/状态链路可用，不是七个上游模型的闭环成功率。
Validation20 聚合器不会从缺失结果推断 0% 或 100%。

## 阻塞项

本地 ACT 原生入口已经实际启动过一次，但在读取数据前因 schema 不兼容停止：
ACT 需要 `traj_*/obs/actions` RoboFactory HDF5，而当前统一 manifest 是
`data/observation/action` WAM HDF5。DP 还缺少对应的 zarr 数据产物；其余五个
方法尚未在仓库中发现可固定的上游训练器、commit、权重及图像/语言预处理合同。
因此当前不能声称“七个 baseline 已完成联合训练和闭环验证”。下一步必须先做
schema adapter/数据导出并固定每个上游版本，再启动真实训练和每任务 20 局验证。

## 监控

本地启动 `python3 web_service/server.py`，然后使用
`ssh -p 10328 -L 8080:127.0.0.1:8088 root@69.176.92.104`，浏览
`http://localhost:8080`。面板只读远程状态和诊断信息。
