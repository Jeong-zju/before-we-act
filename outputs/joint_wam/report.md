# Joint WAM validation

结论：**Joint WAM 验证通过**，`policy_acceptable=true`。

| Suite | Policy | Episodes | Success | Return | Response delay | Coordination error | P50/P95/P99 latency | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | action_prior | 500 | 100.0% | 66.589 | 0.293 | 0.02372 | 1.371 / 1.624 / 4.233 ms | 0.0% |
| standard | joint_wam_direct | 500 | 100.0% | 68.405 | 0.203 | 0.01508 | 9.569 / 14.476 / 17.419 ms | 0.0% |
| standard | joint_wam_with_fallback | 500 | 100.0% | 68.405 | 0.203 | 0.01508 | 9.603 / 14.461 / 17.437 ms | 0.0% |
| standard | scripted_oracle | 500 | 100.0% | 70.292 | 0.150 | 0.00592 | n/a / n/a / n/a ms | 0.0% |
| standard | stationary | 500 | 0.0% | 3.307 | n/a | 0.00000 | n/a / n/a / n/a ms | 0.0% |
| challenge | action_prior | 500 | 100.0% | 57.925 | 0.278 | 0.03509 | 1.321 / 1.586 / 4.249 ms | 0.0% |
| challenge | joint_wam_direct | 500 | 100.0% | 58.111 | 0.263 | 0.03356 | 9.598 / 14.366 / 17.307 ms | 0.0% |
| challenge | joint_wam_with_fallback | 500 | 100.0% | 58.111 | 0.263 | 0.03356 | 9.638 / 14.527 / 17.437 ms | 0.0% |
| challenge | scripted_oracle | 500 | 100.0% | 60.906 | 0.150 | 0.01028 | n/a / n/a / n/a ms | 0.0% |
| challenge | stationary | 500 | 0.0% | -5.904 | n/a | 0.00000 | n/a / n/a / n/a ms | 0.0% |

## Acceptance

- formal protocol: `true`
- policy acceptable: `true`
- held-out seed overlap: `0`
- source checkpoints immutable: `true`
- required videos complete: `true`
- joint benefit: `not evaluated`

- challenge: direct `100.0%`, prior `100.0%`, regression `0.0%`
- standard: direct `100.0%`, prior `100.0%`, regression `0.0%`

本结果证明当前 proprioceptive Joint WAM 在本任务上可接受；未评测未饱和任务上的同预算配对消融，因此不声称 joint world modeling 带来控制增益。
