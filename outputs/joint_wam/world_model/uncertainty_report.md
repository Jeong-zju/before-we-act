# World-model ensemble uncertainty report

结论：**uncertainty acceptance 通过**。

| 判据 | 结论 | 关键数据 |
|---|---|---|
| h20_vs_baseline_mlp | 通过 | ensemble_nrmse=0.18434, baseline_mlp_nrmse=0.32066, relative_improvement=0.42513, minimum=0.2 |
| autoregressive_vs_teacher_forcing | 通过 | member0_nrmse=0.2203, teacher_forcing_nrmse=0.25155, relative_improvement=0.12423, minimum=0.05 |
| uncertainty_error_correlation | 通过 | spearman=0.89098, minimum=0.3 |
| ood_identification | 通过 | auroc=0.98853, auroc_minimum=0.7, epistemic_ratio=29.253, ratio_minimum=1.25 |
| event_aligned_no_average_braking | 通过 | horizon=5, samples=19192, available=True, member_dominant_agent_accuracy=0.97584, member_ambiguous_braking_rate=0.022384, ensemble_mean_dominant_agent_accuracy=0.97864, ensemble_mean_ambiguous_braking_rate=0.018549, accuracy_minimum=0.6, ambiguous_rate_maximum=0.25 |
| independent_members | 通过 | maximum_parameter_rms_distance=0.11254, minimum_parameter_rms_distance=0.11049, pair_count=10 |
| full_test_split | 通过 |  |

阈值仅由版本化配置定义；variance scale 仅使用 validation 拟合，test 不参与校准。