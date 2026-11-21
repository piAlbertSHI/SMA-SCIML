# Figure index / 图表索引

The README now presents the implemented architecture, two temperature animations, a space–time atlas and an offline field explorer. The previous 17 statistical figures are preserved in the [detailed gallery](RESULTS_GALLERY.md). All numerical displays use committed source arrays. The v13 architecture is implemented and execution-checked; its accuracy is not yet evaluated.

README 现展示模型架构、两段温度动画、时空场图谱和离线交互演示。原 17 组统计图保留于详细图集。数值演示来自已提交数组；v13 新架构已实现并验证运行，但精度尚未评价。

| Homepage asset / 首页素材 | Files / 文件 | Source and scope / 来源与范围 |
|---|---|---|
| Hierarchical architecture / 分层架构 | [PNG](../assets/operator_showcase_v2/architecture.png), [SVG](../assets/operator_showcase_v2/architecture.svg) | Implemented v13 code and measured parameter count / 已实现代码与实际参数数目 |
| Three-cycle thermal replay / 三循环热演化 | [GIF](../assets/operator_animation_v2/multicycle_fields.gif) | Trained v12 historical state model + DeepONet, fixed case/seed / 已训练历史参考模型，固定工况和种子 |
| Space–time atlas / 时空场图谱 | [PNG](../assets/operator_showcase_v2/multicycle_field_atlas.png), [SVG](../assets/operator_showcase_v2/multicycle_field_atlas.svg) | Same archived reference, prediction and signed error / 同一归档参考、预测与有符号误差 |
| Offline explorer / 离线交互 | [HTML](../assets/operator_showcase_v2/explorer.html) | Same fields rounded to four decimals; local open; mock-DOM checks, real-browser QA pending / 同源四位小数字段，本地打开，模拟逻辑已验证、真实浏览器验收待完成 |

[Showcase generator / 新展示生成器](../examples/operator_showcase.py) · [Figure manifest / 新素材清单](../assets/operator_showcase_v2/manifest.json)

| # | Question / 展示问题 | Figure / 图片 | Editable / 可编辑 | Numerical source / 数值来源 |
|---|---|---|---|---|
| 01 | Cycle-performance accuracy / 性能预测 | [PNG](../assets/scalar_benchmarks.png) | — | [metrics: scalar](../data/metrics.json) |
| 02 | Autonomous state dynamics / 自主状态推进 | [PNG](../assets/research_v1/state_dynamics.png) | [SVG](../assets/research_v1/state_dynamics.svg) | [metrics: boundary](../data/metrics.json) |
| 03 | Neural vs low-rank field models / 神经与低秩场模型 | [PNG](../assets/operator_comparison.png) | — | [metrics: operator](../data/metrics.json) |
| 04 | Field prediction and error patterns / 温度场与误差分布 | [PNG](../assets/field_replay.png) | — | [replay summary](../assets/demo_summary.json) |
| 05 | Directional-memory transfer / 方向记忆传递 | [PNG](../assets/memory_repair.png) | — | [metrics: memory](../data/metrics.json) |
| 06 | Stage input and hard-IC ablations / 阶段输入与硬初值消融 | [PNG](../assets/research_v1/operator_ablations.png) | [SVG](../assets/research_v1/operator_ablations.svg) | [metrics: operator](../data/metrics.json) |
| 07 | Manufactured diffusion inverse problem / 制造扩散反演 | [PNG](../assets/research_v1/pinn_inverse.png) | [SVG](../assets/research_v1/pinn_inverse.svg) | [PINN summary](../evidence/archive/wp5-pinn-r1/summary.json) |
| 08 | Local thermal identification / 局部热响应识别 | [PNG](../assets/research_v1/local_thermal.png) | [SVG](../assets/research_v1/local_thermal.svg) | [local summary](../evidence/archive/wp5-local-thermal-r2/summary.json) |
| 09 | Calibrated uncertainty intervals / 校准区间 | [PNG](../assets/research_v1/uncertainty_intervals.png) | [SVG](../assets/research_v1/uncertainty_intervals.svg) | [UQ summary](../evidence/archive/wp6-uq-r1/uq_report.json) |
| 10 | Geometry/material/protocol shifts / 几何、材料与协议变化 | [PNG](../assets/research_v1/ood_generalization.png) | [SVG](../assets/research_v1/ood_generalization.svg) | [OOD aggregate](../evidence/archive/ood-audit-r1/aggregate.json) |
| 11 | Physics-loss contributions / 物理损失贡献 | [PNG](../assets/research_v1/physics_ablation.png) | [SVG](../assets/research_v1/physics_ablation.svg) | [physics ablation](../evidence/archive/staged-physics-r1/summary.json) |
| 12 | Historical multicycle rollouts / 历史多循环推演 | [PNG](../assets/research_v1/historical_rollout.png) | [SVG](../assets/research_v1/historical_rollout.svg) | [historical fields](../evidence/archive/historical-fields-r1/summary.json) |
| 13 | New diversity acquisition / 新增多样性选样 | [PNG](../assets/research_v1/diversity_acquisition.png) | [SVG](../assets/research_v1/diversity_acquisition.svg) | [new experiment](../evidence/experiments/acquisition_diversity_v1/summary.json) |
| 14 | Design candidate trade-offs / 设计权衡 | [PNG](../assets/design_tradeoff.png) | — | [candidate verification](../evidence/archive/wp6-design-r1/candidate_verification.json) |
| 15 | Multi-step horizon comparison / 多步训练逐循环对照 | [PNG](../assets/multicycle_v1/multicycle_horizons.png) | [SVG](../assets/multicycle_v1/multicycle_horizons.svg) | [three-seed summary](../evidence/experiments/multicycle_multistep_v1/summary.json) |
| 16 | Training fit and all third-cycle blocks / 训练拟合与第三循环全状态 | [PNG](../assets/multicycle_v1/multicycle_diagnosis.png) | [SVG](../assets/multicycle_v1/multicycle_diagnosis.svg) | [derived metrics](../assets/multicycle_v1/derived_metrics.json) |
| 17 | Both third-cycle cases and all seeds / 两个工况与全部种子 | [PNG](../assets/multicycle_v1/multicycle_cases.png) | [SVG](../assets/multicycle_v1/multicycle_cases.svg) | [protocol and individual reports](MULTICYCLE_MULTISTEP.md) |
| Animation / 动画 | Stage-4 temperature evolution / 第四阶段温度演化 | [GIF](../assets/temperature_evolution.gif) | — | [replay summary](../assets/demo_summary.json) |

The original three-strategy acquisition plot remains available as [historical comparison](../assets/active_learning.png). The new six-target plot includes all four strategies, so results are not selected only for favorable targets.

原三策略主动选样图保留为[历史对照](../assets/active_learning.png)；新增六目标图包含四种策略，完整展示各目标的得失。

For slides, use the SVG exports when possible and retain the dataset, baseline and uncertainty definitions in the caption. Manufacturing problems, current63 simulations and the independent historical profile are separate evidence sources.

制作幻灯片时优先使用 SVG，保留数据范围、基线与不确定性含义。制造问题、当前 63 例和独立历史版本属于不同证据来源。

[Figure generator / 绘图代码](../examples/research_gallery.py) · [Input/output hashes and scopes / 输入输出哈希与范围](../assets/research_v1/gallery_manifest.json)

[Multi-cycle figure generator / 多循环绘图代码](../examples/multicycle_gallery.py) · [Multi-cycle input/output hashes / 多循环输入输出哈希](../assets/multicycle_v1/gallery_manifest.json)
