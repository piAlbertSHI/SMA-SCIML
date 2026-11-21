# Figure index / 图表索引

The README presents the trained architecture, a v12/v13 comparison animation, both three-cycle field atlases and a v13 offline explorer. The previous17 statistical figures and reference animations are preserved. All numerical displays use committed source arrays; see the [formal benchmark](V13_FORMAL_BENCHMARK.md) for accuracy and tradeoffs.

README 展示已训练架构、v12/v13对照动画、两个三循环图谱及v13离线演示。原17组统计图及参考动画保留；全部数值来自已提交数组，精度与取舍见正式训练报告。

| Homepage asset / 首页素材 | Files / 文件 | Source and scope / 来源与范围 |
|---|---|---|
| Hierarchical architecture / 分层架构 | [PNG](../assets/v13_formal_v1/architecture.png), [SVG](../assets/v13_formal_v1/architecture.svg) | Implemented v13 code, actual parameters and formal training status / 代码、实际参数及正式训练状态 |
| Three-cycle comparison / 三循环对照 | [GIF](../assets/v13_formal_v1/v13_comparison.gif) | Reference, old v12, trained v13 and absolute errors; fixed recA case, first preset seeds / 固定recA工况及预定首种子 |
| recA atlas / recA图谱 | [PNG](../assets/v13_formal_v1/field_atlas_1.png), [SVG](../assets/v13_formal_v1/field_atlas_1.svg) | All six exchange segments with shared scales / 六段换热，新旧模型共用色标 |
| recAS atlas / recAS图谱 | [PNG](../assets/v13_formal_v1/field_atlas_2.png), [SVG](../assets/v13_formal_v1/field_atlas_2.svg) | Second three-cycle test case, same seed rule / 第二三循环测试工况，同一种子规则 |
| Offline explorer / 离线交互 | [HTML](../assets/v13_formal_v1/explorer.html) | Trained v13 fields rounded to4decimals; mock-DOM checked, real-browser QA pending / v13四位小数场，模拟逻辑已检查、真实浏览器待验收 |

[Formal gallery generator / 正式结果展示生成器](../examples/formal_gallery.py) · [Source manifest / 素材清单](../assets/v13_formal_v1/manifest.json)

Preserved reference-only displays / 保留原参考模型展示：[animation / 动画](../assets/operator_animation_v2/multicycle_fields.gif), [atlas / 图谱](../assets/operator_showcase_v2/multicycle_field_atlas.png), [explorer / 交互](../assets/operator_showcase_v2/explorer.html).

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
