# Figure index / 图表索引

Figures are generated from recorded numerical arrays. The operator comparison includes the architecture, autonomous field animation and both three-cycle test cases; the statistical gallery covers the reference studies.

图表由记录的数值数组生成。算子对照包含架构、自主场动画及两个三循环工况，统计图集覆盖各参考方法研究。

| Homepage asset / 首页素材 | Files / 文件 | Source and scope / 来源与范围 |
|---|---|---|
| Hierarchical architecture / 分层架构 | [PNG](../assets/showcase/architecture.png), [SVG](../assets/showcase/architecture.svg) | Implemented HRO code, actual parameters and formal training status / 代码、实际参数及正式训练状态 |
| Three-cycle comparison / 三循环对照 | [GIF](../assets/showcase/multicycle_comparison.gif) | Reference, Residual MLP + DeepONet, trained HRO and absolute errors; fixed recA case, first preset seeds / 固定recA工况及预定首种子 |
| Configuration A / 配置 A | [PNG](../assets/showcase/field_atlas_1.png), [SVG](../assets/showcase/field_atlas_1.svg) | All six exchange segments with shared scales / 六段换热，两类模型共用色标 |
| Configuration B / 配置 B | [PNG](../assets/showcase/field_atlas_2.png), [SVG](../assets/showcase/field_atlas_2.svg) | Second three-cycle test case, same seed rule / 第二三循环测试工况，同一种子规则 |
| Offline explorer / 离线交互 | [HTML](../assets/showcase/explorer.html) | Trained HRO fields rounded to4decimals; mock-DOM checked, real-browser QA pending / HRO四位小数场，模拟逻辑已检查、真实浏览器待验收 |
| Error-scaled thermal response / 误差收缩热响应 | [PNG](../assets/sma_risk_limited_readout/architecture.png), [SVG](../assets/sma_risk_limited_readout/architecture.svg) | Frozen HRO and added mean readout / 冻结主干与新增均温读出 |
| Stage-2 SMA fields / 第二阶段 SMA 场 | [PNG](../assets/sma_risk_limited_readout/four_segments.png), [SVG](../assets/sma_risk_limited_readout/four_segments.svg) | Four target segments, three fixed seeds, native grid / 四个目标换热段，三固定种子，原生网格 |

[Formal gallery generator / 正式结果展示生成器](../examples/field_showcase.py) · [Source manifest / 素材清单](../assets/showcase/manifest.json)

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
| 13 | Diversity-aware acquisition / 多样性选样 | [PNG](../assets/research_v1/diversity_acquisition.png) | [SVG](../assets/research_v1/diversity_acquisition.svg) | [experiment](../evidence/experiments/acquisition_diversity_v1/summary.json) |
| 14 | Design candidate trade-offs / 设计权衡 | [PNG](../assets/design_tradeoff.png) | — | [candidate verification](../evidence/archive/wp6-design-r1/candidate_verification.json) |
| 15 | Multi-step horizon comparison / 多步训练逐循环对照 | [PNG](../assets/multicycle_v1/multicycle_horizons.png) | [SVG](../assets/multicycle_v1/multicycle_horizons.svg) | [three-seed summary](../evidence/experiments/multicycle_multistep_v1/summary.json) |
| 16 | Training fit and all third-cycle blocks / 训练拟合与第三循环全状态 | [PNG](../assets/multicycle_v1/multicycle_diagnosis.png) | [SVG](../assets/multicycle_v1/multicycle_diagnosis.svg) | [derived metrics](../assets/multicycle_v1/derived_metrics.json) |
| 17 | Both third-cycle cases and all seeds / 两个工况与全部种子 | [PNG](../assets/multicycle_v1/multicycle_cases.png) | [SVG](../assets/multicycle_v1/multicycle_cases.svg) | [protocol and individual reports](MULTICYCLE_MULTISTEP.md) |
| 18 | Paired solver-version temperature fields / 同工况跨版本温度场 | [PNG](../assets/loading_field_study/cooling_version_difference.png) | — | [54-pair audit / 54 对核验](../evidence/local_studies/solver_version_pairs_20261121/summary.json) |
| 19 | Loading fields and operator error maps / 加载场与算子误差分布 | [PNG](../assets/loading_field_study/loading_benchmark_fields.png) | — | [Fixed loading benchmark / 固定加载基准](LOADING_FIELD_STUDY.md) |
| 20 | Phase violations and transformation-front error / 相分数越界与相变前沿误差 | [PNG](../evidence/local_studies/loading_phase_bounds_20261121/phase_front_diagnostics.png) | — | [Phase diagnostics / 相分数诊断](LOADING_PHASE_CONSTRAINTS.md) |
| Animation / 动画 | Stage-4 temperature evolution / 第四阶段温度演化 | [GIF](../assets/temperature_evolution.gif) | — | [replay summary](../assets/demo_summary.json) |

The original three-strategy acquisition plot remains available as [historical comparison](../assets/active_learning.png). The six-target plot includes all four strategies, so results are not selected only for favorable targets.

原三策略主动选样图保留为[历史对照](../assets/active_learning.png)；六目标图包含四种策略，完整展示各目标的得失。

For slides, use the SVG exports when possible and retain the dataset, baseline and uncertainty definitions in the caption. Manufacturing problems, current63 simulations and the independent historical profile are separate evidence sources.

制作幻灯片时优先使用 SVG，保留数据范围、基线与不确定性含义。制造问题、当前 63 例和独立历史版本属于不同证据来源。

[Figure generator / 绘图代码](../examples/research_gallery.py) · [Input/output hashes and scopes / 输入输出哈希与范围](../assets/research_v1/gallery_manifest.json)

[Multi-cycle figure generator / 多循环绘图代码](../examples/multicycle_gallery.py) · [Multi-cycle input/output hashes / 多循环输入输出哈希](../assets/multicycle_v1/gallery_manifest.json)
