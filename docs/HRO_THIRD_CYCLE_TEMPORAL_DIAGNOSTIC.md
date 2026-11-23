# HRO third-cycle temporal error / HRO 第三循环时间误差

The post-selection audit resolves c3s4 field error along normalized stage time. It compares the protected-state checkpoint with the formal reference on the existing two-case third-cycle cohort. The comparison is diagnostic and was performed after selection.

选择后审计沿归一化阶段时间拆解 c3s4 场误差。它在已有两个第三循环工况上比较状态保护检查点和形式参考，属于选择后的诊断分析。

## Findings / 发现

| Material and channel / 材料与通道 | Protected peak K / 状态保护峰值 | Formal peak K / 形式参考峰值 | Peak time / 峰值时间 | Interpretation / 解释 |
|---|---:|---:|---:|---|
| recA SMA | 1.586 | 1.546 | 0.531 | Mid-stage and late error remain higher |
| recA HTF | 1.379 | 1.478 | 0.041 | Early spike improves, later plateau remains higher |
| recAS SMA | 2.272 | 2.639 | 0.959 | Late-cycle error improves materially |
| recAS HTF | 2.594 | 2.816 | 0.041 | Early entrance spike improves, late error remains elevated |

The protected checkpoint shifts the error pattern rather than removing it. recAS SMA benefits most at late stage-4 time. HTF improvements concentrate at the early entrance spike, while recA SMA and the later HTF plateau remain open.

状态保护检查点改变了误差分布，但没有消除误差。recAS SMA 在第四阶段后段收益最大。HTF 的改善集中在早期入口尖峰，recA SMA 和后段 HTF 平台仍然开放。

## Implication for the next experiment / 对下一实验的含义

An additional global material weight sweep is not the next priority. A useful continuation should separate early-entry and late-stage HTF objectives, keep the displacement and entry gates fixed, and report the four material–channel curves together. The energy boundary audit remains independent of this temperature-only diagnosis.

下一步不再优先进行全局材料权重扫描。更有价值的续实验应分离早期入口和后期 HTF 目标，固定位移与入口门禁，并同时报告四条材料—通道误差曲线。能量边界审计仍独立于该温度场诊断。

## Evidence / 证据

- [Temporal error JSON](../evidence/local_studies/hro_protected_state_confirmation_20261123/temporal_error.json)
- [Temporal error figure](../assets/hro_protected_state/third_cycle_temporal_error.svg)
- [Material-stage diagnostic](../evidence/local_studies/hro_protected_state_confirmation_20261123/material_stage_diagnostics.json)
