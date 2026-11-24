# HRO late-time profile results / HRO 后期时间权重结果

The late-time profile stage tested global and SMA-only temporal weighting under the protected-state gates. The protocol used the fixed six-case fit split, two validation cases and three seeds.

后期时间权重阶段在状态保护门禁下测试了全通道和 SMA-only 时间权重。协议使用固定的 6 个拟合工况、2 个验证工况和 3 个种子。

## Screening / 筛选

| Candidate / 候选 | Late SMA change / 后期 SMA 变化 | State and entry gates / 状态与入口门 | Material field gate / 材料场门 | Decision / 决策 |
|---|---:|---|---|---|
| Global light | −5.7% | Pass | Fail | Reject |
| Global moderate | −10.2% | Pass | Fail | Reject |
| SMA-only micro | −6.3% | Pass | Pass | Pass |
| SMA-only small | **−8.7%** | Pass | Pass | Selected |
| SMA-only moderate | −22.7% | Pass | Fail | Reject |

The selected profile uses `late_time_weight = 0.15` with channel weights `[1, 0]`, so the late profile is applied to the SMA field channel. HTF remains supervised by the base field objective.

选定方案使用 `late_time_weight = 0.15` 和通道权重 `[1, 0]`，后期权重只作用于 SMA 场通道。HTF 继续由基础场目标监督。

## Confirmation / 确认

The selected profile was confirmed with three seeds and 200 updates.

选定方案使用 3 个种子和 200 次更新完成确认。

- SMA entry RMSE **0.621 K**
- HTF entry RMSE **0.203 K**
- SMA future-field RMSE **0.654 K**
- HTF future-field RMSE **0.725 K**
- Combined future-field RMSE **0.689 K**
- Late SMA validation RMSE **0.753 K**
- Exact replay maximum difference **0**

## Strict post-selection audit / 严格选择后审计

The existing two-case c3s4 audit produced SMA field RMSE **1.516 K** and HTF field RMSE **0.827 K**. The preceding protected-state checkpoint produced **1.523 K** and **0.817 K**. The formal reference produced **1.600 K** and **0.775 K**.

在已有两个 c3s4 工况上的审计得到 SMA 场 RMSE **1.516 K**、HTF 场 RMSE **0.827 K**。前一状态保护检查点为 **1.523 K** 和 **0.817 K**。形式参考为 **1.600 K** 和 **0.775 K**。

The SMA-late objective passes the internal and confirmation gates. The HTF value is about **0.010 K** higher than the preceding protected-state checkpoint. The late profile is retained as a specialized research checkpoint. The public all-channel default remains the protected-state reference.

SMA 后段目标通过了内部和确认门禁。HTF 相对于前一状态保护检查点高约 **0.010 K**。后期方案保留为专项研究检查点，公共全通道默认继续使用状态保护参考。

The HTF propagation diagnostic finds strong descriptive co-variation between c3s4 HTF field error and HTF2/HTF4 state errors across the six seed–material rows. The pooled correlations are **0.968** for HTF2 and **0.934** for HTF4. Memory correlation is material-dependent and reverses sign between recA and recAS, so memory is retained as a diagnostic variable rather than a causal explanation.

HTF 传播诊断显示，在 6 个种子—材料行上，c3s4 HTF 场误差与 HTF2/HTF4 状态误差存在较强描述性共变。合并相关系数分别为 **0.968** 和 **0.934**。记忆相关性随材料改变符号，因此记忆只作为诊断变量保留，不能作为因果解释。

## Evidence / 证据

- [Temporal-channel readiness](../evidence/local_studies/hro_temporal_channel_confirmation_20261123/readiness.json)
- [Screening summary](../evidence/local_studies/hro_temporal_channel_confirmation_20261123/screening_summary.json)
- [Confirmation summary](../evidence/local_studies/hro_temporal_channel_confirmation_20261123/confirmation_summary.json)
- [Third-cycle audit](../evidence/local_studies/hro_temporal_channel_confirmation_20261123/test_audit.json)
- [HTF propagation diagnostic](../evidence/local_studies/hro_temporal_channel_confirmation_20261123/htf_propagation_diagnostic.json)
- [HTF propagation report](HRO_HTF_PROPAGATION_DIAGNOSTIC.md)
