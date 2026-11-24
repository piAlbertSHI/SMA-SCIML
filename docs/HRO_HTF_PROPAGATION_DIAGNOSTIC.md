# HRO HTF propagation diagnostic / HRO HTF 误差传播诊断

The selected late-profile checkpoint was audited on the two existing third-cycle cases and three seeds. The diagnostic relates c3s4 HTF field error to the HTF2/HTF4 state errors and the memory block. It is descriptive over six seed–material rows.

对选定后期方案在已有两个第三循环工况和 3 个种子上进行审计。诊断比较 c3s4 HTF 场误差与 HTF2/HTF4 状态误差及记忆块误差，样本为 6 个种子—材料行，属于描述性分析。

## Pooled co-variation / 合并共变

| Predictor / 变量 | Correlation with HTF field error / 与 HTF 场误差相关 |
|---|---:|
| HTF2 state error / HTF2 状态误差 | 0.968 |
| HTF4 state error / HTF4 状态误差 | 0.934 |
| Memory block error / 记忆块误差 | −0.912 |

The memory correlation changes sign by material. It is positive for recA and negative for recAS. The HTF2 and HTF4 relations remain positive in the pooled rows. These values identify co-varying error blocks and do not establish causal attribution or an energy balance.

记忆相关性按材料变号。recA 为正，recAS 为负。HTF2 和 HTF4 在合并行上保持正相关。这些数值用于定位共变误差块，不能作为因果归因或能量闭合证据。

## Decision / 决策

The late SMA profile remains a specialized research checkpoint. The public all-channel default remains the protected-state reference because the post-selection HTF field is about 0.010 K higher than the preceding protected-state checkpoint.

后期 SMA 方案继续作为专项研究检查点。由于选择后 HTF 场相对于前一状态保护检查点高约 0.010 K，公共全通道默认继续使用状态保护参考。

The machine-readable rows and correlations are stored in [htf_propagation_diagnostic.json](../evidence/local_studies/hro_temporal_channel_confirmation_20261123/htf_propagation_diagnostic.json).
