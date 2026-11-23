# HRO third-cycle material diagnostic / HRO 第三循环材料诊断

This audit compares the selected protected-state checkpoint with the formal HRO reference on the existing two-case third-cycle cohort. The protected checkpoint was fitted on six cases, while the formal reference uses its archived eight-case protocol. The comparison is therefore contextual and supports error diagnosis rather than an architecture-only claim.

该审计在已有两个第三循环工况上比较状态保护检查点和形式 HRO 参考。状态保护检查点使用 6 个工况拟合，形式参考使用归档的 8 工况协议。因此该比较用于误差诊断，不构成只改变架构的严格消融。

## c3s4 material split / c3s4 材料分组

| Material / 材料 | Checkpoint / 检查点 | SMA state K | HTF2 state K | SMA field K | HTF field K |
|---|---|---:|---:|---:|---:|
| recA | Protected-state | 1.071 | 0.070 | 1.446 | 0.571 |
| recA | Formal reference | 1.013 | 0.076 | 1.440 | 0.474 |
| recAS | Protected-state | 1.105 | 0.250 | 1.590 | 1.003 |
| recAS | Formal reference | 1.218 | 0.334 | 1.739 | 0.988 |

The protected checkpoint improves recAS SMA state and field error relative to the contextual reference. Its recA SMA field is similar, while recA HTF field remains higher. The recAS HTF field is close to the reference despite a larger absolute error than recA. This identifies the remaining HTF trade-off as material-conditional rather than a uniform stage failure.

状态保护检查点相对于该参考改善了 recAS 的 SMA 状态和场误差。recA 的 SMA 场接近参考，recA 的 HTF 场仍较高。recAS 的 HTF 场接近参考，但绝对误差高于 recA。剩余 HTF 取舍具有材料条件性，不是所有阶段都统一退步。

## Stage pattern / 阶段模式

The internal validation field RMSE for the selected checkpoint is **0.820 K** at c1s2, **0.629 K** at c1s4, **0.652 K** at c2s2 and **0.643 K** at c2s4. The post-selection third-cycle audit remains the difficult boundary, especially for recAS HTF and the recA HTF comparison.

选定检查点在内部验证上的场 RMSE 为 c1s2 **0.820 K**、c1s4 **0.629 K**、c2s2 **0.652 K**、c2s4 **0.643 K**。选择后的第三循环审计仍是困难边界，尤其集中在 recAS HTF 以及 recA 的 HTF 对照。

## Next low-cost diagnostic / 下一步低成本诊断

The next protocol should keep the selected state protection and add a material-conditional HTF term only after fixing its coefficient before fitting. The comparison must retain both material rows, all four exchange stages, the third-cycle per-case rows and the energy boundary audit. A candidate can advance only if the HTF gain does not reopen the displacement or entry gates.

下一协议应保留状态保护，并在训练前固定系数后增加材料条件 HTF 项。比较必须继续保留两种材料、4 个换热阶段、第三循环逐工况行和能量边界审计。只有在 HTF 改善不重新触发位移或入口门禁时，候选才可继续推进。

The material-conditioned follow-up was completed with strong and refined weights. Strong weights reopened state safety, while refined weights preserved state safety but worsened the combined field and entry metrics. The branch is retained as a negative result in [HRO material-conditioned HTF weighting](HRO_MATERIAL_CONDITIONED_HTF.md).

材料条件后续已经完成强权重和轻权重筛选。强权重重新触发状态安全失败，轻权重保持状态安全但使合并场和入口指标退步。该支线作为负面结果保留在[材料条件 HTF 加权审计](HRO_MATERIAL_CONDITIONED_HTF.md)中。

## Evidence / 证据

- [Machine-readable material-stage diagnostic](../evidence/local_studies/hro_protected_state_confirmation_20261123/material_stage_diagnostics.json)
- [Post-selection test audit](../evidence/local_studies/hro_protected_state_confirmation_20261123/test_audit.json)
- [Energy boundary audit](HRO_ENERGY_BOUNDARY_AUDIT.md)
