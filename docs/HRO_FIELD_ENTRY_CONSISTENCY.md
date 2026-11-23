# HRO field-entry consistency / HRO 场入口一致性

This audit separates field error at the predicted stage-entry frame from future-field error. It replays the three formal HRO checkpoints and the matched residual-cascade one-step predictions on the historical test cohort. No training or physical simulation was run.

本审计将预测阶段入口帧的场误差与后续场误差分开。它回放三个正式 HRO 检查点及匹配的残差级联单步预测，使用历史测试队列，没有训练或物理仿真。

## Cycle-3/stage-4 results / 第三循环第四阶段结果

| Material / 材料 | Model / 模型 | SMA entry K | SMA future K | HTF entry K | HTF future K |
|---|---|---:|---:|---:|---:|
| recA | HRO | 1.374 | **1.600** | 0.938 | **0.775** |
| recA | Residual cascade one-step / 残差级联单步 | 1.229 | 2.076 | 0.767 | 0.989 |
| recAS | HRO | 1.460 | **1.739** | 1.176 | **0.988** |
| recAS | Residual cascade one-step / 残差级联单步 | 1.105 | 2.375 | 0.953 | 1.226 |

The HRO field decoder starts from a predicted state-derived entry field. The entry error is already of order 1 K in the third cycle, while the archived stage-4 request/record path-pipe discrepancy is at most **0.016588 K**. This separates the dominant prediction error propagation from the smaller raw interface mismatch.

HRO 场解码器从预测状态重建的入口场开始。第三循环入口误差已经达到 1 K 量级，而归档阶段 4 记录/请求管路差异最大只有 **0.016588 K**。这说明主要误差来自预测状态传播与入口场重建，原始接口差异相对较小。

The entry value is a consistency diagnostic rather than a formal lower bound on model error. The archived field and state semantics include stage-specific projection and restart conventions, so a future consistency loss must respect those semantics.

入口误差是状态—场一致性诊断，不是模型误差的形式化下界。归档场与状态包含阶段特定投影和重启约定，后续一致性损失必须保留这些语义。

## Evidence boundary / 证据边界

The comparison uses a previously examined historical test cohort. It supports a training objective diagnosis and a material-conditional error decomposition. It does not establish independent generalization or justify changing the default checkpoint.

比较使用已经分析过的历史测试队列，可用于训练目标诊断和按材料误差分解，不构成独立泛化证据，也不支持更换默认检查点。

Full per-seed and per-material rows are preserved in [summary.json](../evidence/local_studies/hro_field_entry_consistency_20261123_r2/summary.json). The first run with an incorrect case-insensitive material mask is retained locally under the corresponding `_20261123` directory and is excluded from the curated bundle.

完整逐种子、逐材料记录见 [summary.json](../evidence/local_studies/hro_field_entry_consistency_20261123_r2/summary.json)。第一次由于材料字符串大小写掩码错误的运行保留在本地对应 `_20261123` 目录中，不纳入 curated bundle。

```bash
python -m examples.audit_hro_field_entry_consistency \
  --output evidence/local_studies/hro_field_entry_consistency_20261123_r2
```
