# HRO material split audit / HRO 材料分组审计

This audit replays the three archived formal HRO checkpoints and the matched residual-cascade one-step predictions on the four historical test cases. The test cases are split by material configuration, with two recA and two recAS cases. No weights are updated and no physical simulation is run.

本审计回放三个正式 HRO 检查点及匹配的残差级联单步预测，并按材料配置拆分四个历史测试工况，其中 recA 两例、recAS 两例。没有更新权重，也没有运行物理仿真。

## Cycle-3/stage-4 results / 第三循环第四阶段结果

Metrics are pooled within each seed over the two cases in a material group, then averaged across the three fixed seeds. Field errors omit the stage-entry frame.

指标先在每个种子内汇总同一材料组的两个工况，再对三个固定种子取均值。场误差排除阶段入口帧。

| Material / 材料 | Model / 模型 | SMA state K | HTF2 state K | HTF4 state K | SMA field K | HTF field K |
|---|---|---:|---:|---:|---:|---:|
| recA | HRO | **1.013** | 0.076 | 0.112 | **1.440** | **0.474** |
| recA | Residual cascade one-step / 残差级联单步 | 1.535 | **0.060** | 0.114 | 1.723 | 0.671 |
| recAS | HRO | **1.218** | 0.334 | 0.459 | **1.739** | **0.988** |
| recAS | Residual cascade one-step / 残差级联单步 | 1.552 | **0.177** | **0.196** | 2.375 | 1.226 |

HRO reduces SMA state and SMA field error for both material groups. Its advantage is larger for recAS in SMA state and SMA field, while HTF state errors remain higher than the residual reference. recAS is the harder material group for all HRO field metrics, especially HTF.

HRO 在两种材料上都降低了 SMA 状态和 SMA 场误差。recAS 在 SMA 状态和 SMA 场上的改善幅度更大，但 HTF 状态误差仍高于残差参考。对 HRO 而言，recAS 是更困难的材料组，尤其体现在 HTF 场。

## Evidence boundary / 证据边界

The split is a case-level analysis inside a historical test cohort that has already informed development. It shows material-conditional behavior for the two recorded configurations. It does not establish transfer to an unseen material, independent material calibration or a universal material descriptor.

该分组属于已经参与开发的历史测试队列内的工况级分析，展示两种已记录材料配置的条件性表现。它不能证明对未见材料的迁移、独立材料校准或通用材料描述符有效。

The full per-seed and per-material record is [summary.json](../evidence/local_studies/hro_material_split_20261123/summary.json). The archived HRO and residual checkpoint hashes are retained in the summary.

完整逐种子、逐材料记录见 [summary.json](../evidence/local_studies/hro_material_split_20261123/summary.json)，归档 HRO 与残差检查点哈希保存在摘要中。

```bash
python -m examples.audit_hro_material_split \
  --output evidence/local_studies/hro_material_split_20261123
```
