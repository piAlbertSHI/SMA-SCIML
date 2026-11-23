# Next-stage HRO screening results / 下一阶段 HRO 筛选结果

The fixed internal screening used six fit cases and two material-stratified validation cases from the sealed historical snapshots. Four candidates were fitted with three seeds and 120 updates per seed. The four sealed test cases were not opened and no physical solver was called.

固定内部筛选使用封存历史快照中的 6 个拟合工况和 2 个材料分层验证工况。4 个候选各用 3 个种子和每种子 120 次更新。4 个封存测试工况未打开，也没有调用物理求解器。

## Outcome / 结果

| Candidate / 候选 | Future field reduction / 后续场误差变化 | Entry consistency / 入口一致性 | State safety / 状态安全 | Decision / 决策 |
|---|---:|---|---|---|
| `coordinate_reference` | reference | pass | pass | Reference only |
| `jitter_fluid2` | 17.1% lower | fail, SMA entry increased | fail, maximum block regression 25.6% | Retain as diagnostic candidate |
| `jitter_fluid2_entry` | 16.0% lower | pass after explicit time-zero loss | fail, displacement block regression 10.0% | Retain as research candidate |
| `jitter_fluid3_balance` | 10.4% lower | fail, SMA entry increased | fail, maximum block regression 43.4% | Reject for promotion |

The corrected entry candidate reduced the internal validation entry RMSE from 0.791 K to 0.668 K for SMA and from 0.298 K to 0.226 K for HTF. Its combined future-field RMSE was 0.929 K versus 1.108 K for the matched coordinate reference. The displacement block increased by approximately 10.0%, so the fixed 5% state-safety gate was not met.

修正后的入口候选将内部验证 SMA 入口 RMSE 从 0.791 K 降至 0.668 K，将 HTF 入口 RMSE 从 0.298 K 降至 0.226 K。其合并后续场 RMSE 为 0.929 K，匹配坐标参考为 1.108 K。位移状态块退步约 10.0%，因此未通过固定的 5% 状态安全门。

## Decision / 决策

No candidate is promoted as the default HRO training recipe. The current state-priority recipe remains the default evidence path. The field-balanced candidates remain available for a later multi-objective study with an explicitly negotiated state–field trade-off.

当前筛选没有候选晋级为默认 HRO 训练方案。当前状态优先方案继续作为默认证据路径。场平衡候选保留到后续多目标研究，并在开始前明确状态与场之间的权衡规则。

## Protected-state continuation / 状态保护续训

The continuing twelve-hour stage added normalized displacement, deflection and late-transition protection to the loss. Five candidates were screened for 160 updates per seed. Four passed all internal gates, and `jitter_fluid2_pareto` had the lowest combined future-field RMSE at **0.776 K** with a maximum state-block regression of **3.50%**.

持续十二小时阶段在损失中增加了归一化位移、挠度和后期转移保护。5 个候选各使用每种子 160 次更新。4 个候选通过全部内部门禁，`jitter_fluid2_pareto` 的合并后续场 RMSE 最低，为 **0.776 K**，最大状态块退步为 **3.50%**。

The selected recipe was confirmed with 200 updates and three seeds. Its internal confirmation produced **0.694 K** combined future-field RMSE, **0.622 K** SMA entry RMSE and **0.208 K** HTF entry RMSE. Exact checkpoint replay passed with zero difference.

选定方案使用 200 次更新和 3 个种子完成确认。内部确认得到合并后续场 RMSE **0.694 K**、SMA 入口 RMSE **0.622 K** 和 HTF 入口 RMSE **0.208 K**。检查点精确回放通过，最大差异为 0。

The post-selection third-cycle audit reduced the formal-reference SMA field score from **1.600 K** to **1.523 K** and gave HTF field RMSE **0.817 K** versus **0.775 K** for the formal reference. The protected-state checkpoint is promoted for follow-up research. The public benchmark default remains the state-priority reference until the HTF trade-off and energy boundaries are resolved.

选择后的第三循环审计将形式参考的 SMA 场误差从 **1.600 K** 降至 **1.523 K**，HTF 场 RMSE 为 **0.817 K**，形式参考为 **0.775 K**。状态保护检查点晋级为后续研究方案。公共基准默认仍使用状态优先参考，直到 HTF 取舍和能量边界得到解决。

Full model details and the stage-level loss settings are recorded in [HRO state–field balance](HRO_STATE_FIELD_BALANCE.md). The confirmation readiness record is [readiness.json](../evidence/local_studies/hro_protected_state_confirmation_20261123/readiness.json).

完整模型细节和阶段级损失设置见[HRO 状态—场平衡](HRO_STATE_FIELD_BALANCE.md)。确认 readiness 记录见 [readiness.json](../evidence/local_studies/hro_protected_state_confirmation_20261123/readiness.json)。

The full 200-update confirmation was not run because the prewritten protocol permits confirmation only after an internal candidate passes all gates. This preserves the selection boundary and avoids using the sealed test cohort to justify a promotion.

由于没有候选通过全部内部门禁，当前筛选未启动 200 次更新确认。该处理保留了选择边界，也没有用封存测试队列为晋级提供事后依据。

## Diagnostic repair / 诊断修复

The first completed diagnostic screen exposed a scoring-name bug. The next screen showed that `entry_time_weight` emphasized nonzero times but did not supervise the reported time-zero entry reconstruction. An optional `entry_field_weight` term was added to `joint_loss`, covered by a unit test, and used only by `jitter_fluid2_entry`. The corrected screen passed exact checkpoint replay with zero difference.

第一次诊断筛选发现评分器状态块名称错误。修正后的筛选进一步显示，`entry_time_weight` 只强调非零时间，未直接监督报告中的 time-zero 入口重建。因此在 `joint_loss` 中增加了可选的 `entry_field_weight`，补充单元测试，并只在 `jitter_fluid2_entry` 中启用。修正后的筛选通过精确检查点回放，最大差异为 0。

## Evidence and boundaries / 证据与边界

- Screening output and 12 checkpoints are preserved under `evidence/experiments/hro_internal_validation_20261123_r3`.
- Curated readiness and negative-result records are stored in [readiness.json](../evidence/local_studies/hro_internal_validation_20261123_r3/readiness.json) and [negative_results.json](../evidence/local_studies/hro_internal_validation_20261123_r3/negative_results.json).
- Stage-resolved state and field metrics are stored in [stage_metrics.json](../evidence/local_studies/hro_internal_validation_20261123_r3/stage_metrics.json), covering all nine state transitions and four exchange stages for each candidate and seed aggregate.
- Port-level energy closure, hidden storage ownership, unseen-material transfer and independent-grid behavior remain open.
- This result is an internal historical-data study. It does not establish independent generalization or experimental validity.

筛选输出和 12 个检查点保存在 `evidence/experiments/hro_internal_validation_20261123_r3`。整理后的 readiness 和负面结果记录见 [readiness.json](../evidence/local_studies/hro_internal_validation_20261123_r3/readiness.json) 与 [negative_results.json](../evidence/local_studies/hro_internal_validation_20261123_r3/negative_results.json)。逐阶段状态和场指标见 [stage_metrics.json](../evidence/local_studies/hro_internal_validation_20261123_r3/stage_metrics.json)，覆盖每个候选和种子聚合下的 9 个状态转移及 4 个换热阶段。
