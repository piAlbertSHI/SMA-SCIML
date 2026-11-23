# Next-stage HRO internal validation plan / 下一阶段 HRO 内部验证计划

**Status / 状态:** Protocol prepared before fitting. No physical solver call and no GitHub push are part of this stage.  
**Target window / 目标时长:** 4–8 hours wall time, with a hard stop at 8 hours.  
**Data policy / 数据政策:** Reuse the sealed historical state snapshot and the archived training fields. Do not regenerate physical trajectories.

## Objective / 目标

The next stage will decide whether a field-balanced HRO training recipe should replace the current state-priority candidate for later work. The decision must use a case-level internal validation split, matched seeds and matched update budgets. The four sealed test cases remain untouched until a candidate has passed the internal gates.

下一阶段判断场平衡的 HRO 训练方案是否值得进入后续默认候选。比较使用工况级内部验证、固定随机种子和相同更新预算。4 个封存测试工况在候选通过内部门禁前不参与选择。

## Fixed data split / 固定数据划分

The source is `data/historical12_state.npz` with `data/historical12_training_fields.npz`. The split is fixed before fitting and stored in the companion protocol JSON.

数据源为 `data/historical12_state.npz` 与 `data/historical12_training_fields.npz`。训练前固定划分，并写入配套 protocol JSON。

| Role / 用途 | Material / 材料 | Case IDs / 工况 |
|---|---|---|
| Fit / 拟合 | recA | `...01...`, `...04...`, `...05...` |
| Fit / 拟合 | recAS | `...07...`, `...09...`, `...11...` |
| Internal validation / 内部验证 | recA | `...03...` |
| Internal validation / 内部验证 | recAS | `...08...` |
| Sealed test, held out / 封存测试，留出 | recA and recAS | `...02...`, `...06...`, `...10...`, `...12...` |

The two validation cases cover both archived materials and are not used to update gradients. The four test cases remain reserved for a final confirmation report. No claim about unseen materials or independent grids is made by this split.

两个验证工况覆盖两种归档材料，且不参与梯度更新。4 个测试工况保留到最终确认报告。该划分不支持未见材料或独立网格泛化结论。

## Candidate matrix / 候选矩阵

The runner will be extended only to accept the fixed case split and to report validation metrics. The HRO architecture, frozen prior, normalizers and optimizer remain unchanged. Each candidate uses the same three seeds and the same update count.

训练入口只增加固定工况划分和验证指标输出。HRO 架构、冻结先验、归一化器和优化器保持不变。每个候选使用相同的三个随机种子和相同更新次数。

| Candidate / 候选 | Spatial queries / 空间查询 | Field objective / 场目标 | Entry emphasis / 入口强调 | Purpose / 用途 |
|---|---|---|---|---|
| `coordinate_reference` | 26 uniform points, no jitter | Equal SMA/HTF channels | None | Matched-budget coordinate reference |
| `jitter_fluid2` | 51 points with half-spacing jitter | SMA 1, HTF 2 | None | Current state-priority candidate |
| `jitter_fluid2_entry` | 51 points with half-spacing jitter | SMA 1, HTF 2 | Entry-time weight 1.0 | State–field consistency candidate |
| `jitter_fluid3_balance` | 51 points with half-spacing jitter | SMA 1, HTF 3 | None | Stronger HTF balance candidate |

The candidate names describe the training recipe and are not release or version labels. The existing coordinate HRO checkpoint is also replayed as an audit reference, but it cannot be used to select a candidate.

候选名称只描述训练方案，不使用内部版本标签。已有坐标 HRO 检查点额外做回放审计，但不用于选择候选。

## Fixed training budget / 固定训练预算

- Screening uses 120 updates per seed, batch size 2, AdamW, learning rate `3e-4`, weight decay `1e-4`, gradient clipping at `1.0`, and the existing cosine schedule.
- Screening covers four candidates and three seeds, which is twelve fits. The measured three-seed 200-update confirmation took about 8.6 minutes of CPU time, so the screening estimate is about 30–45 minutes of CPU time plus validation and artifact writing.
- Only the best internally gated candidate receives the 200-update, three-seed confirmation. The measured budget for that confirmation is about 10 minutes of CPU time.
- Documentation, replay checks, tests and local Git review are budgeted for the remaining wall time. A run exceeding the 8-hour wall-time cap stops without an automatic retry.

- 筛选阶段每个种子使用 120 次更新、批大小 2、AdamW、学习率 `3e-4`、权重衰减 `1e-4`、梯度裁剪 `1.0` 及现有余弦调度。
- 筛选覆盖 4 个候选和 3 个种子，共 12 次拟合。已测得的三种子 200 次更新确认约耗时 8.6 分钟 CPU，因此筛选预计需要 30–45 分钟 CPU，另加验证和产物写入时间。
- 只有通过内部门禁的最佳候选进入 200 次更新、三种子确认。该确认预计约 10 分钟 CPU。
- 剩余墙钟时间用于回放、测试、文档和本地 Git 审查。超过 8 小时上限时停止，不自动重跑。

## Metrics and promotion gates / 指标与晋级门

Every fit records state, field, entry and future-field metrics. Reports are stratified by material, cycle, stage and field channel. The primary comparison uses the two internal validation cases only.

每次拟合都记录状态、场、入口和后续场指标，并按材料、循环、阶段和场通道分层。主要比较只使用两个内部验证工况。

1. **State safety gate.** No mandatory state block may regress by more than 5% relative to `coordinate_reference` on the internal validation cases.
2. **Field balance gate.** The candidate must improve or preserve the combined SMA/HTF future-field score and must not trade a material-specific gain for a material-specific failure.
3. **Entry consistency gate.** The candidate must reduce or preserve the normalized stage-entry field error. A lower future-field score alone is insufficient.
4. **Reproducibility gate.** The ranking must be stable across the three seeds, all checkpoints must replay exactly, and the fixed split, source hashes and zero-solver-call count must be present.
5. **Promotion rule.** If no candidate passes all four gates, retain the current state-priority recipe and record the negative result. A candidate is not promoted solely because it improves one channel.

1. **状态安全门。** 相对于 `coordinate_reference`，内部验证上的任何必需状态块退步不得超过 5%。
2. **场平衡门。** 候选必须改善或保持 SMA/HTF 后续场的合并指标，且不能用某一材料的收益换取另一材料的失败。
3. **入口一致性门。** 候选必须降低或保持阶段入口场的归一化误差。仅后续场指标变好不足以晋级。
4. **可复现门。** 三个种子的排序应稳定，所有检查点必须精确回放，并且记录固定划分、源文件哈希和零物理求解调用。
5. **晋级规则。** 若没有候选同时通过四道门，保留当前状态优先方案并记录负面结果。单通道改善不能单独触发晋级。

## Execution stages / 执行阶段

### Stage A — protocol and runner audit / 阶段 A：协议与入口审查

Add case-subset arguments, validation-only scoring and candidate configuration checks. Add unit tests for split disjointness, material balance, held-out test exclusion and deterministic replay. No training is run before these checks pass.

增加工况子集参数、仅验证评分和候选配置校验。增加划分互斥、材料平衡、测试留出和确定性回放的单元测试。检查通过前不启动训练。

### Stage B — internal screening / 阶段 B：内部筛选

Run the four candidates with the fixed 120-update budget. Preserve one protocol, checkpoint, history, validation summary and replay record per candidate and seed. Select at most one candidate by the prewritten gates.

用固定 120 次更新预算运行 4 个候选。每个候选和种子分别保留 protocol、检查点、训练历史、验证摘要和回放记录。按预先写好的门禁至多选出一个候选。

### Stage C — matched confirmation / 阶段 C：同预算确认

Run only the selected candidate for 200 updates and three seeds. The confirmation is still historical-data evidence. It does not open the sealed test cohort or establish independent generalization.

只对选出的候选运行 200 次更新和三个种子的确认。该确认仍属于历史数据证据，不打开封存测试队列，也不建立独立泛化结论。

### Stage D — readiness and local commit / 阶段 D：就绪审查与本地提交

Produce `protocol.json`, `screening_summary.json`, `confirmation_summary.json`, `replay_verification.json`, `readiness.json`, figures or compact tables for the selected metrics, and a negative-results record. Run the existing unit tests and bundle audits, then commit locally. Do not push GitHub.

生成 `protocol.json`、`screening_summary.json`、`confirmation_summary.json`、`replay_verification.json`、`readiness.json`、关键指标图或紧凑表格以及负面结果记录。运行现有单元测试和 bundle 审查后本地提交，不推送 GitHub。

## Readiness boundaries / 就绪边界

The final report will state separately whether the candidate is suitable as the default HRO training recipe, whether it improves the third-cycle field error, and which evidence remains open. Port-level energy closure, hidden storage ownership, unseen-material transfer and independent-grid behavior remain outside this stage unless existing evidence resolves them without new simulation.

最终报告分别说明候选是否适合作为默认 HRO 训练方案、是否改善第三循环场误差，以及哪些证据仍然缺失。端口级能量闭合、隐藏储能归属、未见材料迁移和独立网格行为不属于本阶段范围，除非现有证据能够在不新增仿真的情况下解决。

## Planned artifact / 计划协议文件

The machine-readable protocol is stored at [next_stage_hro_plan_20261123/protocol.json](../evidence/local_studies/next_stage_hro_plan_20261123/protocol.json). It is a pre-run plan and contains no fitted result.

机器可读协议存放在 [next_stage_hro_plan_20261123/protocol.json](../evidence/local_studies/next_stage_hro_plan_20261123/protocol.json)。该文件是运行前计划，不包含拟合结果。
