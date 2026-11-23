# HRO twelve-hour development plan / HRO 十二小时推进计划

**Status / 状态:** Active local development stage.  
**Scope / 范围:** Existing historical snapshots and saved checkpoints only. No physical solver call and no GitHub push.  
**Target work window / 目标工作量:** At least 12 hours of staged engineering, validation, analysis and documentation. The time allocation is a work budget, not a reason to wait after evidence is complete.

## Why the previous stage was insufficient / 上一阶段为何不足

The first internal screen established a field-improvement trade-off but did not produce a safe default candidate. The entry-weighted candidate improved time-zero field error while its displacement block regressed by about 10%. The next stage therefore needs a multi-objective loss that protects state blocks while retaining field and entry supervision.

前一轮内部筛选确认了场误差改善与状态安全之间的权衡，但没有得到可安全替换默认方案的候选。入口加权候选改善了 time-zero 场误差，但位移状态块退步约 10%。因此下一阶段需要在保留场和入口监督的同时显式保护状态块。

## Work allocation / 工作分配

| Stage / 阶段 | Planned effort / 计划用时 | Deliverable / 产物 |
|---|---:|---|
| 0. Evidence and baseline audit / 证据与基线审查 | 1.0 h | Baseline replay, open-boundary register, fixed data protocol |
| 1. Multi-objective loss implementation / 多目标损失实现 | 2.0 h | State-block weights, stage weights, entry-field term, unit tests |
| 2. Protected-state candidate screen / 状态保护候选筛选 | 3.0 h | Five candidates, three seeds, matched budget, raw checkpoints and metrics |
| 3. Stage and material diagnosis / 阶段与材料诊断 | 2.0 h | Stage-resolved error maps, recA/recAS comparison, third-cycle proxy audit |
| 4. Confirmation and robustness / 确认与稳健性 | 2.0 h | At most two shortlisted confirmations, replay and sensitivity records |
| 5. Readiness, figures and local Git history / 就绪报告、图表与本地 Git | 2.0 h | Readiness report, compact figures, model note, follow-up protocol and commits |

The combined budget is 12 hours. A scientific gate can end one sub-experiment early, but the remaining time moves to diagnosis, robustness, documentation and open-boundary analysis. It does not end the overall stage.

总预算为 12 小时。科学门禁可以提前结束某个子实验，但剩余时间转入诊断、稳健性、文档和开放边界分析，不提前结束整个阶段。

## Fixed evidence boundary / 固定证据边界

- Fit data remain the six-case subset and the two material-stratified internal validation cases from the previous protocol.
- The four sealed test cases remain unavailable for candidate selection.
- Seeds remain `20261121`, `20261122` and `20261123`.
- Physical simulation remains at zero solver calls.
- Every candidate records its protocol, source hashes, checkpoint, update history, validation metrics and replay result.

- 拟合数据继续使用上一协议中的 6 个拟合工况和 2 个材料分层内部验证工况。
- 4 个封存测试工况继续不参与候选选择。
- 随机种子继续使用 `20261121`、`20261122` 和 `20261123`。
- 物理仿真调用保持为零。
- 每个候选都记录 protocol、源文件哈希、检查点、更新历史、验证指标和回放结果。

## Technical work packages / 技术工作包

### State protection / 状态保护

Extend `joint_loss` with optional normalized state-block weights and stage weights. The default path remains mathematically equivalent to the current equal-block objective. Candidate settings emphasize displacement and deflection blocks without changing frozen priors, scalers or the HRO architecture.

扩展 `joint_loss`，增加可选的归一化状态块权重和阶段权重。默认路径保持与当前等权状态目标数学等价。候选方案只提高位移和挠度块的权重，不改变冻结先验、归一化器或 HRO 架构。

### Field and entry supervision / 场与入口监督

Retain the explicit time-zero entry-field term introduced after the previous diagnostic. Compare it with future-field channel weighting and with a lower field coefficient that gives state loss more influence.

保留上一轮诊断后新增的 time-zero 入口场项，并与后续场通道加权及较低场损失系数进行对照，让状态目标获得更大影响。

### Stage-aware evaluation / 阶段感知评价

Report every state transition and exchange stage separately. The third-cycle proxy analysis will use existing three-cycle predictions and saved checkpoints without opening the sealed physical test cohort.

逐个报告每个状态转移和换热阶段。第三循环代理分析使用已有三循环预测和已保存检查点，不打开封存物理测试队列。

## Candidate family / 候选族

The first protected-state screen will compare the coordinate reference, the current query-jitter recipe, an entry-field recipe, a displacement-protected recipe and a late-stage balanced recipe. Candidate names describe loss settings and are not release labels.

第一轮状态保护筛选比较坐标参考、当前查询抖动方案、入口场方案、位移保护方案和后期阶段平衡方案。候选名称描述损失设置，不作为发布版本标签。

Promotion requires joint evidence from state blocks, field channels, entry frames, materials, stages, seed stability and exact replay. A field-only improvement cannot replace the default recipe.

晋级必须同时满足状态块、场通道、入口帧、材料、阶段、种子稳定性和精确回放证据。单独的场误差改善不能替换默认方案。

## Completion record / 完成记录

The stage ends with a readiness report that lists completed work, negative results, remaining energy-accounting gaps, unseen-material and independent-grid boundaries, and the exact next protocol. Local Git commits are required throughout the stage. GitHub remains untouched.

阶段结束时生成 readiness 报告，列出已完成工作、负面结果、剩余能量核算缺口、未见材料和独立网格边界以及下一协议。阶段内持续进行本地 Git 提交，GitHub保持不变。
