# WP3 state–field contract audit / WP3 状态—场契约审计

This audit consolidates the archived historical multi-cycle state snapshot, the training-field snapshot and the existing operator replay checks. It uses no new training and no solver calls. / 本审计整合历史多循环状态快照、训练场快照和已有算子回放检查，不重新训练，也不调用物理求解器。 [Audit record / 审计记录](../evidence/local_studies/wp3_state_field_contract_20261123/audit.json)

## Archived corpus / 归档规模

The sealed state snapshot contains 12 cases, 118 transitions and 515 canonical state channels. The split contains 72 training transitions and 46 test transitions. The field snapshot contains 32 training fields from eight state-training cases. The separate evaluation field set contains 20 fields from four cases that also belong to the state-training group. Its field keys are disjoint from the 32 training field keys. / 封存状态快照包含 12 个工况、118 个状态转移和 515 个规范状态通道，划分为 72 个训练转移和 46 个测试转移。训练场快照包含来自 8 个状态训练工况的 32 段场。独立评价场包含 4 个状态训练工况的 20 段场，场键与 32 段训练场不重叠。

| Contract item / 契约项 | Verified value / 已核验值 |
|---|---:|
| Canonical state size / 规范状态维度 | 515 |
| State transitions / 状态转移 | 118 |
| Train/test state transitions / 训练与测试转移 | 72 / 46 |
| Training exchange fields / 训练换热场 | 32 |
| Evaluation exchange fields / 评价换热场 | 20 |
| Training field cases / 训练场工况 | 8 |
| Evaluation field cases / 评价场工况 | 4 |
| Initial-field reconstruction error / 初场重构最大误差 | $5.68\times10^{-14}$ K |

The evaluation fields are held out by field key within cases used for state training. They are useful for checking the state-to-field interface. They do not establish independent case generalization. / 评价场是在参与状态训练的工况内部按场键留出的接口数据，可用于检查状态到场的接口，不构成独立工况泛化证据。 [Transition counts / 转移计数](../evidence/local_studies/wp3_state_field_contract_20261123/transition_counts.csv) · [Field cases / 场工况](../evidence/local_studies/wp3_state_field_contract_20261123/field_case_counts.csv)

## Input boundary findings / 输入边界结果

Requested controls and recorded controls agree exactly. The input-contract audit finds a recorded stage-4 initialization difference in all 63 cases. The largest HTF stage-4 request/record initial difference is **0.016588 K**. Stage-3 and stage-4 mechanical and phase starts also contain nonzero gaps. The request-only sensitivity is a diagnostic of initialization authority, not a retrained causal model. / 请求控制量与记录控制量完全一致。输入契约审计发现 63 个工况的记录阶段 4 初始条件都存在差异，HTF 阶段 4 最大请求/记录初始差为 **0.016588 K**。阶段 3 和阶段 4 的力学、相分数起点也存在非零差异。仅请求输入的敏感性属于初态权限诊断，不是重新训练后的因果模型。 [Input contract source / 输入契约来源](../evidence/archive/wp3-input-contract-r1/summary.json)

Several archived fields are declared inactive or unavailable for a stage transition. Uniform inactive-fluid values are initialization placeholders. Hidden storage ownership and explicit restart maps remain open. / 部分阶段的字段被声明为不活动或不可用，统一的非活动流体值只是初始化占位。隐藏储能归属和显式重启映射仍未闭合。

## Interface-gap distribution / 接口差异分布

The 63-case distribution audit shows that the stage-4 path-pipe inlet discrepancy is systematic. All 63 cases have a nonzero gap, with RMSE **0.010648 K** and maximum **0.016588 K**. The hot-sink and hot-buffer starts are nonzero in all cases with RMSE **0.007106 K**. These values describe the recorded/request interface mismatch. They are diagnostic bounds for the input contract and are not formal model-error lower bounds.

63 例分布审计显示，阶段 4 管路入口差异具有系统性。63 个工况全部存在非零差异，RMSE 为 **0.010648 K**，最大值为 **0.016588 K**。热汇和热缓冲起点在全部工况中均非零，RMSE 为 **0.007106 K**。这些数值描述记录/请求接口差异，是输入契约的诊断边界，不是形式化的模型误差下界。

| Stage-4 interface field / 阶段 4 接口字段 | Nonzero cases / 非零工况 | RMSE K | Maximum K |
|---|---:|---:|---:|
| Path-pipe inlet / 管路入口 | 63 / 63 | 0.010648 | 0.016588 |
| Hot sink / 热汇 | 63 / 63 | 0.007106 | 0.009651 |
| Hot buffer / 热缓冲 | 63 / 63 | 0.007106 | 0.009651 |
| Simscape cold-sink pipe / Simscape 冷汇管路 | 63 / 63 | 0.004304 | 0.009475 |

The full per-case and per-stage records are available in [stage interface gaps / 阶段接口差异](../evidence/local_studies/wp3_interface_gaps_20261123/stage_interface_gaps.csv), [declared initial gaps / 声明初态差异](../evidence/local_studies/wp3_interface_gaps_20261123/declared_initial_gaps.csv) and [distribution audit / 分布审计](../evidence/local_studies/wp3_interface_gaps_20261123/audit.json). The largest path-pipe cases are retained in the audit summary for split-aware review.

## Replay and consistency checks / 回放与一致性检查

The four operator checkpoints reproduce their saved predictions exactly. Poisoning withheld labels does not change the refit result. The radial-versus-strip increment identity replays over 32 training field rows with maximum absolute error $1.28\times10^{-13}$ K. The conditional thermal readout checkpoints and training-only refits reproduce exactly. / 四个算子检查点均逐点复现保存预测，注入被留出标签不会改变回放结果。32 条训练场记录的径向—条带增量恒等关系最大绝对误差为 $1.28\times10^{-13}$ K。条件热读出检查点和仅训练集重拟合均逐点复现。 [Operator audit / 算子审计](../evidence/archive/wp4-operator-audit-r1/summary.json) · [Increment verification / 增量核验](../evidence/local_studies/sma_increment_consistency/verification.json) · [Thermal verification / 热读出核验](../evidence/local_studies/sma_stage_readout/verification.json)

The exact interface checks establish reproducible data flow. They do not close full-system energy accounting. Independent port temperatures, flow, work, storage and restart ownership remain necessary before a field forecast can be promoted to a cycle-performance prediction. / 这些接口检查证明数据流可复现，但没有完成全系统能量核算。在把场预测提升为循环性能预测之前，仍需独立端口温度、流量、功、储能和重启归属。

Replay from the repository root uses only the included snapshots and archived evidence.

```bash
python -m examples.audit_wp3_state_field_contract \
  --output run-output-wp3-state-field-contract
python -m examples.audit_wp3_interface_gaps \
  --output run-output-wp3-interface-gaps
python -m examples.verify_increment_consistency
python -m examples.verify_thermal_readout
python -m examples.verify_hierarchical
python -m examples.verify_modal_evidence
```
