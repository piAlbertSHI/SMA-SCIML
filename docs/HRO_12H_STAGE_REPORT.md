# HRO twelve-hour stage report / HRO 十二小时阶段报告

This report closes the extended local development stage that followed the first internal screen. The work used existing historical snapshots and saved checkpoints. No physical simulation was run and GitHub was not updated.

本报告总结第一次内部筛选后的连续本地开发阶段。工作只使用已有历史快照和保存的检查点，没有运行物理仿真，也没有更新 GitHub。

## Completed work / 已完成工作

### Evidence and protocol

- Fixed the 6-case fit and 2-case material-stratified validation split before each comparison.
- Kept four sealed test cases outside candidate selection.
- Preserved protocols, source hashes, checkpoints, optimizer state, training histories, validation metrics, replay records and negative results.
- Maintained zero fresh solver calls throughout the stage.

### State–field objective

- Added optional normalized state-block weights and transition-stage weights.
- Added an explicit time-zero entry-field term after diagnosing that early-time weighting alone did not supervise the reported entry frame.
- Added optional sample-level material channel weights for a controlled negative experiment.
- Preserved the default equal-block objective and formal checkpoint replay behavior.

### Protected-state screen and confirmation

- Screened 5 candidates with 3 seeds and 160 updates per seed.
- Four candidates passed the internal gates.
- Selected `jitter_fluid2_pareto` with internal combined future-field RMSE **0.776 K** and maximum state-block regression **3.50%**.
- Confirmed the selected recipe with 3 seeds and 200 updates.
- Confirmation internal combined future-field RMSE was **0.694 K**, SMA entry **0.622 K** and HTF entry **0.208 K**.
- All 3 confirmation checkpoints replayed exactly with maximum difference 0.

### Error and boundary analysis

- Post-selection c3s4 audit gave SMA field RMSE **1.523 K** and HTF field RMSE **0.817 K**.
- Material-stage decomposition shows recAS SMA benefits most at c3s4, while recA HTF and later HTF plateaus remain difficult.
- Temporal decomposition shows recAS SMA error peaking late and HTF error peaking near the entrance spike.
- Strong material-conditioned HTF weights improved recAS HTF by 4–5% but reopened state and field gates.
- Refined material weights preserved state safety but produced less than 0.5% recAS HTF improvement while worsening combined field and entry scores. The branch was rejected.
- Energy boundary audit confirms eight missing contract terms, including realized signed flow, pressure trajectory, port ownership, work, storage convention and independent closure.

## Default decision / 默认方案决策

The protected-state checkpoint is promoted as the current research checkpoint for follow-up work. The public benchmark default remains the state-priority reference because the post-selection HTF trade-off and the energy contract remain open. The material-conditioned branch is recorded as a negative result.

状态保护检查点晋级为后续研究检查点。由于选择后 HTF 取舍和能量契约仍未闭合，公共基准默认继续使用状态优先参考。材料条件分支记录为负面结果。

## Evidence map / 证据索引

| Evidence / 证据 | Location / 位置 |
|---|---|
| Twelve-hour protocol / 十二小时协议 | [HRO_12H_DEVELOPMENT_PLAN.md](HRO_12H_DEVELOPMENT_PLAN.md) |
| State–field model note / 状态—场模型说明 | [HRO_STATE_FIELD_BALANCE.md](HRO_STATE_FIELD_BALANCE.md) |
| Protected-state readiness / 状态保护 readiness | [readiness.json](../evidence/local_studies/hro_protected_state_confirmation_20261123/readiness.json) |
| Third-cycle material diagnostic / 第三循环材料诊断 | [HRO_THIRD_CYCLE_MATERIAL_DIAGNOSTIC.md](HRO_THIRD_CYCLE_MATERIAL_DIAGNOSTIC.md) |
| Temporal diagnostic / 时间误差诊断 | [HRO_THIRD_CYCLE_TEMPORAL_DIAGNOSTIC.md](HRO_THIRD_CYCLE_TEMPORAL_DIAGNOSTIC.md) |
| Material-conditioned negative result / 材料条件负面结果 | [HRO_MATERIAL_CONDITIONED_HTF.md](HRO_MATERIAL_CONDITIONED_HTF.md) |
| Energy boundary / 能量边界 | [HRO_ENERGY_BOUNDARY_AUDIT.md](HRO_ENERGY_BOUNDARY_AUDIT.md) |
| Trade-off figure / 权衡图 | [protected_state_tradeoff.svg](../assets/hro_protected_state/protected_state_tradeoff.svg) |
| Temporal figure / 时间图 | [third_cycle_temporal_error.svg](../assets/hro_protected_state/third_cycle_temporal_error.svg) |

## Local Git record / 本地 Git 记录

The stage contains separate local commits for the plan, protected-state objective, confirmation/readiness, material diagnostics, material-conditioned audit and temporal diagnosis.

本阶段按计划、状态保护目标、确认与 readiness、材料诊断、材料条件审计和时间误差诊断分别形成本地提交。

`3a35868` · `e05fbbe` · `dec356d` · `2b0d352` · `3dc66d3` · `b906f9c`

## Remaining work / 尚未完成

1. Add a fixed late-time field profile candidate and test whether recA SMA mid/late error can improve without reopening entry or displacement gates.
2. Define an independent energy contract with signed flow, pressure, port ownership, work and storage terms before any field-to-performance claim.
3. Add independent material and independent-grid coverage before changing the public benchmark default.
4. Use a separately approved physical-simulation budget if new trajectories are required.

下一阶段协议已经写入 [next_stage_protocol.json](../evidence/local_studies/hro_12h_development_20261123/next_stage_protocol.json)。

## Continuation after the stage report / 阶段报告后的继续推进

The late-time follow-up was completed after this report was first written. The SMA-only late profile passed its internal gates and three-seed confirmation. It reduced the internal late-SMA error by **8.7%**. The post-selection c3s4 audit gave **1.516 K** SMA field RMSE and **0.827 K** HTF field RMSE. It is retained as a specialized research checkpoint, while the public all-channel default remains the protected-state reference.

阶段报告形成后继续完成了后期时间权重实验。SMA-only 后期方案通过内部门禁和三种子确认，内部后期 SMA 误差降低 **8.7%**。选择后 c3s4 审计得到 SMA 场 RMSE **1.516 K**、HTF 场 RMSE **0.827 K**。该方案保留为专项研究检查点，公共全通道默认仍使用状态保护参考。

The HTF propagation diagnostic found pooled correlations of **0.968** for HTF2 state error and **0.934** for HTF4 state error with c3s4 HTF field error. Memory correlation changes sign by material, so no causal memory claim is made.

HTF 传播诊断发现 c3s4 HTF 场误差与 HTF2、HTF4 状态误差的合并相关系数分别为 **0.968** 和 **0.934**。记忆相关性按材料变号，因此不作记忆因果归因。
