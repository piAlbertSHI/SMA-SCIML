# Material-conditioned HTF weighting audit / 材料条件 HTF 加权审计

The protected-state checkpoint still shows a material-conditional HTF trade-off. A low-cost follow-up tested known-material channel weights on the same six fit cases and two validation cases. The material label is available from the archived case contract and the weighting is applied only to the supervised field loss.

状态保护检查点仍存在材料条件的 HTF 取舍。低成本后续实验在相同的 6 个拟合工况和 2 个验证工况上测试已知材料的场通道权重。材料标签来自归档工况契约，权重只作用于场监督损失。

## Results / 结果

| Candidate / 候选 | recAS HTF change / recAS HTF 变化 | Combined future-field change / 合并后续场变化 | State gate / 状态门 | Decision / 决策 |
|---|---:|---:|---|---|
| Global Pareto control / 全局 Pareto 对照 | reference | reference | pass | Control |
| recAS HTF guard / recAS HTF 保护 | −4.18% | worsened | fail, 10.4% max state regression | Reject |
| Material HTF balanced / 材料 HTF 平衡 | −4.86% | worsened | fail, 14.7% max state regression | Reject |
| recAS HTF light / recAS HTF 轻权重 | −0.36% | worsened | pass | Reject |
| recAS HTF moderate / recAS HTF 中权重 | −0.46% | worsened | pass | Reject |

The strong weights improve recAS HTF but reopen state, entry and combined-field gates. The lighter weights preserve state safety but produce too little HTF improvement and still worsen the combined field and entry scores. No material-conditioned candidate is promoted.

强权重改善 recAS HTF，但重新触发状态、入口和合并场门禁。轻权重保持状态安全，却只带来很小的 HTF 改善，同时使合并场和入口指标退步。没有材料条件候选晋级。

## Decision / 决策

The selected protected-state checkpoint remains the research candidate. Its global field weighting is retained. Material-conditioned HTF weighting is recorded as a negative result and is not added to the default recipe.

选定的状态保护检查点继续作为研究候选，保留全局场权重。材料条件 HTF 加权记录为负面结果，不加入默认方案。

The next useful direction is a material-aware representation or a longer-cycle data split. A further material weight sweep would repeat the observed trade-off without closing the energy or independent-material boundaries.

下一步更有价值的方向是材料感知表示或更长循环的数据划分。继续扫材料权重会重复当前取舍，无法解决能量契约和独立材料边界。

## Evidence / 证据

- [Curated material-conditioned audit summary](../evidence/local_studies/hro_material_conditioned_20261123/summary.json)
