# Research roadmap / 研究路线图

The [modal-decoder study](HRO_MODAL.md) now provides the sampling comparison and a two-phase refinement result: lower spatial oscillation and whole-test field error, with mixed third-cycle SMA-field results. The directions below remain open for matched-budget tests. The [physics audit](HRO_RESEARCH_DESIGN.md) establishes the historical source/parameter recovery and explains why SMA endpoint consistency cannot yet be imposed.

[模态解码研究](HRO_MODAL.md)已提供采样对照和双阶段训练结果：空间振荡及全测试场误差减小，但第三循环 SMA 场改善不一致。下列方向仍需匹配预算的检验；[物理审查](HRO_RESEARCH_DESIGN.md)记录了历史源码及参数恢复，并解释 SMA 末态一致性为何尚不能直接施加。

| Direction / 方向 | Evidence / 依据 | Evaluation / 评价设计 |
|---|---|---|
| Spatial regularity / 空间平滑性 | HRO improves third-cycle temperature RMSE but exhibits spatial oscillations / HRO 第三循环温度误差改善，仍有空间振荡 | Compare query density/jitter and decoder regularization using validation cases drawn only from the training partition / 仅用训练分区划分验证工况，对照查询密度、抖动及解码正则 |
| State–field consistency / 状态与场一致性 | Field gains coexist with fluid/phase state regressions / 场收益与流体、相分数状态退步并存 | Evaluate loss balancing and endpoint consistency while retaining all state-block metrics / 评价损失权重及端点一致性，保留全部状态块指标 |
| Representation and memory / 表示与记忆 | Spatial encoding, latent attention and GRU memory are trained jointly / 空间编码、注意力与 GRU 联合训练 | Isolate module contributions with matched training budgets and case-level validation / 用同预算及工况级验证隔离模块贡献 |
| Material transfer / 材料迁移 | Two material configurations and structured distribution-shift diagnostics / 两类材料配置及结构化分布变化诊断 | Compare categorical and physical descriptors; establish independent material coverage / 对照类别与物性输入，建立独立材料覆盖 |
| Objective-aware acquisition / 面向目标的选样 | Diversity improves temperature-span prediction but not every performance target / 多样性改善温差预测，未改善全部性能目标 | Specify the primary objective before comparing acquisition rules / 比较选样规则前固定主要目标 |
| Energy accounting / 能量核算 | State/field predictions and scalar performance surrogates are separate / 状态与场预测、标量性能代理分别训练 | Define independent ports, work and storage terms before validating field-derived cycle performance / 先定义独立端口、功及储能项，再评价场到性能读出 |

The multi-cycle historical test cohort has already been examined. Further development should use training-only validation and distinguish subsequent exploratory evaluation from independent confirmation. Numerical accuracy, engineering suitability and experimental validation are separate claims.

历史多循环测试工况已经分析过。后续开发应使用训练集内部验证，并区分探索性评价与独立确认。数值精度、工程适用性及实验验证属于不同结论。

Experimental provenance, adverse results and checkpoint compatibility remain part of the research record. Computational budgets and data requirements belong in each experiment protocol.

实验来源、不利结果及检查点兼容性均应保留，计算预算及数据需求记录于各实验协议。
