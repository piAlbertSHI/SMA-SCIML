# Research roadmap / 研究路线图

The project status maps the original work packages to current model evidence and orders the next experiments. The modal decoder improves spatial regularity, and the thermal response improves the pooled third-cycle SMA field RMSE on the historical benchmark. The directions below describe further HRO studies. [Project status and priorities](PROJECT_STATUS.md) · [Thermal response](HRO_ERROR_SCALED_RESPONSE.md)

项目进度表对照原始工作包与当前证据，给出后续任务顺序。模态解码缓解空间振荡，误差收缩热响应在历史基准上降低了第三循环合并 SMA 场误差。下表集中列出 HRO 的后续研究。[项目进度与优先级](PROJECT_STATUS.md) · [物理接口审查](HRO_RESEARCH_DESIGN.md)

| Direction / 方向 | Evidence / 依据 | Evaluation / 评价设计 |
|---|---|---|
| Spatial regularity / 空间平滑性 | 51-point query jitter with fluid-channel weight 2 gives a balanced single-seed diagnostic, with third-cycle SMA state 1.032 K, SMA field 1.561 K and HTF field 0.854 K / 51 点查询抖动配合流体通道权重 2 形成较平衡的单种子诊断，第三循环 SMA 状态 1.032 K、SMA 场 1.561 K、HTF 场 0.854 K | Confirm the recipe on three fixed seeds, then compare decoder regularization with a training-only validation split / 先用三个固定种子确认，再用训练集内部验证对照解码器正则 |
| State–field consistency / 状态与场一致性 | Field gains coexist with fluid/phase state regressions / 场收益与流体、相分数状态退步并存 | Evaluate loss balancing and endpoint consistency while retaining all state-block metrics / 评价损失权重及端点一致性，保留全部状态块指标 |
| Representation and memory / 表示与记忆 | Inference-only memory reset improves the archived SMA field score by 0.186 K but worsens HTF field error by 0.134 K / 只推理记忆重置使归档 SMA 场改善 0.186 K，但 HTF 场退步 0.134 K | Use matched-budget retraining with separate SMA/HTF objectives and explicit restart evaluation / 用同预算重训、分离 SMA/HTF 目标并显式评价重启阶段 |
| Material transfer / 材料迁移 | Two material configurations and structured distribution-shift diagnostics / 两类材料配置及结构化分布变化诊断 | Compare categorical and physical descriptors; establish independent material coverage / 对照类别与物性输入，建立独立材料覆盖 |
| Objective-aware acquisition / 面向目标的选样 | Diversity improves temperature-span prediction but not every performance target / 多样性改善温差预测，未改善全部性能目标 | Specify the primary objective before comparing acquisition rules / 比较选样规则前固定主要目标 |
| Energy accounting / 能量核算 | State/field predictions and scalar performance surrogates are separate / 状态与场预测、标量性能代理分别训练 | Define independent ports, work and storage terms before validating field-derived cycle performance / 先定义独立端口、功及储能项，再评价场到性能读出 |

The multi-cycle historical test cohort has already been examined. Further development should use training-only validation and distinguish subsequent exploratory evaluation from independent confirmation. Numerical accuracy, engineering suitability and experimental validation are separate claims.

历史多循环测试工况已经分析过。后续开发应使用训练集内部验证，并区分探索性评价与独立确认。数值精度、工程适用性及实验验证属于不同结论。

Experimental provenance, adverse results and checkpoint compatibility remain part of the research record. Computational budgets and data requirements belong in each experiment protocol.

实验来源、不利结果及检查点兼容性均应保留，计算预算及数据需求记录于各实验协议。
