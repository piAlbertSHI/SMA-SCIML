# HRO research design / HRO 研究设计

HRO targets coupled, stage-dependent elastocaloric dynamics with a compact explicit physical state and persistent latent memory. Model development prioritizes autonomous rollout, spatial fidelity, data separation and traceable numerical evidence.

HRO 面向具有阶段切换的弹热耦合动力学，以显式物理状态和持续潜变量记忆描述循环过程。模型开发重点是自主推进、空间精度、数据隔离及可追溯的数值证据。

## Architecture decisions / 架构决策

| Decision / 决策 | Implementation and boundary / 实现与边界 |
|---|---|
| Structured prior / 结构化先验 | A frozen, training-fitted affine transition plus a learned residual. This reduces the burden on the neural transition; it does not enforce a constitutive equation. / 冻结训练数据拟合的仿射转移，由神经网络学习修正；不等同于强制满足本构方程。 |
| Spatial compression / 空间压缩 | Separate radial and axial encoders, followed by latent cross-attention, preserve the distinction between mechanical and transport coordinates. / 分离径向与轴向编码，再进行潜变量交叉注意力，保留力学和输运坐标的区别。 |
| Explicit recurrent state / 显式循环记忆 | GRU memory persists through four physical stages and the mechanical restart. / GRU 记忆跨四个物理阶段及力学重启持续传递。 |
| Modal field representation / 模态场表示 | Time-conditioned attention predicts coefficients of a constant, linear and cosine basis. Initial temperatures remain exact; full-grid supervision constrains spatial derivatives. / 时间条件注意力预测常数、线性及余弦基系数；严格保持初始温度，以全空间网格监督约束导数。 |
| Two-phase optimization / 双阶段优化 | Joint trajectory training followed by cached autonomous decoder refinement. The second phase cannot change state predictions because every non-decoder tensor is frozen and audited. / 完整轨迹联合训练后，缓存自主轨迹并专项优化解码器；第二阶段冻结并核验全部非解码器张量，因此不能改变状态预测。 |

The field correction uses 24 basis functions rather than an independent nonlinear spatial query. This reduces coordinate-attention work from one query per space–time point to one query per time, followed by a matrix contraction. It is not an alias-free CNO implementation, a physical boundary condition, or a stability proof.

场修正采用 24 个基函数，不对每个空间位置单独进行非线性查询。坐标注意力从每个时空点一次查询，变为每个时间一次查询，再作矩阵收缩。这不构成 CNO 的无混叠实现、物理边界条件或稳定性证明。

## Relation to operator research / 与算子研究的关系

| Primary source / 原始论文 | Relevance and distinction / 借鉴点与区别 |
|---|---|
| [CNO, NeurIPS 2023](https://arxiv.org/abs/2302.01178) | Continuous/discrete representation consistency motivates explicit spatial diagnostics. HRO does not reproduce the CNO construction or its universality result. / 连续与离散表示的一致性启发空间诊断；HRO 未复现其结构或理论结论。 |
| [PDE-Refiner, 2023](https://arxiv.org/abs/2308.05732) | Frequency-resolved errors matter for long rollouts. HRO uses supervised gradient losses and a modal decoder, not diffusion refinement. / 频率误差对长时推进重要；HRO 使用监督梯度损失和模态解码，不采用扩散式精炼。 |
| [Poseidon, 2024](https://arxiv.org/abs/2405.19101) | Multiscale operator representations and time conditioning inform a scalable design. HRO has no foundation-model pretraining or demonstrated transfer to unseen physics. / 多尺度算子表示和时间条件机制提供扩展方向；HRO 尚无基础模型预训练或未知物理迁移证据。 |
| [Transolver++, ICML 2025](https://proceedings.mlr.press/v267/luo25o.html) | Attention over compressed physical representations supports computational scaling. HRO uses ordinary latent cross-attention, not Physics-Attention, and has not been tested on million-point geometries. / 压缩物理表示上的注意力支持计算扩展；HRO 使用普通潜变量交叉注意力，未实现 Physics-Attention 或百万网格验证。 |
| [SGNO, 2026 preprint](https://arxiv.org/abs/2602.18801) | Fitted propagation plus a learned correction is a relevant contemporary comparison. HRO uses stage-specific affine maps and persistent memory; no SGNO benchmark comparison has been run. / 拟合传播加学习修正是相关的近期方向；HRO 使用分阶段仿射映射与持续记忆，尚未运行 SGNO 基准对照。 |

These papers inform design choices; their reported improvements cannot be transferred to this dataset. The implemented capacity remains approximately 3.11 million parameters. Increasing width or depth is possible through configuration, but should follow data growth and controlled scaling studies.

上述论文用于指导设计，其性能提升不能直接移用于本数据集。已实现模型约有 311 万参数，宽度和深度可通过配置扩展；实际扩容应随数据增长并通过受控规模实验验证。

## State–field semantics / 状态与场的语义

An audit on all eight training cases finds exact agreement between fluid end fields and their stored state variables, but the reconstructed SMA end field differs from the stored strip field by up to **4.8795 K**. The explicit SMA state includes a radial projection and composite stage semantics; stage 2 also includes mechanical re-equilibration. Therefore HRO does not impose a blanket equality between the decoded field endpoint and the reconstructed state. Initial-field reconstruction is independently exact to floating-point tolerance.

八个训练工况的审查显示，流体末场与相应状态变量完全一致，但由 SMA 末状态重建的场与保存的条带温度场最多相差 **4.8795 K**。显式 SMA 状态包含径向投影及复合阶段语义，第二阶段还包含力学再平衡。因此，HRO 不对全部场强制施加末端状态相等约束；初始场重建已独立验证至浮点精度。

Resolving the SMA projection semantics, auditing energy fluxes and separating exchange endpoints from mechanically equilibrated states are prerequisites for a physically meaningful consistency loss. Simply setting an endpoint mismatch to zero would teach an incorrect identity.

有物理意义的一致性损失需要先厘清 SMA 投影语义、核验能量流，并区分换热末态和力学平衡后状态。直接把末端差异约束为零会引入错误恒等关系。

## Compatible data expansion / 兼容数据扩充

[Historical physics audit](../evidence/hro_legacy_physics.json) verifies all 12 original request hashes and recovers all nine instrumented implementation/configuration files with exact byte hashes. Original material settings include **E_A = 35 GPa, E_M = 25 GPa, H_max = 0.04, density = 6500 kg/m³, heat capacity = 450 J/(kg·K)**. Transformation temperatures are **260/250/275/285 K** for configuration A and **245/235/260/270 K** for configuration B, in the order M_s/M_f/A_s/A_f. Full parameter values are recorded in the audit.

[历史物理模型审查](../evidence/hro_legacy_physics.json)核验了 12 个原始请求哈希，并逐字节恢复了 9 个受记录的实现及配置文件。材料设置包括 **E_A = 35 GPa、E_M = 25 GPa、H_max = 0.04、密度 6500 kg/m³、比热容 450 J/(kg·K)**。按 M_s/M_f/A_s/A_f 顺序，两种配置的相变温度分别为 **260/250/275/285 K** 和 **245/235/260/270 K**，完整参数见审查记录。

The recovered solver package stays isolated from the actively revised physical model. No fresh simulations were performed. Uninstrumented dependencies still require a runtime check, output/cache paths require workspace relocation, and the original raw MAT files remain unavailable. A small, timed compatibility run must precede any expansion. Especially long or large simulation batches require a separate approved budget.

恢复的软件包与正在修改的物理模型隔离保存，未运行新增仿真。未受逐工况哈希记录的依赖仍需运行验证，输出与缓存路径需重新定位到工作区，原始 MAT 文件目前仍不可用。扩充前应先执行小规模、计时的兼容性验证；特别耗时或大批次仿真需要单独批准预算。

For publication-level evidence, reserve new conditions before training, add three-cycle training trajectories without consuming the historical three-cycle evaluation cases, compare matched-budget alternatives, and evaluate extrapolation, uncertainty and energy closure. More data alone does not establish credibility without these checks.

面向论文证据，应在训练前预留新增工况，补充三循环训练轨迹并保留历史三循环评价集，以匹配预算比较方法，同时检验外推、不确定性与能量闭合。可信结论需要这些检验，不能仅靠增加数据量保证。
