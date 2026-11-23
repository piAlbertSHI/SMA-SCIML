# HRO state–field balance / HRO 状态—场平衡

The protected-state continuation keeps the current Hierarchical Recurrent Operator architecture. The model has dual spatial CNN encoders, latent cross-attention, a GRU memory, five FiLM transition heads and a joint coordinate field decoder. The frozen historical prior and train-only normalizers remain outside the trainable update.

状态保护支线继续使用当前分层循环算子架构。模型包含双路空间 CNN 编码器、潜变量交叉注意力、GRU 记忆、5 个 FiLM 状态转移头和联合坐标场解码器。冻结历史先验和仅训练数据拟合的归一化器不参与可训练更新。

## Objective / 目标函数

The state term is calculated over eight normalized state blocks. Optional block and transition weights are normalized to unit mean before multiplication, so the default equal-block objective remains unchanged. The field term supervises future field values and spatial gradients. The entry term supervises the time-zero field reconstructed from the predicted stage-entry state.

状态项对 8 个归一化状态块计算误差。可选状态块权重和转移阶段权重在乘入前归一化为均值 1，因此默认等权目标保持不变。场项监督后续场值和空间梯度。入口项监督由预测阶段入口状态重建出的 time-zero 场。

For a batch, the implemented objective is

\[
\mathcal{L}=\mathcal{L}_{state}+\lambda_f\mathcal{L}_{field}+\lambda_g\mathcal{L}_{gradient}+\lambda_c\mathcal{L}_{curvature}+\lambda_e\mathcal{L}_{entry}.
\]

The selected research candidate uses the following settings.

选定的研究候选使用以下设置。

| Component / 组成 | Setting / 设置 |
|---|---|
| Spatial queries / 空间查询 | 51 points with half-native-spacing jitter |
| Field channel weights / 场通道权重 | SMA 1.0, HTF 1.5 |
| Early-time weight / 早时刻权重 | 0.5 |
| Time-zero entry weight / time-zero 入口权重 | 0.75 |
| State block weights / 状态块权重 | SMA 1, HTF2 1, HTF4 1, side 1, phase 1, displacement 2.5, deflection 2.5, memory 1.25 |
| Transition weights / 转移权重 | 1, 1, 1, 1, 1, 1, 1.25, 1.5, 1.75 |
| Optimizer / 优化器 | AdamW, learning rate 3e-4, weight decay 1e-4 |
| Training / 训练 | Six fit cases, batch size 2, CPU float64, three fixed seeds |

## Internal screening / 内部筛选

The protected-state screen used 5 candidates, 3 seeds and 160 updates per seed. Four candidates passed the fixed internal gates. The Pareto candidate had the lowest combined future-field RMSE while keeping the largest state-block regression below 5%.

状态保护筛选使用 5 个候选、3 个种子和每种子 160 次更新。4 个候选通过固定内部门禁。Pareto 候选的合并后续场 RMSE 最低，且最大状态块退步低于 5%。

| Candidate / 候选 | Combined future field K / 合并后续场 | Max state regression / 最大状态退步 | Decision / 决策 |
|---|---:|---:|---|
| Coordinate reference / 坐标参考 | 0.998 | reference | Reference |
| Fluid-weighted state guard / 流体加权状态保护 | 0.797 | 4.23% | Passed |
| Late-stage guard / 后期阶段保护 | 0.796 | 4.14% | Passed |
| Pareto balance / Pareto 平衡 | **0.776** | **3.50%** | Selected |
| Strong state guard / 强状态保护 | 0.795 | 4.92% | Passed |

## Confirmation and third-cycle audit / 确认与第三循环审计

The selected candidate was retrained for 200 updates with all three seeds. Exact replay passed with zero difference. Its internal-validation combined future-field RMSE was **0.694 K**, with SMA entry **0.622 K** and HTF entry **0.208 K**.

选定候选使用 3 个种子重新训练 200 次更新。精确回放通过，最大差异为 0。内部验证合并后续场 RMSE 为 **0.694 K**，SMA 入口为 **0.622 K**，HTF 入口为 **0.208 K**。

The post-selection audit on the existing two-case third-cycle cohort produced c3s4 state RMSE of **1.088 K** for SMA, **0.184 K** for HTF2 and **0.305 K** for HTF4. Future-field RMSE was **1.523 K** for SMA and **0.817 K** for HTF. The audit is recorded after selection and does not change the candidate choice.

在已有两个第三循环工况上的选择后审计得到 c3s4 状态 RMSE。SMA 为 **1.088 K**，HTF2 为 **0.184 K**，HTF4 为 **0.305 K**。后续场 RMSE 为 SMA **1.523 K**、HTF **0.817 K**。该审计在候选选择后进行，不改变候选选择。

The candidate is retained as the protected-state research checkpoint. The public benchmark default remains the state-priority reference until the HTF trade-off, energy contract and independent-material boundaries are resolved.

该候选保留为状态保护研究检查点。公共基准默认仍使用状态优先参考方案，直到 HTF 取舍、能量契约和独立材料边界得到进一步证据。

## Evidence / 证据

- [Trade-off and stage figure](../assets/hro_protected_state/protected_state_tradeoff.svg)
- [Protected-state screening summary](../evidence/experiments/hro_protected_state_ablation_20261123/summary.json)
- [Protected-state confirmation summary](../evidence/experiments/hro_protected_state_confirmation_20261123/summary.json)
- [Third-cycle post-selection audit](../evidence/local_studies/hro_protected_state_confirmation_20261123/test_audit.json)
- [Stage-resolved confirmation metrics](../evidence/local_studies/hro_protected_state_confirmation_20261123/stage_metrics.json)
- [Energy boundary audit](HRO_ENERGY_BOUNDARY_AUDIT.md)
