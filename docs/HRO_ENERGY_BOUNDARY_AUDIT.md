# HRO energy boundary audit / HRO 能量边界审计

The current HRO state and field snapshots contain 515-dimensional states, nine controls and two-channel temperature fields on the archived exchange stages. A machine-readable audit confirms the information available to the predictor and the terms still absent from the field-to-energy contract.

当前 HRO 状态和场快照包含 515 维状态、9 个控制量以及归档换热阶段上的双通道温度场。机器可读审计确认了预测器可用的信息和场到能量契约仍缺失的项。

## Available evidence / 已有证据

- State snapshot `data/historical12_state.npz` contains initial states, controls, state transitions, stage labels and field truth references.
- Training field snapshot `data/historical12_training_fields.npz` contains 32 two-channel exchange fields with shape `32 × 2 × 50 × 101`.
- The archived 63-case field-energy audit has exact export identities for stored `Qcool`, `Wpump` and COP labels.
- The maximum internal endpoint-to-port temperature gap remains about 13 K, so endpoint ownership cannot be inferred from a temperature field alone.

- 状态快照 `data/historical12_state.npz` 包含初始状态、控制量、状态转移、阶段标签和场真值索引。
- 训练场快照 `data/historical12_training_fields.npz` 包含 32 段双通道换热场，形状为 `32 × 2 × 50 × 101`。
- 归档 63 工况场—能量审计已核验存储 `Qcool`、`Wpump` 和 COP 标签的导出恒等关系。
- 内部端点到端口温差最大约 13 K，因此不能仅从温度场推断端口归属。

## Missing contract terms / 缺失契约项

The machine-readable audit records the following missing terms.

机器可读审计记录以下缺失项。

1. Realized signed mass flow and pressure-command trajectories.
2. Port-temperature ownership and internal pipe-fluid treatment.
3. Mechanical work convention and capacity/latent-energy/restart convention.
4. Independent environment heat and an independent closure residual.

The audit therefore keeps the energy gate open and keeps the HRO checkpoint as a temperature state–field research result. It does not convert field RMSE into autonomous COP evidence.

因此审计保持能量门未闭合，并将 HRO 检查点定位为温度状态—场研究结果。场 RMSE 不转换为自主 COP 证据。

The record is stored in [audit.json](../evidence/local_studies/hro_energy_boundary_20261123/audit.json).
