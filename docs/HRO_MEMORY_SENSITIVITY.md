# HRO memory sensitivity / HRO 记忆敏感性

This is an inference-only ablation of the three archived formal HRO checkpoints. The persistent GRU memory is compared with a path that resets the memory argument to zero before every stage. Neural weights, scalers, affine priors, test cases and evaluation grids remain unchanged. No training or physical simulation was run.

本研究只对三个正式 HRO 检查点做推理消融。对照路径在每个阶段前将 GRU 记忆输入重置为零，与持久记忆路径比较。神经权重、归一化、仿射先验、测试工况和评价网格均保持不变，没有训练或物理仿真。

## Cycle-3/stage-4 results / 第三循环第四阶段结果

Values are pooled over the two historical three-cycle cases within each seed, then averaged across the three fixed seeds. Field errors exclude the stage-entry frame.

数值先在每个种子内汇总两个历史三循环工况，再对三个固定种子取均值。场误差排除阶段入口帧。

| Metric / 指标 | Persistent memory / 持久记忆 | Memory reset each stage / 每阶段重置 | Reset minus persistent / 差值 |
|---|---:|---:|---:|
| SMA state RMSE K / SMA 状态 | 1.120 | 1.110 | -0.010 |
| HTF2 state RMSE K / HTF2 状态 | 0.242 | 0.237 | -0.005 |
| HTF4 state RMSE K / HTF4 状态 | 0.334 | 0.351 | +0.017 |
| SMA field RMSE K / SMA 场 | 1.600 | 1.414 | **-0.186** |
| HTF field RMSE K / HTF 场 | 0.775 | 0.909 | **+0.134** |

Resetting memory improves the archived SMA field score and slightly changes the SMA state score. It worsens the HTF field and HTF4 state scores. The result is a channel-specific trade-off, not evidence that the memory module is globally harmful or globally necessary.

重置记忆改善归档 SMA 场误差，并小幅改变 SMA 状态误差，但会使 HTF 场和 HTF4 状态误差变差。结果表现为通道相关的取舍，不能据此断言记忆模块整体有害或整体必需。

## Evidence boundary / 证据边界

The comparison changes the inference path without retraining the no-memory path. It is therefore a sensitivity diagnostic for the learned representation. It does not isolate causal module contribution under matched training, and it does not justify replacing the persistent-memory checkpoint. A future matched-budget study should retain separate SMA and HTF objectives and evaluate restart stages explicitly.

该比较只改变推理路径，没有重新训练无记忆模型，因此属于已学习表示的敏感性诊断。它没有在同预算训练条件下隔离模块因果贡献，也不能据此替换持久记忆检查点。后续同预算研究应分别保留 SMA 与 HTF 目标，并显式评价重启阶段。

The full per-seed and per-case record is [summary.json](../evidence/local_studies/hro_memory_sensitivity_20261123/summary.json), with saved memory-reset predictions under the same evidence directory. The replay difference for all three archived persistent-memory checkpoints is exactly zero.

完整的逐种子、逐工况记录见 [summary.json](../evidence/local_studies/hro_memory_sensitivity_20261123/summary.json)，记忆重置预测保存在同一证据目录。三个归档持久记忆检查点的回放差异均为 0。

```bash
python -m examples.audit_hro_memory_sensitivity \
  --output evidence/local_studies/hro_memory_sensitivity_20261123
```
