# HRO spatial supervision study / HRO 空间监督研究

This study addresses the spatial oscillations observed in the coordinate Hierarchical Recurrent Operator. The existing model samples 26 radial positions during each update, while the evaluation grid contains 101 positions. The study adds 51-point supervision and an optional supervised curvature term. All runs reuse the historical state and field snapshots. No physical solver was called.

本研究针对坐标分层循环算子中发现的空间振荡。原训练每次更新取 26 个径向点，评价网格包含 101 个点。本研究改为 51 点监督，并加入可选的有监督空间曲率项。所有运行复用历史状态与场快照，没有调用物理求解器。

## Model and objective / 模型与目标函数

The reference HRO has 3,106,989 trainable parameters. Separate dilated radial encoders feed 16 latent tokens, three eight-head attention blocks, a GRU with 256 memory variables, five FiLM-conditioned transition heads and a coordinate temperature decoder. The explicit state has 515 variables. The affine prior, state normalization and projection are frozen.

参考 HRO 包含 3,106,989 个可训练参数。两个带膨胀卷积的径向编码器产生 16 个潜变量 token，经三层八头注意力、256 维 GRU 记忆、五个 FiLM 条件状态头和坐标温度解码器推进。显式状态为 515 维。仿射先验、状态归一化和投影规则保持冻结。

The ablation objective is

\[
  \mathcal{L}=\mathcal{L}_{state}+\mathcal{L}_{field}+0.01\mathcal{L}_{grad}+\lambda_c\mathcal{L}_{curv}.
\]

The curvature term compares predicted and target second radial derivatives. Its error is divided by the RMS target curvature computed from the training batch, with a lower bound of one. Optional channel weights can focus the term on the SMA field and separately increase the fluid-field value and gradient weights. The term is disabled by default, so existing HRO training calls preserve their original objective.

曲率项比较预测场与目标场的径向二阶导数。误差除以训练批次目标曲率的 RMS，尺度下限为 1。可选通道权重可将曲率项集中到 SMA 场，并单独提高流体场值和梯度的权重。默认关闭该项，现有 HRO 训练入口保持原目标函数。

The query-jitter branch keeps 51 nominal radial points and perturbs each interior query by at most half of the native 0.01 radial spacing. The archived 101-point truth is linearly interpolated at those coordinates. The endpoints remain exact, and the model still enforces the learned stage-entry field construction.

查询抖动分支保留 51 个名义径向点，将内部查询点扰动到原生 0.01 径向间距的一半以内，并在这些坐标上对归档 101 点真值做线性插值。两个端点保持精确，模型仍使用学习到的阶段入口场构造。

| Setting / 设置 | Reference HRO / 参考 HRO | Spatial ablation / 空间消融 |
|---|---:|---:|
| Training cases / 训练工况 | 8 | 8 |
| Historical test cases / 历史测试工况 | 4, including 2 three-cycle cases / 4，其中 2 个三循环 | 4, including 2 three-cycle cases / 4，其中 2 个三循环 |
| Updates per seed / 每种子更新 | 200 | 200 |
| Seeds / 种子 | 20261121, 20261122, 20261123 | Screening runs use 20261121 |
| Radial queries per update / 每次径向查询 | 26 | 51 |
| Query geometry / 查询几何 | Fixed native subset / 固定原生子集 | Fixed or half-spacing jitter / 固定或半间距抖动 |
| Field evaluation grid / 场评价网格 | 50 × 101 | 50 × 101 |
| Fresh solver calls / 新物理求解 | 0 | 0 |

## Third-cycle results / 第三循环结果

The primary endpoint is cycle 3, stage 4. Field errors omit the exact stage-entry frame. The reference is a three-seed result. Candidate rows are one-seed 200-update diagnostic runs and are reported without a standard deviation.

主指标为第三循环第四阶段。场误差排除精确的阶段入口帧。参考结果来自三个种子。候选行为单种子、每种子 200 次更新的诊断运行，没有报告标准差。

| Objective / 目标 | SMA state RMSE K | HTF2 state RMSE K | HTF4 state RMSE K | SMA field RMSE K | HTF field RMSE K |
|---|---:|---:|---:|---:|---:|
| Coordinate HRO reference / 坐标 HRO 参考 | 1.120 ± 0.061 | 0.242 ± 0.021 | 0.334 ± 0.022 | 1.600 ± 0.118 | 0.775 ± 0.020 |
| Solid curvature, equal field channels / 固体曲率、等通道场权重 | 1.051 | 0.223 | 0.276 | 1.255 | 0.986 |
| Solid curvature, fluid weight 2 / 固体曲率、流体权重 2 | 1.072 | 0.226 | 0.281 | 1.263 | 0.949 |
| Solid curvature, fluid weight 4 / 固体曲率、流体权重 4 | 1.080 | 0.226 | 0.288 | 1.337 | 0.934 |
| Staged curvature activation / 分阶段开启曲率 | 1.727 | 0.183 | 0.481 | 2.263 | 1.093 |
| Query jitter, equal field channels / 查询抖动、等通道场权重 | 1.054 | 0.189 | 0.364 | 1.486 | 0.932 |
| Query jitter, fluid weight 2 / 查询抖动、流体权重 2 | **1.032** | **0.161** | 0.379 | 1.561 | 0.854 |

Solid-focused curvature reduces third-cycle SMA state and SMA field error in the fixed-seed runs. Query jitter gives a better state/field compromise, and fluid weight 2 further reduces HTF error while retaining the state improvement. The staged schedule does not preserve state accumulation and is retained as a rejected diagnostic.

固体曲率监督在固定种子运行中同时降低第三循环 SMA 状态和 SMA 场误差。查询抖动带来更平衡的状态—场取舍，流体权重 2 进一步降低 HTF 误差，同时保留状态改善。分阶段方案没有保持状态累积精度，作为拒绝的诊断结果保留。

## Decision and evidence boundary / 决策与证据边界

The current coordinate-HRO checkpoint remains the reference model. The new loss, channel-weighting and query-jitter controls are implemented, unit-tested and available for a future field-balanced confirmation. The candidate rows use one test-informed historical cohort and therefore do not support a public performance claim or a default-checkpoint replacement. A three-seed 200-update confirmation would require an additional CPU budget of roughly eight minutes and has not been started.

当前坐标 HRO 检查点继续作为参考模型。新的曲率损失、通道权重和查询抖动控制已经实现并通过单元测试，可用于后续场平衡确认。候选行使用已参与开发的历史测试队列，现阶段不能支持公开性能结论，也不替换默认检查点。三种子、每种子 200 次更新的确认预计需要约 8 分钟 CPU 预算，目前尚未启动。

The compact study record is [summary.json](../evidence/local_studies/hro_spatial_supervision_20261123/summary.json), with source and raw-run hashes in [manifest.json](../evidence/local_studies/hro_spatial_supervision_20261123/manifest.json). Full raw checkpoints remain in the local `evidence/experiments/hro_spatial_ablation_*_20261123` directories for replay. The curated record keeps the large exploratory checkpoints outside the Git commit.

紧凑研究记录见 [summary.json](../evidence/local_studies/hro_spatial_supervision_20261123/summary.json)，源文件和原始运行哈希见 [manifest.json](../evidence/local_studies/hro_spatial_supervision_20261123/manifest.json)。完整检查点仍保留在本地 `evidence/experiments/hro_spatial_ablation_*_20261123` 目录中，可用于回放。整理后的记录将大型探索检查点放在 Git 提交之外。

## Reproduction and verification / 重现与核验

```bash
python -m unittest SciML.tests.test_hierarchical_operator SciML.tests.test_hro_spatial_ablation
python -m examples.hro_spatial_ablation --output evidence/experiments/hro_spatial_ablation_fluid4_long_20261123 --verify-only
```

The verifier reloads the self-contained checkpoint, recomputes all test predictions and scores, and checks exact saved-array replay. The recorded replay difference is zero. The study used no new physical simulations.

核验器重新加载自包含检查点，重算全部测试预测与指标，并检查保存数组的精确回放。记录的回放差异为 0。本研究没有新增物理仿真。
