# Data card / 数据说明

## Datasets / 数据集

| Dataset / 数据集 | Coverage and split / 覆盖与划分 | Use / 用途 |
|---|---|---|
| Single-cycle numerical corpus / 单循环数值数据 | 63 cases; 37 train, 13 calibration, 13 test; radius32–42 mm, thickness160–210 µm; R/t=200 | Performance surrogates, state/field comparisons, calibration and offline acquisition / 性能代理、状态与场对照、校准及离线选样 |
| Historical multi-cycle corpus / 历史多循环数据 | 12 cases; 8 two-cycle train, 4 test including 2 three-cycle cases; 72/46 training/test transitions | Residual-cascade and HRO benchmarks / 残差级联与分层循环算子基准 |
| Historical loading fields / 历史加载场 | 200 distinct single-cycle conditions with a fixed 140/30/30 split and 54 matched conditions in the revised single-cycle archive / 200 个独立单循环工况，固定 140/30/30 划分，54 个工况在修订单循环归档中有配对 | First-loading displacement, deflection, temperature and martensite-fraction benchmarks / 首循环加载位移、挠度、温度和马氏体相分数基准 |

Each corpus has its own configuration history and validation scope. Configuration A/B in the field demonstrations maps to the archived material indicators recA/recAS. The 63-case test partition contains only recA; it does not establish cross-material generalization. Radius and thickness co-vary at fixed R/t, limiting geometric extrapolation claims.

各组数据具有各自的配置历史和验证范围。场演示中的配置 A/B 对应归档材料标识 recA/recAS。63 工况测试分区仅含 recA，不能证明跨材料泛化；半径与厚度按固定 R/t 联动，几何外推结论因此受限。

## Distributed snapshots / 分发快照

| File or collection / 文件或集合 | Contents / 内容 |
|---|---|
| `data/scalar63.npz` | Seven inputs, six labels, fixed case IDs and split / 七输入、六标签、固定工况 ID 与划分 |
| `data/field_demo.npz` | Fixed stage-4 test example with observed entry fields, reference and prediction / 固定第四阶段示例的观测初场、参考与预测 |
| `data/uq_predictions.npz` | Five-member predictions and calibrated intervals / 五成员预测与校准区间 |
| `data/historical12_state.npz` | 118 transitions of515 states, 12 initial states, nine controls per case; 20 test exchange fields / 118个515维状态转移、12个初态、九维控制及20段测试换热场 |
| `data/historical12_training_fields.npz` | 32 exchange fields from the eight training cases; shape32×2×50×101 / 八个训练工况的32段双通道换热场 |
| [Loading-field snapshot / 加载场快照](../evidence/local_studies/loading_field_benchmark_20261121/loading_data.npz) | Controls, fixed case splits and 200×4×50×51 loading fields from one archive / 单一来源的控制量、固定工况划分及 200×4×50×51 加载场 |
| `checkpoints/` and experiment directories / 权重及实验目录 | Saved models, protocols, metrics and training records / 模型、协议、指标及训练记录 |

The residual-cascade fine-tuning study freezes its field decoder. HRO jointly trains the state and field branches using the 32 training exchange fields. The 20 test fields are used only for evaluation. Initial model inputs contain the first mechanical observation and declared thermal defaults; autonomous rollout never consumes later observed states.

残差级联微调冻结场解码器；HRO 使用32段训练换热场联合训练状态与场分支。20段测试场仅用于评价。模型初始输入为首个力学观测及声明的热初值，自主推进不读取后续真实状态。

## Provenance / 来源

The sealed source HDF5 artifact manifest and all12 request hashes were checked. All118 derived after-states and sample IDs match the archived target arrays exactly. Training-field IDs match the original training split; field-entry reconstruction agrees with the state snapshot within1e-9 K. The raw MAT paths are unavailable, so this snapshot verification is not a fresh raw-source audit.

已核验封存 HDF5 产物清单及12份请求哈希；118个派生后态与样本 ID 均和归档目标数组完全一致。训练场 ID 对应训练分区，初场重构与状态快照在1e-9 K内一致。原始 MAT 路径不可用，因此快照核验不等于新的原始来源审核。

[State snapshot provenance / 状态快照来源](../data/historical12_state.json) · [Training-field provenance / 训练场来源](../data/historical12_training_fields.json) · [Distribution manifest / 分发清单](../evidence/bundle_manifest.json)

Original MAT files, full HDF5 archives, solver run directories and unrelated manuscript materials are excluded. Historical source IDs and implementation versions are retained in machine-readable provenance for traceability. The distribution manifest binds the included bytes; it is not a digital signature or a scientific-validity certificate.

不分发原始 MAT、完整 HDF5、求解器运行目录及无关论文资料。来源编号和实现版本保留在机器可读溯源记录中。分发清单绑定文件字节，不构成数字签名或科学有效性认证。

## Interpretation and limitations / 解释与限制

- All labels are numerical. No physical experiment is represented as validation. / 标签均来自数值结果，不作为实体实验验证。
- The historical test cohort has informed development and covers two three-cycle cases. Independent horizon validation remains open. / 历史测试工况参与过开发分析，仅覆盖两个三循环工况。独立的循环长度验证仍待完成。
- All63 cases satisfy the recorded reading/quality criteria; the corpus does not provide both success/failure classes for failure detection. / 63工况符合记录的读取与质量条件，但不同时提供成功、失败两类标签。
- COP uses the archived loading-work, recovered-work quantity and pumping-work convention; it is not net electrical efficiency. / COP 沿用归档加载功、回收功量及泵功口径，不等于净电效率。
- Complete SMA energy accounting needs independent port and storage information. / 完整 SMA 能量核算仍需独立端口与储能信息。
- Data and weight redistribution rights require an explicit release license. / 数据及权重再分发须有明确发布许可。

See the project status for the additional solver archives, exact condition overlap and version-specific evidence. / 其余求解器归档、条件重叠及版本范围见项目进度。[Cross-archive inventory / 跨档案核验](PROJECT_STATUS.md#两项现有数据核验)

The loading snapshot matches its source HDF5. Registered pairs agree pointwise during loading and differ during exchange. / 加载快照与来源 HDF5 完全一致。配对工况在加载阶段逐点一致，在换热阶段有差异。 [Study and provenance / 基准与溯源](LOADING_FIELD_STUDY.md)
