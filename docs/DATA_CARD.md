# 数据与证据范围

主来源为2026-09归档的63个同版本有效单循环数值工况。半径32–42 mm、厚度160–210 µm，R/t恒200；有recA/recAs两种配置。37/13/13按完整工况切分，原测试集只有recA，不可当跨材料证据。全体均为有效工况，不能训练有正负证据的失败分类器。

| 打包内容 | 用途 | 不代表什么 |
|---|---|---|
| scalar63.npz | 7特征、6标签、固定split和ID；可新训标量网络 | 完整数值数据集或独立未查看测试集 |
| field_demo.npz | 原顺序第一个test stage4，branch/初场/真值/旧预测 | 全测试集评价或请求级自主场预测 |
| uq_predictions.npz | 保存的5成员预测和区间 | 真实观测噪声分解 |
| historical12_state.npz | 历史12工况的118个状态转移、初态/控制和20个测试场；支持六次多步/单步续训 | 新的独立测试集、当前63版本或重新核验的原始MAT |
| historical12_training_fields.npz | 同一历史版本八个训练工况的32段换热场，支持新架构联合训练 | 测试标签、新仿真或已完成新模型精度验证 |
| checkpoints | 成功实验权重；排除失败wp3-r1权重 | 每个检查点都能仅靠小样本示例复现其训练 |
| evidence/archive | 原协议、结果、诊断、历史审计的路径清理副本 | 原产物清单可直接用于校验本分发 |
| 详细报告副本 | 保留解释，改写可用链接，标注未打包来源 | 原19文件的字节相同副本 |

不打包原始MAT、大型HDF5、完整仿真运行目录、论文审稿资料、Git历史、凭据和其他项目。本次导出未移动或删除源文件；历史12工况的原始MAT路径当前不可用，可用性如实记录于新数据清单。导出映射保存原/导出摘要，原摘要不能拿来匹配改写过的副本。`evidence/bundle_manifest.json` 是本分发自己的完整性清单。

当前COP标签的分母使用归档的加载功+回收功量+泵功口径，不能称净电效率。结构化OOD、独立历史12例多循环、制造热方程都是分开记录的任务。真实SMA完整能量读出仍缺独立端口/储能信息；不填零、不把残差定义为某个储能后自证闭合。

## Historical multi-cycle snapshot / 历史多循环快照

The portable NPZ preserves the fixed eight training / four test cases. All training cases have two cycles; both three-cycle cases remain in test. State shapes are `(118,515)` before/after and `(12,515)` initial, with nine controls per case; training/test transitions total 72/46. The 20 field targets are test-only, shape `(20,2,50,101)`; no field decoder is trained in the new comparison.

便携 NPZ 保留八训练/四测试的固定工况划分。训练工况均为两循环，两个三循环工况均属于测试。前/后态为 `(118,515)`，初态为 `(12,515)`，各工况有九个控制量；训练/测试分别为72/46个转移。20段温度场仅为测试目标，形状 `(20,2,50,101)`，本次不训练场解码器。

The original HDF5 artifact manifest and all 12 request hashes were checked. Every derived after-state and sample ID matches the historical saved training/test target arrays exactly. Raw MAT availability is explicitly false in this snapshot's provenance; the old quality reports are preserved, not reissued as a fresh raw-source audit. The two test cases have already informed development, limiting claims about independent generalization.

已核验原 HDF5 产物清单及12份请求哈希，全部派生后态与样本ID均与原训练/测试目标数组完全一致。清单明确记录原始 MAT 不可用；保留旧质量报告，不将其重新签发为新的原始来源审核。两个三循环测试工况已经参与开发分析，不能据此声称独立泛化验证。

[Snapshot provenance / 快照来源](../data/historical12_state.json) · [Extraction code / 提取代码](../examples/export_multicycle.py) · [Experiment / 实验说明](MULTICYCLE_MULTISTEP.md)

The additional training-field snapshot contains `(32,2,50,101)` temperatures from the same verified HDF5. Its case IDs exactly match the original eight training cases; field-entry reconstruction agrees with the portable state snapshot within 1e-9 K. Source manifest, HDF5 and both portable snapshots are bound by hashes. No raw MAT re-audit or new physics run was performed.

新增训练场快照为 `(32,2,50,101)` 温度数组，来自同一已核验 HDF5，工况 ID 严格对应原八个训练工况，初场重构与状态快照在 1e-9 K 容差内一致。清单绑定源 manifest、HDF5 和两份便携快照哈希，没有重新审核原始 MAT 或运行新物理仿真。

[Training-field provenance / 训练场来源](../data/historical12_training_fields.json) · [Export code / 提取实现](../examples/export_training_fields.py)
