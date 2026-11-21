# Nomenclature and presentation / 命名与展示规范

| Entity / 对象 | English | 中文 |
|---|---|---|
| Project / 项目 | **SMA-SCIML** | 面向弹热制冷的科学机器学习 |
| Joint state/field model / 联合状态与场模型 | **Hierarchical Recurrent Operator (HRO)** | **分层循环算子** |
| Decoder variants / 解码器方案 | Coordinate HRO; modal HRO | 坐标 HRO；模态 HRO |
| Stage-2 mean readout / 第二阶段均温读出 | HRO with error-scaled thermal response | 误差收缩热响应 HRO |
| Modal training / 模态训练 | Two-phase training: joint trajectory optimization and cached decoder refinement | 双阶段训练：轨迹联合优化与缓存解码器专项训练 |
| State baseline / 状态基线 | Residual MLP state model | 残差 MLP 状态模型 |
| State/field baseline / 状态与场基线 | Residual MLP + DeepONet; shortened to **residual cascade** in figures | 残差级联系统 |
| Baseline training variants / 基线训练方案 | One-step training; one-step fine-tuning; multi-step fine-tuning | 单步训练；单步微调；多步微调 |
| Linear state reference / 线性状态参考 | Affine state baseline | 仿射状态基线 |
| Field references / 场参考 | DeepONet; spatial Fourier operator; POD/Ridge | DeepONet；空间 Fourier 算子；POD/Ridge |
| Scalar predictor / 标量预测器 | Performance surrogate ensemble | 性能代理集成 |
| Data configurations / 数据配置 | Configuration A (recA); Configuration B (recAS) | 配置 A；配置 B |

Use the full method name at first mention and the defined acronym afterward. Figure legends may use “HRO” and “Residual cascade” when the caption defines both. Name training variants explicitly; do not substitute a development sequence number for a method. Software release versions remain appropriate in dependency specifications and immutable provenance.

首次出现使用完整方法名，后续可使用定义过的缩写。图例可以使用 HRO 和 Residual cascade，但图注须给出完整名称。训练方案应明确命名，不能以开发序号代替方法名称。软件发布版本仍可用于依赖约束和不可变溯源记录。

Documents address researchers and software users: explain the problem, mathematical or computational method, evaluation conditions, results, limitations and usage. Keep personal preparation notes, task status and tooling-specific execution commentary out of maintained project pages.

文档面向研究者与软件使用者，说明问题、方法、评价条件、结果、限制及使用方式。个人准备笔记、任务进度和特定执行工具的过程说明不进入维护中的项目页面。

Results must identify the dataset, cohort, baseline, metric, aggregation and input information. Include adverse results and uncertainty definitions. Preserve numerical arrays and original experiment records when revising wording or figure labels.

结果应明确数据集、工况组、基线、指标、汇总方式与输入条件，并保留不利结果和不确定性定义。修改措辞或图例时保留数值数组及原实验记录。

Repository: `SMA-SCIML`. Python modules use lowercase identifiers; documentation filenames use descriptive uppercase names; demonstration assets use stable descriptive filenames. Historical import paths and artifact IDs remain stable for checkpoint and audit compatibility.

仓库名使用 `SMA-SCIML`；Python 模块遵循小写命名惯例，文档使用描述性大写文件名，演示素材使用稳定描述性文件名。历史导入路径及产物编号保持稳定，以兼容检查点和审计记录。
