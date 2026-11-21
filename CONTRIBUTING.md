# Contributing / 开发规范

Changes should identify a concrete problem, the affected model or interface, and evidence that the implementation addresses it. Use focused branches and pull requests. Preserve existing experiment outputs and failed evidence; store each experiment in a new directory.

改动应明确问题、受影响的方法或接口，以及实现有效的证据。使用独立分支与聚焦的合并请求，保留原实验输出和失败证据，各实验写入新目录。

## Models and experiments / 模型与实验

Specify the dataset, permitted inputs, case-level split, baselines, seeds, compute budget, objective and selection rule before training. Prevent test information from entering preprocessing, fitting or model selection. Distinguish numerical software tests from scientific validation.

训练前固定数据集、允许输入、工况级划分、基线、种子、计算预算、目标及选择规则。预处理、拟合和模型选择不读取测试信息，区分软件测试与科学验证。

Run checks appropriate to the change. Model/interface changes need relevant contract tests and checkpoint replay; training changes need a documented experiment. Figure-label or documentation changes should preserve numerical inputs and metrics.

执行与改动相关的检查。模型或接口改动需验证接口及权重回放；训练改动需实验记录；图例或文档改动须保持数值输入与指标不变。

## Documentation and figures / 文档与图表

Follow the [nomenclature](docs/NOMENCLATURE.md). Address researchers and software users. Method names describe architecture or training, and figure captions identify data, metrics and aggregation. Generate numerical figures from recorded arrays; retain adverse results and uncertainty definitions.

遵循命名规范，面向研究者与软件使用者。方法名描述架构或训练，图注注明数据、指标与汇总方式。数值图来自记录数组，并保留不利结果和不确定性定义。

```bash
python -m tools.audit_presentation
python -m tools.finalize_bundle
python -m examples.validate_bundle
```

Review changes before rebuilding the distribution manifest. Finalization normalizes text to LF and records the current environment; do not use it to hide unexplained hash mismatches. Commit the reviewed manifest with its corresponding files.

重建分发清单前审查改动。清单生成统一 LF 并记录环境，不用于掩盖不明哈希差异；清单与对应文件一并提交。

## Distribution / 分发

Access is restricted during release preparation. Code, data and weights do not yet carry an open-source redistribution license. Visibility changes and licensing require the repository owner's explicit authorization.

发布准备期间限制访问。代码、数据与权重尚未获得开源再分发许可，仓库可见性及许可变更需所有者明确授权。
