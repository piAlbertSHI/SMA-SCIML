# 使用说明

## 仅浏览成果

从 README、SHOWCASE 和 MODELS 开始。图像已经随仓库提交，无需运行代码。详细旧报告在 `SciML/docs/learning_closed_loop_report`，里面的原科学复现命令可能需要未打包的原始归档。

新主模型的架构、公式和运行状态见 [HIERARCHICAL_OPERATOR](HIERARCHICAL_OPERATOR.md)。README 聚焦模型与场演示，原统计图保留于 [RESULTS_GALLERY](RESULTS_GALLERY.md)。克隆后可直接用浏览器打开 `assets/operator_showcase_v2/explorer.html`，无需服务器；本环境只完成模拟 DOM/canvas 逻辑检查，真实浏览器验收待完成。

## 环境和最小演示

在 Python 3.12 虚拟环境安装根目录 `requirements.txt`。它是兼容范围声明，不是跨平台逐位复现保证；本次实际版本见 `evidence/current_environment.json`。CPU 即可，不需要 MATLAB、GPU、云服务或 API 密钥。

```bash
python -m pip install -r requirements.txt
python -m examples.showcase --output demo-output
python -m examples.train_scalar --output run-output --epochs 200 --seed 701
python -m examples.research_gallery --output demo-output-research
python -m examples.multicycle_multistep --verify-only
python -m examples.multicycle_gallery --output demo-output-multicycle
python -m examples.verify_hierarchical
python -m examples.train_hierarchical --output run-output-v13 --updates 3 --batch-cases 2 --lr 0.0001
python -m examples.operator_showcase --output demo-output-operator
python -m SciML.evaluation.learning_exercises --output-root learning-exercises
```

- showcase：从本仓库受信检查点重放 DeepONet 和 UQ；由归档指标画比较图。跨批次 float64 最大绝对差检查为 1e-10，不是科学精度门限。
- train_scalar：在便携表上新训 3 成员、6 输出网络；不是把它冒充原 WP2 的 5 输出/5 成员基准。
- learning_exercises：八个微型参考题；运行通过不证明读者本人熟练掌握。

所有输出目录必须是新的；旧结果不覆盖。训练失败先看完整 traceback 和输入来源，不自动重启昂贵程序。

`research_gallery` 从已提交的支线结果生成九组 PNG/SVG，不重新训练。新增多样性选样实验可用 `python -m examples.active_diversity --output run-output-diversity` 完整运行，配置、输入权限与结果见 [实验说明](ACQUISITION_DIVERSITY.md)。

The multi-cycle comparison can be checked with `python -m examples.multicycle_multistep --verify-only`: nine model/decoder combinations are replayed and saved metrics recomputed. To repeat the six fixed-budget continuation fits on the bundled data, use `python -m examples.multicycle_multistep --output run-output-multicycle`. No MATLAB or physical solver is called. See [architecture, protocol and results](MULTICYCLE_MULTISTEP.md).

多循环验证命令回放九组模型/解码组合并重新计算保存指标。用 `python -m examples.multicycle_multistep --output run-output-multicycle` 可在新目录仅依靠仓库数据重新完成六次固定预算续训，不调用 MATLAB 或物理求解器；详见中英文模型与实验说明。

## 验证当前分发

```bash
python -m unittest SciML.tests.test_learning_coordinate_operator SciML.tests.test_learning_ensemble SciML.tests.test_learning_pinn_inverse SciML.tests.test_learning_pool_loop SciML.tests.test_learning_design_loop SciML.tests.test_learning_exercises SciML.tests.test_acquisition_diversity SciML.tests.test_multistep_historical SciML.tests.test_hierarchical_operator -v
node examples/test_field_explorer.cjs
python -m examples.validate_bundle
```

GitHub Actions 执行相同的小测试和无动画重放。历史 292 项测试记录不是当前移植分发的测试数，当前结果单独保存。

## 继续研究

训练和审计源文件保留原模块名。本次多步状态续训可直接使用仓库中的封存快照；若要重新构建全量状态/场原始数据，仍需恢复同版本数据集、请求 JSON、原始 MAT 与协议，然后核验摘要和路径。快照验证不替代新的原始来源审核。

检查点是 pickle 容器形式的 PyTorch 文件，演示使用 `weights_only=True` 的加载实现。只使用自己信任来源、且通过清单哈希核对的检查点，不从陌生仓库下载后任意执行。
