# 使用说明

## 仅浏览成果

从 README、SHOWCASE 和 MODELS 开始。图像已经随仓库提交，无需运行代码。详细旧报告在 `SciML/docs/learning_closed_loop_report`，里面的原科学复现命令可能需要未打包的原始归档。

## 环境和最小演示

在 Python 3.12 虚拟环境安装根目录 `requirements.txt`。它是兼容范围声明，不是跨平台逐位复现保证；本次实际版本见 `evidence/current_environment.json`。CPU 即可，不需要 MATLAB、GPU、云服务或 API 密钥。

```bash
python -m pip install -r requirements.txt
python -m examples.showcase --output demo-output
python -m examples.train_scalar --output run-output --epochs 200 --seed 701
python -m examples.research_gallery --output demo-output-research
python -m SciML.evaluation.learning_exercises --output-root learning-exercises
```

- showcase：从本仓库受信检查点重放 DeepONet 和 UQ；由归档指标画比较图。跨批次 float64 最大绝对差检查为 1e-10，不是科学精度门限。
- train_scalar：在便携表上新训 3 成员、6 输出网络；不是把它冒充原 WP2 的 5 输出/5 成员基准。
- learning_exercises：八个微型参考题；运行通过不证明读者本人熟练掌握。

所有输出目录必须是新的；旧结果不覆盖。训练失败先看完整 traceback 和输入来源，不自动重启昂贵程序。

`research_gallery` 从已提交的支线结果生成九组 PNG/SVG，不重新训练。新增多样性选样实验可用 `python -m examples.active_diversity --output run-output-diversity` 完整运行，配置、输入权限与结果见 [实验说明](ACQUISITION_DIVERSITY.md)。

## 验证当前分发

```bash
python -m unittest SciML.tests.test_learning_coordinate_operator SciML.tests.test_learning_ensemble SciML.tests.test_learning_pinn_inverse SciML.tests.test_learning_pool_loop SciML.tests.test_learning_design_loop SciML.tests.test_learning_exercises -v
python -m examples.validate_bundle
```

GitHub Actions 执行相同的小测试和无动画重放。历史 292 项测试记录不是当前移植分发的测试数，当前结果单独保存。

## 继续研究

训练和审计源文件保留原模块名。若要重跑全量状态/场训练，需从原工作区恢复同版本数据集、请求 JSON、原始 MAT 与协议，然后核验摘要和路径；本仓库的小样本演示不能替代这些来源检查。

检查点是 pickle 容器形式的 PyTorch 文件，演示使用 `weights_only=True` 的加载实现。只使用自己信任来源、且通过清单哈希核对的检查点，不从陌生仓库下载后任意执行。
