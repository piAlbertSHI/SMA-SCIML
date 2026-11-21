# SMA SciML Learning

**从物理仿真数据到状态模型、神经算子与离线设计决策的可复现学习项目。**

Private development workspace · 63 existing single-cycle cases · PyTorch / scikit-learn · No MATLAB needed for the bundled demo

[成果展示](docs/SHOWCASE.md) · [五分钟上手](docs/QUICKSTART.md) · [模型架构](docs/MODELS.md) · [中文面试汇报](docs/INTERVIEW_ZH.md) · [后续路线](docs/ROADMAP.md)

![Fixed held-out temperature field animation](assets/temperature_evolution.gif)

上图来自固定测试案例的已训练 DeepONet 重放，蓝色虚线为预测、绿色实线为归档数值解。输入为真实阶段入口场；不是从循环请求独立生成全部四阶段。

## 做了什么

研究对象是压力驱动 NiTi 形状记忆合金薄圆板的四阶段弹热制冷循环。项目复用已有数值结果，比较性能回归、阶段状态推进和温度场预测，并把预测接到不确定性校准、两轮离线主动学习和候选回查。

| 模块 | 实现 | 代表结果与边界 |
|---|---|---|
| 性能代理 | Ridge、树模型、7→32→32→5 MLP、5 成员集成 | 三种子平均 COP RMSE：Ridge 0.0627、集成 0.0192 |
| 状态模型 | 414 维状态 + 12 维控制，四个 Ridge+神经残差接口 | 请求级自主末阶段 SMA RMSE：0.1260 → 0.0802 K |
| 场算子 | 20,898 参数 DeepONet、7,886 参数一维空间 Fourier、POD | 小数据下 POD 在 4 个场指标中的 3 个优于两个神经主模型；保留负结果 |
| 方向记忆 | 传递预测末场反对称分量，不读取未来真实场 | 固定种子 S4 SMA 级联 RMSE：1.6966 → 0.5927 K |
| 逆问题 | 热方程 PINN + 局部真实曲线有效速率反演 | 制造问题系数误差 0.0458%–0.1906%，不是 SMA 物性标定 |
| 决策闭环 | 5 成员 UQ、12→17→22 标签实验、3 个候选回查 | 复合策略改善 COP/SCP；随机策略的冷量误差更低 |

指标来自 2026-09 的归档学习实验。2026-11-21 本仓库重新组织了代码、便携演示与展示材料，没有新增物理仿真。当前测试集已用于开发，不是新的独立确认集。

## 快速运行

推荐 Python 3.12，在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m examples.showcase --output demo-output
python -m examples.train_scalar --output run-output --epochs 200
```

第一条演示命令加载已有权重，生成 6 张结果图、温度动画和 JSON；第二条用 37 个训练工况进行一次新的小规模神经训练。输出目录必须不存在。首次安装和 PyTorch 下载耗时与网络有关。

## 目录

```text
SciML/models/       原模型实现及必要依赖
SciML/active_learning/  标签查询、选点和候选验证
SciML/evaluation/   原审计/练习代码（部分需原始归档）
SciML/tests/        学习模块测试
SciML/docs/learning_closed_loop_report/  详细中文报告的可移植副本
checkpoints/       成功实验的 PyTorch 检查点
data/              63 例轻量表、固定场样本、历史预测与指标
examples/          可直接运行的重放、训练示例
assets/            本次生成的 PNG/GIF 展示
evidence/          导出来源、历史摘要、当前验证记录
docs/              使用说明、架构、数据卡、路线与公开清单
```

## 可信边界

这是已有数值数据上的学习项目，不是实物验证或工程安全认证。真实主算子包含硬初值，但没有完整 SMA PDE/能量闭合；PINN/PDE 教学分支明确区分制造解。COP 沿用归档的功分母定义，不能解读为设备净电输入效率。只提供有限池候选回查，不声称连续全局最优。

原始 MAT/HDF5、完整 MATLAB/Simulink 工程和大部分全量预测数组不在这份精简分发中。原始科学审计需要这些归档，不能把演示成功称作全套历史审计重跑。详见[数据与证据范围](docs/DATA_CARD.md)。

仓库保持 **private**，没有启用公开 GitHub Pages。未来公开需由所有者明确批准，并完成[公开检查清单](docs/PUBLIC_RELEASE_CHECKLIST.md)。当前未授予开源许可证。
