# 模型架构与训练设置

**Current development / 当前新增主模型:** [Hierarchical recurrent state-space operator](HIERARCHICAL_OPERATOR.md), **3,106,989 trainable parameters**, dual spatial CNNs, latent cross-attention, GRU memory, five FiLM transition heads and a joint coordinate field decoder. Implemented and execution-checked; accuracy evaluation pending. The tables below document the trained reference models.

当前新增主模型采用双路空间 CNN、潜变量交叉注意力、GRU 记忆、五个 FiLM 阶段头及联合坐标场解码，约 **311 万可训练参数**。已实现并完成运行验证，尚未评价精度；下表继续记录已有参考模型。

`a→b` 表示全连接层维度，除注明外在 CPU float64 上训练。下面列的是实际已运行配置，不是拟议架构。

| 模型 | 输入与输出 | 架构 | 训练 |
|---|---|---|---|
| WP2 MLP | 7 个工况特征 → 5 性能标量 | 7→32 SiLU→32 SiLU→5，1,477 参数；ensemble=5 | 全批 AdamW lr=.002，wd=1e-4，300 更新，clip=10；3 seeds |
| WP3 | 414 状态+12 控制 → 活动增量 | 每接口独立 Ridge(alpha=1)+426→32→32→d，SiLU；d=204/313/313/160 | 300 全批更新，AdamW .002/1e-4，clip10；残差 RMS≤1e-10 时取消该通道神经修正 |
| DeepONet | 221 branch + (q,t) → SMA/HTF 温度 | branch 204/9/8 各经32→32→32，Tanh；拼接96→48；trunk2→32→32→48；每场latent24；20,898参数 | 400 AdamW更新，lr=.002，wd=1e-5；batch16阶段×6时刻×51位置 |
| Fourier | 两入口场+位置+时间、221全局条件 → 两场 | lift4→12；两层spectral+local，8低频模式；GELU；投影12→16→2；7,886参数 | 与DeepONet同监督/预算；只沿空间FFT，不是二维时空FNO |
| POD/Ridge | 相同 branch → 粗网格场系数 | 每场4个SVD模态，Ridge alpha1 | 相同26×51监督；固定基底，非神经网络 |
| PINN制造逆问题 | (x,t) → 温度，同时拟合a | 2→24 Tanh→24 Tanh→1 + log(a)，总698参数 | 12观测、128 Sobol配点；loss=10data+PDE；Adam1800 lr.003 + LBFGS≤150 |
| 局部逆ODE | t和已知HTF forcing → SMA温度/有效k | 1→24 Tanh→24 Tanh→1 + log(k)，总674参数 | 8观测、128等距配点；Adam1200 lr.003 + LBFGS≤100；loss=10data+ODE |
| UQ | 同7特征 → 6目标（增加Wtotal） | 5×[7→32 SiLU→32 SiLU→6]；每成员1,510参数 | 同训练数据，不bootstrap；300更新AdamW .002/1e-4；13例conformal校准 |
| 主动学习 | 已揭示标签 → 候选分数 | 每轮3个6输出MLP，200更新 | 12→17→22；random/uncertainty/composite×3seeds；最终22模型用于设计 |

DeepONet 的输出为 `T0_interp + s*(1-exp(-4*tau))*N`，严格承接入口初值。真实模型没有完整 PDE 残差，不能称为 PINN。输入221维=两场去均值202+均温2+stage2+material2+控制5+side8。

场监督为26个时刻×51位置，评价为50×101原生导出网格；新坐标查询不等于已有新求解器网格真值。状态模型先给入口边界，再由场模型解码；方向记忆单向传递，不是全系统端到端可微耦合。

原参数设置及权重分别在 `evidence/archive/*/protocol.json`、`checkpoints/`；完整讲解见 INTERVIEW_ZH 和旧报告第04–11章。

新增主动选样扩展保持原网络、三个种子和 12→17→22 预算，增加“0.5 复合分数 + 0.5 最近输入距离排名”的贪心批次策略。输入尺度仅由已揭示训练样本拟合，选点不读取未揭示标签。四策略 × 三种子已全部运行，协议和结果见 [多样性实验](ACQUISITION_DIVERSITY.md)。

## Historical multi-cycle extension / 历史多循环扩展

This separate historical model carries **515 states = 414 original state channels + 101 directional-memory values**. Its five transitions include next-cycle mechanical restart. Each uses a frozen Ridge map plus a SiLU MLP `529 → 32 → 32 → d`; input is state515 + control9 + stage5, and output sizes are **414 / 414 / 414 / 261 / 204**. The neural maps have **146,411 parameters** in total. Original normalization, residual RMS scales and physical projections are retained; temperatures are not clipped.

该独立历史支线包含 **515 维状态 = 原 414 分量 + 101 维方向记忆**，五个接口包括下一循环的力学重启。各接口使用冻结 Ridge 加 SiLU MLP：`529 → 32 → 32 → d`；输入为状态 515、控制 9、阶段独热 5，输出维度为 **414 / 414 / 414 / 261 / 204**，共 **146,411 神经参数**。保留原标准化、残差 RMS 和物理投影，不裁剪温度。

The completed experiment compares archived one-step weights with **300 joint updates** of either teacher-forced one-step or fully differentiable nine-step continuation. Both use eight complete two-cycle cases, the same block-balanced normalized state MSE, AdamW **lr 0.0005 / weight decay 1e-4 / clip 10**, CPU float64, and three seeds. Only residual MLPs are trained; the same historical DeepONet per seed decodes autonomous field entries without training or feedback. See [the bilingual report](MULTICYCLE_MULTISTEP.md) for formulas, exact interfaces and measured results.

已完成实验比较原单步权重、读取训练真实前态的单步续训，以及完全可微九步续训。两种续训均用八个完整两循环工况、相同分块归一化状态 MSE、**300 次联合更新**、AdamW **lr 0.0005 / wd 1e-4 / clip 10**、CPU float64 和三个种子。仅训练残差 MLP，每个种子使用原历史 DeepONet 解码自主入口，不重训也不反馈。完整公式、接口及结果见中英文报告。
