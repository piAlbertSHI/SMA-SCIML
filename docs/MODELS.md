# 模型架构与训练设置

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
