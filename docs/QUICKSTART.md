# Quickstart / 使用指南

## Installation / 安装

Use Python 3.12 and a virtual environment. The requirements declare compatible ranges; the [recorded environment](../evidence/current_environment.json) documents the evaluated installation.

使用 Python 3.12 与虚拟环境。依赖文件声明兼容范围，已评价的安装版本见环境记录。

```bash
git clone https://github.com/piAlbertSHI/SMA-SCIML.git
cd SMA-SCIML
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -r requirements.txt
python -m examples.validate_bundle
```

The bundled benchmarks run on CPU without MATLAB or a physical solver. Raw simulation archives are not included; see the [data card](DATA_CARD.md).

仓库内基准可在 CPU 上运行，无需 MATLAB 或物理求解器。原始仿真归档不随仓库分发，范围见数据说明。

## Explore fields / 浏览温度场

The [modal HRO report](HRO_MODAL.md) contains the architecture, training commands and stored field comparisons. Verify its three final checkpoints and generate its figures with:

[模态 HRO 报告](HRO_MODAL.md)包含架构、训练命令及温度场对照。核验三组最终检查点并生成图形：

```bash
python -m examples.verify_modal_evidence
python -m examples.modal_showcase --output demo-output-modal
```

The interactive HTML below is the coordinate-HRO reference. / 下方交互 HTML 对应坐标 HRO 参考方案。

Open `assets/showcase/explorer.html` locally. It contains saved HRO predictions and numerical references, with exchange-stage, temperature/error and time controls. Display values are rounded to four decimals; full-precision arrays remain in the experiment record.

在本地打开 HTML，可选择换热阶段、温度/误差及时间。页面包含 HRO 保存预测和数值参考，显示数据保留四位小数，完整精度数组保留于实验记录。

```bash
python -m examples.field_showcase --output demo-output-fields
```

This generates the architecture figure, both test-case atlases, the comparison animation and the offline explorer from committed predictions. It does not train a model.

该命令从保存预测生成架构图、两工况图谱、对照动画及离线浏览器，不训练模型。

## Verify or train HRO / 验证或训练分层循环算子

Replay the recommended thermal-response configuration and recompute its paired metrics with the first command below. For inference, pass CPU float64 initial states of shape `[B, 515]`, controls of shape `[B, 9]`, and the desired cycle count. The example uses the first registered seed and first three-cycle configuration by dataset order. [Checkpoints and data contract](HRO_ERROR_SCALED_RESPONSE.md)

下面的验证命令回放误差收缩热响应配置并复算配对指标。推理输入为 CPU float64 的初态 `[B, 515]`、控制量 `[B, 9]` 和循环数。示例采用登记的首个种子及数据集中首个三循环工况。

```bash
python -m examples.verify_risk_limited_readout
```

```python
import json
from pathlib import Path

import numpy as np
import torch

from SciML.models.v13.risk_limited_readout import RiskLimitedRelaxationOperator

index = json.loads(Path("checkpoints/hro_recommended.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)
model = RiskLimitedRelaxationOperator.load(index["checkpoints"][0]["path"])
with np.load("data/historical12_state.npz", allow_pickle=False) as data:
    i = int(np.flatnonzero((data["split"] == "test") & (data["cycles"] == 3))[0])
    initial = torch.from_numpy(data["initial"][i:i + 1])
    control = torch.from_numpy(data["control"][i:i + 1])

with torch.no_grad():
    prediction = model(
        initial, control, cycles=3,
        times=torch.linspace(0, 1, 50, dtype=torch.float64),
        positions=torch.linspace(0, 1, 101, dtype=torch.float64),
    )
print(prediction["states"].shape, prediction["fields"].shape)
```

The coordinate HRO reference has a separate training and replay entry point. / 坐标 HRO 基线保留独立的训练及回放入口。

```bash
# Replay three checkpoints and recompute all test metrics.
python -m examples.formal_hierarchical --verify-only
# Repeat the fixed three-seed training protocol in a fresh directory.
python -m examples.formal_hierarchical --output run-output-hro
```

The protocol uses three seeds, 200 updates per seed, batch size two, CPU float64 and complete two-cycle trajectories. The recorded training/evaluation elapsed time is 7.1 minutes; runtime depends on hardware. Checkpoints, optimizer/scheduler/RNG state and per-update logs are retained. Output directories must not exist. The training command does not invoke physical simulations.

协议采用三个种子、每种子200次更新、批次2、CPU float64及完整两循环轨迹。记录的训练与评价用时为7.1分钟，实际时间取决于硬件。保存内容包括权重、优化器/调度器/随机数状态及逐次日志。输出目录必须为新目录，训练命令不调用物理仿真。

```python
import torch
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator

model = HierarchicalStateOperator.load(
    "evidence/experiments/v13_formal_v1/s20261121/model.pt"
)
# initial: [B, 515]; control: [B, 9]; both float64.
times = torch.linspace(0, 1, 50, dtype=torch.float64)
positions = torch.linspace(0, 1, 101, dtype=torch.float64)
with torch.no_grad():
    prediction = model(initial, control, cycles=3,
                       times=times, positions=positions)
```

`initial` and `control` are required inputs; their definitions are in the [architecture specification](HIERARCHICAL_OPERATOR.md). States have shape `[B,14,515]`; fields have shape `[B,6,2,50,101]`. Use the verification command for a runnable example with bundled inputs. The class and checkpoint identifiers are stable implementation interfaces.

初态与控制量的定义见架构说明。三循环状态输出为 `[B,14,515]`，场输出为 `[B,6,2,50,101]`。完整可运行的数据示例由验证命令提供；类名及检查点标识为稳定实现接口。

## Reference studies / 参考方法研究

```bash
python -m examples.multicycle_multistep --verify-only
python -m examples.multicycle_gallery --output demo-output-cascade
python -m examples.showcase --output demo-output-reference
python -m examples.research_gallery --output demo-output-research
python -m examples.active_diversity --output run-output-acquisition
```

Reference results cover residual state dynamics, coordinate/Fourier/POD field models, performance ensembles, inverse PINNs and offline acquisition. The scalar training example uses three six-output members; it is distinct from the five-member, five-output performance benchmark.

参考结果覆盖残差状态动力学、坐标/Fourier/POD 场模型、性能集成、逆问题 PINN 与离线选样。标量训练示例使用三个六输出成员，与五成员五输出性能基准分开定义。

## Validation / 验证

```bash
python -m examples.validate_bundle
python -m tools.audit_presentation
python -m examples.formal_hierarchical --verify-only
node examples/test_field_explorer.cjs assets/showcase/explorer.html
```

CI checks Python model contracts, target-free inference, checkpoint replay, evidence integrity, presentation terminology and figure generation. The offline explorer has mock DOM/canvas interaction tests; browser rendering acceptance is tracked in the release policy. Checkpoints use PyTorch's restricted `weights_only=True` loader and should be obtained from trusted, hash-verified sources.

CI 检查模型接口、无目标泄漏推理、权重回放、证据完整性、展示术语与绘图。离线浏览器具有模拟 DOM/canvas 交互测试，真实浏览器渲染验收见发布规范。检查点通过 PyTorch 的 `weights_only=True` 限制加载接口读取，应使用可信且哈希核验的来源。
