# WP4 field-to-energy contract / WP4 场到能量契约

The archived field-energy audit separates export identities, sampled-port diagnostics and a predictive field-to-performance claim. Export identities are exact for the recorded labels. Temperature fields alone do not provide the complete energy contract needed by an autonomous cycle model. / 归档场—能量审计区分了导出恒等关系、端口采样诊断和预测场到性能结论。记录标签的导出恒等关系是精确的。仅有温度场不足以构成自主循环模型所需的完整能量契约。 [Audit summary / 审计摘要](../evidence/archive/field-energy-information-r1/summary.json)

| Check / 检查 | Recorded result / 记录结果 | Interpretation / 解释 |
|---|---:|---|
| Export identity residual for $Q_{cool}$ | 0 J | Derived label matches the archived performance field |
| Export identity residual for $W_{pump}$ | $2.17\times10^{-19}$ J | Arithmetic agreement at floating-point precision |
| Export identity residual for COP | 0 | COP follows the archived work convention |
| Sampled heat quadrature MAE | 0.00793 J | Exported 50-point port sampling approximates stored heat |
| Sampled heat quadrature maximum error | 0.02122 J | Port quadrature is a diagnostic, not a replacement label |
| Sampled pump-work maximum error | $8.67\times10^{-19}$ J | Pump-work sampling is numerically consistent |
| Maximum internal endpoint-to-port temperature gap | 12.999 K | Port ownership and storage state are not interchangeable |

The archive has 63 cases and 126 exchange stages. Its operator outputs include temperature fields, but the current field interface lacks realized signed mass flow, pressure-command history, port-temperature ownership, mechanical work, capacities and latent-energy/restart conventions. The energy export also lacks independent environment heat, an internal-pipe-fluid method and an internal-pipe-fluid temperature estimate. / 归档包含 63 个工况和 126 个换热阶段。当前算子接口输出温度场，但缺少实际有符号质量流量、压力指令轨迹、端口温度归属、机械功、容量以及潜热和重启约定。能量导出还缺少独立环境热、内部管内流体计算方法和内部管内流体温度估计。

The audit therefore keeps the physics gate unchanged and rejects an autonomous field-to-COP claim. A future field-to-performance model needs a declared port/storage boundary, flow and pressure channels, a work convention and an independent closure residual. / 因此审计保持物理门不变，并拒绝自主场到 COP 的结论。后续场到性能模型需要声明端口/储能边界、流量与压力通道、功的定义以及独立闭合残差。

The archived stage-level diagnostics and counterexamples remain the source evidence. / 阶段诊断和反例仍以归档证据为准。 [Stage quadrature diagnostics / 阶段积分诊断](../evidence/archive/field-energy-information-r1/stage_quadrature_diagnostics.json) · [Readout counterexamples / 读出反例](../evidence/archive/field-energy-information-r1/readout_counterexamples.json)
