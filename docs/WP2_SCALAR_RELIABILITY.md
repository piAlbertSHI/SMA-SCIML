# WP2 scalar reliability audit / WP2 标量可靠性审计

The adjacent archived WP2 design-law benchmark is complete as a **static design-law subtask**. This audit verifies its provenance and reorganizes the four published holdout schemes for the current evidence chain. It does not promote the subtask into a full-cycle performance surrogate. / 相邻归档中的 WP2 设计规律标量基准已经完成，范围是**静态设计规律子任务**。本审计核对来源并重新整理四种已发布留出方案，纳入当前证据链。它不把该子任务升级为完整循环性能代理。

## Scope and integrity / 范围与完整性

The source run contains 1,176 raw metric rows and 392 seed-aggregated rows. All four schemes are present, and the resume checkpoint lists every scheme as completed. The source code, configuration, scalar data, split manifest and provenance hashes match the archived run. No fitting or solver call was made by this audit. / 来源运行包含 1,176 条原始指标记录和 392 条种子聚合记录。四种切分均存在，恢复检查点列出全部已完成方案。来源代码、配置、标量数据、切分清单和来源哈希均与归档运行一致。本审计没有重新拟合模型，也没有调用求解器。 [Audit record / 审计记录](../evidence/local_studies/wp2_scalar_reliability_20261123/audit.json)

| Scheme / 方案 | Train | Validation | Test |
|---|---:|---:|---:|
| IID | 2,893 | 365 | 326 |
| Geometry holdout / 几何留出 | 2,659 | 308 | 617 |
| Material holdout / 材料留出 | 1,649 | 208 | 1,727 |
| Combined OOD holdout / 组合外推留出 | 2,977 | 159 | 448 |

The six regression targets are latent heat, loading work, the heat-to-work ratio, mean phase fraction, maximum deflection and maximum stress. The classification target is design validity. Preprocessing and balanced classifier weights are fitted on each training role. / 六个回归目标为潜热、加载功、热功比、平均相分数、最大挠度和最大应力。分类目标为设计有效性。预处理和分类器平衡权重均在各方案训练分区内拟合。

## Held-out regression / 留出回归

The table reports the lowest recorded test RMSE among the archived model families for each scheme and target. It is a descriptive comparison of the fixed archive. / 下表报告每种方案和目标在归档模型族中记录到的最低测试 RMSE，属于固定归档的描述性比较。

| Scheme / 方案 | Latent heat | Loading work | Heat/work ratio | Mean phase | Max deflection | Max stress |
|---|---:|---:|---:|---:|---:|---:|
| IID | 38.96 | 9.49 | 0.398 | 0.0488 | 0.572 | 19.18 |
| Geometry holdout | 741.36 | 194.55 | 0.843 | 0.335 | 5.205 | 160.35 |
| Material holdout | 147.89 | 16.90 | 0.874 | 0.167 | 1.109 | 28.52 |
| Combined OOD holdout | 274.23 | 36.99 | 0.541 | 0.112 | 1.722 | 33.93 |

Geometry holdout is the most difficult regression shift for latent heat, loading work and stress. Material holdout produces a large ratio error and a mean-phase RMSE of 0.167. The combined OOD scheme retains useful ranking for several targets, while its errors remain materially larger than IID. These results establish a structured reliability boundary for the static scalar subtask. / 几何留出对潜热、加载功和应力的回归退化最明显。材料留出下热功比误差较大，平均相分数 RMSE 为 0.167。组合外推方案对若干目标仍有排序能力，但误差明显高于 IID。这些结果为静态标量子任务建立了结构化可靠性边界。

The full target-by-scheme table and the linear Ridge reference are available as [best-model CSV / 最优模型表](../evidence/local_studies/wp2_scalar_reliability_20261123/regression_best_test.csv), [Ridge CSV / Ridge 对照](../evidence/local_studies/wp2_scalar_reliability_20261123/regression_ridge_test.csv) and [shift table / 外推变化表](../evidence/local_studies/wp2_scalar_reliability_20261123/distribution_shift.csv). No single model is declared a universal deployment choice from these test scores.

## Invalid-design detection / 无效设计识别

The invalid class is the minority class. Its test prevalence is 8.17% for IID, 3.59% for geometry holdout, 10.05% for material holdout and 6.67% for combined OOD. The best recorded invalid-class AUPRC is 0.744, 0.714, 0.532 and 0.772 respectively. / 无效设计属于少数类。其测试比例在 IID、几何留出、材料留出和组合外推中分别为 8.17%、3.59%、10.05% 和 6.67%。记录到的无效类最高 AUPRC 分别为 0.744、0.714、0.532 和 0.772。

Material holdout is the weakest invalid-design ranking condition. The archive reports balanced accuracy, Brier score, NLL and ECE on uncalibrated probabilities. These are reliability diagnostics for the recorded cohort. They do not provide a calibrated acceptance probability for new designs. / 材料留出是无效设计排序最弱的条件。归档同时记录平衡准确率、Brier、NLL 和 ECE，这些指标针对未校准概率，仅用于当前工况群的可靠性诊断，不构成新设计的校准接受概率。

## Claim boundary / 结论边界

Dataset v1.0 omits $V^*$, cycle frequency, exchange time, $Q_{cool}$, $W_{pump}$, COP, SCP and cycle temperature span. WP2 full-cycle closure therefore remains incomplete. The existing 63-case full-cycle scalar evidence and this static design-law audit are separate sources with separate input contracts. / 数据集 v1.0 缺少 $V^*$、循环频率、换热时间、$Q_{cool}$、$W_{pump}$、COP、SCP 和循环温差。因此 WP2 的完整循环闭合仍未完成。现有 63 例完整循环标量证据与本静态设计规律审计属于不同来源，输入契约也不同。

The audit can be replayed from the repository root when the adjacent source archive is available.

```bash
python -m examples.audit_wp2_scalar_evidence \
  --solver-repo ../../gx101r1 \
  --output run-output-wp2-scalar-reliability
```
