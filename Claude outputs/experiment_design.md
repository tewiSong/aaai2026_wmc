# 新实验设计规范
**论文**：Modular WMC for Neuro-Symbolic Inference  
**用途**：JAIR 投稿，回应三位审稿人的实验缺失投诉  
**日期**：2026-10-13

---

## E1：模块化分解真正启动时的精确性验证

### 动机
原论文的精确性验证（Table 1）在 GO cc 上做，而 GO cc 的结构导致方法退化为 monolithic JT（modules=1，depth=0，max|B|=0）。审稿人 c36U 正确指出这没有测试模块化分解本身。此实验在 GO mf 和 GO bp 上重复精确性验证，其中方法确实触发非平凡分解。

### 数据准备

**步骤 1**：从 GO mf 和 GO bp 的完整图中采样子图。

筛选条件（AND 关系）：
- 运行 Algorithm 1 后，返回的分解满足 depth ≥ 1（说明递归发生）
- 至少一个模块的 max|B_j| ≤ 12（使得精确参考可行）
- 子图整体大小 ≤ 2,000 个 term（monolithic JT 在合理内存内可完成）
- brute-force 和 monolithic JT 的参考值在固定随机种子下完全一致

采样方法：
- 从每个本体中随机选取一个 term 作为根，取其 depth ≤ 6 的子图
- 重复直到收集到 ≥ 100 个满足筛选条件的实例

**步骤 2**：对每个筛选实例，生成 4 个 score vector（uniform, Beta(2,2), Beta(0.5,0.5), extreme-prior），共 ≥ 400 个（实例×score）组合。

**步骤 3**：记录每个实例的结构信息：depth、modules 数量、max|B_j|、min-fill width。

### 实验协议

对每个（子图，score）对：
1. 运行 brute-force（穷举所有 2^n 个合法赋值）
2. 运行 monolithic JT（作为第二参考，验证 brute-force）
3. 运行 modular WMC（使用精确 separator，exact mode）

确保步骤 3 确实触发分解（depth ≥ 1 的实例），而不是退化到 monolithic。

### 报告指标
与原 Table 1 格式一致：

| Benchmark | MAE | Max error | Completed |
|-----------|-----|-----------|-----------|
| GO mf subgraphs (depth≥1) | | | X/total |
| GO bp subgraphs (depth≥1) | | | X/total |

补充报告：每个实例的（depth, max|B_j|, 模块数）分布（histogram 或 box plot）。

### 预期结果
机器精度误差（MAE ≈ 10^-15 量级），completion = 100%。
这是论文精确性证明的直接实证验证，而非退化验证。

### 论文中的位置
替换原 §6.1（RQ1），或在其后加"Non-trivial decomposition instances"子节。

---

## E2：Rank 自一致性检验（大本体间接精度验证）

### 动机
对于 GO mf/GO bp/UBERON/ChEBI，monolithic JT OOM，无法获得精确参考。但若 rank r 和 rank 2r 的输出非常接近，这间接说明 rank r 已接近收敛——误差来自 TT 表示，而非来自分解本身。命题 2（stability bound）保证了边缘误差上界由边界后验误差决定，rank 自一致性是对这个上界的间接估计。

### 实验协议

对每个大本体（GO mf, GO bp, UBERON, ChEBI），使用相同的输入（同一 score vector）分别运行：
- Modular WMC，TT rank = 2
- Modular WMC，TT rank = 4  
- Modular WMC，TT rank = 8

（如计算资源允许，加入 rank = 16）

使用 uniform score vector（最中性的情形）以获得代表性结果。

### 报告指标

对每对 (rank r, rank 2r)，对所有词条计算：

| 本体 | rank 2→4 MAE | rank 4→8 MAE | rank 8→16 MAE |
|------|-------------|-------------|--------------|
| GO mf | | | |
| GO bp | | | |
| UBERON | | | |
| ChEBI | | | |

同时报告每个本体的完成词条数（应与原 Table 2 完全一致）。

### 分析框架
在论文中加入以下分析：
> "If rank-4 to rank-8 MAE is substantially smaller than rank-2 to rank-4 MAE across all ontologies, this indicates that rank 8 lies in the saturating regime of the rank-error curve. Combined with Proposition 2, this provides indirect evidence that the recovered marginals are close to the ground truth WMC posterior."

如果 rank 自一致性出乎意料地差，需要分析原因（可能是特定模块的边界结构不适合低秩近似，需要针对性处理）。

### 论文中的位置
在 §7.3（Scalability Analysis）中加入，作为 Table 2 的补充。

---

## E3：GO mf 上的下游评测（CAFA 风格）

### 动机
原 Table 4 的 CAFA 评测在 GO cc 上做——而 GO cc 恰好是方法退化为 monolithic JT 的数据集。审稿人 c36U 明确要求"a CAFA-style evaluation on GO bp or GO mf, where the decomposition is required"。

### 数据准备

**蛋白质注释**：
- 使用 CAFA 3 或 CAFA 5 数据集（根据论文使用的是哪个版本）
- 将蛋白质注释映射到 GO mf 词条空间（molecular function）
- 保留有足够注释的蛋白质（至少有 N 个 GO mf 正标签，N 根据 CAFA 协议确定）

**预测器**：
- 使用与原实验相同的预训练 MLP 和 DeepGOCNN
- 这些模型应该已经有 GO mf 的预测输出（如果原始训练覆盖三个命名空间）
- 如果没有，用 GO bp 替代（paper cites that "can be mapped to the GO snapshot"）

### 实验协议

方法对比（与原 Table 4 一致）：
1. Raw predictor scores
2. Max closure（upward propagation）
3. Loopy BP（synchronous sum-product, damping=0.5）
4. Modular WMC，TT rank 8（这才是需要 TT 的场景）

注意：这里**不能**用 monolithic JT（OOM）。TT mode 是唯一的精确框架近似。

### 报告指标
与 Table 4 完全一致：Fmax, mAUPR, S-min, Brier, ECE, Viol.

补充报告：violation rate 细分（违反了多少个 is-a 约束）。

### 论文中的位置
在 §7.6 中，与原 GO cc 实验（Table 4）并列呈现：
> "Table 5 shows the same evaluation on GO mf, the namespace for which modular WMC with TT separators is required. [...]"

### 重要说明
原 Table 4 应该重新定位（在论文 text 中）：不是"WMC 比 BP 好"（差异太小，无法支持这个结论），而是"WMC 保证零约束违反且有最低 Brier score/ECE，BP 有残余违反率"。新增的 GO mf 实验才展示 TT 模式的实用价值。

---

## E4：与主流 WMC Solver 的比较

### 动机
审稿人 KEL1 明确要求："See the results of the model counting competition (https://mccompetition.org/) from this year and previous iterations for suggestions on what solvers should be included. Even if these algorithms run out of memory while reading the input file, that should still be explicitly mentioned in the paper."

### Solver 列表
- ADDMC（Dudek et al. 2020）
- DPMC（Dudek et al. 2021）
- d4（Lagniez & Marquis 2017）
- sharpSAT-TD（如果可获得二进制）

### 数据格式转换
这些 solver 接受 DIMACS WMC 格式。需要将本体约束（is-a 蕴含）转换为 CNF：
- $y_c \Rightarrow y_p$  →  $\neg y_c \vee y_p$（直接转换为 2-clause）
- 权重文件：按 solver 的格式要求

### 实验协议
- 环境：与原论文相同（CPU-only，SLURM）
- 内存上限：300 GB（与 Table 2 的 monolithic JT 一致）
- 时间上限：30 分钟
- 测试集：所有 5 个完整本体，加上 GO cc 子图（验证实现正确性）

**正确性验证**：在 GO cc 的一个小子图上，验证 ADDMC/DPMC/d4 与 brute-force 一致，再用于大本体。

### 报告格式

| Solver | GO cc | GO mf | GO bp | UBERON | ChEBI |
|--------|-------|-------|-------|--------|-------|
| ADDMC | ✓ (Xs, YGB) | OOM | OOM | OOM | OOM |
| DPMC | ✓ | OOM | OOM | OOM | OOM |
| d4 | ✓ | OOM/TO | OOM/TO | OOM | OOM |
| Modular WMC | ✓ | ✓ | ✓ | ✓ | ✓ |

（✓ = 完成并报告时间/内存；OOM = 300 GB 内存溢出；TO = 30 分钟超时）

### 分析框架
这个实验的结论不应该是"我们比 ADDMC 好"（这是不公平的比较，因为 ADDMC 是通用 solver），而是：
> "State-of-the-art WMC solvers that do not exploit ontology-specific structure reach their memory or time limits on complete biomedical ontologies. Modular separator WMC—designed specifically for this structural regime—completes all-term inference on all five ontologies."

### 论文中的位置
§7.5（Comparison with WMC Solvers），独立小节，在可扩展性分析之后。

---

## E5：Loopy BP 精度对比（在有精确参考的子图上）

### 动机
原论文在下游任务中比较了 BP 和 WMC，但没有直接比较两者的 **WMC marginal 精度**。这个实验量化 reconvergence 带来的 double-counting 误差有多大，为 §4（Motivation）的分析提供实验支撑。

### 数据集
- 原 432 个 GO cc 参考子图
- E1 新增的 GO mf/GO bp 参考子图（depth≥1 的实例）

### 实验协议
对每个参考实例，运行：
1. Brute-force reference
2. Modular WMC（exact mode）——应与 reference 一致（见 E1）
3. Loopy BP（synchronous sum-product, damping=0.5, tolerance=1e-9, max 10,000 iter）

记录 BP 收敛状态（是否在 max iter 内收敛）。

### 报告指标

| 数据集 | Method | MAE (med/max) | Viol. rate |
|--------|--------|---------------|------------|
| GO cc subgraphs (432) | Modular WMC | ~0 | 0 |
| GO cc subgraphs (432) | Loopy BP | ? | ? |
| GO mf subgraphs (E1) | Modular WMC | ~0 | 0 |
| GO mf subgraphs (E1) | Loopy BP | ? | ? |

### 分析框架
这个实验的目的是量化 reconvergence 带来的误差，支持 §4 的动机分析：
> "On the 432 GO cc reference subgraphs, loopy BP achieves median MAE of X versus machine precision for modular WMC, confirming that reconvergent paths induce systematic error in path-wise message passing."

如果 BP 误差很小（MAE < 0.01），仍然值得报告：说明 reconvergence 在这些特定子图中影响不大，同时仍然有约束违反（violation rate > 0）。

### 论文中的位置
§7.5 的一部分，或 §7.2（Correctness）的补充实验。

---

## 实验数据管理

### 共享基础设施
- 所有实验使用相同的 WMC 后端（与原论文相同）
- score vectors 使用固定随机种子（与原论文一致：在技术附录中报告）
- 所有实验的原始输出保存为 JSON，按（本体，score_family，rank，实例ID）索引

### 计算资源估算

| 实验 | 预计时间 | 内存需求 | 优先级 |
|------|----------|----------|--------|
| E1（GO mf/bp 精确性）| 1-2 周 | 128 GB | **最高** |
| E2（rank 自一致性）| 2-3 周 | 400 GB（ChEBI）| 高 |
| E3（GO mf CAFA）| 2-3 周 | 128 GB | 高 |
| E4（WMC solver）| 1 周 | 300 GB | 中 |
| E5（BP 精度）| 3-5 天 | 32 GB | 中 |

### 关键路径
E1 的结果是最关键的，因为它直接回应了最有力的拒稿理由。建议先跑 E1，在等待 E2/E3 的同时同步进行论文写作。

---

## 实验结果不如预期时的应对策略

### 如果 E1 发现精确性在 depth≥1 实例上不成立
这意味着代码有 bug 或理论有误。需要在提交前修复。检查点：
- 边界变量是否正确从消元集合中排除？
- backward recovery 是否正确传播？
- score vector 的 clipping 是否在每个 oracle 调用中一致应用？

### 如果 E2 发现 rank 8 和 rank 4 差异很大
说明 rank 8 还没有饱和，需要用更大的 rank（16、32）。在论文中诚实报告这个发现，并将其定位为"对于某些高宽度边界，rank 收敛较慢，需要进一步研究"。

### 如果 E3 发现 TT 模式的下游性能明显差于 loopy BP
这是最严重的情形。分析原因：
- 是 TT 近似精度不够？（看 E2 的 rank 自一致性）
- 还是 CAFA 评测指标对 WMC 的概率校准不敏感？
如果是后者，重新定位贡献为"零约束违反 + 最优校准"而非"最高排名性能"。

### 如果 E4 发现某个 solver 在大本体上能完成
这是好事！在论文中诚实报告，分析它能完成的原因（是否利用了稀疏结构），并比较与 modular WMC 的时间/内存效率。
