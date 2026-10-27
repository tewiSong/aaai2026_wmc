# 修改计划：投稿 JAIR（Journal of Artificial Intelligence Research）

**论文**：Modular Weighted Model Counting for Neuro-Symbolic Inference over Large Ontologies  
**原投**：AAAI-27（拒稿，Phase 1）  
**目标**：JAIR  
**制定日期**：2026-10-13

---

## 一、为什么选 JAIR

AAAI 拒稿的两个根本原因都在 JAIR 框架下更容易修复：

| 问题 | AAAI 约束 | JAIR 优势 |
|------|-----------|-----------|
| 精确验证与方法启用脱节 | 9页硬限制，无法加足够多实验 | 无页数上限，可加完整实验组 |
| 写作对 AAAI 混合受众不够可读 | 摘要/引言篇幅极有限 | 引言可写 3-4 页，充分铺垫背景 |
| 与 ADDMC/DPMC/d4 无比较 | 时间和空间不够 | 可作为独立实验小节 |
| 下游评测只在退化数据集上做 | 只有一个 CAFA 实验槽位 | 可在多个本体上做下游评测 |

JAIR 典型文章长度 25–50 页（双栏），接受周期约 6–12 个月，接受率约 20–25%。

---

## 二、三位审稿人的意见逐条回应策略

### 审稿人 c36U（Rating: 4，核心问题）

**问题 1**：GO cc 的 Table 2 显示 modules=1，depth=0，max |B|=0——方法退化为 monolithic JT，精确性验证没有实际测试模块化分解。

→ **回应**：新增实验 E1（见第四节）：在 GO mf 和 GO bp 中筛选出 depth > 0、|B_j| ≤ 12 的实例，用精确参考验证模块化分解的正确性。在论文中明确说明 GO cc 退化的原因（其结构允许单模块精确处理），并将其定位为"边界条件验证"而非"核心正确性验证"。

**问题 2**：四个大本体只报告 completion，没有精度数据。

→ **回应**：新增实验 E2（rank 自一致性检验）和 E3（下游评测）。对 GO mf/GO bp/UBERON/ChEBI，报告 rank 4 与 rank 8 输出之间的一致性（间接精度上界）。同时在 GO mf 上做 CAFA 风格的下游评测。

**问题 3**：为什么 rank 8 在 1082 维边界下足够用，没有理论依据。

→ **回应**：新增讨论小节，从本体图结构（is-a 蕴含形成的稀疏层级约束）分析为什么 separator 在低秩下可压缩；补充 rank 1/2/4/8/16 的消融实验（如果计算资源允许）以建立经验性的 rank 饱和曲线。

**问题 4**：只比较了 monolithic JT，没有跑 ADDMC/DPMC/loopy BP at scale。

→ **回应**：新增实验 E4（与主流 WMC solver 的比较，包括 OOM/timeout 结果），新增实验 E5（loopy BP 精度对比，在有精确参考的子图上）。

---

### 审稿人 KEL1（Rating: 3，写作问题）

**问题 1**：摘要和引言直接跳入技术细节，没有问题陈述。

→ **回应**：完全重写摘要（从应用问题出发）和引言（第一段讲本体标注预测，第二段讲 WMC 是什么、为什么用它，第三段讲可扩展性瓶颈，第四段才进入我们的方法）。具体见第五节。

**问题 2**：大量未定义术语（monolithic WMC、structural width、endpoint、Horn clause、constraint graph 等）。

→ **回应**：在 Preliminaries 节系统性地添加所有术语的正式定义，并在第一次使用时给出文内引用。具体见第五节的术语表。

**问题 3**：与 tree-decomposition WMC、tensor-network WMC 的区别没讲清楚。

→ **回应**：重写 Related Work（第五节），明确列出与 Dal et al. (2021)、Goral et al. (2024)、Onaka et al. (2025) 的技术差别。

**问题 4**：没有与 model counting competition 的 solver 比较。

→ **回应**：新增实验 E4。

---

### AI Reviewer（综合意见）

**问题 1**：全词条恢复（all-term recovery）的算法没有给出完整操作定义——恢复公式对 |B|=1082 来说是指数的。

→ **回应**：在方法章节加入"Computational scope of backward recovery"段落，明确说明：(a) 精确模式下 calibrated module JT 通过消息传递重用恢复边缘概率，代价是 O(|M_j| · 2^{w_j^∂}) 而非 O(2^{|B_j|}）；(b) TT 近似模式下通过 oracle 查询构建 conditioned marginals。

**问题 2**：Table 4 与 loopy BP 差异太小（0.0006 Fmax），且用的是精确 JT 而不是 TT 模式。

→ **回应**：(a) 报告标准误差和统计检验；(b) 在 GO mf 上做 TT 模式的下游评测（这才是需要 TT 的场景）；(c) 诚实承认在小本体上 WMC 相对 BP 的 ranking 提升边际，重新定位 Table 4 的贡献为"校准质量验证"而非"排名性能突破"。

**问题 3**：novelty boundary 不够清晰，缺少与 Goral et al. (2024) 和 Onaka et al. (2025) 的对比。

→ **回应**：重写 Related Work，明确说明：我们的核心新颖性在于（1）从预测器权重的本体理论中构造 separator，而非编译一个静态知识库；（2）将 TT 近似与精确逻辑支持解耦；（3）递归边界感知核选择算法。

---

## 三、论文结构调整（JAIR 版本）

```
原 AAAI 结构（~9页）          →    JAIR 新结构（~35页）
─────────────────────────────────────────────────────────
§1 Introduction (1p)          →    §1 Introduction (3-4p)
§2 Related Work (0.5p)        →    §2 Related Work (3p，5个子节)
§3 Preliminaries (1p)         →    §3 Problem Formulation (2p)
§4 Motivation (0.5p)          →    §4 Motivation & Failure Modes (1p)
§5 Method (2.5p)              →    §5 Modular Separator WMC (4p)
                                   §6 Theoretical Analysis (2p，含证明)
§6 Experiments (2.5p)         →    §7 Experiments (8-10p，5个实验)
§7 Conclusion (0.5p)          →    §8 Discussion (1.5p)
                                   §9 Conclusion (0.5p)
Appendix                      →    Appendix（保留，更完整）
```

---

## 四、新实验设计（优先级排序）

### E1：模块化分解的精确性验证（最高优先级）
**目标**：直接回应 c36U 的核心问题——Table 1 有没有真正测试模块化分解？

**设计**：
- 数据源：GO mf 和 GO bp（这两个本体确实触发了分解，depth > 0）
- 筛选条件：从各本体中采样子图，要求（1）depth ≥ 1，（2）至少一个模块的 |B_j| ∈ [1, 12]（可精确枚举），（3）子图的 monolithic JT 可在合理时间内完成
- 实验协议：在筛选出的实例上，运行 modular WMC（使用精确 separator），与 brute-force/monolithic JT 比较
- 报告指标：MAE、Max error、completion（与 Table 1 格式一致）
- 预期结果：机器精度误差（与 GO cc 相同量级），但这次在真正触发分解的实例上

**关键说明文字**（在论文中需要加的）：
> "We additionally sample 400 subgraphs from GO mf and GO bp whose selector returns depth ≥ 1 and at least one module boundary of size |Bj| ≤ 12. On these instances, modular WMC with exact separators recovers all reference marginals to machine precision (MAE = X × 10⁻¹⁶), confirming that factor partition, module elimination, separator construction, and backward recovery all operate correctly when the decomposition is non-trivial."

---

### E2：Rank 自一致性检验（高优先级）
**目标**：为大本体上的精度提供间接证据。

**设计**：
- 数据源：GO mf、GO bp、UBERON、ChEBI（与 Table 2 相同）
- 协议：用相同的输入（同一组 score vector）分别以 rank 4 和 rank 8 运行 modular WMC
- 报告指标：rank-4 和 rank-8 输出之间的 MAE（对所有词条）
- 补充：rank 1/2/4/8 的四路比较（如计算资源允许）
- 预期结果：rank 4→8 的差异明显小于 direct product→rank 2 的差异，说明 rank 8 接近收敛

**理论联系**：通过命题 2（stability bound），rank 自一致性是边界后验误差上界的经验代理——如果相邻 rank 的输出非常接近，说明边界后验误差已经很小。

---

### E3：GO mf/GO bp 上的下游评测（高优先级）
**目标**：在方法真正发挥作用的本体上做应用层面的验证。

**设计**：
- 数据源：CAFA 蛋白质功能注释，映射到 GO mf 或 GO bp 词条空间
- 方法对比：Raw scores、Max closure、Loopy BP、Modular WMC（TT rank 8）
- 报告指标：与 Table 4 相同（Fmax、mAUPR、S-min、Brier、ECE、Viol.）
- 注意：这里用的是 TT 模式，因为 GO mf/GO bp 的 monolithic JT OOM

**论文定位**：
- GO cc 实验（原 Table 4）保留，重新定位为"在有精确 WMC 基准的命名空间中验证方法有效性"
- GO mf 实验（新增）定位为"在需要 TT 近似的命名空间中展示方法的实际价值"

---

### E4：与主流 WMC Solver 的比较（中优先级）
**目标**：回应 KEL1 关于 model counting competition 的意见。

**设计**：
- Solver 列表：ADDMC、DPMC、d4（sharpSAT-TD 如果可用）
- 数据源：先在小本体（GO cc 子图）上验证实现正确性，再在完整本体上运行
- 协议：300 GB 内存上限，30 分钟超时（与 Table 2 的 JT 一致）
- 报告：每个 solver 在每个本体上的状态（success/OOM/timeout）及成功时的时间和内存

**预期结果**：这些 solver 在完整本体上会 OOM 或 timeout，因为它们是通用 WMC solver，没有利用本体结构。但这个结果需要明确报告出来。

---

### E5：Loopy BP 精度对比（中优先级）
**目标**：在有精确参考的子图上，比较 loopy BP 和 modular WMC 的精度。

**设计**：
- 数据源：原 432 个参考子图（来自 GO cc），以及 E1 新增的 GO mf/GO bp 子图
- 方法对比：Loopy BP vs Modular WMC（exact mode）vs Brute-force reference
- 报告指标：MAE、Max error
- 动机：原论文已经在下游任务中比较了 BP 和 WMC，但没有在 WMC marginal 层面比较精度

**预期结果**：BP 会有系统性误差（由于 reconvergence），WMC 与精确参考一致。这会量化 reconvergence 带来的 double-counting 误差有多大。

---

## 五、各章节详细修改指南

### §1 Abstract
**核心原则**：读者第一句话就要知道"这是什么问题、为什么重要"。

**现有问题**：
- 第一句即"Weighted model counting (WMC) turns label-wise predictor scores..."——假设读者知道 WMC
- 没有讲为什么这个问题重要、有多少 term 需要处理

**修改要点**（见 abstract_intro.tex）：
- 第1句：应用场景（蛋白质功能预测、层级多标签分类）
- 第2-3句：约束违反问题 + WMC 作为解法的直觉
- 第4-5句：可扩展性瓶颈（structural width, reconvergence）
- 第6-8句：我们的解法（modular decomposition + TT separator）
- 第9-10句：实验结果

---

### §2 Introduction
**核心原则**：一个不了解 WMC 的 ML 读者，读完前两段应该明白这篇文章在做什么。

**结构**（JAIR 版，约3页）：

```
¶1  应用动机：本体预测中的约束违反问题（蛋白质/化学物质/解剖）
¶2  WMC 作为后处理层的直觉：为什么这比 architecture/loss 方法好
¶3  可扩展性瓶颈：treewidth + reconvergence（引用 diamond 例子）
¶4  现有方法为什么不够：monolithic 太贵，loopy BP 不准确
¶5  我们的解法概述：modular decomposition + TT separator
¶6  贡献列表（3-4条，比原来更具体）
¶7  论文组织：每节内容
```

**需要在 Introduction 中定义的术语**：
- weighted model counting（一句话直觉 + 正式定义推迟到 §3）
- treewidth / structural width（一句话直觉）
- reconvergent paths（通过图形化说明）
- modular decomposition（一句话直觉）

---

### §3 Related Work
**新增内容**：

**3.1 Constrained prediction（保留，略扩展）**

**3.2 Exact WMC（扩展）**：
- 加入 ADDMC (Dudek et al. 2020)、DPMC (Dudek et al. 2021)、d4 (Lagniez & Marquis 2017)、sharpSAT-TD
- 明确说明这些方法不针对本体结构的特性

**3.3 Compositional WMC（新增）**：
- Dal et al. (2017) 的 partitioned WMC
- Dal et al. (2021) 的 compositional approach
- 与我们方法的区别：他们分解的是静态公式，我们处理有预测器权重的动态实例；他们的 separator 表示方法不处理宽边界

**3.4 Tensor-based WMC 和知识编译（新增）**：
- Goral et al. (2024)：通过半环扩展做 model counting
- Onaka et al. (2025)：TT 与 OBDD 比较
- Wrigley et al. (2017)：tensor BP
- 与我们方法的区别：他们是静态编译，我们是动态查询拟合；我们将逻辑支持与近似拟合解耦

**3.5 本体预测（新增）**：
- CAFA 评测体系
- DeepGO、DeepGOCNN 等方法
- 后处理一致性方法（max closure 等）

---

### §5 Method：需要补充的段落

在"Recovering posterior marginals inside modules"之后，加入：

**"Computational complexity of backward recovery"**：
> 精确模式下，module j 的边缘恢复通过一个 calibrated module junction tree 完成。该 JT 在 M_j ∪ B_j 上构建，在 w_j^∂ 宽度下，每次查询代价为 O(|M_j| · 2^{w_j^∂})，而不是 O(2^{|B_j|}）。对于 TT 近似模式，每个边界赋值 b 的 h_{j,i}(b) 通过一次 boundary-clamped oracle 查询计算，总恢复代价为 O(|B_j| · r^2 · |M_j| · C_j^∂)，其中 r 是 TT rank，C_j^∂ 是一次 oracle 查询的代价。

---

### §7 Experiments：重组叙事

原论文的三个 RQ 看起来是三篇不同论文的实验。重组为统一叙事：

```
§7.1 实验设置（统一说明数据集、baseline、评测协议）
§7.2 正确性验证（E1 新增 + 原 Table 1：decomposition 真正启动时的精确性）
§7.3 可扩展性分析（原 Table 2 + E2 rank 自一致性）
§7.4 Separator 近似质量（原 Table 3 + rank 饱和曲线）
§7.5 与替代方法的比较（E4 WMC solver + E5 loopy BP 精度）
§7.6 下游应用（原 Table 4 GO cc + E3 GO mf/GO bp CAFA）
```

---

### §8 Discussion（新增）

JAIR 期望有一个讨论节，分析：
- 方法在什么条件下效果最好（本体结构的 modularity 特性）
- 方法的局限性（TT rank 选择、核选择的启发性）
- 与更广泛的 structured prediction 文献的关系
- 未来方向（adaptive decomposition、certified error bounds、直接推理训练）

---

## 六、术语表（Introduction 和 Preliminaries 中需要定义）

| 术语 | 当前处理 | 修改后 |
|------|----------|--------|
| Weighted model counting (WMC) | 直接在摘要使用 | Introduction §1 给直觉，§3 给正式定义 |
| Structural width / treewidth | 在 §3 Preliminaries 定义 | Introduction 给一句直觉 |
| Monolithic WMC | 从未定义 | 在"Method"之前明确定义为"对完整约束图做一次 WMC" |
| Reconvergent path | 图形化说明但未命名 | 在 Motivation 节给出正式定义 |
| Endpoint score clipping | 在实现细节中提到 | 在 Preliminaries 中说明为什么需要 clipping |
| Constraint graph / primal graph | 在 §3 定义 | 第一次出现时加 "(also known as primal graph)" |
| Horn clause (grounded) | 未定义 | 在 §3 中加定义：grounded = no universally quantified variables |
| Boundary-last elimination order | 文中有描述但未命名 | 在 §3 中正式定义 Π(M_j) |
| Min-fill heuristic | 在实现细节中提到 | 在 §3 中解释这是 treewidth 的启发式上界 |

---

## 七、时间估算

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| 阶段 1 | E1（GO mf/GO bp 精确性验证）| 1-2 周（数据处理 + 实验） |
| 阶段 2 | E2（rank 自一致性）+ E3（GO mf CAFA）| 2-3 周 |
| 阶段 3 | E4（WMC solver 比较）+ E5（BP 精度）| 1-2 周 |
| 阶段 4 | 论文重写（Abstract + Introduction + Related Work）| 1-2 周 |
| 阶段 5 | 重写 Experiments 章节 + Discussion | 1-2 周 |
| 阶段 6 | 内部审阅 + 修改 | 2 周 |
| **合计** | | **约 8-12 周** |

JAIR 建议在所有实验完成后再投稿（不像会议有 rebuttal 机制缓冲）。

---

## 八、投稿前检查清单

- [ ] E1：GO mf/GO bp 精确性验证（depth>0 实例）完成
- [ ] E2：大本体 rank 自一致性数据完成
- [ ] E3：GO mf CAFA 下游评测完成
- [ ] E4：ADDMC/DPMC/d4 比较完成（OOM/timeout 也算）
- [ ] E5：loopy BP 精度对比完成
- [ ] Abstract 不假设读者知道 WMC
- [ ] Introduction 前两段对 ML 读者可读
- [ ] 所有术语在第一次使用时有定义
- [ ] Related Work 覆盖 Goral et al. (2024) 和 Onaka et al. (2025)
- [ ] 解释 GO cc 退化原因并重新定位其实验角色
- [ ] Algorithm 1 说明 BUDGET-EXHAUSTED 后的处理
- [ ] TT 模式的 backward recovery 代价有明确分析
- [ ] Table 4 有统计检验（标准误差）
