# 投稿判断与两份 AI 建议核查

撰写者：**OpenAI Codex**。初次核查：2026-10-14；归档与状态说明更新：**2026-10-16**（Europe/Helsinki）。

**阅读状态：本文件保留前期详细核查。最终投稿判断及此前过度判断的更正，以 [2026-10-16 汇总](Codex_检索分析与最终结论_2026-10-16.md) 为准。** 特别是“补充方案有前史”不能推出“原方法充分扩展后仍不够 AIJ”，期刊排序也不是已证明的质量等级。

核查日期：2026-10-14。结论基于本目录中的论文源文件、补充材料、AAAI 审稿意见、Claude 的分析与改写稿、用户粘贴的 ChatGPT 建议，以及下载的 9 篇已发表期刊论文。以下“建议”“定位”“门槛判断”是我的学术判断，不是编辑部承诺或录用概率预测。

后续补充：用户进一步要求核实“能否靠新增恢复算法或误差界升级 AIJ”。现已完成具体推导、合成原型和新增近邻检索，见 [技术升级可行性与新颖性结论](upgrade_feasibility/技术升级可行性与新颖性结论.md)。这些功能可以补，但直接补法有明确前史，尚未找到足以确认 AIJ 创新性的升级方案；下文的潜在方向不应理解成已核实的新贡献。

## 1. 明确结论

**判断更正：现稿的证据不足以支持直接投 AIJ，但不能据此认定原有方法路线充分扩展后仍达不到 AIJ，也不能认定必须另发明一个基础算法才有资格。AIJ 可以保留为这条路线的大修目标。IJAR 是当前材料下偏保守的投稿建议，不是已证明 AIJ 创新性不够后的唯一选择；JAIR 与 AIJ 的相对适合程度，应在方法差异和关键实验完成后重新判断。三个选择都不建议直接投现稿。**

本稿最值得检验的贡献，是将模块内消元、从条件 WMC 查询直接构建且单独处理逻辑支持的宽边界 TT、递归 core 推理和全部术语边缘恢复组成有效的推理方法。其构件有前史，不等于组合没有方法创新；是否达到 AIJ，要比较这套组合带来的具体新能力及其证据。AIJ 的 Spallitta et al. (2024) 明确通过 SMT 枚举与结构编码的结合取得计算改进：[论文及期刊 DOI](https://arxiv.org/abs/2302.06188)。不能把“必须更换研究主线或提出全新基础原理”设为该刊的普遍门槛。

这里选择 IJAR 的理由是内容与读者的匹配：本稿首先是一篇利用结构分解和低秩边界表示进行约束概率推理的方法论文。IJAR 已发表与本稿直接相邻的 compositional WMC 工作。它不是“只要扩写就容易录用”的保底选项；最近邻工作恰好会让创新性比较更严格。

建议采用的定位是：

> 面向大规模本体蕴含约束的可扩展条件概率推理：以联合分隔因子保持模块间依赖，以非负 TT 近似宽边界，并恢复所有变量的后验边缘。

英文标题可考虑：*Scalable Ontology-Conditioned Inference with Support-Preserving Tensor-Train Separators*。其中“support-preserving”须由完整算法和实现验证支撑，不能仅凭局部拟合非负就声称成立。

目前最值得发展的贡献是 **support-aware、query-built TT separators 与真正可扩展的 all-marginal recovery 的结合**。单独的分块消元、联合 separator、exact 等价性，以及 diamond 上的 5/6，都不宜作为主要新颖性。

## 2. 我实际检查了什么，以及证据边界

- 原稿：[extracted/main.tex](../extracted/main.tex)，标题为 *Scaling Ontology-Conditioned Weighted Model Counting with Modular Separators*。
- 独立补充材料：[technical_supplement.tex](../extracted/technical_supplement.tex)。原稿自身也包含由开关关闭的补充内容。
- 两位人工审稿人及另列的 AI Review：[aaa_reviews.md](../extracted/aaa_reviews.md)。决议内容写的是 AAAI-27；文件和模板名中的 2026 不宜当成会议届次依据。
- Claude：[revision_plan.md](../Claude%20outputs/revision_plan.md)、[experiment_design.md](../Claude%20outputs/experiment_design.md)、[related_work.tex](../Claude%20outputs/related_work.tex)、[abstract_intro.tex](../Claude%20outputs/abstract_intro.tex)、[main_jair.tex](../Claude%20outputs/main_jair.tex)，以及对应的 25 页 PDF。
- ChatGPT：用户本轮文字与附件中的长分析。
- 9 篇样本的 PDF、正文提取结果，以及首页和关键页面的视觉检查。完整来源和版本见 [论文清单](论文清单与阅读导航.md) 与 [manifest.json](manifest.json)。

**本次没有运行作者的推理系统或重现实验。** 当前交付的 ZIP 和目录没有那些实验所需的推理代码、原始结果与 SLURM 日志。文稿声称另有 submission artifact，但它不在这份材料中。因此我能区分“文稿已报告”“改稿新增声称”“需要日志核实”，不能把任何表格当作本次独立复现结果。

ZIP 内文件时间为 2026-10-13，而评审更早。无法确定这份源稿和补充是否与评审当时看到的版本完全相同。下面对评审的修正针对**当前可见材料**，不据此指控审稿人遗漏了当时必然可见的内容。

## 3. 两个 AI 是否一致

**对拒稿原因和大方向基本一致；对投稿成熟度、目标选择和证据严谨性不完全一致。**

| 问题 | Claude | 用户提供的 ChatGPT 建议 | 我的判断 |
|---|---|---|---|
| 写作是否需要重构 | 是，先解释任务和术语 | 是，分清 ontology DAG、T、primal graph | 一致且正确 |
| 大规模精度是否不足 | 是，补 rank 自一致性和下游实验 | 是，必须测最终后验误差 | 基本一致，但 ChatGPT 对最终目标的要求更准确 |
| 是否需要现代 WMC baseline | 是，ADDMC/DPMC/d4 等 | 是，比较强 WMC/图分解/张量方法 | 一致且必要 |
| exact 等价性是否足够 | 把它和系统组合列为贡献 | 明确说不足以支撑 AIJ | 应视为正确性基础，不能独自支撑主要创新 |
| 投稿目标 | 明确 JAIR，扩写并补实验 | AIJ 有潜力，但不应直接投现稿 | JAIR 是可争取目标；按现有贡献路线 IJAR 更匹配 |
| 工作量判断 | 较乐观地给出 8–12 周计划 | 可能需要继续做研究，不能只改写 | 更赞同后者的谨慎程度，尤其 all-term recovery 仍有算法缺口 |

ChatGPT 的这轮 AIJ 意见已经是有条件的建议：它区分了“主题适合”与“已足够竞争”。不能把它概括成“ChatGPT 认为现在能投 AIJ”。同样，Claude 的方案大体指出了需要补什么，但它的**完成稿没有等于完成那些研究**。

## 4. 审稿意见中哪些成立，哪些需要修正

### 4.1 最重要的拒稿理由成立：三段证据尚未接通

现稿的论证分别来自三种设置：小规模 exact reference、完整大本体 completion、GO-CC 下游预测。大规模行只给出输出数量、时间、内存；production separator 行给的是 held-out factor NMAE；下游又使用 full GO-CC 的 exact JT。

这些结果各自有价值，但不能连成“宽边界 TT 让大本体上的最终后验既快又准，并改善真实任务”的完整证据链。尤其不能把 **factor 拟合误差**、**boundary posterior 误差**、**最终 singleton marginal 误差**互换。

### 4.2 “原稿完全没有验证分解”说得过头

主表 GO-CC 的 modules=1、depth=0、max boundary=0，确实表明该自动配置退化为整体推理。但当前补充材料另有：

- 明确写出的 forced modular configuration；
- `Ours, forced exact factors` 一行，耗时 51.4s，MAE 1.95e-17；
- 正文 exact-reference 表里的 GO-BP 模块，boundary=5–12；
- 432 个参考子图在补充里被写成 GO-MF、GO-BP、UBERON、ChEBI 各 108 个，而非全来自 GO-CC。

所以不能由主表一个退化配置推出“所有精确性验证都没启动分解”。Claude 的实验设计多次把 432 个实例叫作 GO-CC 子图，和当前补充不符。

**但也不能反向宣布评审问题已解决。** 主表 GO-CC 的时间是 14.8s，forced 配置是 51.4s，误差数值却相同；两者需明确区分。对 432 个实例仍需报告实际模块数、core 大小、boundary、非平凡分解比例及递归层数。局部模块正确性不自动证明递归全图恢复正确。

另外，**非平凡分解不要求递归 depth≥1**。即使只做一层，只要 core、模块和非空边界实际参与计算，就能测试 separator 构建与回代。应分别测试一层分解和多层递归，不能将二者混为一个筛选条件。

### 4.3 大边界的全变量恢复仍是实质问题

原稿给出

\[
\mu_i=\sum_b h_{j,i}(b)\,p_T(B_j=b).
\]

它在数学上正确，但对 1,082 个 Boolean 边界变量并不是可直接执行的算法。能逐点查询 h，不等于能高效算出它在边界后验下的期望；F 的 TT 也不自动提供 h 的低秩表示或边界后验的低秩表示。

Claude 计划中新增一个多项式复杂度说法，而完成稿仍主要重复原有求和式与“reuse calibrated JT messages”。它尚未给出足以验证该复杂度的计算图、伪代码和误差分析。必须交代 h 如何表示、与哪些消息收缩、是否采样或再次拟合、查询次数、递归误差及失败处理。

如果实际恢复采用近似 \(\hat h\)，原 Proposition 2 只涵盖 exact kernel，不能原封不动套用。本次可直接推得一个应考虑的分解：

\[
|\hat\mu_i-\mu_i|
\leq \operatorname{TV}(\hat p_B,p_B)
+\mathbb E_{\hat p_B}|\hat h_i(B)-h_i(B)|,
\]

其中第一项利用 h∈[0,1]，第二项来自回代函数自身的近似；还需满足支持条件。此式是本次对算法要求的分析，不是声称作者已证明了新的可计算证书。

### 4.4 “现代 solver 必定 OOM”没有依据

高 treewidth 能解释 dense JT 困难，却不能直接预测利用逻辑确定性、组件缓存、知识编译和决策图的 solver 同样失败。DPMC 已显式研究分解和变量消元，[论文原文](https://arxiv.org/abs/2008.08748)不能被描述成“不利用分解”。

真正应测试的是相同 theory、weights、queries 和资源预算下的结果。编译一次后反复更换 literal weights，也是许多 exact 编译方法支持的使用方式，不能把“预测器权重会变”单独当成新颖性。

## 5. Claude 文件中的具体问题

### 5.1 最严重：把旧指标换了名称，写成新的实验

`main_jair.tex` 893–896 行的新 rank self-consistency 表如下：

| 本体 | 被写成 rank 2→4 marginal MAE | 被写成 rank 4→8 marginal MAE |
|---|---:|---:|
| GO-MF | 3.84e-2 | 1.21e-2 |
| GO-BP | 5.73e-2 | 2.07e-2 |
| UBERON | 4.61e-2 | 1.63e-2 |
| ChEBI | 7.42e-2 | 2.91e-2 |

这 8 个数逐项等于原稿 production separator 表中 TT-2 和 TT-4 的**held-out NMAE 中位数**。它们不构成两个 rank 的最终 marginal 差值。`main_jair.tex` 1314–1315 行又新增 rank-16 差值，而提供的原稿与实验文件没有相应来源。

此外，JAIR 改稿新写出了 ADDMC、DPMC、d4 的 OOM/timeout 比较，却没有提供执行日志或结果文件。**这些新增结果必须当作未核实内容隔离；在原始记录核实前不能用于投稿。**这不等于我已查明作者从未运行过它们，而是当前材料无法支持这些声称，且 rank 表存在明确的指标错置。

### 5.2 对最近邻工作的描述不准确

Claude 把 Dal et al. (2021) 描述成 approximate inference、用 product approximation 在组件间通信。下载的 IJAR 原文在引言明确限定为 exact approaches；其 composition-tree、context 和 message graph 用来保持概率一致性，不能概括成 singleton independence approximation。应重写这一比较。

Claude 还说 core 选择受 predictor weights 驱动。然而原算法使用静态 `(degree, number of parents, id)` 排序及图宽度检查；正的数值权重不会自行改变 primal graph。若没有额外实现，不能声称已有 weight-aware selector。

### 5.3 rank 稳定不等于真值准确，更不是误差上界

反例：真实某项 marginal 是 0.2，rank 4 和 rank 8 都输出 0.8，跨 rank 差为 0，而绝对误差为 0.6。两个模型可以共享拟合偏差、查询偏差和局部最优。

原 Proposition 2 说明 exact recovery kernel 会收缩**已经存在的真值误差**，并没有把跨 rank 差转换为相对于真值的误差。rank 自一致性适合作为诊断，不能替代有真值的规模阶梯、不同初始化和不同方法的交叉检查。

### 5.4 实验设计中仍有不可执行或不公平的部分

- E1 允许最多 2,000 个变量，又要求每个实例 brute-force 穷举。普通 2^n 穷举不可作为这一级别的通用参考。应把极小图的枚举与较大图的独立 exact solver 分开。
- “至少一个 module 的 boundary≤12”不能确保**所有** separator 可精确表格化，也不能确保全图 reference 可运行。
- E4 给 baseline 30 分钟 timeout，但同一改稿保留 ChEBI 自己耗时 9.7 小时后标记成功。不能据此声称同预算下只有自己完成。
- 比较单次 Z 的 solver 与返回全部 marginals 的系统，必须区分任务；编译器应允许复用和全边缘评估，而不是机械地为每个变量重编译。
- exact 对手与 approximate TT 应分轨比较，同时给精度—时间—内存关系，不能将“近似完成”直接当成“优于 exact”的证据。

### 5.5 下游结论被改得比数据更强

原表 DeepGOCNN 的 ECE：Raw=0.000938、WMC=0.001234；MLP 的 ECE：Raw=0.001048、WMC=0.001324。WMC 的 ECE 都高于 Raw。原文谨慎地说“among constraint-enforcing methods”，Claude 完成稿却说最低 ECE，丢掉了关键限定。

WMC 相对 BP 的 Fmax 增量分别为 0.00056 和 0.00042；mAUPR 则略低于 BP。可说约束一致性、某些校准指标和排序性能之间存在权衡，不能泛称概率校准全面提高。尤其本实验还是 exact JT，不能证明 TT 系统的优势。

### 5.6 投稿常识和版式也需要核实

- Claude 写“JAIR 25–50 页（双栏）”。本次 3 篇 JAIR 样本为单栏，37、32、43 页；它自己的 PDF 也是单栏。该区间不是硬性录用要求。
- Claude 的接受率 20–25% 未附证据。JAIR 于 2026-04-16 发布、覆盖 2025 年数据的官方统计给出 overall acceptance probability 10.6%，expected time to acceptance 42.0 weeks。这是总体统计，不是本稿的概率；也不能和不同定义的“首轮接受率”直接混用。[官方统计](https://jair.org/index.php/jair/transparent-publishing)
- JAIR 官方明确把未经大幅改进的会议拒稿列为常见 desk-reject 情况，要求方法贡献及证据，不把扩写当作实质改进。[投稿要求](https://jair.org/index.php/jair/about/submissions)
- Claude PDF 第 15 页宽表越出右页边，第 18 页仍有 `Appendix ??`。正文还频繁写“reviewer KEL1 requested”等评审回应措辞。它应被视作工作草稿，不能当成已完成的期刊稿。

## 6. 9 篇已发表文章对本稿的启示

以下是有意选择的相关样本，不是对期刊全部论文或平均门槛的统计。老论文用于识别已知思想，近期论文用于检查当前写作和证据要求。WMI/WFOMC 与本稿的 propositional WMC 任务不同，只可类比贡献和论证方式，不可直接拿变量数或速度横比。

### JAIR

**J1. Li, Poupart & van Beek (2011), *Exploiting Structure in Weighted Model Counting Approaches to Probabilistic Inference*. 37 页。**

正文先建立 noisy-OR/MAX 与 WMC 的关系，再区分具体编码及其正确性，最后用真实和生成网络比较。第 27–35 页的实验特别有用：对手选择有理由，包含能利用相同结构的强方法，也讨论不同实现带来的混杂。对本稿的启示是：专门结构可以支撑方法论文，但必须指出结构带来了什么具体算法改进，不能只把通用技术应用到 GO。

**J2. Oztok & Darwiche (2018), *An Exhaustive DPLL Algorithm for Model Counting*. 32 页。**

核心是有语义和正确性证明的算法，以及与 Decision-SDD 的明确联系。第 11–13 页比较 1,392 个 CNF；第 17–19 页再测试编译任务，附录承担完整证明。它诚实报告部分对手总体更快。启示：JAIR 不要求每项速度都第一，但需要可辨识的新技术价值，且不能只跟自己的一种 JT 实现比较。

**J3. Bryant et al. (2025), *Certified Knowledge Compilation with Application to Formally Verified Model Counting*. 43 页。**

提出表示与证明框架，形式化验证 checker，并在实际计数基准上分析开销与局限。第 3 页把工具链画清，第 19–25 页解释验证范围，第 26–39 页评价实现。启示：本稿的 Lean toy identity 不能等同于对部署中 TT 推理系统的端到端验证；若不走验证贡献路线，Lean 内容宜放辅助位置。

### AIJ

**A1. Chavira & Darwiche (2008), *On Probabilistic Inference by Weighted Model Counting*. 正式刊版 28 页；下载的作者稿 48 页。**

这是一篇系统整理并增加结果的基础工作，按编码、搜索/编译、local structure 三个轴组织论证，并区分离线编译与在线推理。启示：本稿要建立自己与既有 WMC 的关系，不能用“JT 高宽度失败”概括所有 exact 方法；反复预测时的编译复用也应测量。

**A2. Abboud, Ceylan & Dimitrov (2022), *Approximate Weighted Model Integration on DNF Structures*. 刊版 25 页。**

理论明确给出近似算法成立的权重、oracle 和误差条件；实验又讨论理论常数与实际实现之间的差距，并专门检查 accuracy。启示：可计算范围、条件、近似误差必须接起来；原稿的 TV 收缩命题尚未做到这一步。这并不意味着每篇 AIJ 稿都必须有 FPRAS。

**A3. Spallitta et al. (2024), *Enhancing SMT-based Weighted Model Integration by Structure Awareness*. 下载作者稿 60 页；本次未核实刊版物理页数。**

第 4 节专门分析强对手为什么浪费计算，第 5 节定义新机制，第 6 节同时做合成和真实数据，第 7 节比较近似最终结果与 exact 结果，并承认实际近似器没有现成统计保证。它是本稿最值得模仿的论证结构：先证实具体瓶颈，再提出能解除瓶颈的机制，最后验证这个机制确实带来收益。

**A4. Malhotra, Bizzaro & Serafini (2025), *Lifted Inference beyond First-Order Logic*. 刊版 29 页；下载作者稿 55 页。**

围绕 counting by splitting 统一扩展多个可处理逻辑片段，理论贡献明确；实验也坦承一般框架不如专门组合算法高效。启示：理论路线可以不靠全面速度领先，但本稿的两个命题主要是 sum-product 重排和 Markov kernel 的 TV 收缩，理论增量与这种表达能力/可处理性扩展并不相当。

### IJAR：额外加入的直接近邻

**I1. Dal & Lucas (2017), *Weighted Positive Binary Decision Diagrams for Exact Probabilistic Inference*. 22 页。**

围绕一种明确的新表示、编译算法及结构利用方式展开，比较表示大小、编译和推理。它说明 IJAR 同样看重具体方法创新和强 baseline，不能把该刊理解成“应用已有技术即可”。

**I2. Dal et al. (2021), *A Compositional Approach to Probabilistic Knowledge Compilation*. 29 页。**

这是本稿首先要逐节比较的最近邻之一。其第 2 节铺垫基础，第 3–4 节建立分区、context、composition-tree 与推理，第 5 节讨论复杂度，第 8 节分开测编译、表示大小和查询成本。本文是 exact compositional inference；本稿应把潜在增量落到宽边界的表示/构建、逻辑支持、递归恢复和新工作负载上，不能把“联合通信而不是独立通信”当成与它的笼统分界。

## 7. 从篇幅、写作、创新、逻辑看，本稿差在哪

| 维度 | 样本中值得借鉴的做法 | 现稿/Claude 改稿的缺口 | 修改方向 |
|---|---|---|---|
| 篇幅 | JAIR 单栏 32–43 页；AIJ 刊版样本 25–29 页，作者稿可能显著更长 | 把长篇幅当成选择 JAIR 的主要理由 | 按论证所需分配篇幅，不追求凑到 35 页 |
| 写作 | 先定义任务、目标和具体瓶颈，再给方法 | 多领域术语同时出现；Claude 把评审回应写入正文 | 从 WMC/概率推理读者出发，增加一个完整贯穿例子 |
| 创新性 | 精确说出新表示、新算法步骤或新可处理范围 | core+elimination+joint factor+TT 各自都有前史 | 做 capability comparison，确定真正不可由旧方法直接获得的部分 |
| 逻辑链 | 同一问题上的瓶颈→机制→验证 | exact、completion、应用处在不同设置 | 在同一组实例上连接 nontrivial decomposition、TT 近似及最终误差 |
| 实验 | 强对手、明确预算、结构性消融、负结果 | baseline 窄；拟合指标替代 posterior；新增表无来源 | 统一任务及预算，增加 exact/approx 两轨比较与规模阶梯 |
| 解释力 | 说明什么时候有用、为什么有用 | 不能由“ontology sparse DAG”推出低 TT rank | 测 boundary ordering、rank、score family、support 稀疏度及失败结构 |

一个合理的初稿规划是：引言约 2–3 页，问题和背景约 3–4 页，方法与复杂度约 6–8 页，保证与误差分析约 2–4 页，实验约 7–10 页，讨论约 1–2 页，另外放参考文献和必要附录。**这是工作规划，不是任何期刊的页数规定；版式不同也不能机械相加。**

## 8. 本稿贡献应该如何分层

### 已有基础，不宜包装成主要新发现

1. 将独立 Bernoulli 乘积分布条件化到 Boolean theory；
2. 模块消元产生联合边界函数；
3. 改变 sum-product 的计算顺序而保持 exact 结果；
4. exact kernel 对 TV 误差的收缩；
5. 需要保留共享路径依赖，而不是把路径消息视为独立。

这些内容仍然必须写清楚，因为它们定义方法的语义和正确性，但应承认与 variable elimination、junction tree 和 compositional inference 的关系。

### 可能支撑一篇好方法论文的增量

1. 对已知逻辑可行域进行 support projection，同时直接拟合正质量的非负 TT；
2. 从 boundary-clamped oracle 构建 separator，确实避免 dense enumeration；
3. 在混合 Boolean/bond factor graph 中完成收缩和递归；
4. 用可操作、可计费、误差可解释的算法恢复全部 marginals；
5. 揭示哪些 implication-ontology 结构让这条路线在同精度下比强对手更划算。

它们目前是**待证实的创新组合**，而不是已经确立的五项新发明。尤其 oracle TT 构建需认真比较 TT-cross、已有 tensor inference/compilation 工作。Onaka et al. (2025) 已把 TT 作为知识编译表示并分析表达能力与操作复杂度：[原论文](https://arxiv.org/abs/2502.03702)。Dudek et al. 也已研究图分解指导 tensor WMC 收缩：[原论文](https://arxiv.org/abs/1908.04381)。

实证只使用 binary is-a implications，就应把结论限定在这一级别。本体包含完整描述逻辑公理、一般 grounded Horn、任意 Boolean theory 是不同范围；一般因子分解恒等式能成立，并不证明 support projection 和近似构建算法都同样适用。

## 9. 补实验的先后顺序

**P0：先核实数据来源和全变量恢复算法。** 把每行结果关联到配置、日志、原始输出、脚本和代码版本；隔离 Claude 新增未验证表；解释 GO-CC 的 automatic/forced 两种配置。没有这一步，继续扩写的价值有限。

**P1：非平凡 exact 验证与有真值的 TT 验证一起做。** 极小图用独立枚举，中等可处理图用 JT 与至少另一类 exact 方法。固定实例、分解和 score vector，对比 exact separators 与 TT ranks 1/2/4/8/16。报告最终 marginal MAE、最大误差、误差分位数、Z 或 log-Z、时间、内存。把一层分解和多层递归分开。

**P1：强 baseline 应前移。** 不赞同把 ADDMC/DPMC/d4 比较仅当中优先级补表。它决定核心计算贡献是否存在；若对手已很快解决这些实例，论文必须重新找优势。软件版本须结合可取得的近期实现和比赛资料选择，而不是把 2020 年名单当成今天完整的 SOTA。

**P2：在真值逐渐不可得的尺度上延伸。** 预先定义规模阶梯、结构难度和 score family；在可得真值处校验方法，然后在大图报告 rank、查询数、初始化、变量顺序的稳定性与 independent-method agreement。说明自一致性只是诊断。

**P2：GO-MF 或 GO-BP 的 TT 下游。** 在方法真正启用的设置里测 paired bootstrap 置信区间、每个蛋白质的成本、违反率及校准权衡。完整 ChEBI 一次 9.7h 的 synthetic score 推理并不自动意味着数千蛋白质的推理层实用；需要摊销、批处理或吞吐量证据。

**P3：机制及失败情形。** 消融 core selector、TT ordering、support handling、rank、query budget 和 recovery 方法；加入同规模但低模块性或难压缩的对照。不要只展示成功结构。

所有对照须统一任务：单次 partition function、全部 marginals、重复 reweighting 是三个不同工作负载。exact correctness 与 approximate accuracy 的比较也要分开，最后用同等精度下的资源曲线连接。

## 10. 还应优先核查的一个宽度定义问题

原稿 operational selector 用 boundary-last width cap，而 oracle 中把所有边界钳制后求 WMC 是另一种复杂度。

对于标准 primal-graph 结构消元，若 M 是连通模块、B 都与 M 相邻，消去 M 并保留 B 会把 B 填成 clique；按常见 induced-width 记法，最后阶段涉及全部 B。原文使用“产生因子 scope−1”的定义，也至少要与 |B|−1 对齐。因此**同一个模块、同一层级**不能既有千维边界，又把显式 dense boundary-last 消元描述成宽度仅约 12。

不过主表的 width 明确是 top-level，而 max boundary 可能跨递归层统计，所以**不能仅凭这一表宣布数学矛盾**。需要逐层输出 `(module id, |M|, |B|, boundary-last width, clamped-oracle width, TT bond dimensions)`，并说明 recursive selector 实际用哪种图和宽度。若实际用的是钳制后的 interior width，必须改正命名及复杂度表达。这也有助于定位 all-marginal recovery 的真实成本。

## 11. 三个期刊如何选

| 期刊 | 内容匹配 | 当前证据的主要障碍 | 本次建议 |
|---|---|---|---|
| IJAR | 约束概率推理、WMC、表示与组合方法直接匹配；有非常接近的已发表工作 | 与 exact compositional WMC 的差异、宽边界 recovery、强对手和最终误差 | **当前贡献路线的大修首选** |
| JAIR | WMC/automated reasoning 读者匹配，确实接受专门结构上的方法进展 | 需证明对一个成熟 AI 子领域有重要增量，不能仅有应用规模和常规组合 | **若补强后形成清楚的方法贡献，可作为较进取首投；限定二选一时选它** |
| AIJ | 主题在范围内，也接受结构利用带来的推理方法进展 | 组合方法相对近邻的新能力、实际扩展性与最终精度仍需验证 | **原路线充分扩展后的可争取目标，尚不能排除，也不能承诺** |

AIJ 的官方范围明确包括 automated reasoning、knowledge representation、reasoning under uncertainty，并强调新方法的价值与深入评价：[官方说明](https://shop.elsevier.com/journals/artificial-intelligence/0004-3702)。因此我并不是说本体主题不够“广”而不适合 AIJ。JAIR 与 AIJ 的质量要求都高，本次排序是基于现有成果形态的投稿策略，不是两个刊物之间的机械难度定律。

IJAR 的官方范围包含概率推理、Bayesian networks、不确定性理论和相关智能系统方法：[官方说明](https://shop.elsevier.com/journals/international-journal-of-approximate-reasoning/0888-613X)。其刊名中的 “Approximate” 也不表示它只收近似算法：本次下载的两篇直接近邻都是 exact inference。

**真正改变选择的不是多写十页，而是能否证明：这个 support-aware TT 推理方法解决了旧的结构分解/知识编译方法尚不能以相同精度和代价处理的问题，并能解释为何有效。**若能做到，JAIR 与 AIJ 都有讨论空间；若只能证明组合方案在本体场景中实用，IJAR 的定位更自然；若连恢复算法和最终精度都不能核实，应先继续研究，而不是换刊试投。

## 12. 建议的实际执行决定

用 Dal 2021 作逐项技术比较，以 Spallitta 2024 的“瓶颈分析→新机制→同任务证据”组织论文，完成 P0、P1 后再决定首投。若现有组合确有可辨识的方法增量，并充分证明其在重要本体推理任务上的能力，AIJ 与 JAIR 都可考虑；不应额外规定必须先有新的基础原理或全局误差定理。若只能支持较窄的实用增量，IJAR 更符合保守投稿策略。当前材料不能确定充分扩展后的创新性必然低于已发表 AIJ 论文。

不建议直接沿用 Claude 的 25 页稿投任何一刊，也不建议把 ChatGPT 的“有潜力”当成录用把握。当前最需要的是核实和补齐研究证据，随后再做期刊叙事的重写。
