# CLAUDE.md：项目交接说明（2026-10-27 从 LUMI 迁出时写）

接手的 Claude 先读这份文件，再按"接手后第一件事"操作。

## 项目是什么

- 作者：Tengwei Song。
- 论文：被 AAAI-27 拒稿的 "Scaling Ontology-Conditioned Weighted Model Counting with Modular Separators"，正在改投 AIJ。
- 原稿和审稿意见：`extracted/main.tex`、`extracted/technical_supplement.tex`、`extracted/aaa_reviews.md`。
- 原始代码、配置和日志都不存在，所有实验都在 `experiments/` 里从头重做。

## 先读什么（按顺序）

1. `AIJ改投_分析与实验计划_Claude_2026-10-16.md`：计划，以及 §7 中核心论点随结果的逐次修订。最新一节是 §7.17。
2. `experiments/RESULTS.md`：每一步的数字、结论和对论文立意的影响。开头的"当前立意"框是现状摘要。
3. `experiments/README.md`：每一步的状态表和目录约定。
4. 两份锁定的协议：`experiments/P5_protocol_*.md`（含修订 1）和 `experiments/P7_protocol_go_bp_time_split_confirmation.md`。
5. `journal_comparison/Codex_*.md`：Codex 对各阶段的独立复核。用户会转来新的复核并问"你同意吗"，回答前要逐条核实。

## 现状（2026-10-27）

- **P0–P4**：
  - 原稿的模块化分隔子路线是计算基础。
  - 下游主线逐步演化成 C2：精确矩匹配得到 θ\*，再在约束模型中学习按深度共享的残差。
- **P5**：GO cc / mf 的时间划分锁定确认通过（H1 3/4，H2 2/4）。只改善排序指标，ECE 变差。
- **P6**：GO bp 上每个蛋白只在热核上做精确后验，可以扩展。CNN 输入下模型参数与 τ 绑定（P6.5）。
- **P7**：GO bp 时间划分锁定确认，主模型 C2+逐术语仿射**未确认**。
  - 次要模型 C2：融合输入下 AP 胜过全部基线，但 Fmax 低 0.009；CNN 输入下 AP 显著变差。
  - 数字见 RESULTS.md 的 "P7.3 结果"。
- **待办，按优先级**：
  1. **P7.4 诊断**：`experiments/scripts/p7_4_go_bp_c2_time_shift_diagnosis.py`，LUMI 上的 sbatch 是 `experiments/slurm/p7_4_*.sbatch`。
     - 当时因 LUMI CPU 机时超额没有运行。
     - 要回答：时间偏移还是过拟合；AP 损失出在热核还是冷项、蛋白内还是跨蛋白；损失与标签稀疏是否相关。
     - 初步线索：新目标的标签数中位 23，旧蛋白是 33；预测总质量与标签数之比从 0.82 升到 1.09。
     - 每个输入约 2–3 小时，128 核。脚本用 `data/scores/p7_core_theta_go_bp_*.npz` 中缓存的 θ\*。如果新机器上 d4 超时导致热核和缓存不一致，会自动重算 θ\*，那会很贵；先检查日志里 `theta_star_recomputed` 的数量。
  2. **新主线由用户决定**。Claude 的建议见计划 §7.17：
     - 以可扩展的精确推理作为计算贡献；
     - 下游部分作为如实报告的案例研究；
     - 不在锁定测试失败后设计新的模型变体去挽救。
  3. 之后改写论文。

## 用户的工作约定（必须遵守，来自之前会话的反馈）

- **用中文回答。**
- **Claude 自己提交所有 sbatch 作业**，用后台轮询等待作业结束、自动唤醒，然后分析和汇报。
  - 不要让用户去提交或运行命令。
  - 原话："不要再交给我提交sbatch了,你自己提交,等任务结束自己唤醒自己"。
  - **提交前检查计算配额。** LUMI 上曾因长作业把 CPU 机时用到 136%。
- **每次汇报都要有大局观**：先说结果对论文创新性、核心论点和方法设计的影响，再给证据，最后说下一步为什么关键。
  - 原话："你每次给我的汇报完全没有大局观"。
  - 结论写进 `RESULTS.md`（含"对论文立意的影响"）和计划 §7。
- **理论和实验对不上时，先诊断根因，不要直接建议换方向。**
  - 原话："理论和实际对不上你不分析原因直接就让我换方向是为什么?"。
  - 先确认被测的参数化能表达理论上存在的解，并且包含了被比较的基线。
- **下论断前排除伪影**：
  - 平凡的输入性质能否解释这个现象；
  - 分组报告，给出明确分母（含失败和超时）；
  - 检查求解器容差；
  - 失败的基线先在它必须精确的情形（比如树上的 BP）测一遍；
  - 不发明无法验证的原因；
  - 汇报中引用的快速实验，脚本和输出都要保存。
- **脚本命名**：`pX_Y_<目的>.py`，sbatch 用同名的 `.sbatch`。一律通过 `experiments/run.sh scripts/<脚本> [参数]` 运行，它会写带 sha256、参数、主机、作业号的日志。
- **锁定协议**：协议文件的 sha256 记录在 `results/*_lock.txt`，评估脚本会核对。锁定后不能修改，偏离必须写成带日期的修订。

## 环境与数据（新服务器）

人工迁移步骤见 `MIGRATION.md`。要点：

- `experiments/data` 不在 git 里，有两个来源：
  - 原始数据（6.5 GB）由 `experiments/scripts/p0_0_fetch_all_raw_data_new_server.sh` 下载，并按 `experiments/data_manifest_sha256.txt` 校验；
  - 派生数据（分数、理论、θ\* 缓存，约 0.7 GB）来自数据包 `wmc_derived_data_2026-10-27.tar`。**不要重算 θ\* 缓存**，P7.3 的 θ\* 用了约 2,000 核时。
- **Python 环境**：用 `experiments/env/wmc_env.yml` 建 conda 环境 `wmc`。`env/run_in_wmc_env.sh` 在没有 `.sif` 容器时直接用当前激活的环境。
- **二进制**：
  - d4 由 `experiments/tools/build_exact_solvers.sh` 构建，固定 commit，需要在 `wmc` 环境里运行（用到 `$CONDA_PREFIX`）；
  - DIAMOND v2.2.6 由 `scripts/p1_0b_download_swissprot_2021_04_and_diamond.sh` 下载；
  - 两者都放在 `experiments/tools/bin/`。
- **DeepGOPlus 的 TF 环境**（`experiments/env/tf/deepgoplus_tf_env.yml`）只在重算 CNN 分数时需要。现有分数已在数据包里，P7.4 用不到它。
- **`experiments/slurm/*.sbatch` 是 LUMI 专用的**：账户 `project_462001792`、分区 `small`、绝对路径 `/pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected/...`。在新集群上要改成新的账户、分区和路径；原文件保留作为 LUMI 运行记录。
- 旧会话的对话记录和 Claude 记忆文件没有放进 git，记忆的内容已经并入上面的"用户的工作约定"。如果用户另外把对话记录（`.jsonl`）拷到新服务器，需要追溯细节时用 `grep` 查找，不要整份读入。
