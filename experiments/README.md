# 实验目录（LUMI）

计划与判断见 [../AIJ改投_分析与实验计划_Claude_2026-10-16.md](../AIJ改投_分析与实验计划_Claude_2026-10-16.md)。
自检和冒烟测试由 Claude 在登录节点运行；sbatch 作业也由 Claude 提交（2026-10-19 起）；脚本名前缀 = 计划编号，名字说明实验目的。
**每一步的结果、结论和对计划的影响记录在 [RESULTS.md](RESULTS.md)**；下表只记状态。

## 目录约定

| 路径 | 内容 |
|---|---|
| `env/` | 容器定义 `wmc_env.yml`、构建脚本、`run_in_wmc_env.sh`、构建产物 `wmc_env.sif` |
| `scripts/` | 实验脚本（`pX_Y_<目的>.py/.sh`），共享代码 `wmc_common.py` |
| `slurm/` | 批处理脚本（`pX_Y_<目的>.sbatch`），从任意目录 `sbatch` 均可 |
| `logs/` | `run.sh` 生成的运行日志（含脚本 sha256、参数、主机、作业号、耗时与内存）；`logs/slurm/` 为 Slurm 输出 |
| `results/` | 小型结果表（tsv/json），随项目保存 |
| `data` → `/scratch/project_462001792/WMC_aaai_rejected/data` | 原始本体、理论、消元序等大文件（scratch 不备份，可由脚本重建） |

**环境**：按 LUMI 建议用 cotainr 把 conda 环境打包成单个 `.sif`，不直接在 Lustre 上装 conda。
**运行**：一律 `experiments/run.sh scripts/<脚本> [参数]`，它在容器内运行并写日志。计算任务用 `slurm/` 中的 sbatch，不在登录节点上跑。

## 运行记录

| 步骤 | 目的 | 命令 | 状态 |
|---|---|---|---|
| ENV | 构建项目容器 | `bash experiments/env/build_wmc_container.sh` | ✅ 2026-10-16，`container OK`（kahypar 未装，cotengra 自动退回，不影响 P0） |
| P0.1 | 下载原稿所用确切版本本体并记录校验和 | `experiments/run.sh scripts/p0_1_download_paper_ontology_releases.sh` | ✅ 2026-10-16，三个版本头与原稿一致，见 `results/p0_1_ontology_release_manifest.tsv` |
| P0.2 | 重建 is-a 蕴含理论，对照原稿术语数/结构统计 | `sbatch experiments/slurm/p0_2_rebuild_isa_theories.sbatch` | ✅ 2026-10-16（作业 22908152）。ChEBI 完全一致；UBERON 14,975 vs 14,977；GO 比原稿多 3.4%（cc）–9.3%（mf），但最大父数完全一致。原稿 GO 数字可能来自另一快照或过滤，无代码无法确认；后续统一用本表的理论 |
| P0.3 | 全图 min-fill/min-degree 宽度对照 12/37/269/46/738 | `sbatch experiments/slurm/p0_3_minfill_width_check.sbatch`（自检由 Claude 运行） | ✅ 2026-10-16（作业 22918811）。ChEBI 738 与原稿一致；GO bp 339 vs 269 |
| P0.4 | 确定性画像：bag 上合法状态数 vs 2^\|S\| | `sbatch experiments/slurm/p0_4_determinism_profile.sbatch` | ✅ 2026-10-16（作业 22919437）。GO mf 稀疏表总量 2^29.6，可精确；UBERON 边缘；GO bp / ChEBI 下界 2^174 / 2^483 |
| ~~P0.5~~ | 原方法溯源 | — | 取消：原始代码不存在 |
| ~~P0.5'~~ | 重新实现原精确模式 | — | 撤销：重心是重新定创新点，新方法待 P0.4 + P1 结果后设计 |
| P1.0 准备 | 编译 d4、下载 Swiss-Prot 2021_04 与 DIAMOND、引擎自检 | Claude 已完成 | ✅ 自检全部通过（见 RESULTS） |
| P1.0a | 难度诊断：合成分数下 BP / Gibbs 对精确 d4 边缘的误差 | `sbatch experiments/slurm/p1_0a_hardness_synthetic_scores.sbatch` | ✅ 作业 22954989；GO bp 2,000 项 d4 全部超时 |
| P1.0c | 构建真实分数（DiamondScore，prop / raw） | `sbatch experiments/slurm/p1_0c_build_diamondscore.sbatch` | ✅ 作业 22954986：77,638 个注释蛋白，测试集 15,523 |
| P1.0a' | 难度诊断：真实 DiamondScore 分数 | `sbatch experiments/slurm/p1_0a_hardness_real_diamond_scores.sbatch` | ✅ 作业 22955060（Claude 提交） |
| P1.1 | 生死关卡：d4 编译完整 GO mf / GO bp / UBERON / ChEBI（46 h，224 GB） | `sbatch experiments/slurm/p1_1_d4_full_ontology_exact_gate.sbatch` | 运行中（作业 22955071，Claude 提交） |
| P1.2 | 完整本体上带证书的全部边缘（冷集剪枝 + 活跃核心 d4），以及 BP 的证书误差 | `sbatch experiments/slurm/p1_2_certified_marginals_full_ontology.sbatch` | ✅ 作业 22955320：GO bp 93% 蛋白得到证书（中位 3.6 s），BP 对 71% 随机蛋白可证误差 > 1% |
| P1.2b | 基线：d4 + 零分数单元子句，在完整理论上求解（与 P1.2 相同的蛋白） | Claude 提交，作业 22969307 | ✅ 与 P1.2 管线等价，剪枝管线作为方法放弃 |
| P1.4 | 构建 DeepGOPlus 1.0.6 稠密分数与词表投影理论 | Claude 运行 | ✅ GO bp 词表 4,474 项，分数和中位 30，剪枝无效 |
| P1.4a | 词表理论 d4 编译一次 + 全部 3,874 个蛋白的精确边缘与 BP 误差 | Claude 提交，作业 22969414 | GO cc / GO mf ✅（精确比 BP 快）；GO bp 编译中 |
| P1.4b | GO bp 词表：BP 对比单调夹逼 Gibbs，200 个随机蛋白 | Claude 提交，作业 22969415 | ✅ 27% 的蛋白 BP 误差 > 0.01（超出参照不确定度） |
| P1.4c | GO bp 词表的宽度与确定性画像 | Claude 提交，作业 22974189 | ✅ 宽度 61，最大 bag 合法状态 2^41.9 |
| P4.0 | 下游：原始 / max-closure / 精确 WMC / BP 对比真实标签（DeepGOPlus，GO cc / GO mf） | Claude 提交，作业 22974188 | ✅ 精确条件化显著降低 Fmax 和 AUPR（口径已由 P4.1b 取代） |
| P4.1a | 规范化输入：别名映射、CNN 与融合分数、标签闭包 | Claude 运行 | ✅ |
| P4.1b | 规范化下游：加入保序回归，3 个命名空间 × 2 种输入 | Claude 提交，作业 22997008 | ✅ **条件化全部显著有害；保序回归与 max-closure 只有千分位级改进** |
| P4.2 | 机制：按深度分层的校准分析 | Claude 运行 | ✅ 条件化深层压低、浅层抬高，随深度加重 |
| P4.3 | 学习式约束模型 C 对比公平的非 WMC 校准基线（探索性划分） | Claude 提交，作业 23032412 | ✅ C 劣于全部基线，但 **C 的参数化有误、与基线不嵌套（P4.4），停止结论已撤回** |
| P4.4 | 诊断 C 的参数化：矩匹配 θ\* 与 C 形式的差距 | Claude 运行 | ✅ θ\* − logit(s) 随深度增至 +8；按深度仿射形式 R² 只有 0.19 |
| P4.5 | 修正后的嵌套模型 C2（θ\* + 学到的残差） | Claude 提交，作业 23037331 | ✅ 对保序回归 micro AP 4/4 显著更好；对最好的校准基线 3/4 显著更好（GO cc CNN 例外） |
| P4.6 | 消融：同特征、同参数量的非 WMC 模型 | Claude 提交，作业 23038917 | v1 数值有误，已由 v2 取代 |
| P4.5 / P4.6 v2 | 修复稳定 log Z 后重跑，新增 θ\* 特征与全向量 PCA 对照 | Claude 提交，作业 23044088 / 23044090 | ✅ 结论不变；3/4 组 C2 显著胜过所有非 WMC 方法；收益来自约束下的联合预测；GO cc CNN 为弱点 |

（命令均在项目根目录 `/projappl/project_462001792/WMC_aaai_rejected` 下执行。）
| P4.7a–d | (b) BP / Bethe 能否替代精确推理（诊断） | Claude 运行 | ✅ 矩匹配病态；给定 θ\* 时 BP 误差仍有 0.2–1.0 |
| P4.5b / P4.6b | 未收敛的两组用 1,000 步复跑 | Claude 提交，作业 23050076 / 23050077 | ✅ 收敛，结论不变 |
| P4.7 | C2 精确 vs BP（同一训练器） | Claude 提交，作业 23050078 | v1：BP 实现有缺陷，结论已撤回 |
| P4.8 | log-odds BP 修复与自检；BP 原生矩匹配检查 | Claude 运行 | ✅ 两节点反例和强权重树都正确；BP 原生矩匹配在 GO mf 上 6/6 不收敛 |
| P4.7d v2 / P4.7 v2 | 用 log-odds BP 重跑：固定 θ\* 处的保真度，以及两种起点的端到端对比 | Claude 提交，作业 23062916 / 23062910 | ✅ C2 收益依赖精确矩匹配；跳过矩匹配时 BP≈精确，但两者都不如保序回归 |
| P5 | 按时间划分的独立确认（Swiss-Prot 2021_04 → 2024_01，无先验知识目标；协议先锁定，修订 1 处理了长序列） | Claude 提交，作业 23087245–23087247 | ✅ **按预注册判据确认**（H1 3/4，H2 2/4）：GO mf 全面胜出；GO cc 主要改善 AP |
| P6.0 | (d) 热核诊断：核大小、核外赔率质量 Λ、d4 编译热核的时间（GO bp vs GO cc/mf） | Claude 提交，作业 23100728 | ✅ GO bp 热核在 τ = 0.01 时大多数秒级可编译，约 1,000 项以上多数超时；Λ 很大，截断证书不可用 |
| P6.1 | 混合推理（热核精确 + 核外 BP）自检 | Claude 运行 | ✅ 否定：误差 0.02–0.15，核外矩匹配不收敛 |
| P6.2 | 只在热核上的 C2 vs 完整精确 C2（GO cc/mf，探索性划分） | Claude 提交，作业 23101076 | ✅ τ ≤ 0.01 时 4/4 与完整 C2 无差异；只需逐蛋白热核的精确推理 |
| P6.3 | GO bp 上的热核 C2（自适应 τ）vs 9 个方法，探索性划分 | Claude 提交，作业 23105446 / 23134223 | 融合 ✅：AP 和 Brier 对全部 9 个方法显著更好，Fmax 持平；CNN ✅：AP 显著优于保序回归，但不胜逐术语 Platt / PCA32 / 邻域 LR，logloss 输给逐术语 Platt |
| P6.4 | 诊断：C2 在 CNN 输入上输给逐术语 Platt，是否因为残差只按深度共享？比较 C2 + 逐术语偏置 / 仿射、以逐术语 Platt 输出为输入的 C2；GO cc / GO mf × CNN / 融合，探索性划分 | Claude 提交，作业 23148482 | ✅ C2+逐术语仿射：AP 在 4/4 设置显著优于逐术语 Platt，3/4 设置 ≥ 原 C2（GO mf 融合略差）；先 Platt 再 C2 不可取 |
| P6.5 | GO bp 热核 C2 的截断敏感性：固定 P6.3 的参数 w，在 500 个评估蛋白上比较 P6.3 热核与更大热核（τ ≤ 0.007）的预测 | Claude 提交，作业 23148739 | ✅ 融合：固定 w 时对截断不敏感（AP −0.0005）；CNN：固定 w 扩核后 AP −0.144，参数与 τ 绑定 |
| P7.1 | GO bp 时间划分目标（与 P5 相同规则） | Claude 提交，作业 23160691 | ✅ 2,677 个目标，标签数中位 24 |
| P7.2a/b | GO bp 目标的 CNN / 融合分数（重跑 gate a，GO bp 的 gate b） | Claude 提交，作业 23160706 / 23160707 | ✅ gate a 完全复现；GO bp gate b 通过（中位差 0，旧集合 Fmax 差 0.0004）；2,206/2,677 个目标有 DIAMOND 命中 |
| P7.3 | GO bp 锁定确认：热核 C2+逐术语仿射（主）、C2（次）vs 9 个方法；协议 `P7_protocol_go_bp_time_split_confirmation.md` | Claude 提交，作业 23162248；协议已于 2026-10-26 00:14 锁定 | ❌ 未确认：融合 H2/H3 不成立（Fmax −0.015）；CNN 的 AP 显著低于 isotonic。次要 C2：融合 AP 优于全部基线，但 Fmax −0.009 |
| P7.4 | P7.3 失败的事后诊断（时间偏移 vs 过拟合、热核/冷项、蛋白内/跨蛋白） | 脚本已写；CPU 机时超额，未能提交 | ⏸ |
