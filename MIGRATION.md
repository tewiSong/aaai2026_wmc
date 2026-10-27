# 从 LUMI 迁移到新服务器（2026-10-27）

## 什么在 git 里，什么不在

| 内容 | 大小 | 怎么带过去 |
|---|---|---|
| 代码、协议、结果 JSON、日志、计划和复核文档、原稿 | 约 40 MB | git |
| 派生数据 `experiments/data/{scores,theories,orders,compiled}`，含 θ\* 缓存 | 0.73 GB | 数据包 `wmc_derived_data_2026-10-27.tar` |
| 原始数据 `experiments/data/raw`（UniProt、GO、DeepGOPlus 发布包等） | 6.5 GB | 在新服务器上用脚本重新下载并校验 |
| d4 / DIAMOND 二进制 | — | 在新服务器上用脚本构建或下载 |
| 容器 `.sif`（LUMI 专用） | 1.8 GB | 不带，改用 conda 环境 |
| `experiments/core`（9.2 GB 的崩溃 core dump） | — | 不带，可以删掉 |

数据包在 LUMI 上的位置：`/scratch/project_462001792/WMC_aaai_rejected/data_bundle/wmc_derived_data_2026-10-27.tar`
- sha256：`85cd60324fba7c97a7215fb95f7a65f00c74eb06117c25397bc00a0ab2348950`
- **LUMI 的 /scratch 会被定期清理**，请尽快把它拷走。

## 在 LUMI 上（迁出前）

```bash
cd /pfs/lustrep1/projappl/project_462001792/WMC_aaai_rejected
git remote add origin git@github.com:<你的账号>/<仓库名>.git   # 先在 GitHub 网页上新建一个私有空仓库
git push -u origin main
```

- 建议用**私有**仓库：里面有未发表的稿件和审稿意见。
- 数据包可以选下面任一种方式带走：
  1. **直接拷到新服务器**（最简单）：
     ```bash
     scp <你>@lumi.csc.fi:/scratch/project_462001792/WMC_aaai_rejected/data_bundle/wmc_derived_data_2026-10-27.tar .
     ```
  2. **作为 GitHub Release 附件**（单个附件上限 2 GB，这个包 0.73 GB）：在仓库网页的 Releases 里新建一个 release（比如 `data-2026-10-27`），把 tar 文件拖进去上传。

## 在新服务器上

```bash
# 1. 代码
git clone git@github.com:<你的账号>/<仓库名>.git WMC_aaai_rejected
cd WMC_aaai_rejected/experiments

# 2. 数据目录：放在大容量盘上，用软链接接到 experiments/data
mkdir -p /<大容量盘>/wmc_data && ln -s /<大容量盘>/wmc_data data

# 3. 派生数据
tar -xf /<路径>/wmc_derived_data_2026-10-27.tar -C data/
sha256sum /<路径>/wmc_derived_data_2026-10-27.tar   # 应为 85cd6032…8950

# 4. Python 环境（需要 conda / mamba）
conda env create -f env/wmc_env.yml        # 环境名 wmc
conda activate wmc

# 5. d4（固定 commit，从源码编译；DIAMOND v2.2.6 在第 6 步下载）
bash tools/build_exact_solvers.sh

# 6. 原始数据（约 6.5 GB；也会下载 DIAMOND）并按清单校验
bash scripts/p0_0_fetch_all_raw_data_new_server.sh

# 7. 校验派生数据和整个环境
cd data && grep -v '  raw/' ../data_manifest_sha256.txt | sha256sum -c --quiet && cd ..
env/run_in_wmc_env.sh python -c "import sys; sys.path.insert(0,'scripts'); import wmc_inference, p6_3_go_bp_core_c2_exploratory; from wmc_common import load_theory; print('OK', len(load_theory('dgp2_go_bp')[0]))"   # 应打印 OK 4475
ls tools/bin   # 应有 d4 和 diamond
```

第 6 步结束时会打印缺失或不一致的原始文件。没有输出就说明全部一致。

## 让新的 Claude 接手

在新服务器上，于仓库根目录 `WMC_aaai_rejected/` 启动 Claude Code。它会自动读取 `CLAUDE.md`，里面有：
- 项目现状；
- 待办事项（第一项是 P7.4 诊断）；
- 你的工作约定（中文、自己提交作业、汇报要有大局观等）；
- 环境说明。

第一句话可以这样说：

> 读 CLAUDE.md，按 MIGRATION.md 检查新服务器的环境和数据是否就绪，把 slurm/ 下的 sbatch 改成这台服务器的账户、分区和路径，然后提交 P7.4。

## 备注

- 旧会话的对话记录和 Claude 的记忆文件存放在 LUMI 的 `~/.claude/projects/-pfs-lustrep1-projappl-project-462001792-WMC-aaai-rejected/` 下，没有放进 git，因为对话记录可能含有不宜上传 GitHub 的内容。记忆的要点已经写进 `CLAUDE.md`。如果需要，可以用 scp 私下拷贝。
- `experiments/slurm/*.sbatch` 和 `logs/` 中的绝对路径是 LUMI 的，保留它们作为运行记录。
