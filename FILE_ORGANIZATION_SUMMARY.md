# 系统消融实验 - 最终整理总结

## 当前状态

✅ 所有实验脚本已迁移到 `experiments/` 目录
⚠️ 根目录下有冗余的临时文件需要清理

## 正式的实验文件（在 experiments/ 目录）

### 执行脚本
- **execute_system_ablation.py** - 主执行脚本，支持命令行参数
- **run_system_ablation.py** - 核心实验实现（已存在，无需修改）
- **run_dry_run.bat** - Dry run 批处理（100样本测试）
- **run_full_experiment.bat** - 完整实验批处理（全量数据）

### 文档
- **SYSTEM_ABLATION_README.md** - 详细使用说明

## 根目录下的冗余文件（需要删除）

以下文件都是临时创建的，功能已被 `experiments/` 目录下的文件替代：

**Python 脚本：**
- run_experiment_wrapper.py
- run_dry_run_test.py
- run_full_experiment.py
- run_experiments.py
- run_now.py

**批处理文件：**
- run_dry_run.bat
- run_full_experiment.bat
- RUN_ALL_EXPERIMENTS.bat
- RUN_NOW.bat

**文档：**
- EXPERIMENT_GUIDE.md
- EXPERIMENT_SUMMARY.md

## 清理冗余文件

运行以下命令删除所有冗余文件：
```bash
DELETE_TEMP_FILES.bat
```

这个脚本会自动删除上述所有临时文件。

## 清理后的目录结构

```
D:\PromptInjectionDetection\
├── experiments/
│   ├── execute_system_ablation.py      # 主执行脚本 ✓
│   ├── run_system_ablation.py          # 核心实验实现 ✓
│   ├── run_dry_run.bat                 # Dry run 批处理 ✓
│   ├── run_full_experiment.bat         # 完整实验批处理 ✓
│   └── SYSTEM_ABLATION_README.md       # 详细文档 ✓
├── outputs/
│   └── results/
│       └── system/                     # 实验结果输出目录
├── RUN_EXPERIMENT.md                   # 快速开始指南
├── FILE_ORGANIZATION_SUMMARY.md        # 本文档
└── DELETE_TEMP_FILES.bat               # 清理脚本
```

## 如何运行实验

### 方式1：使用批处理文件（推荐）

**Dry Run 测试（100样本）：**
```bash
cd experiments
run_dry_run.bat
```

**完整实验（全量数据）：**
```bash
cd experiments
run_full_experiment.bat
```

### 方式2：使用 Python 命令

**Dry Run：**
```bash
python experiments/execute_system_ablation.py --dry-run
```

**完整实验：**
```bash
python experiments/execute_system_ablation.py
```

**自定义参数：**
```bash
python experiments/execute_system_ablation.py --dpi-limit 500 --ipi-limit 500 --batch-size 16
```

## 输出文件

实验完成后会在 `outputs/results/system/` 生成 5 个文件：

1. **system_ablation_<timestamp>.json** - 完整结果和元数据
2. **system_ablation_<timestamp>.csv** - 6个配置的性能指标
3. **system_ablation_compare_<timestamp>.png** - 性能对比图（黑白风格）
4. **system_ablation_timing_<timestamp>.csv** - 耗时统计（4行，仅IPI attack）
5. **system_ablation_timing_compare_<timestamp>.png** - 耗时对比图（黑白风格）

## 实验配置

### 6个系统配置
1. DPI-only
2. IPI warning-only
3. IPI BERT-only
4. **DPI + IPI warning** ← 新增
5. DPI + IPI BERT
6. DPI + IPI warning + IPI BERT unified

### 关键特性
- ✅ 使用 tuned IPI warning 参数（非默认参数）
- ✅ 耗时统计只针对 IPI attack 样本
- ✅ unified 系统区分早期提醒和最终判定
- ✅ 黑白论文风格图表（灰阶、hatch纹理、220 DPI）

## 下一步操作

### 1. 清理冗余文件（推荐）
```bash
DELETE_TEMP_FILES.bat
```

### 2. 运行实验
```bash
cd experiments
run_full_experiment.bat
```

### 3. 查看结果
检查 `outputs/results/system/` 目录下的 5 个输出文件。

---

**总结：**
- ✅ 实验脚本已整理到 `experiments/` 目录
- ⚠️ 运行 `DELETE_TEMP_FILES.bat` 清理根目录冗余文件
- ✅ 使用 `experiments/run_full_experiment.bat` 运行实验
