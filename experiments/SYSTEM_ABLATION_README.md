# System Ablation Experiment

## Overview

This experiment evaluates the system-level performance of different prompt injection detection configurations, including:

1. **DPI-only**: Direct prompt injection detection only
2. **IPI warning-only**: Indirect prompt injection warning system only
3. **IPI BERT-only**: Indirect prompt injection BERT classifier only
4. **DPI + IPI warning**: Combined DPI and warning system
5. **DPI + IPI BERT**: Combined DPI and BERT classifier
6. **DPI + IPI warning + IPI BERT unified**: Full unified system with all three components

## Quick Start

### Dry Run (Recommended First)
Test with limited samples (100 DPI + 100 IPI):
```bash
cd experiments
run_dry_run.bat
```

Or using Python directly:
```bash
python experiments/execute_system_ablation.py --dry-run
```

### Full Experiment
Run with complete dataset:
```bash
cd experiments
run_full_experiment.bat
```

Or using Python directly:
```bash
python experiments/execute_system_ablation.py
```

## Command Line Options

```bash
python experiments/execute_system_ablation.py [OPTIONS]

Options:
  --dry-run              Run with 100 DPI + 100 IPI samples for testing
  --dpi-limit N          Limit DPI samples to N (default: None = all)
  --ipi-limit N          Limit IPI samples to N (default: None = all)
  --batch-size N         Batch size for DPI predictions (default: 32)
  --progress-interval N  Print progress every N samples (default: 1000)
```

### Examples

```bash
# Dry run
python experiments/execute_system_ablation.py --dry-run

# Full experiment
python experiments/execute_system_ablation.py

# Custom sample limits
python experiments/execute_system_ablation.py --dpi-limit 500 --ipi-limit 500

# Adjust batch size for memory constraints
python experiments/execute_system_ablation.py --batch-size 16
```

## Output Files

The experiment generates 5 files in `outputs/results/system/`:

### 1. system_ablation_<timestamp>.json
Complete experiment results including:
- Timestamp and data file paths
- Sample counts (DPI, IPI, direct, indirect)
- Tuned IPI warning parameters
- Performance metrics for all 6 configurations
- Timing statistics with explanations
- Output file paths

### 2. system_ablation_<timestamp>.csv
Performance metrics table with columns:
- config, accuracy, precision, recall, f1
- direct_recall, indirect_recall
- tp, fp, tn, fn
- support, direct_support, indirect_support

6 rows for each configuration.

### 3. system_ablation_compare_<timestamp>.png
Black-white bar chart comparing 6 configurations:
- Metrics: Accuracy, Precision, Recall, F1
- Style: Grayscale colors with hatch patterns
- High DPI (220) for publication quality

### 4. system_ablation_timing_<timestamp>.csv
Latency statistics for IPI attack samples only:
- Columns: config, timing_scope, mean_ms, median_ms, p95_ms, min_ms, max_ms, support
- 4 rows showing:
  - DPI + IPI warning (final_decision)
  - DPI + IPI BERT (final_decision)
  - DPI + IPI warning + IPI BERT unified (early_warning)
  - DPI + IPI warning + IPI BERT unified (final_decision)

### 5. system_ablation_timing_compare_<timestamp>.png
Black-white bar chart comparing latencies:
- Metrics: Mean, Median, P95
- Shows early warning vs final decision timing for unified system

## Key Features

### Performance Evaluation
- Evaluates 6 different system configurations
- Measures accuracy, precision, recall, F1 score
- Separate recall metrics for direct vs indirect attacks
- Confusion matrix statistics (TP, FP, TN, FN)

### Timing Analysis
- Focuses on IPI attack scenarios only
- Compares early warning latency (DPI + IPI warning)
- Compares final decision latency (DPI + IPI warning + IPI BERT)
- Shows unified system can provide early alerts while BERT acts as backstop

### IPI Warning Parameters
Uses optimized parameters (not defaults):
- instructionality_weight: 0.334
- authority_conflict_weight: 0.428
- execution_affinity_weight: 0.238
- warn_threshold: 0.125
- block_threshold: 0.446
- Plus pattern bonuses and context bonuses

### Visualization
- Black-white style suitable for academic papers
- Grayscale colors with hatch patterns
- High DPI (220) for publication quality
- Clear labels and legends

## Data Requirements

The experiment requires:
- `data/dpi/test-00000-of-00001.parquet` - DPI test set
- `data/ipi/ipi_test_group_source.json` - IPI test set with scenario types

## Environment

- Python environment: `D:\AnaConda\envs\prompt-injection`
- Required packages: torch, transformers, pandas, numpy, matplotlib
- GPU recommended for faster execution

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python experiments/execute_system_ablation.py --batch-size 16
```

### Missing Data Files
Ensure data files exist:
- `data/dpi/test-00000-of-00001.parquet`
- `data/ipi/ipi_test_group_source.json`

### Import Errors
Make sure you're running from project root or the script will handle path setup automatically.

### Model Not Found
Ensure DPI model exists in `outputs/model/dpi_detector/`

## Implementation Details

### Timing Measurement
- **DPI**: Batch-level timing evenly distributed to samples
- **IPI warning**: Per-sample timing
- **IPI BERT**: Per-sample timing with CUDA synchronization
- **Scope**: Only IPI attack samples (source == "ipi" and attack_type == "indirect")

### Decision Logic
- **Single component**: Direct prediction
- **Combined systems**: OR logic (attack if any component detects attack)
- **Unified system**: DPI → IPI warning → IPI BERT (sequential evaluation)

### Early Warning vs Final Decision
- **Early warning**: System can alert after DPI + IPI warning (faster)
- **Final decision**: BERT provides additional validation (more accurate)
- **Use case**: Early warning for time-sensitive scenarios, BERT for high-confidence decisions

## Related Files

- `run_system_ablation.py` - Core experiment implementation
- `context_utils.py` - Context parsing utilities
- `../src/dpi/dpi_detector.py` - DPI detector
- `../src/ipi/ipi_detector.py` - IPI warning system
- `../src/ipi/ipi_bert_detector.py` - IPI BERT classifier
