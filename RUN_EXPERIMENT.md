# System Ablation Experiment - Quick Start

## Run the Experiment

All experiment scripts have been moved to the `experiments/` directory.

### Quick Commands

**Dry Run (100 samples, recommended first):**
```bash
cd experiments
run_dry_run.bat
```

**Full Experiment (all samples):**
```bash
cd experiments
run_full_experiment.bat
```

### Documentation

See `experiments/SYSTEM_ABLATION_README.md` for:
- Detailed usage instructions
- Command line options
- Output file descriptions
- Troubleshooting guide

### Output Files

Results will be saved to `outputs/results/system/`:
- `system_ablation_<timestamp>.json` - Complete results
- `system_ablation_<timestamp>.csv` - Performance metrics (6 configs)
- `system_ablation_compare_<timestamp>.png` - Performance comparison chart
- `system_ablation_timing_<timestamp>.csv` - Timing statistics (IPI attacks only)
- `system_ablation_timing_compare_<timestamp>.png` - Timing comparison chart

### What This Experiment Does

Evaluates 6 system configurations:
1. DPI-only
2. IPI warning-only
3. IPI BERT-only
4. **DPI + IPI warning** (new)
5. DPI + IPI BERT
6. DPI + IPI warning + IPI BERT unified

Measures:
- Performance metrics (accuracy, precision, recall, F1)
- Timing for IPI attack scenarios
- Early warning vs final decision latency

---

**Note:** Temporary run scripts in the root directory can be deleted. Use the scripts in `experiments/` instead.
