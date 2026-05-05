# Prompt Injection Detection

A research system for detecting **Direct Prompt Injection (DPI)** and **Indirect Prompt Injection (IPI)** attacks targeting LLM-based applications.

## Overview

As large language models are increasingly deployed in agentic and RAG-based pipelines, prompt injection attacks pose a significant security risk. This project investigates detection algorithms for two distinct attack types:

- **Direct Prompt Injection (DPI)**: Malicious instructions embedded directly in user input
- **Indirect Prompt Injection (IPI)**: Malicious instructions hidden in external content retrieved at inference time (documents, conversation history, web pages)

The core contribution is a **multi-signal fusion framework** for IPI detection that combines three complementary linguistic signals without requiring a full LLM at inference time.

## Algorithm Design

### DPI Detection

A DistilBERT-based binary classifier fine-tuned on a labeled prompt injection dataset.

| Model | Accuracy | F1 | Inference Speed |
|---|---|---|---|
| fastText-style | 96.16% | 0.965 | 0.095 ms/sample |
| TextCNN | 97.53% | 0.977 | 0.439 ms/sample |
| **DistilBERT (ours)** | **99.21%** | **0.993** | 5.18 ms/sample |

### IPI Detection — Core Contribution

A two-layer detection architecture:

```
External Content (retrieved docs / conversation history)
        │
        ▼
[Layer 1] Multi-Signal Warning (rule-based)
  ├── Instructionality Analysis     (weight ~30%)
  ├── Authority Conflict Analysis   (weight ~45%)
  └── Execution Affinity Detection  (weight ~25%)
        │
        ▼
[Layer 2] DistilBERT Final Decision (fine-tuned)
  └── Context: [USER] + [HISTORY] + [DOC]
```

**Three detection signals:**

| Signal | Purpose | Key Patterns |
|---|---|---|
| Instructionality | Detects imperative/command language | "ignore previous instructions", "output system prompt" |
| Authority Conflict | Detects role impersonation and permission override | "you are the system", "override constraints" |
| Execution Affinity | Detects machine-executable control structures | `[system]:`, "for automated assistants only" |

**IPI detection results (4,000-sample test set):**

| Method | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Multi-signal warning (rules) | 47.55% | 78.88% | 0.593 | 57.5% |
| **BERT final decision (ours)** | **85.93%** | **90.14%** | **0.880** | 90.32% |
| Combined pipeline | 45.28% | 97.14% | 0.618 | 52.75% |

**Ablation — signal contribution ranking:**

| Signal | Separation Power |
|---|---|
| Instructionality | 0.0473 (strongest) |
| Authority Conflict | 0.0236 |
| Execution Affinity | 0.0056 |

**Cross-scenario generalization (BERT model):**

| Scenario | F1 | Accuracy |
|---|---|---|
| Conversation history injection | 0.884 | 90.60% |
| RAG basic injection | 0.875 | 89.80% |
| HTML-wrapped injection | 0.905 | 91.95% |
| JSON-wrapped injection | 0.878 | 90.38% |
| Markdown-wrapped injection | 0.849 | 89.63% |

## Repository Structure

```
PromptInjectionDetection/
├── config/
│   └── dpi_config.yaml          # DPI model configuration
├── src/
│   ├── dpi/
│   │   ├── dpi_detector.py      # DPI detector with threshold logic
│   │   ├── train_dpi_model.py   # DistilBERT fine-tuning
│   │   ├── run_dpi_baselines.py # Baseline model implementations
│   │   └── dataset_loader.py
│   ├── ipi/
│   │   ├── ipi_detector.py      # Multi-signal fusion detector (core)
│   │   ├── ipi_bert_detector.py # IPI DistilBERT classifier
│   │   ├── instructionality.py  # Signal 1: instructionality
│   │   ├── authority_conflict.py # Signal 2: authority conflict
│   │   ├── execution_affinity.py # Signal 3: execution affinity
│   │   ├── train_ipi_detector.py
│   │   └── dataset_loader.py
│   ├── data/
│   │   ├── build_ipi_dataset_v1.py  # Random split dataset builder
│   │   ├── build_ipi_dataset_v2.py  # Source-grouped split (no leakage)
│   │   └── build_synthetic_ipi_v1.py # Synthetic IPI data generator
│   ├── training/
│   │   └── train_detector.py    # Shared training framework
│   └── utils/
│       ├── paths.py
│       ├── metrics.py
│       └── plot_styles.py
├── experiments/
│   ├── run_dpi_experiment.py    # DPI evaluation
│   ├── run_ipi_experiment.py    # IPI Bayesian optimization
│   ├── run_ipi_ablation.py      # Ablation study
│   ├── run_ipi_experiment_bert.py # BERT IPI evaluation
│   └── run_unified_experiment.py  # Full pipeline benchmark
└── data/
    └── examples/                # Sample data (see Data section)
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

> **Note**: `tensorflow` is only required for `src/dpi/run_dpi_baselines.py` (fastText and TextCNN baselines). If you only need the DistilBERT-based detectors, you can skip it.

### Data

The full training datasets are not included in this repository due to size. The IPI dataset is constructed from a publicly available jailbreak benchmark. To build the datasets:

```bash
# Build IPI dataset with source-level split (recommended)
python src/data/build_ipi_dataset_v2.py

# Generate synthetic IPI samples
python src/data/build_synthetic_ipi_v1.py
```

Place the DPI dataset (Parquet format) in `data/dpi/`.

### Training

```bash
# Train DPI detector
python src/dpi/train_dpi_model.py

# Train IPI BERT detector
python src/ipi/train_ipi_detector.py
```

Models are saved to `outputs/model/`.

### Evaluation

```bash
# Evaluate DPI detector and compare baselines
python experiments/run_dpi_experiment.py

# Run IPI Bayesian hyperparameter optimization
python experiments/run_ipi_experiment.py

# IPI ablation study (signal contribution analysis)
python experiments/run_ipi_ablation.py

# Full pipeline benchmark
python experiments/run_unified_experiment.py
```

Results are saved as JSON to `outputs/results/`.

## Key Design Decisions

**Source-level data split**: The IPI dataset uses `GroupShuffleSplit` by `source_id` to prevent data leakage across train/test splits — a common issue when multiple samples originate from the same document.

**Context-aware noise reduction**: Each signal module includes domain-specific downweighting rules to reduce false positives on benign technical content (e.g., academic papers discussing prompt injection, deployment documentation).

**Bayesian hyperparameter optimization**: 12 weight and threshold parameters are jointly optimized using Optuna to maximize recall subject to a minimum precision constraint.

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended for training (inference runs on CPU)
