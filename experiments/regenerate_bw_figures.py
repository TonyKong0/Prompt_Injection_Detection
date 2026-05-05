import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_styles import apply_bw_axis_style, bar_style, bw_output_path
from experiments.run_ipi_experiment import IPI_DATA_DIR, RESULTS_IPI_DIR, load_dataset, optimize


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _threshold_line_style(index: int) -> dict:
    styles = [
        {"color": "black", "linestyle": "-", "linewidth": 2.6, "marker": "o", "markevery": 6, "markersize": 5.0},
        {"color": "black", "linestyle": (0, (7, 3)), "linewidth": 2.4, "marker": "s", "markevery": 7, "markersize": 4.8},
        {"color": "black", "linestyle": (0, (1, 2)), "linewidth": 2.4, "marker": "^", "markevery": 8, "markersize": 4.8},
    ]
    return styles[index]


def regenerate_threshold_sweep(json_path: Path) -> Path:
    data = _load_json(json_path)
    thresholds = [row["threshold"] for row in data]
    recalls = [row["recall"] for row in data]
    precisions = [row["precision"] for row in data]
    f1_scores = [row["f1"] for row in data]

    out_path = bw_output_path(json_path.with_suffix(".png"))
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.plot(thresholds, recalls, label="Recall", **_threshold_line_style(0))
    ax.plot(thresholds, precisions, label="Precision", **_threshold_line_style(1))
    ax.plot(thresholds, f1_scores, label="F1", **_threshold_line_style(2))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Threshold Sweep")
    ax.legend(frameon=True, edgecolor="black", facecolor="white")
    apply_bw_axis_style(ax)
    ax.grid(axis="both", linestyle=":", color="0.82", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return out_path


def regenerate_dpi_scores(json_path: Path) -> Path:
    results = _load_json(json_path)
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = [item["model_name"] for item in results]
    x = np.arange(len(labels))
    width = 0.18

    out_path = bw_output_path(json_path.with_name(json_path.name.replace("comparison", "scores")).with_suffix(".png"))
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, metric in enumerate(metric_names):
        values = [item[metric] for item in results]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=metric.upper(),
            **bar_style(idx),
        )

    ax.set_ylim(0.85, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("DPI Model Comparison")
    ax.legend()
    apply_bw_axis_style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return out_path


def rerun_opt_progress(output_stem: str = "opt_progress_20260401_113802", n_trials: int = 50) -> Path:
    train_path = IPI_DATA_DIR / "ipi_train_group_source.json"
    train_dataset = load_dataset(str(train_path))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective_study = optimize(train_dataset, n_trials=n_trials, min_precision=0.50)

    recalls = []
    best = []
    best_so_far = 0.0

    for trial in objective_study.trials:
        if trial.value is None:
            continue
        recalls.append(trial.value)
        best_so_far = max(best_so_far, trial.value)
        best.append(best_so_far)

    out_path = RESULTS_IPI_DIR / f"{output_stem}_bw_hd.png"
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(
        range(1, len(recalls) + 1),
        recalls,
        label="Recall",
        color="black",
        linestyle=(0, (5, 2)),
        linewidth=2.2,
        marker="o",
        markersize=3.8,
        markevery=max(1, len(recalls) // 16),
    )
    ax.plot(
        range(1, len(best) + 1),
        best,
        label="Best Recall",
        color="black",
        linestyle="-",
        linewidth=2.8,
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Optimization Convergence")
    ax.legend(frameon=True, edgecolor="black", facecolor="white")
    apply_bw_axis_style(ax)
    ax.grid(axis="both", linestyle=":", color="0.82", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=360, bbox_inches="tight")
    return out_path


def main() -> None:
    threshold_json = PROJECT_ROOT / "outputs" / "results" / "ipi" / "threshold_sweep_v2_20260331_212853.json"
    dpi_json = PROJECT_ROOT / "outputs" / "results" / "dpi" / "dpi_baseline_comparison_20260402_231547.json"

    print("Generated:", regenerate_threshold_sweep(threshold_json))
    print("Generated:", rerun_opt_progress())
    print("Generated:", regenerate_dpi_scores(dpi_json))


if __name__ == "__main__":
    main()
