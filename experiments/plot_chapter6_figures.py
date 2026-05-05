import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_styles import apply_bw_axis_style, bar_style, save_bw_figure
from src.utils.paths import RESULTS_IPI_DIR, ensure_dir


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _latest(pattern: str) -> Path:
    files = sorted(RESULTS_IPI_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files[0]


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_overall_compare(unified_data: Dict[str, Any], out_png: Path):
    models = ["ipi_warning_only", "ipi_bert_only", "unified"]
    metrics = ["precision", "recall", "f1"]
    x = range(len(models))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.22
    offsets = [-width, 0.0, width]

    for i, m in enumerate(metrics):
        values = [unified_data["overall"][k][m] for k in models]
        ax.bar([xi + offsets[i] for xi in x], values, width=width, label=m, **bar_style(i + 1))

    ax.set_xticks(list(x), models, rotation=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall Comparison on IPI Test Set")
    ax.legend()
    apply_bw_axis_style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    save_bw_figure(fig, out_png, dpi=200)
    plt.close(fig)


def _plot_scenario_f1(unified_data: Dict[str, Any], out_png: Path):
    models = ["ipi_warning_only", "ipi_bert_only", "unified"]
    scenario_names = sorted(unified_data["by_scenario"]["unified"].keys())
    x = range(len(scenario_names))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.25
    offsets = [-width, 0.0, width]

    for i, model_name in enumerate(models):
        values = [unified_data["by_scenario"][model_name][s]["f1"] for s in scenario_names]
        ax.bar([xi + offsets[i] for xi in x], values, width=width, label=model_name, **bar_style(i))

    ax.set_xticks(list(x), scenario_names, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("F1 by Scenario Type")
    ax.legend()
    apply_bw_axis_style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    save_bw_figure(fig, out_png, dpi=200)
    plt.close(fig)


def _plot_ablation_auc(ablation_data: List[Dict[str, Any]], out_png: Path):
    # Focus on core_signal_only rows for clearer signal-level attribution.
    core_rows = [r for r in ablation_data if r["config"].startswith("core_signal_only::")]
    if not core_rows:
        core_rows = ablation_data

    labels = [r["config"].split("::")[-1] for r in core_rows]
    roc = [r["score_metrics"]["roc_auc"] for r in core_rows]
    pr = [r["score_metrics"]["pr_auc"] for r in core_rows]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.35
    ax.bar([i - width / 2 for i in x], roc, width=width, label="ROC-AUC", **bar_style(2))
    ax.bar([i + width / 2 for i in x], pr, width=width, label="PR-AUC", **bar_style(3))
    ax.set_xticks(list(x), labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Signal Ablation (core_signal_only)")
    ax.legend()
    apply_bw_axis_style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    save_bw_figure(fig, out_png, dpi=200)
    plt.close(fig)


def main():
    ensure_dir(RESULTS_IPI_DIR)
    ts = get_timestamp()

    unified_file = _latest("unified_benchmark_*.json")
    ablation_file = _latest("ipi_ablation_results_*.json")

    unified_data = _load_json(unified_file)
    ablation_data = _load_json(ablation_file)

    fig1 = RESULTS_IPI_DIR / f"chapter6_overall_compare_{ts}.png"
    fig2 = RESULTS_IPI_DIR / f"chapter6_scenario_f1_{ts}.png"
    fig3 = RESULTS_IPI_DIR / f"chapter6_ablation_auc_{ts}.png"

    _plot_overall_compare(unified_data, fig1)
    _plot_scenario_f1(unified_data, fig2)
    _plot_ablation_auc(ablation_data, fig3)

    print("Unified source:", unified_file)
    print("Ablation source:", ablation_file)
    print("Generated:", fig1)
    print("Generated:", fig2)
    print("Generated:", fig3)


if __name__ == "__main__":
    main()
