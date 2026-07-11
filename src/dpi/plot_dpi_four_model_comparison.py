import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import RESULTS_DPI_DIR, ensure_dir
from src.utils.plot_styles import apply_bw_axis_style, bar_style, save_bw_figure


def load_latest_json(pattern: str) -> tuple[Path, object]:
    candidates = sorted(RESULTS_DPI_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No result file matched {pattern} in {RESULTS_DPI_DIR}")

    latest = candidates[-1]
    with latest.open("r", encoding="utf-8") as handle:
        return latest, json.load(handle)


def load_baseline_models() -> list[dict]:
    source_path, rows = load_latest_json("dpi_baseline_comparison_*.json")
    wanted = {"fastText-style", "TextCNN"}
    results = []

    for row in rows:
        if row.get("model_name") in wanted:
            item = dict(row)
            item["source_result_file"] = str(source_path)
            results.append(item)

    missing = wanted - {item["model_name"] for item in results}
    if missing:
        raise ValueError(f"Missing baseline metrics for: {', '.join(sorted(missing))}")

    return results


def load_timing_result(display_name: str, pattern: str) -> dict:
    source_path, metrics = load_latest_json(pattern)
    return {
        "model_name": display_name,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "train_seconds": None,
        "predict_seconds": float(metrics["predict_seconds"]),
        "ms_per_sample": float(metrics["ms_per_sample"]),
        "parameter_count": int(metrics["parameter_count"]),
        "sample_count": int(metrics.get("sample_count", 10000)),
        "device": metrics.get("device"),
        "batch_size": metrics.get("batch_size"),
        "source_result_file": str(source_path),
    }


def collect_results() -> list[dict]:
    return [
        *load_baseline_models(),
        load_timing_result("DistilBERT", "distilbert-base-uncased_inference_timing_*.json"),
        load_timing_result("BERT", "bert-base-uncased_inference_timing_*.json"),
    ]


def save_summary(results: list[dict], output_base: Path) -> tuple[Path, Path]:
    json_path = output_base.with_suffix(".json")
    csv_path = output_base.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return json_path, csv_path


def add_bar_labels(ax, bars, values: list[float], fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_chart(results: list[dict], output_path: Path) -> Path:
    labels = [item["model_name"] for item in results]
    x = np.arange(len(labels))

    accuracy = [item["accuracy"] for item in results]
    f1 = [item["f1"] for item in results]
    speed = [item["ms_per_sample"] for item in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    score_width = 0.34
    acc_bars = axes[0].bar(
        x - score_width / 2,
        accuracy,
        score_width,
        label="Accuracy",
        **bar_style(1),
    )
    f1_bars = axes[0].bar(
        x + score_width / 2,
        f1,
        score_width,
        label="F1",
        **bar_style(3),
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0.94, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Detection Quality")
    axes[0].legend(frameon=False)
    apply_bw_axis_style(axes[0])
    add_bar_labels(axes[0], acc_bars, accuracy, "{:.4f}")
    add_bar_labels(axes[0], f1_bars, f1, "{:.4f}")

    speed_bars = []
    for idx, (label, value) in enumerate(zip(labels, speed)):
        container = axes[1].bar(label, value, **bar_style(idx + 1))
        speed_bars.append(container[0])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Milliseconds per sample (log scale)")
    axes[1].set_title("Inference Speed")
    axes[1].tick_params(axis="x", rotation=15)
    apply_bw_axis_style(axes[1])
    add_bar_labels(axes[1], speed_bars, speed, "{:.3f}")

    fig.suptitle("DPI Four-Model Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    bw_path = save_bw_figure(fig, output_path, dpi=200)
    plt.close(fig)
    return bw_path


def main() -> None:
    ensure_dir(RESULTS_DPI_DIR)
    results = collect_results()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = RESULTS_DPI_DIR / f"dpi_four_model_comparison_{timestamp}"

    json_path, csv_path = save_summary(results, output_base)
    chart_path = output_base.with_suffix(".png")
    bw_chart_path = save_chart(results, chart_path)

    print("Saved summary json:", json_path)
    print("Saved summary csv:", csv_path)
    print("Saved chart:", chart_path)
    print("Saved black-white chart:", bw_chart_path)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
