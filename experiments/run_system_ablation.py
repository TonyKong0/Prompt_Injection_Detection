import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in the experiment env.
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.context_utils import parse_structured_context
from src.dpi.dpi_detector import DPIDetector
from src.ipi.ipi_bert_detector import IPIBertDetector
from src.ipi.ipi_detector import IPIDetector
from src.utils.paths import DPI_DATA_DIR, IPI_DATA_DIR, RESULTS_DIR, ensure_dir


TUNED_IPI_WARNING_PARAMS = {
    "instructionality_weight": 0.33402774140688785,
    "authority_conflict_weight": 0.4275146911846204,
    "execution_affinity_weight": 0.23845756740849178,
    "warn_threshold": 0.12465943559777146,
    "block_threshold": 0.44639509181172915,
    "strong_signal_multiplier": 1.2537570435538332,
    "pattern_bonus_single": 0.1258764421444331,
    "pattern_bonus_multi": 0.23699826495664625,
    "context_bonus_retrieved": 0.11890409066378241,
    "context_bonus_history": 0.07091063530159541,
    "context_bonus_benign": 0.04082465585401465,
}


CONFIGS = [
    "DPI-only",
    "IPI warning-only",
    "IPI BERT-only",
    "DPI + IPI warning",
    "DPI + IPI BERT",
    "DPI + IPI warning + IPI BERT unified",
]


CSV_FIELDS = [
    "config",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "direct_recall",
    "indirect_recall",
    "tp",
    "fp",
    "tn",
    "fn",
    "support",
    "direct_support",
    "indirect_support",
]


TIMING_CSV_FIELDS = [
    "config",
    "timing_scope",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "support",
]


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def bool_label(raw_label: Any) -> bool:
    s = str(raw_label).strip().lower()
    return s in {"1", "2", "jailbreak", "attack", "true"}


def is_attack_decision(decision: str) -> bool:
    return decision in {"WARN", "BLOCK"}


def synchronize_cuda() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def limited(items: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    if limit is None:
        return items
    return items[: max(0, limit)]


def load_dpi_samples(limit: Optional[int]) -> List[Dict[str, Any]]:
    path = DPI_DATA_DIR / "test-00000-of-00001.parquet"
    df = pd.read_parquet(path, columns=["text", "label"])
    if limit is not None:
        df = df.head(max(0, limit))

    samples = []
    for idx, row in df.iterrows():
        text = str(row["text"] or "")
        is_attack = bool_label(row["label"])
        samples.append(
            {
                "id": f"dpi_{idx}",
                "source": "dpi",
                "attack_type": "direct" if is_attack else "benign",
                "ground_truth": is_attack,
                "user_prompt": text,
                "conversation_history": [],
                "retrieved_materials": [],
                "dpi_text": text,
            }
        )
    return samples


def load_ipi_samples(limit: Optional[int]) -> List[Dict[str, Any]]:
    path = IPI_DATA_DIR / "ipi_test_group_source.json"
    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    samples = []
    for item in limited(raw_items, limit):
        structured = parse_structured_context(item.get("context", ""))
        is_attack = bool_label(item.get("label", 0))
        samples.append(
            {
                "id": item.get("id"),
                "source": "ipi",
                "attack_type": "indirect" if is_attack else "benign",
                "ground_truth": is_attack,
                "user_prompt": structured["user_prompt"],
                "conversation_history": structured["conversation_history"],
                "retrieved_materials": structured["retrieved_materials"],
                "dpi_text": structured["user_prompt"],
                "scenario_type": item.get("scenario_type", "unknown"),
            }
        )
    return samples


def iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_dpi_predictions(
    samples: List[Dict[str, Any]],
    detector: DPIDetector,
    batch_size: int,
    progress_interval: int,
) -> Tuple[List[bool], List[float]]:
    predictions: List[bool] = []
    timings_ms: List[float] = []
    texts = [s["dpi_text"] for s in samples]
    processed = 0

    for batch in iter_batches(texts, batch_size):
        synchronize_cuda()
        start = perf_counter()
        labels, _ = detector.predict_batch(batch)
        synchronize_cuda()
        elapsed_per_sample_ms = ((perf_counter() - start) * 1000.0) / len(batch)
        predictions.extend(label == 1 for label in labels)
        timings_ms.extend([elapsed_per_sample_ms] * len(batch))
        processed += len(batch)
        if progress_interval and processed % progress_interval == 0:
            print(f"DPI predictions: {processed}/{len(samples)}")

    if progress_interval:
        print(f"DPI predictions: {len(samples)}/{len(samples)}")
    return predictions, timings_ms


def run_warning_predictions(
    samples: List[Dict[str, Any]],
    detector: IPIDetector,
    progress_interval: int,
) -> Tuple[List[bool], List[float]]:
    predictions: List[bool] = []
    timings_ms: List[float] = []
    for idx, sample in enumerate(samples, 1):
        start = perf_counter()
        result = detector.analyze(
            user_prompt=sample["user_prompt"],
            conversation_history=sample["conversation_history"],
            retrieved_materials=sample["retrieved_materials"],
            metadata={"sample_id": sample.get("id"), "source": sample["source"]},
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        predictions.append(is_attack_decision(result["decision"]))
        timings_ms.append(elapsed_ms)
        if progress_interval and idx % progress_interval == 0:
            print(f"IPI warning predictions: {idx}/{len(samples)}")

    if progress_interval:
        print(f"IPI warning predictions: {len(samples)}/{len(samples)}")
    return predictions, timings_ms


def run_bert_predictions(
    samples: List[Dict[str, Any]],
    detector: IPIBertDetector,
    progress_interval: int,
) -> Tuple[List[bool], List[float]]:
    predictions: List[bool] = []
    timings_ms: List[float] = []
    for idx, sample in enumerate(samples, 1):
        synchronize_cuda()
        start = perf_counter()
        result = detector.detect(
            user_prompt=sample["user_prompt"],
            conversation_history=sample["conversation_history"],
            retrieved_materials=sample["retrieved_materials"],
        )
        synchronize_cuda()
        elapsed_ms = (perf_counter() - start) * 1000.0
        predictions.append(is_attack_decision(result["decision"]))
        timings_ms.append(elapsed_ms)
        if progress_interval and idx % progress_interval == 0:
            print(f"IPI BERT predictions: {idx}/{len(samples)}")

    if progress_interval:
        print(f"IPI BERT predictions: {len(samples)}/{len(samples)}")
    return predictions, timings_ms


def binary_metrics(
    samples: List[Dict[str, Any]],
    predictions: List[bool],
    config: str,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    direct_hits = direct_support = 0
    indirect_hits = indirect_support = 0

    for sample, pred in zip(samples, predictions):
        gt = bool(sample["ground_truth"])
        if gt and pred:
            tp += 1
        elif (not gt) and pred:
            fp += 1
        elif (not gt) and (not pred):
            tn += 1
        else:
            fn += 1

        if sample["attack_type"] == "direct":
            direct_support += 1
            direct_hits += int(pred)
        elif sample["attack_type"] == "indirect":
            indirect_support += 1
            indirect_hits += int(pred)

    support = len(samples)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / support if support else 0.0
    direct_recall = direct_hits / direct_support if direct_support else 0.0
    indirect_recall = indirect_hits / indirect_support if indirect_support else 0.0

    return {
        "config": config,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "direct_recall": round(direct_recall, 4),
        "indirect_recall": round(indirect_recall, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": support,
        "direct_support": direct_support,
        "indirect_support": indirect_support,
    }


def build_config_predictions(
    dpi_preds: List[bool],
    warning_preds: List[bool],
    bert_preds: List[bool],
) -> Dict[str, List[bool]]:
    return {
        "DPI-only": dpi_preds,
        "IPI warning-only": warning_preds,
        "IPI BERT-only": bert_preds,
        "DPI + IPI warning": [d or w for d, w in zip(dpi_preds, warning_preds)],
        "DPI + IPI BERT": [d or b for d, b in zip(dpi_preds, bert_preds)],
        "DPI + IPI warning + IPI BERT unified": [
            d or w or b for d, w, b in zip(dpi_preds, warning_preds, bert_preds)
        ],
    }


def timing_stats(config: str, timing_scope: str, values_ms: List[float]) -> Dict[str, Any]:
    arr = np.array(values_ms, dtype=float)
    if arr.size == 0:
        return {
            "config": config,
            "timing_scope": timing_scope,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "support": 0,
        }
    return {
        "config": config,
        "timing_scope": timing_scope,
        "mean_ms": round(float(np.mean(arr)), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "support": int(arr.size),
    }


def build_timing_rows(
    samples: List[Dict[str, Any]],
    dpi_ms: List[float],
    warning_ms: List[float],
    bert_ms: List[float],
) -> List[Dict[str, Any]]:
    indices = [
        idx
        for idx, sample in enumerate(samples)
        if sample["source"] == "ipi" and sample["attack_type"] == "indirect"
    ]
    dpi_warning = [dpi_ms[i] + warning_ms[i] for i in indices]
    dpi_bert = [dpi_ms[i] + bert_ms[i] for i in indices]
    dpi_warning_bert = [dpi_ms[i] + warning_ms[i] + bert_ms[i] for i in indices]

    return [
        timing_stats("DPI + IPI warning", "final_decision", dpi_warning),
        timing_stats("DPI + IPI BERT", "final_decision", dpi_bert),
        timing_stats(
            "DPI + IPI warning + IPI BERT unified",
            "early_warning",
            dpi_warning,
        ),
        timing_stats(
            "DPI + IPI warning + IPI BERT unified",
            "final_decision",
            dpi_warning_bert,
        ),
    ]


def save_csv(rows: List[Dict[str, Any]], path: Path, fields: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_black_white(rows: List[Dict[str, Any]], path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = [r["config"] for r in rows]
    x = np.arange(len(labels))
    width = 0.16

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "hatch.linewidth": 0.9,
        }
    )

    fig, ax = plt.subplots(figsize=(16.5, 8.4), dpi=220)
    grays = ["#f2f2f2", "#cfcfcf", "#969696", "#595959"]
    hatches = ["//", "\\\\", "xx", ".."]

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in rows]
        positions = x + (i - 1.5) * width
        ax.bar(
            positions,
            values,
            width,
            label=metric.capitalize(),
            color=grays[i],
            edgecolor="black",
            linewidth=0.9,
            hatch=hatches[i],
        )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("System configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_yticks(np.linspace(0, 1.0, 11))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#777777", alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        edgecolor="black",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_timing_black_white(rows: List[Dict[str, Any]], path: Path) -> None:
    metrics = ["mean_ms", "median_ms", "p95_ms"]
    metric_labels = ["Mean", "Median", "P95"]
    labels = [f"{r['config']}\n({r['timing_scope']})" for r in rows]
    x = np.arange(len(labels))
    width = 0.22

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.1,
            "hatch.linewidth": 0.9,
        }
    )

    fig, ax = plt.subplots(figsize=(14.5, 8.0), dpi=220)
    grays = ["#f2f2f2", "#bdbdbd", "#636363"]
    hatches = ["//", "\\\\", "xx"]

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in rows]
        positions = x + (i - 1) * width
        ax.bar(
            positions,
            values,
            width,
            label=metric_labels[i],
            color=grays[i],
            edgecolor="black",
            linewidth=0.9,
            hatch=hatches[i],
        )

    ax.set_ylabel("Latency per IPI attack sample (ms)")
    ax.set_xlabel("System configuration and timing scope")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#777777", alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=True,
        edgecolor="black",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = ensure_dir(RESULTS_DIR / "system")
    print("Loading datasets...")
    dpi_samples = load_dpi_samples(args.dpi_limit)
    ipi_samples = load_ipi_samples(args.ipi_limit)
    samples = dpi_samples + ipi_samples
    print(
        f"Loaded {len(dpi_samples)} DPI samples and {len(ipi_samples)} IPI samples "
        f"({len(samples)} total)."
    )

    print("Loading detectors...")
    dpi_detector = DPIDetector()
    warning_detector = IPIDetector(**TUNED_IPI_WARNING_PARAMS)
    bert_detector = IPIBertDetector()

    print("Running base predictions...")
    dpi_preds, dpi_ms = run_dpi_predictions(
        samples=samples,
        detector=dpi_detector,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
    )
    warning_preds, warning_ms = run_warning_predictions(
        samples,
        warning_detector,
        args.progress_interval,
    )
    bert_preds, bert_ms = run_bert_predictions(samples, bert_detector, args.progress_interval)

    config_predictions = build_config_predictions(dpi_preds, warning_preds, bert_preds)
    rows = [binary_metrics(samples, config_predictions[name], name) for name in CONFIGS]
    timing_rows = build_timing_rows(samples, dpi_ms, warning_ms, bert_ms)

    timestamp = get_timestamp()
    json_path = output_dir / f"system_ablation_{timestamp}.json"
    csv_path = output_dir / f"system_ablation_{timestamp}.csv"
    png_path = output_dir / f"system_ablation_compare_{timestamp}.png"
    timing_csv_path = output_dir / f"system_ablation_timing_{timestamp}.csv"
    timing_png_path = output_dir / f"system_ablation_timing_compare_{timestamp}.png"

    result = {
        "timestamp": timestamp,
        "data_files": {
            "dpi": str(DPI_DATA_DIR / "test-00000-of-00001.parquet"),
            "ipi": str(IPI_DATA_DIR / "ipi_test_group_source.json"),
        },
        "limits": {"dpi_limit": args.dpi_limit, "ipi_limit": args.ipi_limit},
        "sample_counts": {
            "dpi_samples": len(dpi_samples),
            "ipi_samples": len(ipi_samples),
            "support": len(samples),
            "direct_support": sum(1 for s in samples if s["attack_type"] == "direct"),
            "indirect_support": sum(1 for s in samples if s["attack_type"] == "indirect"),
        },
        "tuned_ipi_warning_params": TUNED_IPI_WARNING_PARAMS,
        "overall": rows,
        "timing": {
            "sample_scope": "source == 'ipi' and ground_truth == attack",
            "dpi_timing_note": "Batch-level DPI elapsed time is evenly assigned to samples in the batch.",
            "rows": timing_rows,
        },
        "outputs": {
            "json": str(json_path),
            "csv": str(csv_path),
            "png": str(png_path),
            "timing_csv": str(timing_csv_path),
            "timing_png": str(timing_png_path),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    save_csv(rows, csv_path, CSV_FIELDS)
    save_csv(timing_rows, timing_csv_path, TIMING_CSV_FIELDS)
    plot_black_white(rows, png_path)
    plot_timing_black_white(timing_rows, timing_png_path)

    print("Saved JSON:", json_path)
    print("Saved CSV:", csv_path)
    print("Saved PNG:", png_path)
    print("Saved timing CSV:", timing_csv_path)
    print("Saved timing PNG:", timing_png_path)
    print("\nMetrics:")
    for row in rows:
        print(row)
    print("\nTiming metrics:")
    for row in timing_rows:
        print(row)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run system-level prompt injection ablation.")
    parser.add_argument("--dpi-limit", type=int, default=None)
    parser.add_argument("--ipi-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
