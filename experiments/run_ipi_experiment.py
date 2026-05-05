import json
import optuna
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ipi.ipi_detector import IPIDetector
from src.utils.plot_styles import apply_bw_axis_style, line_style, save_bw_figure
from src.utils.paths import RESULTS_IPI_DIR, IPI_DATA_DIR, ensure_dir


# =========================================================
# Utils
# =========================================================

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_json_field(value: str, default):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


# =========================================================
# Dataset
# =========================================================

def _normalize_label(raw_label: Any) -> str:
    label = str(raw_label).strip().lower()
    if label in {"1", "jailbreak", "attack"}:
        return "jailbreak"
    return "benign"


def _normalize_row(row: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    retrieved = row.get("retrieved_materials", [])
    history = row.get("conversation_history", [])

    # CSV usually stores JSON as strings; JSON split files are already lists.
    if isinstance(retrieved, str):
        retrieved = _parse_json_field(retrieved, [])
    if isinstance(history, str):
        history = _parse_json_field(history, [])

    context_text = row.get("context") or row.get("flat_text") or ""

    return {
        "id": str(row.get("id", f"sample_{fallback_idx}")),
        "label": _normalize_label(row.get("label", "benign")),
        "user_prompt": row.get("user_prompt", ""),
        "retrieved_materials": retrieved,
        "conversation_history": history,
        "context": context_text,
        "source_id": row.get("source_id"),
        "scenario_type": row.get("scenario_type", "unknown"),
    }


def load_dataset(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            raw_rows = json.load(f)
        return [_normalize_row(row, i) for i, row in enumerate(raw_rows)]

    if suffix == ".csv":
        import csv

        rows = []
        with open(path_obj, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(_normalize_row(row, i))
        return rows

    raise ValueError(f"Unsupported dataset format: {path_obj}")


# =========================================================
# Metrics
# =========================================================

def evaluate_predictions(results):
    tp = fp = tn = fn = 0

    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted"]

        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif not gt and not pred:
            tn += 1
        elif gt and not pred:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# =========================================================
# Experiment
# =========================================================

def run_experiment(dataset, detector):

    results = []

    for sample in dataset:
        gt = sample["label"] == "jailbreak"
        user_prompt = sample.get("user_prompt", "")
        retrieved_materials = sample.get("retrieved_materials", [])
        conversation_history = sample.get("conversation_history", [])
        context_text = sample.get("context", "")

        if user_prompt or retrieved_materials or conversation_history:
            detection = detector.analyze(
                user_prompt=user_prompt,
                retrieved_materials=retrieved_materials,
                conversation_history=conversation_history,
            )
        else:
            detection = detector.analyze(text=context_text)

        pred = detection["decision"] in ("WARN", "BLOCK")

        results.append({
            "ground_truth": gt,
            "predicted": pred
        })

    return evaluate_predictions(results)


# =========================================================
# Bayesian Optimization
# =========================================================

def optimize(dataset, n_trials=200, min_precision=0.75):

    def objective(trial):

        # ---- weights ----
        w1 = trial.suggest_float("instructionality_weight", 0.2, 0.5)
        w2 = trial.suggest_float("authority_conflict_weight", 0.2, 0.5)
        w3 = 1 - w1 - w2

        if w3 <= 0:
            raise optuna.exceptions.TrialPruned()

        # ---- thresholds ----
        warn = trial.suggest_float("warn_threshold", 0.05, 0.25)
        block = trial.suggest_float("block_threshold", 0.30, 0.50)

        if warn >= block:
            raise optuna.exceptions.TrialPruned()

        # ---- other params ----
        params = {
            "instructionality_weight": w1,
            "authority_conflict_weight": w2,
            "execution_affinity_weight": w3,

            "warn_threshold": warn,
            "block_threshold": block,

            "strong_signal_multiplier": trial.suggest_float("strong_signal_multiplier", 1.0, 1.5),

            "pattern_bonus_single": trial.suggest_float("pattern_bonus_single", 0.1, 0.2),
            "pattern_bonus_multi": trial.suggest_float("pattern_bonus_multi", 0.2, 0.35),

            "context_bonus_retrieved": trial.suggest_float("context_bonus_retrieved", 0.05, 0.15),
            "context_bonus_history": trial.suggest_float("context_bonus_history", 0.03, 0.1),
            "context_bonus_benign": trial.suggest_float("context_bonus_benign", 0.03, 0.1),
        }

        if params["pattern_bonus_multi"] < params["pattern_bonus_single"]:
            raise optuna.exceptions.TrialPruned()

        detector = IPIDetector(**params)
        metrics = run_experiment(dataset, detector)

        precision = metrics["precision"]
        recall = metrics["recall"]

        # 保存用于画图
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)

        if precision < min_precision:
            raise optuna.exceptions.TrialPruned()

        return recall

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study


# =========================================================
# Plotting
# =========================================================

def plot_progress(study, timestamp):
    recalls = []
    best = []
    best_so_far = 0

    for t in study.trials:
        if t.value is None:
            continue
        recalls.append(t.value)
        best_so_far = max(best_so_far, t.value)
        best.append(best_so_far)

    fig, ax = plt.subplots()
    ax.plot(recalls, label="Recall", **line_style(0))
    ax.plot(best, label="Best Recall", **line_style(1))
    ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Recall")
    ax.set_title("Optimization Convergence")
    apply_bw_axis_style(ax)

    file = RESULTS_IPI_DIR / f"opt_progress_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(file)
    save_bw_figure(fig, file, dpi=200)
    plt.close(fig)


def plot_pr(study, timestamp):
    p, r = [], []

    for t in study.trials:
        if "precision" in t.user_attrs:
            p.append(t.user_attrs["precision"])
            r.append(t.user_attrs["recall"])

    fig, ax = plt.subplots()
    ax.scatter(r, p, color="black", marker="o", s=28)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Scatter")
    apply_bw_axis_style(ax)

    file = RESULTS_IPI_DIR / f"pr_curve_{timestamp}.png"
    fig.tight_layout()
    fig.savefig(file)
    save_bw_figure(fig, file, dpi=200)
    plt.close(fig)

def threshold_sweep(
    dataset,
    detector,
    num_thresholds=50,
    save_path=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    scores = []
    labels = []

    # 收集 risk score 和 ground truth
    for sample in dataset:
        gt = sample["label"] == "jailbreak"

        detection = detector.analyze(
            user_prompt=sample["user_prompt"],
            retrieved_materials=sample["retrieved_materials"],
            conversation_history=sample["conversation_history"],
        )

        scores.append(float(detection["final_risk_score"]))
        labels.append(gt)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    sweep_results = []

    print("\nThreshold Sweep Results:")
    print("threshold | recall | precision | f1")

    for t in thresholds:
        tp = fp = tn = fn = 0

        for s, gt in zip(scores, labels):
            pred = s >= t

            if gt and pred:
                tp += 1
            elif not gt and pred:
                fp += 1
            elif not gt and not pred:
                tn += 1
            elif gt and not pred:
                fn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (tp + tn) / len(labels) if labels else 0.0

        row = {
            "threshold": round(float(t), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        }
        sweep_results.append(row)

        print(
            f"{row['threshold']:.2f} | "
            f"{row['recall']:.3f} | "
            f"{row['precision']:.3f} | "
            f"{row['f1']:.3f}"
        )

    # 绘图
    thresholds_plot = [r["threshold"] for r in sweep_results]
    recalls_plot = [r["recall"] for r in sweep_results]
    precisions_plot = [r["precision"] for r in sweep_results]
    f1_plot = [r["f1"] for r in sweep_results]

    fig, ax = plt.subplots()
    ax.plot(thresholds_plot, recalls_plot, label="Recall", **line_style(0))
    ax.plot(thresholds_plot, precisions_plot, label="Precision", **line_style(1))
    ax.plot(thresholds_plot, f1_plot, label="F1", **line_style(2))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep")
    ax.legend()
    apply_bw_axis_style(ax)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        save_bw_figure(fig, Path(save_path), dpi=200)
        print(f"\nThreshold sweep plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)

    return sweep_results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    ensure_dir(RESULTS_IPI_DIR)

    TRAIN_PATH = IPI_DATA_DIR / "ipi_train_group_source.json"
    TEST_PATH = IPI_DATA_DIR / "ipi_test_group_source.json"
    train_dataset = load_dataset(str(TRAIN_PATH))
    test_dataset = load_dataset(str(TEST_PATH))

    print(f"Train dataset: {TRAIN_PATH} ({len(train_dataset)} samples)")
    print(f"Test dataset : {TEST_PATH} ({len(test_dataset)} samples)")

    timestamp = get_timestamp()

    # # =========================
    # # Step 1: threshold sweep
    # # =========================
    # detector = IPIDetector()
    # sweep_results = threshold_sweep(
    #     dataset=test_dataset,
    #     detector=detector,
    #     num_thresholds=100,
    #     save_path=str(RESULTS_IPI_DIR / f"threshold_sweep_{timestamp}.png"),
    # )
    #
    # with open(
    #     RESULTS_IPI_DIR / f"threshold_sweep_{timestamp}.json",
    #     "w",
    #     encoding="utf-8",
    # ) as f:
    #     json.dump(sweep_results, f, indent=2, ensure_ascii=False)
    #
    # print("Threshold sweep done.")

    # =========================
    # Step 2: Bayesian optimization
    # =========================
    study = optimize(train_dataset, n_trials=100, min_precision=0.50)

    best_params = dict(study.best_params)
    best_params["execution_affinity_weight"] = (
        1.0
        - best_params["instructionality_weight"]
        - best_params["authority_conflict_weight"]
    )

    best_detector = IPIDetector(**best_params)
    test_metrics = run_experiment(test_dataset, best_detector)

    output_file = RESULTS_IPI_DIR / f"bayes_opt_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_file": str(TRAIN_PATH),
                "test_file": str(TEST_PATH),
                "best_params": best_params,
                "best_train_recall": study.best_value,
                "test_metrics": test_metrics,
                "n_trials": len(study.trials),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved results to {output_file}")
    print("Test metrics:", test_metrics)

    plot_progress(study, timestamp)
    plot_pr(study, timestamp)

    print("Done.")
