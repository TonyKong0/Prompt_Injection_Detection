import json
from datetime import datetime
from typing import Dict, List, Any

from sklearn.metrics import average_precision_score, roc_auc_score

from src.ipi.ipi_detector import IPIDetector
from experiments.run_ipi_experiment import (
    load_dataset,
    evaluate_predictions,
)
from src.utils.paths import RESULTS_IPI_DIR, IPI_DATA_DIR, ensure_dir


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_auc_metrics(y_true: List[int], y_score: List[float]) -> Dict[str, Any]:
    if len(set(y_true)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "note": "AUC undefined: only one class present in y_true.",
        }

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def _compute_signal_diagnostics(signal_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    signal_names = [
        "instructionality",
        "authority_conflict",
        "execution_affinity",
        "density_bonus",
        "pattern_bonus",
        "context_bonus",
        "position_bonus",
    ]

    diagnostics: Dict[str, Any] = {}
    separation_rank: List[Dict[str, Any]] = []

    for name in signal_names:
        pos_values = [
            float(r["signals"].get(name, 0.0))
            for r in signal_rows
            if r["ground_truth"] == 1
        ]
        neg_values = [
            float(r["signals"].get(name, 0.0))
            for r in signal_rows
            if r["ground_truth"] == 0
        ]

        pos_mean = sum(pos_values) / len(pos_values) if pos_values else 0.0
        neg_mean = sum(neg_values) / len(neg_values) if neg_values else 0.0
        delta = pos_mean - neg_mean

        diagnostics[name] = {
            "positive_mean": round(pos_mean, 6),
            "negative_mean": round(neg_mean, 6),
            "delta": round(delta, 6),
        }

        separation_rank.append(
            {
                "signal": name,
                "abs_delta": abs(delta),
                "delta": delta,
            }
        )

    separation_rank.sort(key=lambda x: x["abs_delta"], reverse=True)

    return {
        "signal_stats": diagnostics,
        "separation_rank": [
            {
                "signal": r["signal"],
                "delta": round(r["delta"], 6),
                "abs_delta": round(r["abs_delta"], 6),
            }
            for r in separation_rank
        ],
        "top_signal": separation_rank[0]["signal"] if separation_rank else None,
    }


def run_ablation(
    dataset_path: str,
    config_name: str,
    instructionality_weight: float,
    authority_conflict_weight: float,
    execution_affinity_weight: float,
    decision_threshold: str = "WARN",
    detector_overrides: Dict[str, Any] = None,
) -> Dict[str, Any]:
    total = (
        instructionality_weight
        + authority_conflict_weight
        + execution_affinity_weight
    )

    if abs(total - 1.0) > 1e-6:
        raise ValueError("Ablation weights must sum to 1.0")

    detector_kwargs = {
        "instructionality_weight": instructionality_weight,
        "authority_conflict_weight": authority_conflict_weight,
        "execution_affinity_weight": execution_affinity_weight,
    }
    if detector_overrides:
        detector_kwargs.update(detector_overrides)

    detector = IPIDetector(**detector_kwargs)

    dataset = load_dataset(dataset_path)
    results: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_score: List[float] = []
    signal_rows: List[Dict[str, Any]] = []

    for sample in dataset:
        ground_truth = sample["label"] == "jailbreak"
        user_prompt = sample.get("user_prompt", "")
        retrieved_materials = sample.get("retrieved_materials", [])
        conversation_history = sample.get("conversation_history", [])
        context_text = sample.get("context", "")

        metadata = {
            "sample_id": sample["id"],
            "source_id": sample.get("source_id"),
            "scenario_type": sample.get("scenario_type", "unknown"),
            "ablation_config": config_name,
        }

        if user_prompt or retrieved_materials or conversation_history:
            detection = detector.analyze(
                user_prompt=user_prompt,
                retrieved_materials=retrieved_materials,
                conversation_history=conversation_history,
                metadata=metadata,
            )
        else:
            detection = detector.analyze(
                text=context_text,
                metadata=metadata,
            )

        decision = detection["decision"]

        if decision_threshold == "WARN":
            predicted = decision in ("WARN", "BLOCK")
        else:
            predicted = decision == "BLOCK"

        score = float(detection["final_risk_score"])
        component_scores = detection.get("component_scores", {})

        results.append(
            {
                "ground_truth": ground_truth,
                "predicted": predicted,
            }
        )
        y_true.append(1 if ground_truth else 0)
        y_score.append(score)
        signal_rows.append(
            {
                "ground_truth": 1 if ground_truth else 0,
                "signals": component_scores,
            }
        )

    decision_metrics = evaluate_predictions(results)
    score_metrics = _safe_auc_metrics(y_true, y_score)
    signal_diagnostics = _compute_signal_diagnostics(signal_rows)

    return {
        "config": config_name,
        "weights": {
            "instructionality": instructionality_weight,
            "authority_conflict": authority_conflict_weight,
            "execution_affinity": execution_affinity_weight,
        },
        "detector_overrides": detector_overrides or {},
        "decision_threshold": decision_threshold,
        "num_samples": len(dataset),
        "decision_metrics": decision_metrics,
        "score_metrics": score_metrics,
        "signal_diagnostics": signal_diagnostics,
    }


if __name__ == "__main__":
    dataset_path = IPI_DATA_DIR / "ipi_test_group_source.json"

    base_experiments = [
        {
            "name": "Full Model",
            "iw": 0.30,
            "aw": 0.45,
            "ew": 0.25,
        },
        {
            "name": "No Authority Conflict",
            "iw": 0.60,
            "aw": 0.00,
            "ew": 0.40,
        },
        {
            "name": "No Instructionality",
            "iw": 0.00,
            "aw": 0.70,
            "ew": 0.30,
        },
        {
            "name": "No Execution Affinity",
            "iw": 0.40,
            "aw": 0.60,
            "ew": 0.00,
        },
    ]

    ablation_modes = [
        {
            "name": "full_pipeline",
            "overrides": {},
        },
        {
            "name": "core_signal_only",
            "overrides": {
                "strong_signal_multiplier": 0.0,
                "pattern_bonus_single": 0.0,
                "pattern_bonus_multi": 0.0,
                "context_bonus_retrieved": 0.0,
                "context_bonus_history": 0.0,
                "context_bonus_benign": 0.0,
            },
        },
    ]

    all_results: List[Dict[str, Any]] = []

    print("=" * 80)
    print("Running IPI Ablation Study (Indirect Prompt Injection)")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")

    for mode in ablation_modes:
        print(f"\n{'-' * 80}")
        print(f"Mode: {mode['name']}")
        print(f"{'-' * 80}")

        for exp in base_experiments:
            config_name = f"{mode['name']}::{exp['name']}"
            print(f"\n[Configuration: {config_name}]")

            result = run_ablation(
                dataset_path=str(dataset_path),
                config_name=config_name,
                instructionality_weight=exp["iw"],
                authority_conflict_weight=exp["aw"],
                execution_affinity_weight=exp["ew"],
                decision_threshold="WARN",
                detector_overrides=mode["overrides"],
            )

            print("decision_metrics:")
            for k, v in result["decision_metrics"].items():
                print(f"{k}: {v}")

            print("score_metrics:")
            for k, v in result["score_metrics"].items():
                print(f"{k}: {v}")

            print(
                "top_signal_by_abs_delta:",
                result["signal_diagnostics"]["top_signal"],
            )

            all_results.append(result)

    ensure_dir(RESULTS_IPI_DIR)
    timestamp = get_timestamp()
    output_file = RESULTS_IPI_DIR / f"ipi_ablation_results_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAblation results saved to {output_file}")
