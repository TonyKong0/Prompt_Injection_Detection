import json
from datetime import datetime
from typing import Dict, List, Any

from src.ipi.ipi_detector import IPIDetector
from src.ipi.ipi_bert_detector import IPIBertDetector
from src.system.unified_detector import UnifiedPromptInjectionDetector
from src.utils.paths import IPI_DATA_DIR, RESULTS_IPI_DIR, ensure_dir
from experiments.context_utils import parse_structured_context


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _bool_label(raw_label: Any) -> bool:
    s = str(raw_label).strip().lower()
    return s in {"1", "jailbreak", "attack", "true"}


def _binary_metrics(rows: List[Dict[str, bool]]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in rows:
        gt = r["ground_truth"]
        pd = r["predicted"]
        if gt and pd:
            tp += 1
        elif (not gt) and pd:
            fp += 1
        elif (not gt) and (not pd):
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": len(rows),
    }


def _metrics_by_scenario(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_scenario: Dict[str, List[Dict[str, bool]]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario_type"], []).append(
            {
                "ground_truth": r["ground_truth"],
                "predicted": r["predicted"],
            }
        )
    return {k: _binary_metrics(v) for k, v in sorted(by_scenario.items())}


def run():
    ensure_dir(RESULTS_IPI_DIR)

    test_path = IPI_DATA_DIR / "ipi_test_group_source.json"
    data = load_json(test_path)

    # Keep consistent with tuned pre-warning settings.
    warning_detector = IPIDetector(
        instructionality_weight=0.33402774140688785,
        authority_conflict_weight=0.4275146911846204,
        execution_affinity_weight=0.23845756740849178,
        warn_threshold=0.12465943559777146,
        block_threshold=0.44639509181172915,
        strong_signal_multiplier=1.2537570435538332,
        pattern_bonus_single=0.1258764421444331,
        pattern_bonus_multi=0.23699826495664625,
        context_bonus_retrieved=0.11890409066378241,
        context_bonus_history=0.07091063530159541,
        context_bonus_benign=0.04082465585401465,
    )
    bert_detector = IPIBertDetector()
    unified = UnifiedPromptInjectionDetector(dpi_warn_as_stop=False)

    rows_warning: List[Dict[str, Any]] = []
    rows_bert: List[Dict[str, Any]] = []
    rows_unified: List[Dict[str, Any]] = []

    for item in data:
        structured = parse_structured_context(item.get("context", ""))
        gt = _bool_label(item.get("label", 0))
        scenario_type = item.get("scenario_type", "unknown")

        w = warning_detector.analyze(
            user_prompt=structured["user_prompt"],
            conversation_history=structured["conversation_history"],
            retrieved_materials=structured["retrieved_materials"],
            metadata={"sample_id": item.get("id")},
        )
        w_pred = w["decision"] in {"WARN", "BLOCK"}
        rows_warning.append(
            {
                "ground_truth": gt,
                "predicted": w_pred,
                "scenario_type": scenario_type,
            }
        )

        b = bert_detector.detect(
            user_prompt=structured["user_prompt"],
            conversation_history=structured["conversation_history"],
            retrieved_materials=structured["retrieved_materials"],
        )
        b_pred = b["decision"] in {"WARN", "BLOCK"}
        rows_bert.append(
            {
                "ground_truth": gt,
                "predicted": b_pred,
                "scenario_type": scenario_type,
            }
        )

        u = unified.detect(
            user_prompt=structured["user_prompt"],
            conversation_history=structured["conversation_history"],
            retrieved_materials=structured["retrieved_materials"],
            metadata={"sample_id": item.get("id")},
        )
        u_pred = u["final_decision"] in {"WARN", "BLOCK"}
        rows_unified.append(
            {
                "ground_truth": gt,
                "predicted": u_pred,
                "scenario_type": scenario_type,
            }
        )

    result = {
        "test_file": str(test_path),
        "num_samples": len(data),
        "overall": {
            "ipi_warning_only": _binary_metrics(rows_warning),
            "ipi_bert_only": _binary_metrics(rows_bert),
            "unified": _binary_metrics(rows_unified),
        },
        "by_scenario": {
            "ipi_warning_only": _metrics_by_scenario(rows_warning),
            "ipi_bert_only": _metrics_by_scenario(rows_bert),
            "unified": _metrics_by_scenario(rows_unified),
        },
    }

    ts = get_timestamp()
    out_path = RESULTS_IPI_DIR / f"unified_benchmark_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print("Overall metrics:")
    for k, v in result["overall"].items():
        print(k, v)


if __name__ == "__main__":
    run()
