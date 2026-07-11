import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import DPI_MODEL_DIR, DPI_BERT_MODEL_DIR, MODEL_DIR, RESULTS_DPI_DIR, dpi_parquet_path, ensure_dir


def load_test_data() -> tuple[list[str], list[int]]:
    frame = pd.read_parquet(dpi_parquet_path("test"))
    frame["label"] = frame["label"].apply(lambda value: 0 if value == 0 else 1)
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame["text"].tolist(), frame["label"].tolist()


def get_model_dir(model_name: str, model_dir: Optional[str]) -> Path:
    if model_dir:
        return Path(model_dir)
    if model_name == "distilbert-base-uncased":
        return DPI_MODEL_DIR
    if model_name == "bert-base-uncased":
        return DPI_BERT_MODEL_DIR
    safe_name = model_name.replace("/", "_")
    return MODEL_DIR / f"dpi_detector_{safe_name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased",
                        help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--model-dir", default=None,
                        help="Path to trained model directory (default: auto-derived)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default: 32)")
    args = parser.parse_args()

    ensure_dir(RESULTS_DPI_DIR)

    texts, labels = load_test_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = get_model_dir(args.model_name, args.model_dir)
    display_name = "DistilBERT" if "distilbert" in args.model_name else args.model_name

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    batch_size = args.batch_size
    predictions = []
    confidences = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            batch_predictions = torch.argmax(probs, dim=1)
            batch_confidences = torch.max(probs, dim=1).values

            predictions.extend(batch_predictions.cpu().tolist())
            confidences.extend(batch_confidences.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - start_time

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predictions)

    result = {
        "model_name": display_name,
        "device": str(device),
        "batch_size": batch_size,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predict_seconds": float(total_seconds),
        "ms_per_sample": float((total_seconds / len(texts)) * 1000),
        "parameter_count": int(model.num_parameters()),
        "sample_count": len(texts),
        "average_confidence": float(sum(confidences) / len(confidences)),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.model_name.replace("/", "_")
    output_path = RESULTS_DPI_DIR / f"{safe_name}_inference_timing_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("Saved timing json:", output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
