"""
Evaluate a trained BERT DPI detector on the test set.
Output: outputs/results/dpi/dpi_experiment_bert_{timestamp}.json
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import DPI_BERT_MODEL_DIR, RESULTS_DPI_DIR, dpi_parquet_path, ensure_dir
from src.utils.dataset import load_parquet_dataset


def map_label(label: int) -> str:
    return "safe" if label == 0 else "attack"


def main() -> None:
    ensure_dir(RESULTS_DPI_DIR)

    # 1. Load test dataset
    test_dataset = load_parquet_dataset(dpi_parquet_path("test"))
    texts = [x["text"] for x in test_dataset]
    labels = [x["label"] for x in test_dataset]

    y_true_mapped = [map_label(l) for l in labels]

    # 2. Load model
    model_dir = DPI_BERT_MODEL_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading BERT model from:", model_dir)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    # 3. Batch inference with timing
    batch_size = 32
    y_pred_mapped = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    predict_start = time.perf_counter()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i: i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for p in preds.cpu():
                y_pred_mapped.append(map_label(p.item()))

    if device.type == "cuda":
        torch.cuda.synchronize()
    predict_seconds = time.perf_counter() - predict_start

    # 4. Metrics
    results_dict = classification_report(
        y_true_mapped,
        y_pred_mapped,
        output_dict=True,
        zero_division=0,
    )

    # 5. Enrich with timing and model info
    results_dict["predict_seconds"] = round(predict_seconds, 3)
    results_dict["ms_per_sample"] = round((predict_seconds / len(texts)) * 1000, 4)
    results_dict["parameter_count"] = model.num_parameters()
    results_dict["sample_count"] = len(texts)

    # 6. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DPI_DIR / f"dpi_experiment_bert_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    print(f"\nAccuracy: {results_dict.get('accuracy', 0):.4f}")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
