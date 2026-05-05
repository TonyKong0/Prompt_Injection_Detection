import json
import torch
from datetime import datetime

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.ipi.dataset_loader import IPIDataset
from src.utils.paths import IPI_DATA_DIR, MODEL_DIR, RESULTS_IPI_DIR, ensure_dir


MODEL_NAME = "distilbert-base-uncased"

# 新模型目录名
MODEL_SUBDIR = "ipi_detector_v2"

# 新测试集文件名（group split）
TEST_FILE = "ipi_test_group_source.json"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------
# Load JSON
# ------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------
# Evaluate
# ------------------------------------------------

def evaluate(model, dataset, device):
    model.eval()

    preds = []
    labels = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            labels.extend(y.cpu().tolist())

    return preds, labels


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    ensure_dir(RESULTS_IPI_DIR)

    test_path = IPI_DATA_DIR / TEST_FILE
    model_path = MODEL_DIR / MODEL_SUBDIR

    print("Loading test dataset...")
    print("Test file:", test_path)

    test_data = load_json(test_path)

    print("Loading tokenizer from:", model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    test_dataset = IPIDataset(
        test_data,
        tokenizer
    )

    print("Test samples:", len(test_dataset))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    print("Loading trained model from:", model_path)

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    preds, labels = evaluate(
        model,
        test_dataset,
        device
    )

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, zero_division=0)

    print("\n===== Evaluation Results =====")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    result = {
        "model_dir": str(model_path),
        "test_file": str(test_path),
        "num_test_samples": len(test_dataset),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    timestamp = get_timestamp()
    output_path = RESULTS_IPI_DIR / f"{MODEL_SUBDIR}_eval_group_source_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nSaved evaluation results to:", output_path)


if __name__ == "__main__":
    main()
