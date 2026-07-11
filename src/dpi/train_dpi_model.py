# src/dpi/train_dpi_model.py
import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from src.dpi.dataset_loader import load_dpi_dataset, tokenize_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.paths import DPI_DATA_DIR, DPI_MODEL_DIR, MODEL_DIR, LOGS_DIR, ensure_dir

warnings.filterwarnings("ignore", category=FutureWarning)


def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased",
                        help="HuggingFace model name (default: distilbert-base-uncased)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: auto-derived from model name)")
    args = parser.parse_args()
    model_name = args.model_name

    # Derive output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif model_name == "distilbert-base-uncased":
        output_dir = DPI_MODEL_DIR
    else:
        safe_name = model_name.replace("/", "_")
        output_dir = MODEL_DIR / f"dpi_detector_{safe_name}"

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    # 自动创建目录
    ensure_dir(output_dir)
    ensure_dir(LOGS_DIR)

    print("Loading dataset...")

    dataset = load_dpi_dataset(str(DPI_DATA_DIR))

    dataset, tokenizer = tokenize_dataset(dataset, model_name)

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(

        output_dir=str(output_dir),

        learning_rate=2e-5,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        gradient_accumulation_steps=2,

        num_train_epochs=3,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        fp16=True,

        logging_dir=str(LOGS_DIR),
        logging_steps=50,

        load_best_model_at_end=True,

        save_total_limit=2
    )

    trainer = Trainer(

        model=model,
        args=training_args,

        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],

        tokenizer=tokenizer,

        compute_metrics=compute_metrics
    )

    print("Training DPI detector...")

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Model saved to:", output_dir)
    print("Logs saved to:", LOGS_DIR)


if __name__ == "__main__":
    main()