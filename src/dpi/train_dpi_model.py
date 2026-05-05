# src/dpi/train_dpi_model.py
import warnings
import torch
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from src.dpi.dataset_loader import load_dpi_dataset, tokenize_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.paths import DPI_DATA_DIR, DPI_MODEL_DIR, LOGS_DIR, ensure_dir

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "distilbert-base-uncased"


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
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    # 自动创建目录
    ensure_dir(DPI_MODEL_DIR)
    ensure_dir(LOGS_DIR)

    print("Loading dataset...")

    dataset = load_dpi_dataset(str(DPI_DATA_DIR))

    dataset, tokenizer = tokenize_dataset(dataset, MODEL_NAME)

    print("Loading model...")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(

        output_dir=str(DPI_MODEL_DIR),

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=3,

        evaluation_strategy="epoch",
        save_strategy="epoch",

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

    trainer.save_model(str(DPI_MODEL_DIR))
    tokenizer.save_pretrained(str(DPI_MODEL_DIR))

    print("Model saved to:", DPI_MODEL_DIR)
    print("Logs saved to:", LOGS_DIR)


if __name__ == "__main__":
    main()