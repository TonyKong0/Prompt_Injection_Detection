import warnings
import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "distilbert-base-uncased"


# ------------------------------------------------
# Metrics
# ------------------------------------------------

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
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ------------------------------------------------
# Train Function
# ------------------------------------------------

def train_detector(train_dataset,
                   eval_dataset,
                   output_dir,
                   log_dir):

    print("Loading tokenizer...")

    tokenizer = DistilBertTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # ------------------------------------------------
    # Training Arguments (与 DPI 保持一致)
    # ------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(output_dir),

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        num_train_epochs=3,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        logging_dir=str(log_dir),
        logging_steps=50,

        load_best_model_at_end=True,

        save_total_limit=2
    )

    # ------------------------------------------------
    # Trainer
    # ------------------------------------------------

    trainer = Trainer(

        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        tokenizer=tokenizer,

        compute_metrics=compute_metrics
    )

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

    print("\nStarting training...\n")

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nModel saved to:", output_dir)