import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot_styles import apply_bw_axis_style, bar_style, save_bw_figure
from src.utils.paths import RESULTS_DPI_DIR, dpi_parquet_path, ensure_dir


SEED = 42
MAX_TOKENS = 20000
SEQUENCE_LENGTH = 256
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_dpi_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(dpi_parquet_path("train"))
    test_df = pd.read_parquet(dpi_parquet_path("test"))

    for frame in (train_df, test_df):
        frame["label"] = frame["label"].apply(lambda value: 0 if value == 0 else 1)
        frame["text"] = frame["text"].fillna("").astype(str)

    return train_df, test_df


def build_vectorizer(train_texts: np.ndarray) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    vectorizer.adapt(train_texts)
    return vectorizer


def build_fasttext_model(vectorizer: layers.TextVectorization) -> keras.Model:
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = layers.Embedding(MAX_TOKENS, EMBED_DIM, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="fasttext_style")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_textcnn_model(vectorizer: layers.TextVectorization) -> keras.Model:
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = layers.Embedding(MAX_TOKENS, EMBED_DIM, name="embedding")(x)

    conv_3 = layers.Conv1D(128, 3, activation="relu")(x)
    conv_4 = layers.Conv1D(128, 4, activation="relu")(x)
    conv_5 = layers.Conv1D(128, 5, activation="relu")(x)

    pool_3 = layers.GlobalMaxPooling1D()(conv_3)
    pool_4 = layers.GlobalMaxPooling1D()(conv_4)
    pool_5 = layers.GlobalMaxPooling1D()(conv_5)

    x = layers.concatenate([pool_3, pool_4, pool_5])
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="textcnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(
    model: keras.Model,
    model_name: str,
    train_texts: np.ndarray,
    train_labels: np.ndarray,
    test_texts: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    train_x, val_x, train_y, val_y = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=SEED,
        stratify=train_labels,
        shuffle=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    train_start = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    train_seconds = time.perf_counter() - train_start

    predict_start = time.perf_counter()
    probabilities = model.predict(test_texts, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    predict_seconds = time.perf_counter() - predict_start
    predictions = (probabilities >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(test_labels, predictions)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "train_seconds": float(train_seconds),
        "predict_seconds": float(predict_seconds),
        "ms_per_sample": float((predict_seconds / len(test_labels)) * 1000),
        "parameter_count": int(model.count_params()),
        "epochs_ran": len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
    }


def _load_hf_result(
    model_display_name: str,
    param_count_fallback: int,
    epochs_ran_fallback: int,
    glob_pattern: str,
) -> dict:
    result_candidates = sorted(RESULTS_DPI_DIR.glob(glob_pattern))
    if not result_candidates:
        raise FileNotFoundError(
            f"No {model_display_name} result json found matching '{glob_pattern}' "
            f"in {RESULTS_DPI_DIR}."
        )

    latest_result = result_candidates[-1]
    with latest_result.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    return {
        "model_name": model_display_name,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["attack"]["precision"]),
        "recall": float(metrics["attack"]["recall"]),
        "f1": float(metrics["attack"]["f1-score"]),
        "train_seconds": None,
        "predict_seconds": metrics.get("predict_seconds"),
        "ms_per_sample": metrics.get("ms_per_sample"),
        "parameter_count": metrics.get("parameter_count", param_count_fallback),
        "epochs_ran": metrics.get("epochs_ran", epochs_ran_fallback),
        "best_val_accuracy": float(metrics["accuracy"]),
        "source_result_file": str(latest_result),
    }


def load_distilbert_result() -> dict:
    return _load_hf_result("DistilBERT", 66_000_000, 3, "dpi_experiment_*.json")


def load_bert_result() -> dict:
    return _load_hf_result("BERT", 110_000_000, 3, "dpi_experiment_bert_*.json")


def save_comparison_chart(results: list[dict], output_path: Path) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = [item["model_name"] for item in results]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, metric in enumerate(metric_names):
        values = [item[metric] for item in results]
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=metric.upper(),
            **bar_style(idx),
        )

    ax.set_ylim(0.85, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("DPI Model Comparison")
    ax.legend()
    apply_bw_axis_style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    save_bw_figure(fig, output_path, dpi=200)
    plt.close(fig)


def save_efficiency_chart(results: list[dict], output_path: Path) -> None:
    efficiency_results = [item for item in results if item["ms_per_sample"] is not None]
    labels = [item["model_name"] for item in efficiency_results]
    ms_values = [item["ms_per_sample"] for item in efficiency_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = []
    for idx, (label, value) in enumerate(zip(labels, ms_values)):
        container = ax.bar(label, value, **bar_style(idx + 1))
        bars.append(container[0])
    ax.set_ylabel("Milliseconds per sample")
    ax.set_title("DPI Baseline Inference Efficiency")
    apply_bw_axis_style(ax)

    for bar, value in zip(bars, ms_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    save_bw_figure(fig, output_path, dpi=200)
    plt.close(fig)


def save_summary_table(results: list[dict], output_path: Path) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    set_seed()
    ensure_dir(RESULTS_DPI_DIR)

    train_df, test_df = load_dpi_data()
    train_texts = train_df["text"].to_numpy()
    train_labels = train_df["label"].to_numpy()
    test_texts = test_df["text"].to_numpy()
    test_labels = test_df["label"].to_numpy()

    vectorizer = build_vectorizer(train_texts)

    fasttext_model = build_fasttext_model(vectorizer)
    textcnn_model = build_textcnn_model(vectorizer)

    fasttext_result = evaluate_model(
        fasttext_model,
        "fastText-style",
        train_texts,
        train_labels,
        test_texts,
        test_labels,
    )
    textcnn_result = evaluate_model(
        textcnn_model,
        "TextCNN",
        train_texts,
        train_labels,
        test_texts,
        test_labels,
    )
    distilbert_result = load_distilbert_result()
    bert_result = load_bert_result()

    all_results = [fasttext_result, textcnn_result, distilbert_result, bert_result]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_json_path = RESULTS_DPI_DIR / f"dpi_baseline_comparison_{timestamp}.json"
    summary_csv_path = RESULTS_DPI_DIR / f"dpi_baseline_comparison_{timestamp}.csv"
    score_chart_path = RESULTS_DPI_DIR / f"dpi_baseline_scores_{timestamp}.png"
    efficiency_chart_path = RESULTS_DPI_DIR / f"dpi_baseline_efficiency_{timestamp}.png"

    with result_json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)

    save_summary_table(all_results, summary_csv_path)
    save_comparison_chart(all_results, score_chart_path)
    save_efficiency_chart(all_results, efficiency_chart_path)

    print("Saved result json:", result_json_path)
    print("Saved summary csv:", summary_csv_path)
    print("Saved score chart:", score_chart_path)
    print("Saved efficiency chart:", efficiency_chart_path)
    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
