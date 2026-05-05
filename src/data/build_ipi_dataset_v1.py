
"""
build_ipi_dataset_v1.py

Dataset split script for the semi-synthetic IPI dataset.

Split strategy:
- Uses standard random train/test split at the sample level.
- Applies label stratification to preserve the benign/jailbreak ratio.

Important limitation:
- This split does NOT control for source-level grouping.
- In dataset_ipi.csv, multiple IPI samples are derived from the same original
  DPI source sample and share the same source_id.
- Therefore, samples originating from the same source_id may appear in both
  train and test sets, which can introduce source-level leakage and lead to
  overly optimistic evaluation results, especially for neural classifiers.

Recommended usage:
- Keep this version as a baseline random-split setting.
- For more reliable evaluation, use build_ipi_dataset_v2.py, which performs
  group-based splitting by source_id.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.paths import DATA_DIR, IPI_DATA_DIR, ensure_dir


def convert_dataset():

    input_file = DATA_DIR / "ipi" / "dataset_ipi.csv"

    df = pd.read_csv(input_file)

    df.columns = df.columns.str.strip()

    print("Loaded rows:", len(df))
    print("Columns:", df.columns.tolist())

    # label mapping
    df["label"] = df["label"].map({
        "benign": 0,
        "jailbreak": 1
    })

    # use flat_text directly
    df["context"] = df["flat_text"]

    dataset = df[["context", "label"]]

    train_df, test_df = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"]
    )

    ensure_dir(IPI_DATA_DIR)

    train_path = IPI_DATA_DIR / "ipi_train.json"
    test_path = IPI_DATA_DIR / "ipi_test.json"

    train_df.to_json(train_path, orient="records", indent=2)
    test_df.to_json(test_path, orient="records", indent=2)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    print("Saved to:", IPI_DATA_DIR)


if __name__ == "__main__":
    convert_dataset()
