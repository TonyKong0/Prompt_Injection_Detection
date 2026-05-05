import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.utils.paths import DATA_DIR, IPI_DATA_DIR, ensure_dir


def convert_dataset():
    """
    Build train/test splits for the semi-synthetic IPI dataset using
    group-based splitting by source_id.

    Split strategy:
    - Uses GroupShuffleSplit with groups=source_id.
    - Ensures all samples derived from the same original DPI source are placed
      entirely in either the train set or the test set.

    Why this matters:
    - dataset_ipi.csv is constructed by converting each original DPI sample into
      multiple IPI variants (e.g., rag_basic, rag_wrapped, structured_source,
      conversation).
    - Random sample-level splitting may place variants from the same source_id
      into both train and test sets, leading to source-level leakage and
      over-optimistic evaluation.

    Output files:
    - ipi_train_group_source.json
    - ipi_test_group_source.json
    """

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

    if df["label"].isnull().any():
        raise ValueError("Found unmapped labels in dataset_ipi.csv")

    # use flat_text directly as model input
    df["context"] = df["flat_text"]

    # keep metadata for later analysis
    dataset = df[
        ["context", "label", "source_id", "scenario_type", "id", "original_text"]
    ].copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            dataset,
            y=dataset["label"],
            groups=dataset["source_id"],
        )
    )

    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx].reset_index(drop=True)

    ensure_dir(IPI_DATA_DIR)

    train_path = IPI_DATA_DIR / "ipi_train_group_source.json"
    test_path = IPI_DATA_DIR / "ipi_test_group_source.json"

    train_df.to_json(train_path, orient="records", indent=2, force_ascii=False)
    test_df.to_json(test_path, orient="records", indent=2, force_ascii=False)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts(normalize=True).sort_index())

    print("\nTest label distribution:")
    print(test_df["label"].value_counts(normalize=True).sort_index())

    train_sources = set(train_df["source_id"])
    test_sources = set(test_df["source_id"])
    overlap = train_sources & test_sources

    print("\nUnique source_id in train:", len(train_sources))
    print("Unique source_id in test:", len(test_sources))
    print("Overlap source_id count:", len(overlap))

    if overlap:
        print("Warning: overlapping source_id found:", sorted(list(overlap))[:10])
    else:
        print("No overlapping source_id between train and test.")

    print("\nSaved to:", IPI_DATA_DIR)
    print("Train file:", train_path)
    print("Test file:", test_path)


if __name__ == "__main__":
    convert_dataset()
