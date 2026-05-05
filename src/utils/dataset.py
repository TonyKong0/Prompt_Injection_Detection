from datasets import load_dataset

def load_parquet_dataset(path):

    dataset = load_dataset(
        "parquet",
        data_files={"test": str(path)}
    )

    return dataset["test"]