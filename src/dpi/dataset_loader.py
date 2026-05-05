# src/dpi/dataset_loader.py

from datasets import load_dataset
from transformers import DistilBertTokenizer


def load_dpi_dataset(data_dir):
    """
    加载DPI数据集
    """
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{data_dir}/train-00000-of-00001.parquet",
            "test": f"{data_dir}/test-00000-of-00001.parquet",
        }
    )

    return dataset


def convert_labels(example):
    """
    将标签转为二分类
    0 -> safe
    1,2 -> attack
    """
    example["label"] = 0 if example["label"] == 0 else 1
    return example


def tokenize_dataset(dataset, model_name="distilbert-base-uncased", max_length=256):

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    dataset = dataset.map(convert_labels)
    dataset = dataset.map(tokenize, batched=True)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return dataset, tokenizer