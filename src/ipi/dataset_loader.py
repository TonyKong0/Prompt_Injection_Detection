import json
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"


# ------------------------------------------------
# Dataset
# ------------------------------------------------

class IPIDataset(Dataset):

    def __init__(self, data, tokenizer, max_len=512):

        texts = [d["context"] for d in data]
        labels = [d["label"] for d in data]

        self.labels = labels

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item


# ------------------------------------------------
# JSON Loader
# ------------------------------------------------

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------
# Main dataset loader
# ------------------------------------------------

def load_ipi_dataset(data_dir):

    print("Loading IPI dataset...")

    train_data = load_json(data_dir / "ipi_train.json")
    test_data = load_json(data_dir / "ipi_test.json")

    print("Train samples:", len(train_data))
    print("Test samples:", len(test_data))

    tokenizer = DistilBertTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = IPIDataset(train_data, tokenizer)
    test_dataset = IPIDataset(test_data, tokenizer)

    return train_dataset, test_dataset