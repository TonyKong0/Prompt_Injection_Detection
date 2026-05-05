import json

from src.training.train_detector import train_detector
from src.utils.paths import IPI_DATA_DIR, MODEL_DIR
from transformers import DistilBertTokenizer
from src.ipi.dataset_loader import IPIDataset

MODEL_NAME = "distilbert-base-uncased"

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    train_data = load_json(IPI_DATA_DIR / "ipi_train_group_source.json")
    test_data = load_json(IPI_DATA_DIR / "ipi_test_group_source.json")

    tokenizer = DistilBertTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = IPIDataset(train_data, tokenizer)
    test_dataset = IPIDataset(test_data, tokenizer)

    train_detector(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        output_dir=MODEL_DIR / "ipi_detector_v2",
        log_dir=MODEL_DIR / "ipi_detector_v2/logs"
    )

if __name__ == "__main__":
    main()