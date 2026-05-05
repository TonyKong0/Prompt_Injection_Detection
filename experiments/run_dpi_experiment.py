
import json
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from datetime import datetime
from src.dpi.dpi_detector import DPIDetector
from src.utils.paths import RESULTS_DPI_DIR, dpi_parquet_path, ensure_dir
from src.utils.dataset import load_parquet_dataset

# ========= 1. 加载数据集 =========

test_dataset = load_parquet_dataset(dpi_parquet_path("test"))

print("Dataset size:", len(test_dataset))


# ========= 2. 加载检测器 =========

print("Loading DPI Detector...")

detector = DPIDetector()

print("Model loaded.")


# ========= 3. 运行预测 =========

texts = [x["text"] for x in test_dataset]
labels = [x["label"] for x in test_dataset]

y_true = labels
y_pred = []

# 0 -> "safe"
# 1,2 -> "attack"

def map_label(label):
    return "safe" if label == 0 else "attack"


# 对 y_true 和 y_pred 映射
y_true_mapped = [map_label(l) for l in y_true]
y_pred_mapped = []

batch_size = 32

print("Running DPI detection...")

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]

    preds, _ = detector.predict_batch(batch_texts)  # preds 是数字 0/1/2

    # 映射 preds -> "safe"/"attack"
    preds_mapped = [map_label(p) for p in preds]

    y_pred_mapped.extend(preds_mapped)

# ========= 4. 输出结果 =========

print("\nAccuracy:", accuracy_score(y_true_mapped, y_pred_mapped))

print("\nClassification Report:\n")
print(classification_report(y_true_mapped, y_pred_mapped, zero_division=0))

# ========= 5. 保存实验结果 =========

results_dict = classification_report(
    y_true_mapped,
    y_pred_mapped,
    output_dict=True,
    zero_division=0
)

ensure_dir(RESULTS_DPI_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = RESULTS_DPI_DIR / f"dpi_experiment_{timestamp}.json"

with open(output_path, "w") as f:
    json.dump(results_dict, f, indent=4)

print("Results saved to:", output_path)