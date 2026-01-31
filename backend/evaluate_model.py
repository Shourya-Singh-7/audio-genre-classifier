import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

from inference import extract_features, model, scaler, GENRES

# Path to GTZAN data (adjust if needed)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "Raw" / "genres_original"

y_true = []
y_pred = []

print("🔍 Evaluating model on GTZAN dataset...")

count = 0

for label_idx, genre in enumerate(GENRES):
    genre_dir = DATA_DIR / genre

    for audio_file in genre_dir.glob("*.wav"):
        try:
            feats = extract_features(str(audio_file))
            feats = scaler.transform(feats)

            probs = model.predict(feats, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

            y_true.append(label_idx)
            y_pred.append(pred_idx)

            count += 1
            if count % 50 == 0:
                print(f"Processed {count} files...")

        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")


y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

metrics = {
    "accuracy": float(accuracy),
    "labels": GENRES,
    "confusion_matrix": cm.tolist()
}

# Save metrics
METRICS_PATH = Path(__file__).resolve().parent / "metrics.json"
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"📁 Metrics saved to {METRICS_PATH}")
