from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
from pathlib import Path
import uuid

from backend.inference import predict_genre

app = FastAPI(title="Audio Genre Classification API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Audio Genre Classifier API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        genre, confidence = predict_genre(str(temp_path))

        return {
            "predicted_genre": genre,
            "confidence": round(confidence, 4)
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()

import json
from pathlib import Path

METRICS_PATH = Path("backend/metrics.json")

@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        return {"error": "Metrics not available"}

    with open(METRICS_PATH) as f:
        return json.load(f)
