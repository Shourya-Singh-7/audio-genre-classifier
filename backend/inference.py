import numpy as np
import librosa
import joblib
import tensorflow as tf
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model" / "audio_genre"

# Load model + scaler ONCE (important for performance)
model = tf.keras.models.load_model(MODEL_DIR / "audio_genre_mlp.keras")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

def extract_features(file_path, sr=22050, duration=30):
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)

    expected_length = sr * duration
    if len(y) < expected_length:
        y = np.pad(y, (0, expected_length - len(y)))
    else:
        y = y[:expected_length]

    features = []

    # MFCC (26)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Spectral (8)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=200.0)

    for spec in [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast]:
        features.append(np.mean(spec))
        features.append(np.std(spec))

    # ZCR + RMS (4)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    # Tempo (1)
    tempo = float(np.squeeze(librosa.beat.beat_track(y=y, sr=sr)[0]))
    features.append(tempo)

    return np.array(features, dtype=np.float32).reshape(1, -1)

def predict_genre(audio_path: str):
    feats = extract_features(audio_path)
    feats = scaler.transform(feats)
    probs = model.predict(feats, verbose=0)[0]

    idx = int(np.argmax(probs))
    return GENRES[idx], float(probs[idx])
