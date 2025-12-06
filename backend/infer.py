"""
backend/infer.py

Central inference function used by FastAPI.

Pipeline:
1. Read uploaded audio bytes
2. Convert to waveform
3. Fix duration (10 seconds)
4. Convert waveform → log-mel spectrogram (1,128,T)
5. Run model prediction
6. Format JSON-ready output
"""

import time
from backend.preprocess import preprocess_audio_bytes
from backend.model_loader import ModelWrapper

# Load model once globally (best practice)
MODEL = ModelWrapper()


def predict_audio_bytes(audio_bytes: bytes):
    """
    Main inference function used by FastAPI.

    Returns:
        dict with:
        - distress_score
        - label
        - confidence
        - timestamp
        - explain
    """

    start = time.time()

    # Step 1–3: preprocessing
    mel = preprocess_audio_bytes(audio_bytes)
    if mel is None:
        return {
            "error": "audio_processing_failed",
            "message": "Unable to decode audio bytes"
        }

    # Step 4: run prediction
    distress_score, label_idx, probs, explain = MODEL.predict(mel)

    # Step 5: convert to API JSON
    # label mapping — can adjust later
    LABEL_MAP = {
        0: "normal",
        1: "cough",
        2: "wheeze",
        3: "crackles",
        4: "agonal"
    }

    response = {
        "distress_score": float(distress_score),
        "label": LABEL_MAP.get(label_idx, "unknown"),
        "confidence": float(explain.get("confidence", 0.0)),
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latency_ms": int((time.time() - start) * 1000),
        "explain": explain,
    }

    return response

