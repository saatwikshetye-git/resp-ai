"""
backend/main.py

FastAPI server for Resp-AI inference.
Exposes:
- GET /health
- POST /predict  (file upload: audio/wav or mp3)

Uses backend.infer.predict_audio_bytes() for ML logic.
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.infer import predict_audio_bytes

# ---------------------------------------------------
# FastAPI initialization
# ---------------------------------------------------

app = FastAPI(
    title="Respiratory Distress Detection API",
    version="0.1.0",
    description="Infer respiratory distress from 10s breathing audio.",
)

# Allow Streamlit frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow all for demo; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Health Endpoint
# ---------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------
# Predict Endpoint
# ---------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an audio file (.wav or .mp3) and returns:
    - distress_score
    - label
    - confidence
    - probabilities
    - timestamp
    - explainability placeholder
    """

    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file.content_type}. Please upload WAV or MP3.",
        )

    # Read raw bytes
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read file: {e}")

    # Run inference pipeline
    result = predict_audio_bytes(audio_bytes)

    if result is None:
        raise HTTPException(status_code=500, detail="Inference failed internally.")

    return result


# ---------------------------------------------------
# Local dev entry point
# ---------------------------------------------------

if __name__ == "__main__":
    # Run locally: python backend/main.py
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
