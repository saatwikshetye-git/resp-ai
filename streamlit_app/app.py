"""
streamlit_app/app.py

Main Streamlit UI for Respiratory Distress Detection.
Allows:
- Upload audio file
- Visualize waveform + mel spectrogram
- Send to FastAPI backend for inference
- Show distress score, predicted label, and metadata
"""

import streamlit as st
import numpy as np

from audio_utils import load_waveform_from_fileobj, display_waveform
from components.spectrogram import plot_mel
from api_client import send_audio_for_prediction

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------

st.set_page_config(
    page_title="Respiratory Distress Detector",
    layout="centered",
)

st.title("🩺 Respiratory Distress Detection Demo")
st.markdown(
    """
Upload a **10-second breathing audio** (WAV/MP3) and the model will analyze it for
respiratory distress indicators such as **wheezing, crackles, cough, or agonal breathing**.
    """
)

# ---------------------------------------------------------
# Audio Upload Section
# ---------------------------------------------------------

st.sidebar.header("Upload Audio")
uploaded = st.sidebar.file_uploader(
    "Choose audio file (.wav or .mp3)", 
    type=["wav", "mp3"],
)

# ---------------------------------------------------------
# Once file is uploaded
# ---------------------------------------------------------

if uploaded is not None:
    st.subheader("🔊 Audio Playback")
    st.audio(uploaded)

    # Load waveform
    waveform, sr = load_waveform_from_fileobj(uploaded)

    # Display waveform
    st.subheader("📈 Waveform")
    display_waveform(waveform, sr)

    # Display spectrogram
    st.subheader("📡 Mel Spectrogram")
    plot_mel(waveform, sr)

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    st.subheader("🤖 Model Prediction")

    if st.button("Analyze Audio", use_container_width=True):
        with st.spinner("Analyzing..."):
            response = send_audio_for_prediction(uploaded)

        if response is None:
            st.error("Inference failed.")
        else:
            # Score + label
            st.metric(
                label="Distress Score",
                value=f"{response['distress_score']:.3f}",
            )

            st.write(f"**Predicted Label:** {response['label']}")
            st.write(f"**Confidence:** {response['confidence']:.3f}")

            # Color-coded alert
            if response["distress_score"] >= 0.7:
                st.error("🚨 High Respiratory Distress Detected!")
            elif response["distress_score"] >= 0.4:
                st.warning("⚠️ Moderate distress — consider monitoring.")
            else:
                st.success("✅ No significant distress detected.")

            # Probabilities table
            st.markdown("### Class Probabilities")
            st.json(response["probabilities"])

            # Metadata
            st.markdown("### Metadata")
            st.json({
                "timestamp": response.get("timestamp"),
                "latency_ms": response.get("latency_ms"),
            })

else:
    st.info("Upload an audio file to begin analysis.")
