"""
streamlit_app/audio_utils.py

Utility functions for:
- Loading waveform from uploaded audio file
- Displaying waveform
- Optional audio recording simulation
"""

import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO


SAMPLE_RATE = 16000
DURATION = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)


def load_waveform_from_fileobj(file_obj, sr=SAMPLE_RATE):
    """
    Load audio from a Streamlit UploadedFile or BytesIO object.
    Converts to mono, resamples to 16 kHz.
    Returns waveform (np.float32) and sample_rate.
    """
    try:
        # file_obj.read() moves file pointer → so we wrap in BytesIO
        data = file_obj.read()
        waveform, sr = librosa.load(BytesIO(data), sr=sr, mono=True)
        return waveform.astype(np.float32), sr
    except Exception as e:
        st.error(f"Failed to load audio: {e}")
        return np.zeros(TARGET_SAMPLES, dtype=np.float32), sr


def display_waveform(waveform, sr=SAMPLE_RATE, title="Waveform"):
    """
    Display waveform in Streamlit using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    t = np.linspace(0, len(waveform) / sr, len(waveform))
    ax.plot(t, waveform)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


def record_audio_widget(duration=10):
    """
    Placeholder for an audio recording component.

    Streamlit Cloud does not support direct microphone input reliably,
    so this simulates a "record" button by generating silence or sample noise.
    You can replace this later with a proper audio recorder component.

    Returns: BytesIO audio file for inference.
    """
    st.warning("Microphone recording not supported in this demo. Using silence as placeholder.")

    waveform = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)

    import soundfile as sf
    memfile = BytesIO()
    sf.write(memfile, waveform, SAMPLE_RATE, format="WAV")
    memfile.seek(0)

    return memfile
