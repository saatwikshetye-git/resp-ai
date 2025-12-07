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

    Important: This function rewinds the file_obj after reading so
    the same object can be uploaded/sent afterwards.
    """
    try:
        # Ensure we're at start of file
        try:
            file_obj.seek(0)
        except Exception:
            pass

        data = file_obj.read()
        waveform, actual_sr = librosa.load(BytesIO(data), sr=sr, mono=True)

        # Rewind uploaded file so it can be reused (e.g., for sending to backend)
        try:
            file_obj.seek(0)
        except Exception:
            pass

        return waveform.astype(np.float32), actual_sr

    except Exception as e:
        st.error(f"Failed to load audio: {e}")
        # Ensure the file pointer is rewound before returning
        try:
            file_obj.seek(0)
        except Exception:
            pass
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
    plt.close(fig)


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

    # Give the BytesIO a name attribute so downstream code can use it like a file
    memfile.name = "recording.wav"
    return memfile
