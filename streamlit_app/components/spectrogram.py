"""
streamlit_app/components/spectrogram.py

Displays mel-spectrograms in Streamlit.
Matches training settings exactly.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import streamlit as st

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


def plot_mel(waveform: np.ndarray, sr: int = SAMPLE_RATE, title: str = "Mel Spectrogram"):
    """
    Convert waveform → mel spectrogram → log scale
    and plot using matplotlib inside Streamlit.
    """

    try:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(
            logmel,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=ax,
        )
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Failed to generate spectrogram: {e}")

