"""
backend/preprocess.py

Converts uploaded audio bytes → waveform → normalized log-mel spectrogram.
This MUST match training preprocessing exactly.
"""

import numpy as np
import librosa
from io import BytesIO
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 10.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION)

# Mel parameters (must match training)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


def read_audio_bytes(audio_bytes: bytes):
    """
    Load audio from raw bytes.
    Returns: waveform np.array(float32), sample_rate
    """
    try:
        # librosa can load from BytesIO when soundfile is available
        with BytesIO(audio_bytes) as f:
            waveform, sr = librosa.load(f, sr=SAMPLE_RATE, mono=True)
        return waveform.astype(np.float32), sr
    except Exception:
        # fallback
        try:
            data, sr = sf.read(BytesIO(audio_bytes))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != SAMPLE_RATE:
                data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
            return data.astype(np.float32), SAMPLE_RATE
        except Exception as e:
            print("[read_audio_bytes] FAILED:", e)
            return None, None


def ensure_duration(waveform: np.ndarray, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """
    Pad or trim waveform to exact duration (16000 * 10 = 160000 samples)
    """
    if waveform is None:
        return np.zeros(TARGET_SAMPLES, dtype=np.float32)

    target_len = int(sr * duration)
    cur_len = len(waveform)

    if cur_len > target_len:
        waveform = waveform[:target_len]
    elif cur_len < target_len:
        pad_len = target_len - cur_len
        waveform = np.pad(waveform, (0, pad_len), mode="constant")

    return waveform.astype(np.float32)


def waveform_to_mel(waveform: np.ndarray):
    """
    Convert waveform to normalized log-mel spectrogram (128 x T)
    EXACT match with training preproc.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    # normalize (same as training)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-9)

    return logmel.astype(np.float32)


def preprocess_audio_bytes(audio_bytes: bytes):
    """
    Full pipeline:
    1. load bytes → waveform
    2. enforce duration
    3. waveform → normalized mel (1, 128, T)
    Returns numpy array
    """
    waveform, sr = read_audio_bytes(audio_bytes)
    if waveform is None:
        return None

    waveform = ensure_duration(waveform, sr)
    mel = waveform_to_mel(waveform)

    # final shape expected by model: (1, 128, T)
    return mel[np.newaxis, :, :]  # shape = (1, N_MELS, time)
