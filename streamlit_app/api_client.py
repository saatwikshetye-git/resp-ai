import requests
import streamlit as st

# Use deployed backend URL
API_URL = "https://resp-ai-backend.onrender.com/predict"

def send_audio_for_prediction(file_obj):
    """
    Sends audio file to FastAPI backend.
    """

    try:
        files = {
            "file": (
                file_obj.name if hasattr(file_obj, "name") else "audio.wav",
                file_obj,
                "audio/wav",
            )
        }

        resp = requests.post(API_URL, files=files, timeout=20)

        if resp.status_code != 200:
            st.error(f"API error: {resp.text}")
            return None

        return resp.json()

    except Exception as e:
        st.error(f"Request failed: {e}")
        return None
