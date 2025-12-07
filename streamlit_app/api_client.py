"""
streamlit_app/api_client.py

Handles communication between Streamlit frontend and the FastAPI backend.
"""

import requests
import streamlit as st


# -------------------------------------------------------------------
# Load API URL from Streamlit Secrets
# -------------------------------------------------------------------

# Primary source = Streamlit Cloud secret
if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
else:
    # Local fallback (useful when testing on your own PC)
    API_URL = "http://localhost:8000/predict"


def send_audio_for_prediction(file_obj):
    """
    Sends audio file to FastAPI backend for inference.

    Parameters:
    - file_obj: Streamlit UploadedFile or BytesIO

    Returns:
    - JSON response (dict) or None on error
    """

    try:
        files = {
            "file": (
                getattr(file_obj, "name", "audio.wav"),
                file_obj,
                "audio/wav",
            )
        }

        response = requests.post(API_URL, files=files, timeout=20)

        # Backend returned an error
        if response.status_code != 200:
            st.error(f"API Error ({response.status_code}): {response.text}")
            return None

        # Success
        return response.json()

    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out. Backend might be sleeping — try again.")
        return None

    except requests.exceptions.ConnectionError:
        st.error("❌ Failed to connect to backend. Check API_URL or backend status.")
        return None

    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None
