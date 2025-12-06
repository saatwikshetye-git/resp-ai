"""
model_loader.py
---------------
Loads trained model (Torch / TFLite) for inference.
Final logic added after training & export.
"""

class ModelWrapper:
    def predict(self, mel_tensor):
        # placeholder prediction
        return 0.0, 0, {"explain": "placeholder"}
