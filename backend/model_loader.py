"""
backend/model_loader.py

Loads the trained PyTorch model for CPU inference.
Provides ModelWrapper.predict() which accepts a mel spectrogram (numpy array)
of shape (1, 128, T) and returns:

- distress_score
- predicted label index
- raw probabilities
- placeholder explainability dict (filled later)

This file must NOT import training-only components.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from model.resnet_spec import ResNetSpec

# Distress class mapping (you can adjust later)
# 0 normal, 1 cough, 2 wheeze, 3 crackles, 4 agonal
DISTRESS_CLASSES = [1, 2, 3, 4]  # anything except normal


class ModelWrapper:
    """
    Loads model/checkpoints/best.pt and performs prediction.
    """

    def __init__(self, checkpoint_path: str = "model/checkpoints/best.pt"):
        ckpt_path = Path(checkpoint_path)

        if not ckpt_path.exists():
            print("[ModelLoader] WARNING: Checkpoint not found:", ckpt_path)
            print("              API will load an untrained model.")
            self.model = ResNetSpec(num_classes=5)
        else:
            print(f"[ModelLoader] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu")

            self.model = ResNetSpec(num_classes=5)
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def predict(self, mel_array: np.ndarray):
        """
        mel_array shape: (1, 128, T)
        Returns:
            - distress_score: float
            - label_index: int
            - probs: list[float]
            - explain: dict (placeholder)
        """
        if mel_array is None:
            return None

        # Convert numpy -> torch
        with torch.no_grad():
            x = torch.from_numpy(mel_array).float().unsqueeze(0)  # (1,1,128,T)
            x = x.to(self.device)

            logits = self.model(x)  # (1, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # predicted label
        label_idx = int(np.argmax(probs))

        # distress_score = sum of probabilities of distress classes
        distress_score = float(np.sum([probs[i] for i in DISTRESS_CLASSES]))

        # placeholder explainability (to be added later)
        explain = {
            "confidence": float(probs[label_idx]),
            "gradcam": None,
        }

        return distress_score, label_idx, probs.tolist(), explain
