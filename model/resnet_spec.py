"""
resnet_spec.py
---------------
Final implementation for ResNet18_mod adapted for Mel-spectrogram classification.

Key features:
- Accepts spectrograms: (Batch, 1, 128, Time)
- Uses ResNet18 backbone with modified stem (1-channel input)
- Supports dynamic Time dimension
- Exportable to TorchScript and ONNX
- Lightweight for CPU inference (<500ms target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetSpec(nn.Module):
    """
    ResNet18 backbone adapted for 1-channel spectrogram input.

    - Input:  (B, 1, 128, T)
    - Output: (B, num_classes)
    - ONNX + TorchScript compatible
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()

        # Load a clean ResNet18 backbone (no pretrained weights)
        self.backbone = resnet18(weights=None)

        # -------- MODIFY STEM FOR 1-CHANNEL INPUT --------
        # Replace conv1 to accept 1-channel instead of 3-channel RGB
        # Use same kernel/stride/padding as original for ResNet compatibility
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # -------- OPTIONAL: Retain MaxPool --------
        # With input H=128, conv1(s=2)->64, maxpool(s=2)->32.
        # Works cleanly through layers:
        #   32 -> 16 -> 8 -> 4 spatial height
        # Good for adaptive pooling at the end.
        # No modification required here.

        # -------- MODIFY FINAL FC LAYER --------
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spectrograms.

        Args:
            x: Tensor (B, 1, 128, T)
        Returns:
            logits: Tensor (B, num_classes)
        """
        return self.backbone(x)


# ------------------------------ TESTING SECTION ------------------------------ #

if __name__ == "__main__":
    print("Testing ResNetSpec model...")

    model = ResNetSpec(num_classes=5)
    model.eval()

    # Dummy input: (batch=1, channel=1, mel_bins=128, time_frames=256)
    dummy = torch.randn(1, 1, 128, 256)
    print("Input shape:", dummy.shape)

    # --- Test forward pass ---
    try:
        with torch.no_grad():
            out = model(dummy)
        print("Output shape:", out.shape)  # Expected: (1, 5)
    except Exception as e:
        print("Forward pass FAILED:", e)
        exit()

    # --- Test TorchScript ---
    try:
        scripted = torch.jit.script(model)
        test_out = scripted(dummy)
        print("TorchScript export: SUCCESS")
    except Exception as e:
        print("TorchScript export FAILED:", e)

    # --- Test ONNX export (dry) ---
    try:
        import io
        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            dummy,
            buffer,
            opset_version=12,
            input_names=["mel_spectrogram"],
            output_names=["logits"],
            dynamic_axes={
                "mel_spectrogram": {0: "batch", 3: "time"},
                "logits": {0: "batch"},
            },
        )
        print("ONNX export: SUCCESS")
    except Exception as e:
        print("ONNX export FAILED:", e)
