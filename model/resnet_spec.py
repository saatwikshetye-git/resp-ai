"""
resnet_spec.py
---------------
Skeleton for custom ResNet model that accepts spectrogram input.
Final implementation in Step 4.
"""

import torch.nn as nn

class ResNet18_mod(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()
        # Placeholder layers (real architecture comes later)
        self.dummy = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.dummy(x)
