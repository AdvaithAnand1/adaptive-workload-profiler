# model.py
"""
Simple feed-forward network to classify system state from telemetry only.
"""

import torch
import torch.nn as nn

MODEL_NAME = "SystemStateNet"


class SystemStateNet(nn.Module):
    """
    MLP:
      input_dim  ~ len(FEATURE_NAMES)
      hidden     64 -> 64
      outputs    num_classes (Idle, Gaming, Rendering, etc.)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
