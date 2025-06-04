import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary

class CNNEncoder(nn.Module):
    def __init__(self, input_features=5, output_dim=64):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        out = self.encoder(x).squeeze(-1)  
        return out


model = CNNEncoder()
summary(model, input_size=(8, 15, 5))