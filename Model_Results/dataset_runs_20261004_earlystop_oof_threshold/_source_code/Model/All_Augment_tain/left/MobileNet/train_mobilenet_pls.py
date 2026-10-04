import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import copy

class InvertedResidual1D(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual1D, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend([
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        layers.extend([
            # dw
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_1D(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(MobileNetV2_1D, self).__init__()
        
        # Initial conv: 1 -> 16
        self.features = [
            nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU6(inplace=True)
        ]
        
        # Inverted residual blocks
        # t: expand_ratio, c: output_channels, n: num_blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 2, 1],
            [6, 64, 1, 2],
        ]
        
        input_channel = 16
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual1D(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                
        # Last layer
        self.features.append(nn.Conv1d(input_channel, 128, 1, 1, 0, bias=False))
        self.features.append(nn.BatchNorm1d(128))
        self.features.append(nn.ReLU6(inplace=True))
        
        self.features = nn.Sequential(*self.features)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


def train_mobilenet_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, device='cpu'):
    # Reshape for 1D CNN: (batch, channels, sequence_length) -> (batch, 1, n_components)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = MobileNetV2_1D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        losses.append(epoch_loss / len(train_loader.dataset))
        
    # Evaluate on val if provided
    val_acc = 0.0
    if X_val is not None and y_val is not None:
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
            
            outputs = model(X_val_t)
            preds = (outputs > 0.5).float()
            val_acc = (preds == y_val_t).float().mean().item()
            
    return model, losses, val_acc


def run_pipeline():
    """Run shared 10-fold CV and save OOF, Test results and logs."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dataset_cross_validation import legacy_entrypoint
    legacy_entrypoint('left', 'MobileNet')

if __name__ == "__main__":
    run_pipeline()
