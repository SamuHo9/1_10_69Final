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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import copy

class ResNetAutoencoder1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, seq_length=10):
        super(ResNetAutoencoder1D, self).__init__()
        
        self.target_seq_length = seq_length
        
        # Initial block: 3x3 conv, 32 -> batch norm, relu -> 2x2 max pool, /2
        # Adapted to 1D
        self.init_conv = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm1d(32)
        self.init_pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Block 1: 32 -> 32
        self.b1_conv1 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.b1_bn1 = nn.BatchNorm1d(32)
        self.b1_conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.b1_bn2 = nn.BatchNorm1d(32)
        
        # Block 2: 32 -> 64, stride 2
        self.b2_conv1 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.b2_bn1 = nn.BatchNorm1d(64)
        self.b2_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.b2_bn2 = nn.BatchNorm1d(64)
        self.b2_skip_conv = nn.Conv1d(32, 64, kernel_size=1, stride=2)
        self.b2_skip_bn = nn.BatchNorm1d(64)
        
        # Block 3: 64 -> 128, stride 2
        self.b3_conv1 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.b3_bn1 = nn.BatchNorm1d(128)
        self.b3_conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.b3_bn2 = nn.BatchNorm1d(128)
        self.b3_skip_conv = nn.Conv1d(64, 128, kernel_size=1, stride=2)
        self.b3_skip_bn = nn.BatchNorm1d(128)
        
        # Output
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # --- CLASSIFIER HEAD ---
        self.fc_class = nn.Linear(128, num_classes)
        
        # --- DECODER HEAD ---
        self.dec_fc = nn.Linear(128, 128 * 2) 
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) 
        self.dec_out = nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # x is (batch, in_channels, sequence_length)
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = F.relu(x)
        
        if x.shape[2] < 2:
            x = F.pad(x, (0, 2 - x.shape[2]))
        x = self.init_pool(x)
        
        # Block 1
        identity = x
        out = self.b1_conv1(x)
        out = self.b1_bn1(out)
        out = F.relu(out)
        out = self.b1_conv2(out)
        out = self.b1_bn2(out)
        out += identity
        out = F.relu(out)
        
        # Block 2
        identity = self.b2_skip_conv(out)
        identity = self.b2_skip_bn(identity)
        
        out2 = self.b2_conv1(out)
        out2 = self.b2_bn1(out2)
        out2 = F.relu(out2)
        out2 = self.b2_conv2(out2)
        out2 = self.b2_bn2(out2)
        
        if out2.shape[2] != identity.shape[2]:
            diff = identity.shape[2] - out2.shape[2]
            out2 = F.pad(out2, (0, diff))
            
        out2 += identity
        out2 = F.relu(out2)
        
        # Block 3
        identity = self.b3_skip_conv(out2)
        identity = self.b3_skip_bn(identity)
        
        out3 = self.b3_conv1(out2)
        out3 = self.b3_bn1(out3)
        out3 = F.relu(out3)
        out3 = self.b3_conv2(out3)
        out3 = self.b3_bn2(out3)
        
        if out3.shape[2] != identity.shape[2]:
            diff = identity.shape[2] - out3.shape[2]
            out3 = F.pad(out3, (0, diff))
            
        out3 += identity
        out3 = F.relu(out3)
        
        # Latent Space
        out_pool = self.global_avg_pool(out3)
        latent = out_pool.view(out_pool.size(0), -1)
        
        # CLASSIFIER HEAD
        class_out = torch.sigmoid(self.fc_class(latent))
        
        # DECODER HEAD
        dec = self.dec_fc(latent)
        dec = dec.view(dec.size(0), 128, 2)
        dec = F.relu(self.dec_conv1(dec))
        dec = F.relu(self.dec_conv2(dec))
        reconstruction = self.dec_out(dec)
        
        if reconstruction.shape[2] > self.target_seq_length:
            reconstruction = reconstruction[:, :, :self.target_seq_length]
        elif reconstruction.shape[2] < self.target_seq_length:
            reconstruction = F.pad(reconstruction, (0, self.target_seq_length - reconstruction.shape[2]))
            
        return class_out, reconstruction


def train_resnet_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, device='cpu'):
    # Reshape for 1D CNN: (batch, channels, sequence_length) -> (batch, 1, n_components)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = ResNetAutoencoder1D(seq_length=X_train_t.shape[2]).to(device)
    criterion_cls = nn.BCELoss()
    criterion_recon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            class_out, recon_out = model(batch_x)
            
            loss_cls = criterion_cls(class_out, batch_y)
            loss_recon = criterion_recon(recon_out, batch_x)
            
            # Combine losses (adjust weight as needed)
            loss = loss_cls + 0.5 * loss_recon
            
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
            
            outputs, _ = model(X_val_t)
            preds = (outputs > 0.5).float()
            val_acc = (preds == y_val_t).float().mean().item()
            
    return model, losses, val_acc


def run_pipeline():
    """Run shared 10-fold CV and save OOF, Test results and logs."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dataset_cross_validation import legacy_entrypoint
    legacy_entrypoint('right', 'ResNetAE')

if __name__ == "__main__":
    run_pipeline()
