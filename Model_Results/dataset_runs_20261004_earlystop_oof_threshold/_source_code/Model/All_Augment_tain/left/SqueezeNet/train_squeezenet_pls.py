import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class Fire1D(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire1D, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv1d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv1d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv1d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(SqueezeNet1D, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Fire1D(16, 8, 16, 16),
            Fire1D(32, 16, 32, 32),
            Fire1D(64, 32, 64, 64),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv1d(128, self.num_classes, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

def train_with_checkpoint(model, X_tr, y_tr, X_val, y_val, epochs=80, batch_size=32, lr=0.001, device='cpu', pos_weight=1.0, patience=25):
    pos_w_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 1
    no_improve = 0
    actual_epochs = 1
    
    for epoch in range(1, epochs + 1):
        actual_epochs = epoch
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
                total += bx.size(0)
        val_loss /= total
        
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
            
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        preds = (torch.sigmoid(model(X_val_t)).cpu().numpy().flatten() > 0.5).astype(int)
        val_acc = accuracy_score(y_val, preds)
        
    return model, val_acc, best_epoch, actual_epochs

def run_pipeline():
    """Run shared 10-fold CV and save OOF, Test results and logs."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dataset_cross_validation import legacy_entrypoint
    legacy_entrypoint('left', 'SqueezeNet')

if __name__ == '__main__':
    run_pipeline()
