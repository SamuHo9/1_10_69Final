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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score, f1_score, roc_auc_score

import sys

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

def extract_features_labels(df):
    meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType', 'Group_Name', 'Group_Label', 'Unnamed: 0']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    features_3d_raw = df[feature_cols].values
    
    num_points = features_3d_raw.shape[1] // 3
    features_3d = features_3d_raw.reshape(-1, num_points, 3)
    
    mean_val = np.mean(features_3d, axis=1, keepdims=True)
    std_val = np.std(features_3d, axis=1, keepdims=True) + 1e-8
    features_3d = (features_3d - mean_val) / std_val
    
    X = np.transpose(features_3d, (0, 2, 1))
    
    if 'BinaryClass' in df.columns:
        y = df['BinaryClass'].values
    elif 'Group_Label' in df.columns:
        y = df['Group_Label'].values
    elif 'Class' in df.columns:
        y = df['Class'].values
    else:
        raise ValueError("No binary target label column found")
        
    return X, y

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 512)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = (x + iden).view(-1, self.k, self.k)
        return x

class PointNetDual(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetDual, self).__init__()
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        trans3 = self.input_transform(x)
        x = torch.bmm(trans3, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x_max = torch.max(x, 2)[0]
        x_mean = torch.mean(x, 2)
        global_feat = torch.cat([x_max, x_mean], dim=1)
        
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

def run_pipeline():
    """Run shared 10-fold CV and save OOF, Test results and logs."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dataset_cross_validation import legacy_entrypoint
    legacy_entrypoint('left', 'PointNet')

if __name__ == "__main__":
    run_pipeline()
