import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class PLSWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components)
        
    def fit(self, X, y):
        self.pls.fit(X, y)
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        return self.pls.transform(X)
        
    def predict(self, X):
        return self.pls.predict(X)


def run_pipeline():
    """Run shared 10-fold CV and save OOF, Test results and logs."""
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from dataset_cross_validation import legacy_entrypoint
    legacy_entrypoint('right', 'MLP')

if __name__ == "__main__":
    run_pipeline()
