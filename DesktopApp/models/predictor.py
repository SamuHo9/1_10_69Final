import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress harmless scikit-learn unpickle version and feature name warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

# =============================================================================
# ResNet1D Neural Network Architecture (Matching Training Model)
# =============================================================================
class ResNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(ResNet1D, self).__init__()
        
        # Initial block: 3x3 conv, 32 -> batch norm, relu -> 2x2 max pool
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
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch, in_channels, sequence_length)
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
        
        # Output
        out_pool = self.global_avg_pool(out3)
        out_flat = out_pool.view(out_pool.size(0), -1)
        out_final = self.fc(out_flat)
        
        return torch.sigmoid(out_final)


# =============================================================================
# Helper: Parse SPHARM-PDM .coef file
# =============================================================================
def parse_coef_file(file_path: str, expected_coeffs: int = 169) -> np.ndarray:
    """
    Parses a SPHARM-PDM .coef file into a 1D numpy array of 507 features (169 * 3).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"SPHARM .coef file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    pattern = re.compile(r"\{([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+),\s*([-+]?[\d\.eE+-]+)\}")
    matches = pattern.findall(content)
    
    coeffs = []
    for m in matches:
        coeffs.append([float(x) for x in m])
        
    num_match = re.search(r"\{\s*(\d+)", content)
    if num_match:
        num_coeffs = int(num_match.group(1))
        coeffs = coeffs[:num_coeffs]
    elif len(coeffs) > expected_coeffs:
        coeffs = coeffs[:expected_coeffs]
        
    flat = np.array(coeffs).ravel()
    if len(flat) != expected_coeffs * 3:
        raise ValueError(
            f"Expected {expected_coeffs * 3} features from .coef, but parsed {len(flat)}"
        )
    return flat


# =============================================================================
# HippocampalPredictor: Desktop App Inference Engine
# =============================================================================
class HippocampalPredictor:
    """
    Inference helper for Epilepsy Detection from Hippocampal Shape Features.
    Supports Left and Right models with ResNet1D + PLS-DA pipeline.
    """
    _shared_models = {}
    _shared_scalers = {}
    _shared_pls_models = {}
    _shared_pipelines = {}

    def __init__(self, models_root: str = None, device: str = None):
        if models_root is None:
            models_root = os.path.dirname(os.path.abspath(__file__))
        self.models_root = models_root
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.models = HippocampalPredictor._shared_models
        self.scalers = HippocampalPredictor._shared_scalers
        self.pls_models = HippocampalPredictor._shared_pls_models
        self.pipelines = HippocampalPredictor._shared_pipelines

    def is_model_available(self, side: str) -> bool:
        side = side.lower()
        if side in self.models:
            return True
        side_dir = os.path.join(self.models_root, side)
        pth_path = os.path.join(side_dir, f"resnet_model_{side}.pth")
        pipeline_path = os.path.join(side_dir, f"pipeline_{side}.joblib")
        scaler_path = os.path.join(side_dir, f"scaler_{side}.joblib")
        pls_path = os.path.join(side_dir, f"pls_model_{side}.joblib")
        has_sklearn = (os.path.isfile(scaler_path) and os.path.isfile(pls_path)) or os.path.isfile(pipeline_path)
        return os.path.isfile(pth_path) and has_sklearn

    def load_model(self, side: str):
        side = side.lower()
        if side in self.models:
            return
            
        side_dir = os.path.join(self.models_root, side)
        pth_path = os.path.join(side_dir, f"resnet_model_{side}.pth")
        pipeline_path = os.path.join(side_dir, f"pipeline_{side}.joblib")
        scaler_path = os.path.join(side_dir, f"scaler_{side}.joblib")
        pls_path = os.path.join(side_dir, f"pls_model_{side}.joblib")

        if not os.path.exists(pth_path):
            raise FileNotFoundError(
                f"ResNet weights not found for side '{side}' at: {pth_path}"
            )

        # 1. Load scaler and PLS-DA safely (prefer standalone sklearn files to avoid torch CUDA deserialization)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if os.path.isfile(scaler_path) and os.path.isfile(pls_path):
                self.scalers[side] = joblib.load(scaler_path)
                self.pls_models[side] = joblib.load(pls_path)
            elif os.path.isfile(pipeline_path):
                try:
                    pipeline_info = joblib.load(pipeline_path)
                    self.pipelines[side] = pipeline_info
                    self.scalers[side] = pipeline_info.get('scaler')
                    self.pls_models[side] = pipeline_info.get('pls')
                except Exception as e:
                    raise RuntimeError(f"Failed to load pipeline for {side}: {e}")
            else:
                raise FileNotFoundError(f"Scaler/PLS files not found for side '{side}' in {side_dir}")

        # 2. Load PyTorch model with safe map_location
        model = ResNet1D().to(self.device)
        state_dict = torch.load(pth_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        self.models[side] = model

    def get_template_mesh_path(self, side: str = "left") -> str:
        """Find the canonical SPHARM mean template mesh."""
        side = side.lower()
        repo_root = os.path.abspath(os.path.join(self.models_root, "..", ".."))
        candidates = [
            os.path.join(repo_root, "Templates", "SPHARM", f"template_spharm_{side}.vtk"),
            os.path.join(repo_root, "Templates", "SPHARM", f"template_spharm_{side}.ply"),
            os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "All_Augment_tain", side, "PLS1", "PLS1_Mean.vtk"),
            os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Ds005602", side, "PLS1", "PLS1_Mean.vtk")
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def get_gradcam_mesh_path(self, side: str = "left", component: str = "PLS1", milestone: str = "Mean", cohort: str = "All_Augment_tain") -> str:
        """Find the pre-computed Grad-CAM / PLS-DA deformation mesh."""
        side = side.lower()
        repo_root = os.path.abspath(os.path.join(self.models_root, "..", ".."))
        cohort_dir = os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", cohort, side, component)
        target_file = os.path.join(cohort_dir, f"{component}_{milestone}.vtk")
        if os.path.isfile(target_file):
            return target_file
        
        # Fallback to any matching component in All_Augment_tain or Ds005602
        fallback_dirs = [
            os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "All_Augment_tain", side, "PLS1"),
            os.path.join(repo_root, "Model", "Output_GradCAM_PLSDA", "Ds005602", side, "PLS1")
        ]
        for fdir in fallback_dirs:
            mean_f = os.path.join(fdir, "PLS1_Mean.vtk")
            if os.path.isfile(mean_f):
                return mean_f

        return self.get_template_mesh_path(side)

    def _prepare_input(self, input_data, side: str) -> pd.DataFrame:
        if isinstance(input_data, str):
            # Input is file path to .coef
            arr = parse_coef_file(input_data).reshape(1, -1)
        elif isinstance(input_data, pd.DataFrame):
            # Filter meta columns if present
            meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType']
            drop_cols = [c for c in meta_cols if c in input_data.columns]
            arr = input_data.drop(columns=drop_cols).values
        elif isinstance(input_data, pd.Series):
            meta_cols = ['Subject', 'Group', 'Class', 'BinaryClass', 'DataType']
            series = input_data.drop(labels=[c for c in meta_cols if c in input_data.index])
            arr = series.values.reshape(1, -1)
        elif isinstance(input_data, (list, tuple)):
            arr = np.array(input_data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                arr = input_data.reshape(1, -1)
            else:
                arr = input_data
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

        # Match scaler feature names if available to eliminate feature name mismatch warning
        scaler = self.scalers.get(side)
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_names = scaler.feature_names_in_
            if arr.shape[1] == len(expected_names):
                return pd.DataFrame(arr, columns=expected_names)

        feat_names = self.pipelines.get(side, {}).get('feature_names')
        if feat_names and arr.shape[1] == len(feat_names):
            return pd.DataFrame(arr, columns=feat_names)
        return pd.DataFrame(arr)

    def predict(self, input_data, side: str = 'left'):
        """
        Run inference for single or multiple samples.
        Returns dictionary (single sample) or list of dictionaries (multiple samples).
        """
        side = side.lower()
        if side not in self.models:
            self.load_model(side)

        X_df = self._prepare_input(input_data, side)
        
        # 1. StandardScaler Transform & 2. PLS-DA Transform
        scaler = self.scalers[side]
        pls = self.pls_models[side]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_sc = scaler.transform(X_df)
            X_pls = pls.transform(X_sc)

        # 3. ResNet1D Inference
        model = self.models[side]
        with torch.no_grad():
            X_t = torch.tensor(X_pls, dtype=torch.float32).unsqueeze(1).to(self.device)
            probs = model(X_t).cpu().numpy().flatten()

        results = []
        for prob in probs:
            prob_val = float(prob)
            pred_class = 1 if prob_val > 0.5 else 0
            label = "Epilepsy" if pred_class == 1 else "Healthy"
            
            # Risk stratification
            if prob_val < 0.35:
                risk_level = "Low Risk (Healthy-aligned)"
            elif prob_val > 0.65:
                risk_level = "High Risk (Epilepsy-aligned)"
            else:
                risk_level = "Borderline / Moderate Risk"

            confidence = prob_val if pred_class == 1 else (1.0 - prob_val)

            results.append({
                'side': side,
                'probability': prob_val,
                'prediction': pred_class,
                'label': label,
                'risk_level': risk_level,
                'confidence': float(confidence),
                'n_components': self.pls_models[side].n_components
            })

        if len(results) == 1:
            return results[0]
        return results

    def predict_both(self, left_input=None, right_input=None) -> dict:
        """
        Run combined prediction for both Left and Right hippocampus if available.
        """
        out = {'left': None, 'right': None, 'combined_summary': None}
        
        if left_input is not None:
            out['left'] = self.predict(left_input, side='left')
        if right_input is not None:
            out['right'] = self.predict(right_input, side='right')

        if out['left'] and out['right']:
            avg_prob = (out['left']['probability'] + out['right']['probability']) / 2.0
            pred_class = 1 if avg_prob > 0.5 else 0
            label = "Epilepsy" if pred_class == 1 else "Healthy"
            out['combined_summary'] = {
                'average_probability': avg_prob,
                'prediction': pred_class,
                'label': label,
                'highest_risk_side': 'left' if out['left']['probability'] >= out['right']['probability'] else 'right'
            }
        return out
