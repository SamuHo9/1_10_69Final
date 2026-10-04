"""
Hippocampal Shape Analysis Model Package for Desktop App.
Provides ResNet1D + PLS-DA models and prediction utilities for Left and Right hippocampus.
"""

from .predictor import HippocampalPredictor, ResNet1D

__all__ = ["HippocampalPredictor", "ResNet1D"]
