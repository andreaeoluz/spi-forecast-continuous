"""Inference package - Continuous SPI inference and prediction."""

from .predictor import InferencePredictor
from .run import main as inference_main

try:
    from .analyze import run_analysis
except ImportError:
    def run_analysis(*args, **kwargs):
        print("⚠️  Analysis module not available")
        return None, None

__all__ = [
    "InferencePredictor",
    "inference_main",
    "run_analysis",
]
