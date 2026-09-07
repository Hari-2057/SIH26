"""
Lunar Image Registration Pipeline
Specialized for Chandrayaan-2 Optical Images & Lunar Reference Maps
Sub-pixel precision (<0.5 px RMSE) alignment under extreme illumination and non-rigid relief.
"""

__version__ = "1.0.0"

from .io import load_geospatial_image, load_lunar_image, LunarImageData
from .preprocessing import preprocess_lunar_image
from .features import extract_features, LunarFeatures
from .matching import match_and_filter_keypoints, match_and_filter, MatchResult
from .grid_uniformity import enforce_grid_uniformity_and_refine, SubPixelRefinementResult
from .warping_evaluation import register_and_evaluate, RegistrationMetrics, RegistrationResult
from .pipeline import LunarRegistrationPipeline
from .synthetic_data import generate_lunar_crater_scene

__all__ = [
    "load_geospatial_image",
    "load_lunar_image",
    "LunarImageData",
    "preprocess_lunar_image",
    "extract_features",
    "LunarFeatures",
    "match_and_filter_keypoints",
    "match_and_filter",
    "MatchResult",
    "enforce_grid_uniformity_and_refine",
    "SubPixelRefinementResult",
    "register_and_evaluate",
    "RegistrationMetrics",
    "RegistrationResult",
    "LunarRegistrationPipeline",
    "generate_lunar_crater_scene",
]
