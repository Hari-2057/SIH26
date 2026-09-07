"""
MODULE 2: Preprocessing & Contrast Normalization (OpenCV)
Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) for crater shadow enhancement
and Gaussian/Bilateral denoising for sensor and cosmic ray noise suppression.
"""

from typing import Tuple, Union, Optional
import numpy as np
import cv2
from .io import LunarImageData


def preprocess_lunar_image(
    image_input: Union[LunarImageData, np.ndarray],
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    gaussian_ksize: int = 3,
    gaussian_sigma: float = 0.8,
    use_bilateral: bool = False,
    percentile_stretch: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies adaptive contrast normalization and edge-preserving denoising
    optimized for lunar topography, deep crater shadows, and bright rim highlights.

    Parameters:
        image_input: LunarImageData instance or 2D/3D NumPy array (uint8 or float32).
        clip_limit: Threshold for contrast limiting in CLAHE (default: 3.0).
        tile_grid_size: Grid size for local histogram equalization (default: (8, 8)).
        gaussian_ksize: Kernel size for Gaussian blur (must be odd, default: 3).
        gaussian_sigma: Standard deviation for Gaussian kernel (default: 0.8).
        use_bilateral: If True, applies Bilateral Filter instead of Gaussian to preserve sharp crater edges.
        percentile_stretch: If True, performs 1%-99% dynamic range expansion prior to CLAHE.

    Returns:
        (preprocessed_uint8, preprocessed_float32):
            - preprocessed_uint8: uint8 array [0, 255] ready for OpenCV feature detectors.
            - preprocessed_float32: float32 array [0.0, 1.0] for radiometric metrics.
    """
    # Extract 2D single-channel uint8 array
    if isinstance(image_input, LunarImageData):
        gray_u8 = image_input.to_grayscale_uint8()
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] in [3, 4]:
            gray_u8 = cv2.cvtColor(image_input[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif image_input.ndim == 2:
            if image_input.dtype == np.uint8:
                gray_u8 = image_input.copy()
            elif np.issubdtype(image_input.dtype, np.floating):
                scaled = np.clip(image_input, 0.0, 1.0) * 255.0
                gray_u8 = scaled.astype(np.uint8)
            else:
                norm = (image_input - image_input.min()) / max(float(image_input.max() - image_input.min()), 1e-6)
                gray_u8 = (norm * 255.0).astype(np.uint8)
        else:
            raise ValueError(f"Invalid array shape for preprocessing: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    # Step 1: Dynamic Range Percentile Stretch (1% to 99%)
    if percentile_stretch:
        p1, p99 = np.percentile(gray_u8, (1.0, 99.0))
        if p99 > p1:
            stretched = np.clip((gray_u8.astype(np.float32) - p1) * 255.0 / (p99 - p1), 0, 255)
            gray_u8 = stretched.astype(np.uint8)

    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    clahe_enhanced = clahe.apply(gray_u8)

    # Step 3: Denoising (Gaussian Filter or Bilateral Filter)
    if use_bilateral:
        # Preserves crisp crater rims while smoothing low-texture regolith
        denoised = cv2.bilateralFilter(clahe_enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    else:
        # Mild Gaussian blur (3x3) to remove cosmic ray hits & sensor noise
        ksize = gaussian_ksize if (gaussian_ksize % 2 == 1 and gaussian_ksize > 0) else 3
        denoised = cv2.GaussianBlur(clahe_enhanced, (ksize, ksize), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)

    preprocessed_uint8 = denoised
    preprocessed_float32 = (denoised.astype(np.float32) / 255.0)

    return preprocessed_uint8, preprocessed_float32
