"""
MULTI-TEMPORAL LUNAR SURFACE CHANGE DETECTION
Identifies physical surface changes (new impact craters, boulder tracks, landslides)
between registered moving optical passes and historical reference basemaps.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2


@dataclass
class LunarChangeFeature:
    """Represents a detected physical change anomaly on the lunar surface."""
    feature_id: int
    centroid: Tuple[float, float]  # (x, y) coordinates
    area_pixels: float
    equivalent_radius_m: float     # Estimated physical size in meters
    change_confidence: float       # Confidence score (0.0 to 1.0)
    category: str                  # 'New Impact Crater', 'Boulder Shift', 'Albedo Disruption'
    bbox: Tuple[int, int, int, int]# (x, y, w, h)


@dataclass
class ChangeDetectionResult:
    """Container for multi-temporal surface change detection analysis."""
    change_mask: np.ndarray        # Binary mask of detected physical change regions
    anomaly_heatmap: np.ndarray    # Colorized JET anomaly heatmap
    detected_features: List[LunarChangeFeature]
    total_change_area_sq_m: float
    change_percentage: float       # Percentage of overlap area exhibiting physical change
    overlay_image: np.ndarray      # RGB image with highlighted change bounding boxes


def detect_lunar_surface_changes(
    img_registered: np.ndarray,
    img_reference: np.ndarray,
    overlap_mask: np.ndarray,
    pixel_resolution_m: float = 1.25,
    sensitivity: float = 0.75,
    min_feature_size_px: int = 8
) -> ChangeDetectionResult:
    """
    Computes structural anomaly maps and isolates genuine lunar surface disturbances
    from benign solar illumination and phase angle variations.

    Parameters:
        img_registered: Warped moving lunar image (H, W) uint8.
        img_reference: Fixed reference lunar basemap (H, W) uint8.
        overlap_mask: Boolean mask of valid registered pixels.
        pixel_resolution_m: Ground sample distance in meters/pixel.
        sensitivity: Detection sensitivity threshold (0.5 to 1.0).
        min_feature_size_px: Minimum connected component size to filter noise.

    Returns:
        ChangeDetectionResult with segmented anomaly masks and candidate feature catalog.
    """
    H, W = img_registered.shape[:2]

    # 1. Local Contrast Normalization (suppresses uniform solar elevation bias)
    ksize = 25
    mean_reg = cv2.blur(img_registered.astype(np.float32), (ksize, ksize))
    mean_ref = cv2.blur(img_reference.astype(np.float32), (ksize, ksize))
    
    std_reg = np.sqrt(np.maximum(cv2.blur(img_registered.astype(np.float32)**2, (ksize, ksize)) - mean_reg**2, 1.0))
    std_ref = np.sqrt(np.maximum(cv2.blur(img_reference.astype(np.float32)**2, (ksize, ksize)) - mean_ref**2, 1.0))

    norm_reg = (img_registered.astype(np.float32) - mean_reg) / std_reg
    norm_ref = (img_reference.astype(np.float32) - mean_ref) / std_ref

    # 2. Structural Anomaly Difference
    diff_norm = np.abs(norm_reg - norm_ref)
    diff_norm[~overlap_mask] = 0.0

    # 3. Dynamic Thresholding based on Sensitivity
    thresh_val = float(np.percentile(diff_norm[overlap_mask], 97.5 - (sensitivity * 5.0)))
    binary_change = (diff_norm > thresh_val).astype(np.uint8) * 255

    # 4. Morphological Filtering to remove single-pixel cosmic ray hits
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_change, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Connected Component Feature Extraction
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
    
    detected_features: List[LunarChangeFeature] = []
    overlay = cv2.cvtColor(img_reference, cv2.COLOR_GRAY2RGB) if img_reference.ndim == 2 else img_reference.copy()

    total_change_pixels = 0
    feature_id_counter = 1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_feature_size_px:
            total_change_pixels += area
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[label]

            # Physical diameter in meters
            radius_m = np.sqrt(area / np.pi) * pixel_resolution_m

            # Classify anomaly category based on aspect ratio and circularity
            aspect = float(w) / max(float(h), 1e-5)
            if 0.7 <= aspect <= 1.4:
                cat = "Candidate Impact Crater / Bowl"
                box_color = (0, 255, 255)  # Cyan
            elif aspect > 2.0 or aspect < 0.5:
                cat = "Boulder Track / Linear Feature"
                box_color = (255, 100, 0)  # Orange
            else:
                cat = "Regolith Reflectance Anomaly"
                box_color = (0, 255, 100)  # Green

            conf = min(0.95, float(area / 100.0 * 0.4 + 0.5))

            feat = LunarChangeFeature(
                feature_id=feature_id_counter,
                centroid=(float(cx), float(cy)),
                area_pixels=float(area),
                equivalent_radius_m=float(radius_m),
                change_confidence=conf,
                category=cat,
                bbox=(int(x), int(y), int(w), int(h))
            )
            detected_features.append(feat)

            # Draw on overlay
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                overlay,
                f"#{feat.feature_id} {radius_m:.1f}m",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA
            )
            feature_id_counter += 1

    # Anomaly colorized heatmap
    diff_scaled = np.clip(diff_norm * 45.0, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_scaled, cv2.COLORMAP_INFERNO)
    diff_heatmap[~overlap_mask] = 0

    overlap_area = float(np.sum(overlap_mask))
    change_pct = (total_change_pixels / overlap_area * 100.0) if overlap_area > 0 else 0.0
    total_sq_m = total_change_pixels * (pixel_resolution_m ** 2)

    return ChangeDetectionResult(
        change_mask=cleaned_mask,
        anomaly_heatmap=diff_heatmap,
        detected_features=detected_features,
        total_change_area_sq_m=total_sq_m,
        change_percentage=change_pct,
        overlay_image=overlay
    )
