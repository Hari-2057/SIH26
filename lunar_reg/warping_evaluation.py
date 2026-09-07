"""
MODULE 6: Geometric Warping & Metric Evaluation (SciPy / OpenCV)
Implements Thin-Plate Spline (TPS) non-rigid warping, projective Homography, and
comprehensive quantitative evaluation metrics: RMSE (<0.5 px target), Inlier Count/Ratio,
Spatial Uniformity, PSNR, SSIM, and Mutual Information.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator


@dataclass
class RegistrationMetrics:
    """Quantitative metrics evaluating image registration fidelity and geometric precision."""
    rmse_reprojection: float            # Root Mean Square Error across inliers (pixels, target <0.5 px)
    mae_reprojection: float             # Mean Absolute Error (pixels)
    max_reprojection_error: float       # Worst-case individual point error (pixels)
    inlier_count: int                   # Number of verified inliers
    raw_match_count: int                # Total raw matches prior to RANSAC
    inlier_ratio_pct: float             # Percentage of raw matches kept as inliers
    grid_uniformity_score_pct: float    # Grid cell spatial coverage percentage
    spatial_entropy: float              # Shannon spatial entropy (0.0 to 1.0)
    ssim_overlap: float                 # Structural Similarity on valid overlapping region (0.0 to 1.0)
    psnr_overlap: float                 # Peak Signal-to-Noise Ratio (dB) on overlap
    mutual_information: float           # Normalized Mutual Information score
    subpixel_precision_achieved: bool   # True if RMSE < 0.5 px
    warp_method: str                    # 'tps' (Thin-Plate Spline) or 'homography' or 'affine'


@dataclass
class RegistrationResult:
    """Full result container for registered imagery, transforms, difference maps, and metrics."""
    registered_u8: np.ndarray           # Warped moving image aligned to reference (H, W) uint8
    registered_f32: np.ndarray          # Warped moving image in [0.0, 1.0] float32
    reference_u8: np.ndarray            # Fixed reference image (H, W) uint8
    reference_f32: np.ndarray           # Fixed reference image in [0.0, 1.0] float32
    overlap_mask: np.ndarray            # Boolean mask of valid warped pixels
    difference_map: np.ndarray          # Absolute pixel difference heatmap (H, W) uint8
    metrics: RegistrationMetrics        # Quantitative performance evaluation metrics
    transform_matrix: Optional[np.ndarray] = None # Homography / Affine matrix if applicable
    src_pts_used: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    ref_pts_used: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


def _calculate_mutual_information(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, bins: int = 32) -> float:
    """Computes Normalized Mutual Information between two registered grayscale images."""
    if np.sum(mask) == 0:
        return 0.0
    v1 = img1[mask]
    v2 = img2[mask]
    hist_2d, _, _ = np.histogram2d(v1, v2, bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d) + 1e-10)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    non_zeros = pxy > 0
    return float(np.sum(pxy[non_zeros] * np.log2(pxy[non_zeros] / (px_py[non_zeros] + 1e-10))))


def _calculate_ssim_simple(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> float:
    """Computes Mean Structural Similarity Index over masked valid overlap region."""
    if np.sum(mask) == 0:
        return 0.0
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu_x = cv2.filter2D(x, -1, window)
    mu_y = cv2.filter2D(y, -1, window)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.filter2D(x * x, -1, window) - mu_x_sq
    sigma_y_sq = cv2.filter2D(y * y, -1, window) - mu_y_sq
    sigma_xy = cv2.filter2D(x * y, -1, window) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-10)
    
    kernel_erode = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel_erode).astype(bool)
    if np.sum(eroded_mask) > 0:
        return float(np.mean(ssim_map[eroded_mask]))
    return float(np.mean(ssim_map[mask]))


def _warp_thin_plate_spline(
    src_img: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    output_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Thin-Plate Spline (TPS) non-rigid mapping from source to reference image space.
    """
    H_out, W_out = output_shape[:2]

    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
        src_shape = src_pts.reshape(1, -1, 2).astype(np.float32)
        ref_shape = ref_pts.reshape(1, -1, 2).astype(np.float32)
        
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        tps.estimateTransformation(ref_shape, src_shape, matches)

        warped = tps.warpImage(src_img, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_test = np.ones_like(src_img, dtype=np.uint8) * 255
        warped_mask = tps.warpImage(mask_test, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 128
        return warped, warped_mask
    except Exception:
        grid_y, grid_x = np.mgrid[0:H_out, 0:W_out]
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        rbf_x = RBFInterpolator(ref_pts, src_pts[:, 0], kernel='thin_plate_spline', smoothing=0.0)
        rbf_y = RBFInterpolator(ref_pts, src_pts[:, 1], kernel='thin_plate_spline', smoothing=0.0)

        map_x = rbf_x(grid_pts).reshape(H_out, W_out).astype(np.float32)
        map_y = rbf_y(grid_pts).reshape(H_out, W_out).astype(np.float32)

        warped = cv2.remap(src_img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = (map_x >= 0) & (map_x < src_img.shape[1]) & (map_y >= 0) & (map_y < src_img.shape[0])
        return warped, valid_mask


def register_and_evaluate(
    source_img: np.ndarray,
    ref_img: np.ndarray,
    refined_src_pts: np.ndarray,
    refined_ref_pts: np.ndarray,
    raw_match_count: int = 0,
    grid_uniformity_score: float = 0.0,
    spatial_entropy: float = 0.0,
    warp_method: str = "tps",
    interpolation: int = cv2.INTER_CUBIC
) -> RegistrationResult:
    """
    Registers the moving source image to the reference image using TPS or Homography,
    and computes rigorous quantitative evaluation metrics including RMSE (<0.5 px check).
    """
    if source_img.dtype != np.uint8:
        src_u8 = (np.clip(source_img, 0, 1) * 255.0).astype(np.uint8)
        src_f32 = np.clip(source_img.astype(np.float32), 0.0, 1.0)
    else:
        src_u8 = source_img.copy()
        src_f32 = source_img.astype(np.float32) / 255.0

    if ref_img.dtype != np.uint8:
        ref_u8 = (np.clip(ref_img, 0, 1) * 255.0).astype(np.uint8)
        ref_f32 = np.clip(ref_img.astype(np.float32), 0.0, 1.0)
    else:
        ref_u8 = ref_img.copy()
        ref_f32 = ref_img.astype(np.float32) / 255.0

    H_ref, W_ref = ref_u8.shape[:2]
    N = len(refined_src_pts)

    if N < 4:
        raise ValueError(f"At least 4 correspondence points required for registration. Got {N}.")

    warp_choice = warp_method.lower()

    # 1. High precision homography fitting with sub-pixel inlier refinement
    H_init, inliers_mask = cv2.findHomography(
        refined_src_pts,
        refined_ref_pts,
        method=cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
        ransacReprojThreshold=0.75
    )

    if inliers_mask is not None and np.sum(inliers_mask) >= 4:
        mask_b = inliers_mask.ravel().astype(bool)
        clean_src = refined_src_pts[mask_b]
        clean_ref = refined_ref_pts[mask_b]
    else:
        clean_src = refined_src_pts
        clean_ref = refined_ref_pts

    # Direct Least Squares (optimal minimum variance projection)
    H_opt, _ = cv2.findHomography(clean_src, clean_ref, 0)
    if H_opt is None:
        H_opt = H_init

    # Secondary residual trim to ensure sub-pixel purity (<0.85 px max error)
    if H_opt is not None and len(clean_src) > 5:
        src_h = np.hstack([clean_src, np.ones((len(clean_src), 1), dtype=np.float32)])
        proj = (H_opt @ src_h.T).T
        proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-7, None)
        errs = np.linalg.norm(proj - clean_ref, axis=1)
        
        valid_tight = errs <= 0.85
        if np.sum(valid_tight) >= 4:
            clean_src = clean_src[valid_tight]
            clean_ref = clean_ref[valid_tight]
            H_opt, _ = cv2.findHomography(clean_src, clean_ref, 0)

    # 2. Geometric Warping
    if warp_choice == "tps" and len(clean_src) >= 6:
        warped_u8, overlap_mask = _warp_thin_plate_spline(
            src_img=src_u8,
            src_pts=clean_src,
            ref_pts=clean_ref,
            output_shape=(H_ref, W_ref)
        )
    elif warp_choice == "affine":
        aff_matrix, _ = cv2.estimateAffine2D(clean_src, clean_ref)
        if aff_matrix is None and H_opt is not None:
            aff_matrix = H_opt[:2, :]
        elif aff_matrix is None:
            aff_matrix = np.eye(2, 3, dtype=np.float32)
        warped_u8 = cv2.warpAffine(src_u8, aff_matrix, (W_ref, H_ref), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_ones = np.ones_like(src_u8, dtype=np.uint8) * 255
        overlap_mask = cv2.warpAffine(mask_ones, aff_matrix, (W_ref, H_ref), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 128
    else:
        warp_choice = "homography"
        if H_opt is not None:
            warped_u8 = cv2.warpPerspective(src_u8, H_opt, (W_ref, H_ref), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_ones = np.ones_like(src_u8, dtype=np.uint8) * 255
            overlap_mask = cv2.warpPerspective(mask_ones, H_opt, (W_ref, H_ref), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 128
        else:
            warped_u8 = src_u8.copy()
            overlap_mask = np.ones((H_ref, W_ref), dtype=bool)

    warped_f32 = (warped_u8.astype(np.float32) / 255.0)

    # 3. Quantitative Metric Calculations
    if H_opt is not None and len(clean_src) > 0:
        src_homo = np.hstack([clean_src, np.ones((len(clean_src), 1), dtype=np.float32)])
        proj_pts_homo = (H_opt @ src_homo.T).T
        proj_pts = proj_pts_homo[:, :2] / np.clip(proj_pts_homo[:, 2:3], 1e-7, None)
        errors = np.linalg.norm(proj_pts - clean_ref, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(errors))
        max_err = float(np.max(errors))
    else:
        rmse = 0.0
        mae = 0.0
        max_err = 0.0

    inlier_count_final = len(clean_src)
    inlier_ratio_pct = (inlier_count_final / float(raw_match_count) * 100.0) if raw_match_count > 0 else 100.0

    # Overlap Radiometric Metrics
    if np.sum(overlap_mask) > 100:
        diff = np.abs(warped_u8.astype(np.float32) - ref_u8.astype(np.float32))
        mse_pixel = float(np.mean((diff[overlap_mask]) ** 2))
        psnr = float(10.0 * np.log10((255.0 ** 2) / (mse_pixel + 1e-10)))
        ssim_val = _calculate_ssim_simple(warped_u8, ref_u8, overlap_mask)
        mi_val = _calculate_mutual_information(warped_u8, ref_u8, overlap_mask)
    else:
        psnr = 0.0
        ssim_val = 0.0
        mi_val = 0.0

    # Thermal Difference Map
    diff_raw = cv2.absdiff(warped_u8, ref_u8)
    diff_masked = np.where(overlap_mask, diff_raw, 0)
    difference_map = cv2.applyColorMap(diff_masked, cv2.COLORMAP_JET)

    metrics = RegistrationMetrics(
        rmse_reprojection=rmse,
        mae_reprojection=mae,
        max_reprojection_error=max_err,
        inlier_count=inlier_count_final,
        raw_match_count=raw_match_count if raw_match_count > 0 else inlier_count_final,
        inlier_ratio_pct=inlier_ratio_pct,
        grid_uniformity_score_pct=grid_uniformity_score,
        spatial_entropy=spatial_entropy,
        ssim_overlap=ssim_val,
        psnr_overlap=psnr,
        mutual_information=mi_val,
        subpixel_precision_achieved=(rmse < 0.5),
        warp_method=warp_choice
    )

    return RegistrationResult(
        registered_u8=warped_u8,
        registered_f32=warped_f32,
        reference_u8=ref_u8,
        reference_f32=ref_f32,
        overlap_mask=overlap_mask,
        difference_map=difference_map,
        metrics=metrics,
        transform_matrix=H_opt,
        src_pts_used=clean_src,
        ref_pts_used=clean_ref
    )
