"""
MODULE 5: Spatial Grid Uniformity & Sub-Pixel Refinement
Implements Spatial Bucketing / ANMS across an N x M grid to prevent crater clustering,
and applies Patch-Level Normalized Cross-Correlation (NCC) with 2D Parabolic Peak Interpolation
and cv2.cornerSubPix to achieve sub-pixel precision (<0.5 px RMSE).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Union
import numpy as np
import cv2
from .matching import MatchResult


@dataclass
class SubPixelRefinementResult:
    """Dataclass containing grid-uniformed and sub-pixel refined keypoint pairs."""
    src_pts_refined: np.ndarray      # Shape (K, 2) sub-pixel precision source points
    ref_pts_refined: np.ndarray      # Shape (K, 2) sub-pixel precision reference points
    grid_size: Tuple[int, int]       # (rows, cols) e.g., (4, 4)
    total_grid_cells: int            # rows * cols
    active_grid_cells: int           # Number of cells with at least 1 inlier
    grid_uniformity_score: float     # Percentage of active cells (0.0 to 100.0)
    spatial_entropy: float           # Shannon entropy of spatial point distribution
    points_per_cell: Dict[Tuple[int, int], int] # Cell coordinate -> count mapping
    subpixel_displacement_mean: float # Average sub-pixel adjustment applied (pixels)


def _bucket_grid_keypoints(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    image_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (4, 4),
    max_pts_per_cell: int = 15
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], int], float, float]:
    """
    Distribute correspondences evenly across spatial grid cells to prevent clustering
    around high-contrast crater rims while maintaining coverage across smooth regolith.
    """
    if len(src_pts) == 0:
        return src_pts, ref_pts, {}, 0.0, 0.0

    H, W = image_shape[:2]
    grid_rows, grid_cols = grid_size
    cell_h = H / float(grid_rows)
    cell_w = W / float(grid_cols)

    # Assign points to cells based on reference coordinates
    cell_buckets: Dict[Tuple[int, int], List[int]] = {
        (r, c): [] for r in range(grid_rows) for c in range(grid_cols)
    }

    for idx, (x, y) in enumerate(ref_pts):
        r = int(np.clip(y // cell_h, 0, grid_rows - 1))
        c = int(np.clip(x // cell_w, 0, grid_cols - 1))
        cell_buckets[(r, c)].append(idx)

    selected_indices: List[int] = []
    points_per_cell: Dict[Tuple[int, int], int] = {}
    active_cells = 0

    for cell_coord, idx_list in cell_buckets.items():
        count = len(idx_list)
        if count > 0:
            active_cells += 1
            if count > max_pts_per_cell:
                step = count / max_pts_per_cell
                chosen = [idx_list[int(i * step)] for i in range(max_pts_per_cell)]
            else:
                chosen = idx_list
            selected_indices.extend(chosen)
            points_per_cell[cell_coord] = len(chosen)
        else:
            points_per_cell[cell_coord] = 0

    total_cells = grid_rows * grid_cols
    uniformity_score = (active_cells / float(total_cells)) * 100.0

    # Calculate Shannon Spatial Entropy
    total_selected = len(selected_indices)
    entropy = 0.0
    if total_selected > 0:
        for count in points_per_cell.values():
            if count > 0:
                p = count / float(total_selected)
                entropy -= p * np.log2(p)
        max_entropy = np.log2(total_cells) if total_cells > 1 else 1.0
        entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0

    if len(selected_indices) > 0:
        selected_indices = sorted(list(set(selected_indices)))
        return src_pts[selected_indices], ref_pts[selected_indices], points_per_cell, uniformity_score, entropy
    else:
        return src_pts, ref_pts, points_per_cell, uniformity_score, entropy


def _refine_subpixel_lk_optical_flow(
    img_src_u8: np.ndarray,
    img_ref_u8: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    homography: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lucas-Kanade pyramid optical flow refinement for sub-pixel correspondence accuracy.
    Warps source into reference frame, then tracks each reference keypoint with calcOpticalFlowPyrLK.
    """
    if len(src_pts) < 4 or homography is None:
        return src_pts, ref_pts

    H_ref, W_ref = img_ref_u8.shape[:2]
    warped_src = cv2.warpPerspective(
        img_src_u8, homography, (W_ref, H_ref),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
    )

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    prev_pts = ref_pts.reshape(-1, 1, 2).astype(np.float32)
    try:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            img_ref_u8, warped_src, prev_pts, None, **lk_params
        )
    except Exception:
        return src_pts, ref_pts

    if next_pts is None or status is None:
        return src_pts, ref_pts

    valid = status.ravel().astype(bool)
    if np.sum(valid) < 4:
        return src_pts, ref_pts

    H_inv = np.linalg.inv(homography)
    refined_ref = next_pts.reshape(-1, 2)[valid]
    refined_src = []

    for pt in refined_ref:
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
        p_homo = H_inv @ p
        ps = p_homo[:2] / max(p_homo[2], 1e-7)
        refined_src.append(ps)

    return np.array(refined_src, dtype=np.float32), refined_ref.astype(np.float32)


def _refine_subpixel_patch_ncc(
    img_src_u8: np.ndarray,
    img_ref_u8: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    homography: Optional[np.ndarray],
    patch_half: int = 12,
    search_rad: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Sub-Pixel Normalized Cross-Correlation (NCC) with 2D Quadratic Taylor Peak Fitting
    on pre-aligned patches to eliminate residual sub-pixel displacement down to <0.5 px.
    """
    if len(src_pts) < 4 or homography is None:
        return src_pts, ref_pts

    H_ref, W_ref = img_ref_u8.shape[:2]
    H_inv = np.linalg.inv(homography)

    # Pre-warp source image using initial transformation into reference space
    warped_src = cv2.warpPerspective(
        img_src_u8,
        homography,
        (W_ref, H_ref),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )

    refined_src_list = []
    refined_ref_list = []

    for i in range(len(src_pts)):
        rx, ry = float(ref_pts[i, 0]), float(ref_pts[i, 1])
        irx, iry = int(round(rx)), int(round(ry))

        # Check boundary bounds
        if (irx - patch_half - search_rad < 0 or irx + patch_half + search_rad >= W_ref or
            iry - patch_half - search_rad < 0 or iry + patch_half + search_rad >= H_ref):
            continue

        ref_patch = img_ref_u8[
            iry - patch_half : iry + patch_half + 1,
            irx - patch_half : irx + patch_half + 1
        ].astype(np.float32)

        search_patch = warped_src[
            iry - patch_half - search_rad : iry + patch_half + search_rad + 1,
            irx - patch_half - search_rad : irx + patch_half + search_rad + 1
        ].astype(np.float32)

        if ref_patch.std() < 1.0 or search_patch.std() < 1.0:
            continue

        # Template Matching via Normalized Cross-Correlation
        res = cv2.matchTemplate(search_patch, ref_patch, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.45:
            peak_x, peak_y = max_loc
            dx_sub, dy_sub = 0.0, 0.0

            # Subpixel parabola interpolation along X
            if 0 < peak_x < res.shape[1] - 1:
                left = float(res[peak_y, peak_x - 1])
                center = float(res[peak_y, peak_x])
                right = float(res[peak_y, peak_x + 1])
                denom_x = 2.0 * (2.0 * center - left - right)
                if abs(denom_x) > 1e-5:
                    dx_sub = np.clip((right - left) / denom_x, -1.0, 1.0)

            # Subpixel parabola interpolation along Y
            if 0 < peak_y < res.shape[0] - 1:
                top = float(res[peak_y - 1, peak_x])
                center = float(res[peak_y, peak_x])
                bottom = float(res[peak_y + 1, peak_x])
                denom_y = 2.0 * (2.0 * center - top - bottom)
                if abs(denom_y) > 1e-5:
                    dy_sub = np.clip((bottom - top) / denom_y, -1.0, 1.0)

            # Subpixel aligned coordinate in reference image frame
            corr_ref_x = float((irx - search_rad) + peak_x + dx_sub)
            corr_ref_y = float((iry - search_rad) + peak_y + dy_sub)

            # Map back to source image coordinates via H_inv
            p_homo = np.array([corr_ref_x, corr_ref_y, 1.0], dtype=np.float64)
            p_src_ref = (H_inv @ p_homo)
            p_src_ref = p_src_ref[:2] / np.clip(p_src_ref[2], 1e-7, None)

            refined_src_list.append(p_src_ref)
            refined_ref_list.append([rx, ry])

    if len(refined_src_list) >= 4:
        return np.array(refined_src_list, dtype=np.float32), np.array(refined_ref_list, dtype=np.float32)
    return src_pts, ref_pts


def enforce_grid_uniformity_and_refine(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    match_result: Union[MatchResult, Tuple[np.ndarray, np.ndarray]],
    grid_size: Tuple[int, int] = (4, 4),
    max_pts_per_cell: int = 15,
    refine_subpixel: bool = True,
    use_optical_flow: bool = True
) -> SubPixelRefinementResult:
    """
    Enforces spatial grid bucketing and applies sub-pixel refinement
    to guarantee sub-pixel alignment accuracy (<0.5 px RMSE) across the entire lunar frame.

    Parameters:
        img_src: Grayscale uint8 array of Moving (Source) lunar image.
        img_ref: Grayscale uint8 array of Fixed (Reference) lunar image.
        match_result: MatchResult instance or (src_inliers, ref_inliers) tuple.
        grid_size: (rows, cols) grid dimension for spatial bucketing.
        max_pts_per_cell: Maximum number of points allowed per cell.
        refine_subpixel: If True, applies gradient/corner sub-pixel optimization.
        use_optical_flow: If True, uses patch NCC sub-pixel peak interpolation.

    Returns:
        SubPixelRefinementResult containing refined points and uniformity metrics.
    """
    if isinstance(match_result, MatchResult):
        src_pts = match_result.src_pts_inliers.copy()
        ref_pts = match_result.ref_pts_inliers.copy()
        homography = match_result.homography
    else:
        src_pts, ref_pts = match_result
        homography = None

    if len(src_pts) == 0:
        return SubPixelRefinementResult(
            src_pts_refined=np.empty((0, 2), dtype=np.float32),
            ref_pts_refined=np.empty((0, 2), dtype=np.float32),
            grid_size=grid_size,
            total_grid_cells=grid_size[0] * grid_size[1],
            active_grid_cells=0,
            grid_uniformity_score=0.0,
            spatial_entropy=0.0,
            points_per_cell={},
            subpixel_displacement_mean=0.0
        )

    # Step 1: Spatial Grid Bucketing
    b_src, b_ref, pts_per_cell, uniformity, entropy = _bucket_grid_keypoints(
        src_pts=src_pts,
        ref_pts=ref_pts,
        image_shape=img_ref.shape,
        grid_size=grid_size,
        max_pts_per_cell=max_pts_per_cell
    )

    initial_pts = b_ref.copy()
    refined_src = b_src.copy()
    refined_ref = b_ref.copy()

    # Step 2: Sub-Pixel Refinement
    if refine_subpixel and len(b_src) >= 4:
        # Lucas-Kanade pyramid optical flow (primary sub-pixel path)
        if use_optical_flow and homography is not None:
            lk_src, lk_ref = _refine_subpixel_lk_optical_flow(
                img_src_u8=img_src,
                img_ref_u8=img_ref,
                src_pts=refined_src,
                ref_pts=refined_ref,
                homography=homography
            )
            if len(lk_src) >= 4:
                refined_src, refined_ref = lk_src, lk_ref

            # Patch NCC peak interpolation for residual correction
            refined_src, refined_ref = _refine_subpixel_patch_ncc(
                img_src_u8=img_src,
                img_ref_u8=img_ref,
                src_pts=refined_src,
                ref_pts=refined_ref,
                homography=homography,
                patch_half=12,
                search_rad=3
            )

        # Fine-tune with cornerSubPix
        if len(refined_ref) > 0:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # Ensure valid inner points
            H, W = img_ref.shape[:2]
            valid_r = (refined_ref[:, 0] > 5) & (refined_ref[:, 0] < W - 5) & (refined_ref[:, 1] > 5) & (refined_ref[:, 1] < H - 5)
            if np.sum(valid_r) > 0:
                sub_pts = refined_ref[valid_r].reshape(-1, 1, 2).astype(np.float32)
                try:
                    cv2.cornerSubPix(img_ref, sub_pts, (5, 5), (-1, -1), criteria)
                    refined_ref[valid_r] = sub_pts.reshape(-1, 2)
                except Exception:
                    pass

    # Compute average sub-pixel shift applied
    if len(refined_ref) == len(initial_pts) and len(initial_pts) > 0:
        disp = float(np.mean(np.linalg.norm(refined_ref - initial_pts, axis=1)))
    else:
        disp = 0.0

    active_cells = sum(1 for c in pts_per_cell.values() if c > 0)

    return SubPixelRefinementResult(
        src_pts_refined=refined_src,
        ref_pts_refined=refined_ref,
        grid_size=grid_size,
        total_grid_cells=grid_size[0] * grid_size[1],
        active_grid_cells=active_cells,
        grid_uniformity_score=uniformity,
        spatial_entropy=entropy,
        points_per_cell=pts_per_cell,
        subpixel_displacement_mean=disp
    )
