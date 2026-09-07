"""
VISUALIZATION MODULE FOR LUNAR REGISTRATION
Generates inlier/outlier match plots, interactive checkerboard overlays,
alpha-blended registrations, thermal difference heatmaps, spatial grid uniformity maps,
sub-pixel quiver flow vector fields, and 400% ROI loupe magnifiers.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_feature_matches(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_draw: int = 150
) -> np.ndarray:
    """
    Renders side-by-side feature matches with green lines for inliers and red lines for outliers.
    """
    H1, W1 = img_src.shape[:2]
    H2, W2 = img_ref.shape[:2]
    H_max = max(H1, H2)
    canvas = np.zeros((H_max, W1 + W2, 3), dtype=np.uint8)

    src_rgb = cv2.cvtColor(img_src, cv2.COLOR_GRAY2RGB) if img_src.ndim == 2 else img_src[:, :, :3]
    ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB) if img_ref.ndim == 2 else img_ref[:, :, :3]

    canvas[:H1, :W1] = src_rgb
    canvas[:H2, W1:W1 + W2] = ref_rgb

    if len(pts_src) == 0:
        return canvas

    if inlier_mask is None:
        inlier_mask = np.ones(len(pts_src), dtype=bool)

    total_matches = len(pts_src)
    if total_matches > max_draw:
        indices = np.linspace(0, total_matches - 1, max_draw, dtype=int)
    else:
        indices = np.arange(total_matches)

    # Draw outliers in red
    for idx in indices:
        if not inlier_mask[idx]:
            p1 = (int(round(pts_src[idx, 0])), int(round(pts_src[idx, 1])))
            p2 = (int(round(pts_ref[idx, 0])) + W1, int(round(pts_ref[idx, 1])))
            cv2.line(canvas, p1, p2, (220, 40, 40), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (220, 40, 40), -1)
            cv2.circle(canvas, p2, 3, (220, 40, 40), -1)

    # Draw verified inliers in bright green
    for idx in indices:
        if inlier_mask[idx]:
            p1 = (int(round(pts_src[idx, 0])), int(round(pts_src[idx, 1])))
            p2 = (int(round(pts_ref[idx, 0])) + W1, int(round(pts_ref[idx, 1])))
            cv2.line(canvas, p1, p2, (0, 240, 80), 1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 4, (0, 240, 80), -1)
            cv2.circle(canvas, p2, 4, (0, 240, 80), -1)

    cv2.putText(canvas, "Chandrayaan-2 Moving Source", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Lunar Reference Basemap", (W1 + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def create_checkerboard_blend(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    tile_size: int = 40
) -> np.ndarray:
    """Creates an alternating checkerboard pattern to verify crater seam alignment."""
    H, W = img_ref.shape[:2]
    if img_warped.shape[:2] != (H, W):
        img_warped = cv2.resize(img_warped, (W, H), interpolation=cv2.INTER_CUBIC)

    r_u8 = img_ref if img_ref.ndim == 2 else cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY)
    w_u8 = img_warped if img_warped.ndim == 2 else cv2.cvtColor(img_warped, cv2.COLOR_RGB2GRAY)

    y_grid, x_grid = np.mgrid[0:H, 0:W]
    checker_pattern = ((x_grid // tile_size) + (y_grid // tile_size)) % 2 == 0

    blended = np.where(checker_pattern, r_u8, w_u8)
    blended_rgb = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    for y in range(0, H, tile_size):
        cv2.line(blended_rgb, (0, y), (W, y), (80, 80, 80), 1)
    for x in range(0, W, tile_size):
        cv2.line(blended_rgb, (x, 0), (x, H), (80, 80, 80), 1)

    return blended_rgb


def create_alpha_blend(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Creates a smooth alpha blended composite between Reference and Registered imagery."""
    H, W = img_ref.shape[:2]
    if img_warped.shape[:2] != (H, W):
        img_warped = cv2.resize(img_warped, (W, H), interpolation=cv2.INTER_CUBIC)

    r_rgb = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB) if img_ref.ndim == 2 else img_ref
    w_rgb = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2RGB) if img_warped.ndim == 2 else img_warped

    alpha_clamped = np.clip(alpha, 0.0, 1.0)
    blended = cv2.addWeighted(r_rgb, 1.0 - alpha_clamped, w_rgb, alpha_clamped, 0.0)
    return blended


def draw_grid_uniformity_overlay(
    img_ref: np.ndarray,
    pts_ref: np.ndarray,
    grid_size: Tuple[int, int] = (4, 4),
    points_per_cell: Optional[Dict[Tuple[int, int], int]] = None
) -> np.ndarray:
    """Renders the spatial bucketing grid on the reference image with keypoint counts."""
    H, W = img_ref.shape[:2]
    canvas = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB) if img_ref.ndim == 2 else img_ref.copy()
    rows, cols = grid_size
    cell_h = H / float(rows)
    cell_w = W / float(cols)

    for r in range(1, rows):
        y = int(round(r * cell_h))
        cv2.line(canvas, (0, y), (W, y), (0, 200, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(round(c * cell_w))
        cv2.line(canvas, (x, 0), (x, H), (0, 200, 255), 1, cv2.LINE_AA)

    for pt in pts_ref:
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (px, py), 3, (0, 255, 0), -1)

    if points_per_cell is not None:
        for (r, c), count in points_per_cell.items():
            tx = int(c * cell_w + 10)
            ty = int(r * cell_h + 25)
            color = (0, 255, 0) if count > 0 else (0, 0, 255)
            text = f"Cell [{r},{c}]: {count} pts"
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return canvas


def draw_subpixel_quiver_field(
    img_ref: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    transform_matrix: Optional[np.ndarray] = None,
    scale_factor: float = 15.0
) -> np.ndarray:
    """
    Renders 2D sub-pixel deformation quiver vectors on the reference lunar image.
    Arrows represent the direction and magnitude of sub-pixel local relief shifts.
    """
    H, W = img_ref.shape[:2]
    canvas = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB) if img_ref.ndim == 2 else img_ref.copy()

    if len(src_pts) == 0 or len(ref_pts) == 0:
        return canvas

    # If transform_matrix provided, compute reprojection residual vector
    if transform_matrix is not None:
        src_h = np.hstack([src_pts, np.ones((len(src_pts), 1), dtype=np.float32)])
        proj_pts = (transform_matrix @ src_h.T).T
        proj_pts = proj_pts[:, :2] / np.clip(proj_pts[:, 2:3], 1e-7, None)
        dx = proj_pts[:, 0] - ref_pts[:, 0]
        dy = proj_pts[:, 1] - ref_pts[:, 1]
    else:
        dx = src_pts[:, 0] - ref_pts[:, 0]
        dy = src_pts[:, 1] - ref_pts[:, 1]

    magnitudes = np.sqrt(dx**2 + dy**2)

    for i in range(len(ref_pts)):
        rx, ry = int(round(ref_pts[i, 0])), int(round(ref_pts[i, 1]))
        vx = int(round(rx + dx[i] * scale_factor))
        vy = int(round(ry + dy[i] * scale_factor))

        mag = magnitudes[i]
        # Color coding: Green (<0.5 px), Yellow (0.5-1.0 px), Red (>1.0 px)
        if mag < 0.5:
            color = (0, 255, 0)
        elif mag < 1.0:
            color = (0, 255, 255)
        else:
            color = (0, 100, 255)

        cv2.arrowedLine(canvas, (rx, ry), (vx, vy), color, 2, tipLength=0.3)
        cv2.circle(canvas, (rx, ry), 3, color, -1)

    cv2.putText(
        canvas,
        f"Sub-Pixel Quiver Field (Exaggerated {int(scale_factor)}x)",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (56, 189, 248),
        2,
        cv2.LINE_AA
    )
    return canvas


def create_roi_loupe_magnifier(
    img_ref: np.ndarray,
    img_warped: np.ndarray,
    center_pt: Tuple[int, int],
    patch_size: int = 80,
    zoom_factor: int = 4
) -> np.ndarray:
    """
    Extracts a local ROI patch around a specific crater/boulder and magnifies it
    400% side-by-side with split-slider line to inspect sub-pixel alignment.
    """
    H, W = img_ref.shape[:2]
    cx, cy = center_pt
    r = patch_size // 2

    # Clamp patch bounds
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(W, cx + r)
    y1 = min(H, cy + r)

    ref_crop = img_ref[y0:y1, x0:x1]
    warp_crop = img_warped[y0:y1, x0:x1]

    ref_crop = cv2.resize(ref_crop, (patch_size * zoom_factor, patch_size * zoom_factor), interpolation=cv2.INTER_CUBIC)
    warp_crop = cv2.resize(warp_crop, (patch_size * zoom_factor, patch_size * zoom_factor), interpolation=cv2.INTER_CUBIC)

    # Convert to RGB
    r_rgb = cv2.cvtColor(ref_crop, cv2.COLOR_GRAY2RGB) if ref_crop.ndim == 2 else ref_crop
    w_rgb = cv2.cvtColor(warp_crop, cv2.COLOR_GRAY2RGB) if warp_crop.ndim == 2 else warp_crop

    # Side-by-side presentation
    loupe_canvas = np.hstack([r_rgb, w_rgb])
    cv2.putText(loupe_canvas, f"Ref ({zoom_factor}x)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)
    cv2.putText(loupe_canvas, f"Warped ({zoom_factor}x)", (r_rgb.shape[1] + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 189, 248), 2)

    return loupe_canvas
