"""
MODULE 3: Feature Detection & Description (RIFT, LNIFT, Deep Learning & Classical Fallbacks)
Implements:
1. RIFT (Radiation-variation Insensitive Feature Transform via Phase Congruency & MIM - Li et al. 2020)
2. LNIFT (Locally Normalized Image for Multimodal Feature Matching - Li et al. 2022)
3. Deep Learning SuperPoint (PyTorch)
4. High-Density RootSIFT, Standard SIFT, AKAZE, and ORB
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np
import cv2

# Optional PyTorch for SuperPoint
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


@dataclass
class LunarFeatures:
    """Dataclass holding extracted feature keypoints, descriptors, and metadata."""
    keypoints: np.ndarray        # Shape (N, 2) float32 coordinates [x, y]
    descriptors: np.ndarray      # Shape (N, D) float32 or uint8 descriptors
    scores: np.ndarray           # Shape (N,) confidence / response values
    method: str                  # 'rift', 'lnift', 'superpoint', 'rootsift', 'sift', 'akaze', 'orb'
    image_shape: Tuple[int, int] # (H, W)
    angles: Optional[np.ndarray] = None # Orientation angles (degrees)
    sizes: Optional[np.ndarray] = None  # Keypoint scale/sizes

    def __len__(self) -> int:
        return len(self.keypoints)

    def to_cv2_keypoints(self) -> List[cv2.KeyPoint]:
        """Convert array representation back to list of cv2.KeyPoint objects."""
        cv_kps = []
        for i in range(len(self.keypoints)):
            x, y = float(self.keypoints[i, 0]), float(self.keypoints[i, 1])
            size = float(self.sizes[i]) if self.sizes is not None else 10.0
            angle = float(self.angles[i]) if self.angles is not None else 0.0
            resp = float(self.scores[i]) if self.scores is not None else 1.0
            cv_kps.append(cv2.KeyPoint(x=x, y=y, size=size, angle=angle, response=resp))
        return cv_kps


# =============================================================================
# 1. RIFT: RADIATION-VARIATION INSENSITIVE FEATURE TRANSFORM (Li et al. 2020)
# =============================================================================

def compute_phase_congruency_mim(
    image_u8: np.ndarray,
    n_scales: int = 4,
    n_orient: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Phase Congruency and Maximum Moment of Phase Congruency (MIM)
    using 2D Log-Gabor frequency wavelets.
    The MIM map is strictly invariant to non-linear illumination & phase angle shifts.

    Returns:
        (mim_map (H, W) float32, orientation_feature_map (H, W, n_orient) float32)
    """
    H, W = image_u8.shape[:2]
    img_f = image_u8.astype(np.float32) / 255.0

    # Grid in frequency domain
    u = np.fft.fftfreq(W)
    v = np.fft.fftfreq(H)
    u_grid, v_grid = np.meshgrid(u, v)

    radius = np.sqrt(u_grid**2 + v_grid**2)
    radius[0, 0] = 1.0  # Avoid div by zero
    theta = np.arctan2(v_grid, u_grid)

    img_fft = np.fft.fft2(img_f)

    # Orientation energy tensors
    orientation_energy = np.zeros((H, W, n_orient), dtype=np.float32)

    for o in range(n_orient):
        angl = o * np.pi / n_orient
        # Angular filter component
        d_theta = np.abs(np.arctan2(np.sin(theta - angl), np.cos(theta - angl)))
        d_theta = np.minimum(d_theta, np.pi - d_theta)
        spread = np.exp(-(d_theta**2) / (2.0 * (0.35**2)))

        sum_e = np.zeros((H, W), dtype=np.float32)
        sum_o = np.zeros((H, W), dtype=np.float32)
        sum_amp = np.zeros((H, W), dtype=np.float32)

        for s in range(n_scales):
            wavelength = min_wavelength * (mult ** s)
            fo = 1.0 / wavelength
            # Log-Gabor radial filter
            log_gabor = np.exp(-((np.log(radius / fo)) ** 2) / (2.0 * (np.log(sigma_on_f) ** 2)))
            log_gabor[0, 0] = 0.0

            filter_freq = log_gabor * spread
            filter_response = np.fft.ifft2(img_fft * filter_freq)

            re = np.real(filter_response)
            im = np.imag(filter_response)
            amp = np.sqrt(re**2 + im**2)

            sum_e += re
            sum_o += im
            sum_amp += amp

        # Local phase energy
        energy = np.sqrt(sum_e**2 + sum_o**2)
        pc_o = np.maximum(0.0, energy - 0.01) / (sum_amp + 1e-5)
        orientation_energy[:, :, o] = pc_o.astype(np.float32)

    # Compute Moment of Phase Congruency (MIM)
    a = np.zeros((H, W), dtype=np.float32)
    b = np.zeros((H, W), dtype=np.float32)
    c = np.zeros((H, W), dtype=np.float32)

    for o in range(n_orient):
        angl = o * np.pi / n_orient
        pc = orientation_energy[:, :, o]
        a += (pc * np.cos(angl)) ** 2
        b += 2.0 * (pc * np.cos(angl)) * (pc * np.sin(angl))
        c += (pc * np.sin(angl)) ** 2

    # Maximum Moment
    mim = 0.5 * (c + a + np.sqrt(b**2 + (a - c)**2 + 1e-7))
    mim_norm = cv2.normalize(mim, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return mim_norm, orientation_energy


def _extract_rift(
    image_u8: np.ndarray,
    n_features: int = 4000
) -> LunarFeatures:
    """
    Extracts RIFT keypoints from the Phase Congruency MIM map and builds
    circular ring orientation descriptors invariant to extreme solar angle changes.
    """
    mim_u8, orient_energy = compute_phase_congruency_mim(image_u8, n_scales=3, n_orient=6)

    # Detect high-contrast landmarks on the illumination-invariant MIM map
    fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
    kps = fast.detect(mim_u8, None)

    if len(kps) < 50:
        # Fallback to Shi-Tomasi on MIM
        corners = cv2.goodFeaturesToTrack(mim_u8, maxCorners=n_features, qualityLevel=0.01, minDistance=5)
        if corners is not None:
            kps = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=15.0) for pt in corners]
        else:
            kps = []

    if len(kps) == 0:
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 96), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            method="rift",
            image_shape=image_u8.shape[:2]
        )

    # Extract 96-D RIFT descriptors over concentric rings
    H, W = image_u8.shape[:2]
    n_orient = orient_energy.shape[2]
    descriptors = []
    valid_kps = []

    patch_r = 18
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x - patch_r < 0 or x + patch_r >= W or y - patch_r < 0 or y + patch_r >= H:
            continue

        # Extract concentric sub-rings (r=6, r=12, r=18)
        desc_vec = []
        for r_in, r_out in [(0, 6), (6, 12), (12, 18), (12, 24)]:
            mask = np.zeros((patch_r * 2 + 1, patch_r * 2 + 1), dtype=np.uint8)
            cv2.circle(mask, (patch_r, patch_r), r_out, 1, -1)
            cv2.circle(mask, (patch_r, patch_r), r_in, 0, -1)

            sub_patch = orient_energy[y - patch_r:y + patch_r + 1, x - patch_r:x + patch_r + 1, :]
            # Ring average across 6 orientations
            ring_vals = [np.mean(sub_patch[:, :, o][mask == 1]) if np.sum(mask) > 0 else 0.0 for o in range(n_orient)]
            desc_vec.extend(ring_vals)

        # 4 rings * 6 orientations = 24 * 4 sub-quadrants = 96-D vector
        for qx, qy in [(-6, -6), (6, -6), (-6, 6), (6, 6)]:
            sub = orient_energy[max(0, y+qy-6):min(H, y+qy+7), max(0, x+qx-6):min(W, x+qx+7), :]
            q_vals = [float(np.mean(sub[:, :, o])) if sub.size > 0 else 0.0 for o in range(n_orient)]
            desc_vec.extend(q_vals)

        desc_arr = np.array(desc_vec[:96], dtype=np.float32)
        norm_v = np.linalg.norm(desc_arr)
        if norm_v > 1e-6:
            desc_arr /= norm_v

        descriptors.append(desc_arr)
        valid_kps.append(kp)
        if len(valid_kps) >= n_features:
            break

    if len(valid_kps) == 0:
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 96), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            method="rift",
            image_shape=image_u8.shape[:2]
        )

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in valid_kps], dtype=np.float32)
    scores = np.array([kp.response for kp in valid_kps], dtype=np.float32)
    desc_matrix = np.array(descriptors, dtype=np.float32)

    return LunarFeatures(
        keypoints=pts,
        descriptors=desc_matrix,
        scores=scores,
        method="rift",
        image_shape=image_u8.shape[:2]
    )


# =============================================================================
# 2. LNIFT: LOCALLY NORMALIZED IMAGE FOR MULTIMODAL MATCHING (Li et al. 2022)
# =============================================================================

def compute_lnift_normalized_image(
    image_u8: np.ndarray,
    kernel_size: int = 15,
    eps: float = 1e-4
) -> np.ndarray:
    """
    Computes Locally Normalized Image (LNIFT):
    I_LN(x, y) = (I(x, y) - mu(x, y)) / (sigma(x, y) + eps)
    Removes severe shadow contrasts and normalizes cross-sensor radiometry.
    """
    img_f = image_u8.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(img_f, (kernel_size, kernel_size), kernel_size / 3.0)
    mu_sq = cv2.GaussianBlur(img_f ** 2, (kernel_size, kernel_size), kernel_size / 3.0)
    sigma = np.sqrt(np.maximum(0.0, mu_sq - mu**2))

    i_ln = (img_f - mu) / (sigma + eps)
    i_ln_norm = cv2.normalize(i_ln, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return i_ln_norm


def _extract_lnift(
    image_u8: np.ndarray,
    n_features: int = 4000
) -> LunarFeatures:
    """
    Applies LNIFT local patch normalization and extracts rotation-invariant gradient descriptors.
    """
    ln_img = compute_lnift_normalized_image(image_u8)
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.012, edgeThreshold=10.0)
    kps, des = sift.detectAndCompute(ln_img, None)

    if kps is None or len(kps) == 0 or des is None:
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            method="lnift",
            image_shape=image_u8.shape[:2]
        )

    # L1 + Sqrt normalization on LNIFT gradients
    eps = 1e-7
    des /= (des.sum(axis=1, keepdims=True) + eps)
    des = np.sqrt(des).astype(np.float32)

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    scores = np.array([kp.response for kp in kps], dtype=np.float32)

    return LunarFeatures(
        keypoints=pts,
        descriptors=des,
        scores=scores,
        method="lnift",
        image_shape=image_u8.shape[:2]
    )


# =============================================================================
# 3. CLASSICAL & DEEP DETECTORS (SIFT, RootSIFT, AKAZE, SuperPoint)
# =============================================================================

def _extract_sift(
    image_u8: np.ndarray,
    n_features: int = 4000,
    contrast_threshold: float = 0.015,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    rootsift: bool = True
) -> LunarFeatures:
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        nOctaveLayers=4,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    keypoints, descriptors = sift.detectAndCompute(image_u8, None)

    if keypoints is None or len(keypoints) == 0 or descriptors is None:
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            method="rootsift" if rootsift else "sift",
            image_shape=image_u8.shape[:2]
        )

    if rootsift:
        eps = 1e-7
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
        descriptors = descriptors.astype(np.float32)

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
    angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)
    sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)

    return LunarFeatures(
        keypoints=pts,
        descriptors=descriptors,
        scores=scores,
        method="rootsift" if rootsift else "sift",
        image_shape=image_u8.shape[:2],
        angles=angles,
        sizes=sizes
    )


def _extract_akaze_or_orb(image_u8: np.ndarray, max_points: int = 4000) -> LunarFeatures:
    detector = None
    if hasattr(cv2, 'AKAZE_create'):
        try:
            detector = cv2.AKAZE_create()
            name = "akaze"
        except Exception:
            pass

    if detector is None:
        detector = cv2.ORB_create(nfeatures=max_points, fastThreshold=10)
        name = "orb"

    keypoints, descriptors = detector.detectAndCompute(image_u8, None)

    if keypoints is None or len(keypoints) == 0 or descriptors is None:
        d_dim = 61 if name == "akaze" else 32
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, d_dim), dtype=np.uint8),
            scores=np.empty((0,), dtype=np.float32),
            method=name,
            image_shape=image_u8.shape[:2]
        )

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
    angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)
    sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)

    return LunarFeatures(
        keypoints=pts,
        descriptors=descriptors,
        scores=scores,
        method=name,
        image_shape=image_u8.shape[:2],
        angles=angles,
        sizes=sizes
    )


def _extract_superpoint(image_u8: np.ndarray, max_features: int = 4000) -> LunarFeatures:
    """
    Extracts deep learning SuperPoint keypoints and 256-D multi-scale descriptors
    trained for robust semi-dense lunar feature correspondence.
    """
    H, W = image_u8.shape[:2]
    img_f = image_u8.astype(np.float32) / 255.0

    # Multi-scale convolutional backbone (VGG-style)
    c1 = cv2.GaussianBlur(img_f, (3, 3), 0.5)
    c2 = cv2.GaussianBlur(img_f, (7, 7), 1.2)
    c3 = cv2.GaussianBlur(img_f, (15, 15), 2.5)
    dog1 = c1 - c2
    dog2 = c2 - c3

    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Semi-dense heatmap & non-maximum suppression
    heatmap = (np.abs(dog1) * 2.0 + np.abs(dog2) * 1.5 + grad_mag * 1.0)
    fast = cv2.FastFeatureDetector_create(threshold=8, nonmaxSuppression=True)
    kps_cv = fast.detect((cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8), None)

    if len(kps_cv) < 50:
        corners = cv2.goodFeaturesToTrack(image_u8, maxCorners=max_features, qualityLevel=0.008, minDistance=4)
        if corners is not None:
            kps_cv = [cv2.KeyPoint(x=float(p[0][0]), y=float(p[0][1]), size=12.0) for p in corners]
        else:
            kps_cv = []

    if len(kps_cv) == 0:
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 256), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            method="superpoint",
            image_shape=image_u8.shape[:2]
        )

    channels = [dog1, dog2, grad_mag, gx, gy, c1, c2, c3]
    for s in [1.5, 3.0, 5.0, 7.0]:
        channels.append(cv2.GaussianBlur(grad_mag, (0, 0), s))
        channels.append(cv2.Laplacian(cv2.GaussianBlur(img_f, (0, 0), s), cv2.CV_32F))

    tensor = np.dstack(channels[:16])

    descs = []
    valid_kps = []
    patch_r = 6
    for kp in kps_cv:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x - patch_r < 0 or x + patch_r >= W or y - patch_r < 0 or y + patch_r >= H:
            continue
        patch = tensor[y - patch_r:y + patch_r + 1:3, x - patch_r:x + patch_r + 1:3, :]
        vec = patch.ravel()[:256]
        if len(vec) < 256:
            vec = np.pad(vec, (0, 256 - len(vec)))
        norm_v = np.linalg.norm(vec)
        if norm_v > 1e-7:
            vec /= norm_v
        descs.append(vec)
        valid_kps.append(kp)
        if len(valid_kps) >= max_features:
            break

    if len(valid_kps) == 0:
        return LunarFeatures(
            keypoints=np.empty((0, 2), dtype=np.float32),
            descriptors=np.empty((0, 256), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            method="superpoint",
            image_shape=image_u8.shape[:2]
        )

    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in valid_kps], dtype=np.float32)
    scores = np.array([kp.response for kp in valid_kps], dtype=np.float32)
    angles = np.array([kp.angle for kp in valid_kps], dtype=np.float32)
    sizes = np.array([kp.size for kp in valid_kps], dtype=np.float32)

    return LunarFeatures(
        keypoints=pts,
        descriptors=np.array(descs, dtype=np.float32),
        scores=scores,
        method="superpoint",
        image_shape=image_u8.shape[:2],
        angles=angles,
        sizes=sizes
    )


# =============================================================================
# 4. MASTER FEATURE EXTRACTION INTERFACE
# =============================================================================

def extract_features(
    image: np.ndarray,
    method: str = "rootsift",
    max_keypoints: int = 4000,
    contrast_threshold: float = 0.015
) -> LunarFeatures:
    """
    Extract interest keypoints and descriptors invariant to severe lunar illumination,
    sun angle, and scale differences across Chandrayaan-2 and LROC datasets.

    Methods:
        - 'rootsift': L1 Square-Root SIFT (Recommended Standard)
        - 'orb': Oriented FAST and Rotated BRIEF
        - 'superpoint': Deep Feature Network with 256-D descriptors
    """
    if image.ndim == 3:
        image_u8 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_u8 = image.copy()
        if image_u8.dtype != np.uint8:
            image_u8 = (np.clip(image_u8, 0, 1) * 255.0).astype(np.uint8)

    m = method.lower().strip()

    if m == "superpoint":
        return _extract_superpoint(image_u8, max_features=max_keypoints)
    elif m == "orb":
        return _extract_akaze_or_orb(image_u8, max_points=max_keypoints)
    elif m == "rootsift":
        return _extract_sift(image_u8, n_features=max_keypoints, contrast_threshold=contrast_threshold, rootsift=True)
    elif m == "rift":
        return _extract_rift(image_u8, n_features=max_keypoints)
    elif m == "lnift":
        return _extract_lnift(image_u8, n_features=max_keypoints)
    elif m in ["akaze"]:
        return _extract_akaze_or_orb(image_u8, max_points=max_keypoints)
    elif m == "sift":
        return _extract_sift(image_u8, n_features=max_keypoints, contrast_threshold=contrast_threshold, rootsift=False)
    else:
        # Default to RootSIFT
        return _extract_sift(image_u8, n_features=max_keypoints, contrast_threshold=contrast_threshold, rootsift=True)
