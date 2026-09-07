"""
3D LUNAR TOPOGRAPHY & INTERACTIVE ELEVATION PROFILER
Implements Photoclinometric Shape-From-Shading to reconstruct 3D crater digital elevation models
and generates interactive Plotly 3D terrain meshes and cross-sectional topographic profiles.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
import plotly.graph_objects as go


def reconstruct_photoclinometric_dem(
    image_u8: np.ndarray,
    sun_azimuth_deg: float = 45.0,
    sun_elevation_deg: float = 30.0,
    albedo_scale: float = 1.0,
    smoothing_iter: int = 5
) -> np.ndarray:
    """
    Reconstructs a relative 3D digital elevation model (DEM) from lunar surface shading
    using Frankot-Chellappa / Shape-From-Shading surface gradient integration.

    Parameters:
        image_u8: Grayscale lunar image (H, W) uint8.
        sun_azimuth_deg: Solar azimuth angle in degrees.
        sun_elevation_deg: Solar elevation angle in degrees.
        albedo_scale: Relative surface reflectance scale factor.
        smoothing_iter: Number of relaxation iterations for Poisson surface solver.

    Returns:
        2D elevation matrix (H, W) float32 in meters.
    """
    H, W = image_u8.shape[:2]
    img_f = (image_u8.astype(np.float32) / 255.0) * albedo_scale
    img_f = cv2.GaussianBlur(img_f, (5, 5), 1.0)

    # Solar illuminant unit vector
    az_rad = np.radians(sun_azimuth_deg)
    el_rad = np.radians(sun_elevation_deg)
    ps = np.cos(az_rad) / (np.tan(el_rad) + 1e-5)
    qs = np.sin(az_rad) / (np.tan(el_rad) + 1e-5)

    # Gradient estimation from image intensity derivatives
    Ix = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

    # Initial surface slope estimates
    denom = ps * ps + qs * qs + 1e-5
    p = (Ix * ps) / denom
    q = (Iy * qs) / denom

    # Integrable surface recovery via Frankot-Chellappa algorithm in Fourier domain
    wx = 2.0 * np.pi * np.fft.fftfreq(W)
    wy = 2.0 * np.pi * np.fft.fftfreq(H)
    Wx, Wy = np.meshgrid(wx, wy)

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    # Poisson solver in frequency domain: Z = (-j*Wx*P - j*Wy*Q) / (Wx^2 + Wy^2)
    denom_freq = Wx**2 + Wy**2
    denom_freq[0, 0] = 1.0  # Avoid zero division at DC component

    Z = (-1j * Wx * P - 1j * Wy * Q) / denom_freq
    Z[0, 0] = 0.0

    dem = np.real(np.fft.ifft2(Z))
    # Normalize elevation scale (typical crater depth ~100m to 800m)
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-7) * 450.0

    return dem.astype(np.float32)


def create_3d_terrain_mesh_plotly(
    dem: np.ndarray,
    texture_u8: np.ndarray,
    subsample: int = 4,
    title: str = "3D Lunar Crater Topography"
) -> go.Figure:
    """
    Generates a 3D Plotly interactive surface mesh textured with the registered lunar image.
    """
    H, W = dem.shape[:2]
    # Subsample for smooth 60fps rendering in browser
    dem_sub = dem[::subsample, ::subsample]
    tex_sub = texture_u8[::subsample, ::subsample]

    h_sub, w_sub = dem_sub.shape
    x = np.arange(w_sub) * subsample
    y = np.arange(h_sub) * subsample

    fig = go.Figure(data=[
        go.Surface(
            z=dem_sub,
            x=x,
            y=y,
            surfacecolor=tex_sub,
            colorscale='Greys_r',
            showscale=False,
            lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.5, specular=0.2),
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="#38bdf8", project_z=True))
        )
    ])

    fig.update_layout(
        title=dict(text=f"🌖 {title}", font=dict(family="Outfit", size=18, color="#f1f5f9")),
        autosize=True,
        height=550,
        margin=dict(l=10, r=10, b=10, t=40),
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        scene=dict(
            xaxis=dict(title="X (meters)", backgroundcolor="#0f172a", gridcolor="#334155", showbackground=True),
            yaxis=dict(title="Y (meters)", backgroundcolor="#0f172a", gridcolor="#334155", showbackground=True),
            zaxis=dict(title="Elevation (meters)", backgroundcolor="#0f172a", gridcolor="#334155", showbackground=True),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.35),
            camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9))
        )
    )

    return fig


def extract_transect_profile(
    dem: np.ndarray,
    start_pt: Tuple[int, int],
    end_pt: Tuple[int, int],
    num_samples: int = 150
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a 1D elevation profile along a 2D line transect across the crater diameter.

    Returns:
        (distances_along_transect (meters), elevations (meters))
    """
    x0, y0 = start_pt
    x1, y1 = end_pt

    xs = np.linspace(x0, x1, num_samples)
    ys = np.linspace(y0, y1, num_samples)

    H, W = dem.shape[:2]
    xs_clamped = np.clip(xs, 0, W - 1)
    ys_clamped = np.clip(ys, 0, H - 1)

    elevations = cv2.remap(
        dem,
        xs_clamped.astype(np.float32),
        ys_clamped.astype(np.float32),
        interpolation=cv2.INTER_LINEAR
    ).ravel()

    # Total distance in meters
    total_dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    distances = np.linspace(0, total_dist, num_samples)

    return distances, elevations
