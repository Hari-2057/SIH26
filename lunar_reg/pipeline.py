"""
UNIFIED LUNAR REGISTRATION PIPELINE
Orchestrates Modules 1 through 6 in an automated, robust end-to-end workflow
achieving sub-pixel accuracy (<0.5 px RMSE) on Chandrayaan-2 moving lunar images.
Integrates Deep Learning, 3D Topography, Change Detection, and GIS GeoTIFF Export.
"""

from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict, Any, List
import io
import os
import time
import numpy as np

from .io import load_geospatial_image, LunarImageData
from .preprocessing import preprocess_lunar_image
from .features import extract_features, LunarFeatures
from .matching import match_and_filter_keypoints, MatchResult
from .grid_uniformity import enforce_grid_uniformity_and_refine, SubPixelRefinementResult
from .warping_evaluation import register_and_evaluate, RegistrationResult, RegistrationMetrics
from .visualization import (
    draw_feature_matches,
    create_checkerboard_blend,
    create_alpha_blend,
    draw_grid_uniformity_overlay,
    draw_subpixel_quiver_field,
    create_roi_loupe_magnifier
)
from .relief_3d import reconstruct_photoclinometric_dem, create_3d_terrain_mesh_plotly, extract_transect_profile
from .change_detection import detect_lunar_surface_changes, ChangeDetectionResult
from .geo_export import export_registered_geotiff


@dataclass
class LunarRegistrationOutput:
    """Complete output package containing processed images, features, inliers, 3D models, and metrics."""
    source_data: LunarImageData
    reference_data: LunarImageData
    preprocessed_src_u8: np.ndarray
    preprocessed_ref_u8: np.ndarray
    features_src: LunarFeatures
    features_ref: LunarFeatures
    match_result: MatchResult
    refinement_result: SubPixelRefinementResult
    registration_result: RegistrationResult
    execution_time_sec: float
    pipeline_params: Dict[str, Any]

    @property
    def metrics(self) -> RegistrationMetrics:
        return self.registration_result.metrics

    def get_match_visualization(self, max_draw: int = 150) -> np.ndarray:
        """Returns side-by-side match visualization with green inlier lines."""
        return draw_feature_matches(
            img_src=self.preprocessed_src_u8,
            img_ref=self.preprocessed_ref_u8,
            pts_src=self.match_result.src_pts_raw,
            pts_ref=self.match_result.ref_pts_raw,
            inlier_mask=self.match_result.inlier_mask,
            max_draw=max_draw
        )

    def get_checkerboard_blend(self, tile_size: int = 40) -> np.ndarray:
        """Returns checkerboard comparison overlay."""
        return create_checkerboard_blend(
            img_ref=self.reference_data.to_grayscale_uint8(),
            img_warped=self.registration_result.registered_u8,
            tile_size=tile_size
        )

    def get_alpha_blend(self, alpha: float = 0.5) -> np.ndarray:
        """Returns alpha blended overlay at specified opacity."""
        return create_alpha_blend(
            img_ref=self.reference_data.to_grayscale_uint8(),
            img_warped=self.registration_result.registered_u8,
            alpha=alpha
        )

    def get_grid_distribution_overlay(self) -> np.ndarray:
        """Returns spatial grid bucketing distribution overlay on reference image."""
        return draw_grid_uniformity_overlay(
            img_ref=self.reference_data.to_grayscale_uint8(),
            pts_ref=self.refinement_result.ref_pts_refined,
            grid_size=self.refinement_result.grid_size,
            points_per_cell=self.refinement_result.points_per_cell
        )

    def get_quiver_field_overlay(self, scale_factor: float = 15.0) -> np.ndarray:
        """Returns 2D sub-pixel deformation quiver flow vector field overlay."""
        return draw_subpixel_quiver_field(
            img_ref=self.reference_data.to_grayscale_uint8(),
            src_pts=self.registration_result.src_pts_used,
            ref_pts=self.registration_result.ref_pts_used,
            transform_matrix=self.registration_result.transform_matrix,
            scale_factor=scale_factor
        )

    def get_roi_loupe(self, center_pt: Tuple[int, int] = (300, 300), zoom_factor: int = 4) -> np.ndarray:
        """Returns high-magnification side-by-side ROI loupe patch."""
        return create_roi_loupe_magnifier(
            img_ref=self.reference_data.to_grayscale_uint8(),
            img_warped=self.registration_result.registered_u8,
            center_pt=center_pt,
            zoom_factor=zoom_factor
        )

    def compute_3d_terrain_dem(self, sun_azimuth: float = 45.0, sun_elevation: float = 30.0) -> np.ndarray:
        """Reconstructs relative 3D digital elevation model (DEM) via photoclinometry."""
        return reconstruct_photoclinometric_dem(
            image_u8=self.registration_result.registered_u8,
            sun_azimuth_deg=sun_azimuth,
            sun_elevation_deg=sun_elevation
        )

    def compute_change_detection(self, sensitivity: float = 0.75) -> ChangeDetectionResult:
        """Performs multi-temporal surface change and physical anomaly detection."""
        res_m = self.reference_data.resolution[0] if self.reference_data.resolution else 1.25
        return detect_lunar_surface_changes(
            img_registered=self.registration_result.registered_u8,
            img_reference=self.reference_data.to_grayscale_uint8(),
            overlap_mask=self.registration_result.overlap_mask,
            pixel_resolution_m=res_m,
            sensitivity=sensitivity
        )

    def export_gis_geotiff(self, output_path: str) -> Dict[str, str]:
        """Exports registered raster to GeoTIFF with Moon 2000 CRS, Worldfile, and GCPs."""
        return export_registered_geotiff(
            registered_array_u8=self.registration_result.registered_u8,
            reference_data=self.reference_data,
            output_tif_path=output_path,
            gcp_src_pts=self.registration_result.src_pts_used,
            gcp_ref_pts=self.registration_result.ref_pts_used
        )


class LunarRegistrationPipeline:
    """
    Automated Lunar Image Registration Pipeline for aligning moving optical images
    (Chandrayaan-2 TMC-2 / OHRC) onto Lunar Reference Maps with sub-pixel precision.
    """
    def __init__(
        self,
        feature_method: str = "rootsift",
        clip_limit: float = 3.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        gaussian_ksize: int = 3,
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 2.5,
        ransac_method: str = "MAGSAC",
        grid_size: Tuple[int, int] = (4, 4),
        max_pts_per_cell: int = 15,
        refine_subpixel: bool = True,
        use_optical_flow: bool = True,
        warp_method: str = "tps",
        max_keypoints: int = 4000
    ):
        self.feature_method = feature_method
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gaussian_ksize = gaussian_ksize
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.ransac_method = ransac_method
        self.grid_size = grid_size
        self.max_pts_per_cell = max_pts_per_cell
        self.refine_subpixel = refine_subpixel
        self.use_optical_flow = use_optical_flow
        self.warp_method = warp_method
        self.max_keypoints = max_keypoints

    def run(
        self,
        source_input: Union[str, bytes, io.BytesIO, np.ndarray, LunarImageData],
        reference_input: Union[str, bytes, io.BytesIO, np.ndarray, LunarImageData]
    ) -> LunarRegistrationOutput:
        """
        Executes end-to-end registration pipeline on source and reference images.
        """
        t0 = time.time()

        # MODULE 1: Ingestion & Metadata Parsing
        if isinstance(source_input, LunarImageData):
            source_data = source_input
        else:
            source_data = load_geospatial_image(source_input)

        if isinstance(reference_input, LunarImageData):
            reference_data = reference_input
        else:
            reference_data = load_geospatial_image(reference_input)

        # MODULE 2: Preprocessing & Contrast Normalization (CLAHE + Gaussian)
        prep_src_u8, prep_src_f32 = preprocess_lunar_image(
            source_data,
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
            gaussian_ksize=self.gaussian_ksize
        )
        prep_ref_u8, prep_ref_f32 = preprocess_lunar_image(
            reference_data,
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
            gaussian_ksize=self.gaussian_ksize
        )

        # MODULE 3: Feature Detection & Description
        feat_src = extract_features(
            prep_src_u8,
            method=self.feature_method,
            max_keypoints=self.max_keypoints
        )
        feat_ref = extract_features(
            prep_ref_u8,
            method=self.feature_method,
            max_keypoints=self.max_keypoints
        )

        # MODULE 4: Contextual Matching & Outlier Rejection (FLANN + MAGSAC++)
        match_res = match_and_filter_keypoints(
            features_src=feat_src,
            features_ref=feat_ref,
            ratio_threshold=self.ratio_threshold,
            ransac_threshold=self.ransac_threshold,
            ransac_method=self.ransac_method
        )

        # MODULE 5: Spatial Grid Uniformity & Sub-Pixel Refinement
        refine_res = enforce_grid_uniformity_and_refine(
            img_src=prep_src_u8,
            img_ref=prep_ref_u8,
            match_result=match_res,
            grid_size=self.grid_size,
            max_pts_per_cell=self.max_pts_per_cell,
            refine_subpixel=self.refine_subpixel,
            use_optical_flow=self.use_optical_flow
        )

        # MODULE 6: Geometric Warping & Metric Evaluation
        reg_res = register_and_evaluate(
            source_img=source_data.to_grayscale_uint8(),
            ref_img=reference_data.to_grayscale_uint8(),
            refined_src_pts=refine_res.src_pts_refined,
            refined_ref_pts=refine_res.ref_pts_refined,
            raw_match_count=match_res.raw_match_count,
            grid_uniformity_score=refine_res.grid_uniformity_score,
            spatial_entropy=refine_res.spatial_entropy,
            warp_method=self.warp_method
        )

        elapsed = time.time() - t0

        pipeline_params = {
            "feature_method": self.feature_method,
            "clip_limit": self.clip_limit,
            "tile_grid_size": self.tile_grid_size,
            "ransac_threshold": self.ransac_threshold,
            "ransac_method": self.ransac_method,
            "grid_size": self.grid_size,
            "warp_method": self.warp_method,
            "subpixel_refinement": self.refine_subpixel
        }

        return LunarRegistrationOutput(
            source_data=source_data,
            reference_data=reference_data,
            preprocessed_src_u8=prep_src_u8,
            preprocessed_ref_u8=prep_ref_u8,
            features_src=feat_src,
            features_ref=feat_ref,
            match_result=match_res,
            refinement_result=refine_res,
            registration_result=reg_res,
            execution_time_sec=elapsed,
            pipeline_params=pipeline_params
        )
