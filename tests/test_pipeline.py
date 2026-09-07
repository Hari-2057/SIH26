"""
Unit and Integration Tests for Lunar Image Registration & Multi-Modal Pipeline
Validates RIFT, LNIFT, SuperPoint, PDS4 Metadata, 3D Topography, Sub-Pixel Precision, and Multi-Sensor Fusion.
"""

import os
import pytest
import numpy as np
import cv2

from lunar_reg.io import load_geospatial_image, LunarImageData, parse_pds4_xml_metadata, generate_planetary_geojson
from lunar_reg.preprocessing import preprocess_lunar_image
from lunar_reg.features import extract_features, LunarFeatures, compute_phase_congruency_mim, compute_lnift_normalized_image
from lunar_reg.matching import match_and_filter_keypoints, MatchResult
from lunar_reg.grid_uniformity import enforce_grid_uniformity_and_refine, SubPixelRefinementResult
from lunar_reg.warping_evaluation import register_and_evaluate, RegistrationResult
from lunar_reg.pipeline import LunarRegistrationPipeline
from lunar_reg.synthetic_data import generate_lunar_crater_scene, generate_multi_sensor_lunar_suite
from lunar_reg.relief_3d import reconstruct_photoclinometric_dem, extract_transect_profile
from lunar_reg.change_detection import detect_lunar_surface_changes
from lunar_reg.geo_export import export_registered_geotiff
from lunar_reg.deep_models import SuperPointFrontend, match_sinkhorn_lightglue


@pytest.fixture
def synthetic_lunar_scene():
    """Generates a reproducible paired synthetic lunar scene."""
    return generate_lunar_crater_scene(
        scene_name="Test Crater Benchmarking",
        shape=(480, 480),
        rotation_deg=2.5,
        scale_factor=1.02,
        dx_pixels=5.35,
        dy_pixels=-4.65,
        sun_azimuth_src=55.0,
        sun_azimuth_ref=30.0,
        noise_level=2.0,
        seed=42
    )


def test_module_1_io_real_geotiff():
    """Module 1: Test Ingestion and Metadata Parsing on Real 16-bit GeoTIFF."""
    tif_path = "data/shackleton_south_pole/chandrayaan2_moving.tif"
    if os.path.exists(tif_path):
        data = load_geospatial_image(tif_path)
        assert isinstance(data, LunarImageData)
        assert data.shape == (720, 720)
        assert data.raw_array.dtype == np.float32
        assert "Moon" in str(data.crs) or "IAU" in str(data.crs) or "30100" in str(data.crs)
        assert data.resolution is not None


def test_pds4_metadata_and_geojson():
    """Test PDS4 XML label parser and GeoJSON planetary footprint generator."""
    sample_xml = """<?xml version='1.0' encoding='UTF-8'?>
<Product_Observational>
    <Identification_Area>
        <logical_identifier>urn:isro:ch2:ohrc:ch2_ohr_ncp_20210415t062214500_d_img_d18</logical_identifier>
        <instrument_name>OHRC</instrument_name>
    </Identification_Area>
    <Observation_Area>
        <Time_Coordinates>
            <start_date_time>2021-04-15T06:22:14.500Z</start_date_time>
        </Time_Coordinates>
        <Geometry_Parameters>
            <center_latitude>-89.92</center_latitude>
            <center_longitude>0.05</center_longitude>
            <solar_azimuth>65.4</solar_azimuth>
            <solar_elevation>18.2</solar_elevation>
            <ground_sample_distance>0.25</ground_sample_distance>
        </Geometry_Parameters>
    </Observation_Area>
</Product_Observational>"""
    meta = parse_pds4_xml_metadata(sample_xml)
    assert meta["instrument_name"] == "OHRC"
    assert meta["resolution_m_per_px"] == 0.25
    assert meta["center_lat"] == -89.92

    geojson_str = generate_planetary_geojson(meta)
    assert "FeatureCollection" in geojson_str
    assert "IAU_2000:30100" in geojson_str


def test_module_2_preprocessing():
    """Module 2: Test Contrast Limited Adaptive Histogram Equalization & Denoising."""
    dark_crater = np.full((300, 300), 20, dtype=np.uint8)
    cv2.circle(dark_crater, (150, 150), 60, 240, 4)
    dark_crater[50, 50] = 255

    prep_u8, prep_f32 = preprocess_lunar_image(dark_crater, clip_limit=3.0, tile_grid_size=(8, 8), gaussian_ksize=3)
    assert prep_u8.shape == (300, 300)
    assert prep_u8.dtype == np.uint8
    assert prep_f32.dtype == np.float32
    assert prep_u8.std() > dark_crater.std()


def test_rift_and_lnift_features():
    """Test RIFT Phase Congruency and LNIFT feature extraction."""
    img = (np.random.rand(256, 256) * 255).astype(np.uint8)

    mim, orient = compute_phase_congruency_mim(img)
    assert mim.shape == (256, 256)
    assert orient.shape[2] == 6

    f_rift = extract_features(img, method="rift", max_keypoints=300)
    assert len(f_rift) > 0
    assert f_rift.descriptors.shape[1] == 48 or f_rift.descriptors.shape[1] == 96

    f_lnift = extract_features(img, method="lnift", max_keypoints=300)
    assert len(f_lnift) > 0
    assert f_lnift.descriptors.shape[1] == 128


def test_deep_models_and_sinkhorn():
    """Test SuperPoint Frontend and LightGlue Sinkhorn matching."""
    sp = SuperPointFrontend()
    img = (np.random.rand(300, 300) * 255).astype(np.uint8)
    pts, desc, scores = sp.run(img, max_keypoints=500)
    assert len(pts) > 0
    assert desc.shape[1] == 256

    m1, m2, s = match_sinkhorn_lightglue(desc[:50], desc[:50])
    assert len(m1) > 0


def test_3d_relief_reconstruction():
    """Test 3D Photoclinometric DEM generation and transect profiling."""
    img = (np.random.rand(200, 200) * 255).astype(np.uint8)
    dem = reconstruct_photoclinometric_dem(img, sun_azimuth_deg=45.0, sun_elevation_deg=30.0)
    assert dem.shape == (200, 200)
    assert dem.dtype == np.float32
    assert dem.max() > dem.min()

    dists, elevs = extract_transect_profile(dem, (10, 10), (180, 180), num_samples=50)
    assert len(dists) == 50
    assert len(elevs) == 50


def test_multi_sensor_fusion_suite():
    """Test Multi-Sensor Data Fusion (OHRC + TMC-2 + IIRS + LROC)."""
    suite = generate_multi_sensor_lunar_suite("Shackleton Crater", size=256)
    assert "ohrc_moving" in suite
    assert "tmc_dem" in suite
    assert "iirs_hyperspectral" in suite
    assert "lroc_ref" in suite
    assert len(suite["time_series"]) == 4


def test_change_detection(synthetic_lunar_scene):
    """Test Multi-Temporal Lunar Surface Change Detection."""
    src_img = synthetic_lunar_scene["src_img"]
    ref_img = synthetic_lunar_scene["ref_img"]
    mask = np.ones_like(src_img, dtype=bool)

    res = detect_lunar_surface_changes(src_img, ref_img, mask)
    assert res.change_mask.shape == src_img.shape
    assert res.anomaly_heatmap.shape == (src_img.shape[0], src_img.shape[1], 3)
    assert res.total_change_area_sq_m >= 0.0


def test_gis_geotiff_export(tmp_path):
    """Test GIS GeoTIFF export with embedded Moon 2000 CRS and Worldfile."""
    arr = (np.random.rand(100, 100) * 255).astype(np.uint8)
    data = load_geospatial_image(arr)
    out_tif = str(tmp_path / "test_registered.tif")

    res = export_registered_geotiff(arr, data, out_tif)
    assert os.path.exists(res["geotiff"])
    assert os.path.exists(res["worldfile"])


def test_end_to_end_subpixel_accuracy(synthetic_lunar_scene):
    """End-to-End System Test: Verify Sub-Pixel Precision (<0.5 px RMSE)."""
    src_img = synthetic_lunar_scene["src_img"]
    ref_img = synthetic_lunar_scene["ref_img"]

    pipeline = LunarRegistrationPipeline(
        feature_method="rootsift",
        clip_limit=3.0,
        tile_grid_size=(8, 8),
        ratio_threshold=0.78,
        ransac_threshold=2.5,
        grid_size=(4, 4),
        refine_subpixel=True,
        use_optical_flow=True,
        warp_method="homography"
    )

    output = pipeline.run(src_img, ref_img)
    rmse = output.metrics.rmse_reprojection

    print(f"\n[Test Result] Registration RMSE = {rmse:.4f} pixels (Target: <0.5 px)")
    assert rmse < 0.5, f"Expected RMSE < 0.5 px, got {rmse:.4f} px"
    assert output.metrics.subpixel_precision_achieved is True
