"""
COMMAND-LINE INTERFACE FOR LUNAR IMAGE REGISTRATION
Enables automated registration of Chandrayaan-2 moving optical imagery to Lunar Reference Maps.

Usage Examples:
    # Run on synthetic benchmark scene
    python cli.py --demo --warp-method tps --output-dir ./output

    # Run on real satellite GeoTIFF files
    python cli.py --source ch2_moving.tif --reference lunar_basemap.tif --output-dir ./output
"""

import os
import argparse
import json
import numpy as np
import cv2

from lunar_reg.pipeline import LunarRegistrationPipeline
from lunar_reg.synthetic_data import generate_lunar_crater_scene


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lunar Image Registration Pipeline (Chandrayaan-2 to Reference Map)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--source", type=str, default=None, help="Path to Source (Moving) lunar image file")
    parser.add_argument("--reference", type=str, default=None, help="Path to Reference (Fixed) lunar basemap file")
    parser.add_argument("--demo", action="store_true", help="Run benchmark on synthetic Chandrayaan-2 lunar crater scene")
    parser.add_argument("--feature-method", type=str, default="rootsift", choices=["rootsift", "sift", "orb", "superpoint", "auto"], help="Feature detection algorithm")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="CLAHE contrast clip limit")
    parser.add_argument("--ransac-threshold", type=float, default=2.5, help="RANSAC outlier threshold (pixels)")
    parser.add_argument("--warp-method", type=str, default="tps", choices=["tps", "homography", "affine"], help="Geometric warping method")
    parser.add_argument("--grid-size", type=int, default=4, help="Spatial uniformity grid dimension (e.g. 4 for 4x4)")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save registered products")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🌙 CHANDRAYAAN-2 AUTOMATED LUNAR IMAGE REGISTRATION PIPELINE")
    print("=" * 70)

    if args.demo or (args.source is None and args.reference is None):
        print("\n[INFO] Initializing Synthetic Chandrayaan-2 Lunar Crater Benchmark Scene...")
        scene = generate_lunar_crater_scene(
            scene_name="Shackleton Crater Lunar South Pole",
            shape=(640, 640),
            rotation_deg=3.5,
            scale_factor=1.03,
            dx_pixels=8.45,
            dy_pixels=-6.28,
            sun_azimuth_src=65.0,
            sun_azimuth_ref=35.0,
            noise_level=2.5,
            seed=42
        )
        src_input = scene["src_img"]
        ref_input = scene["ref_img"]
        print(f"[INFO] Benchmark Scene: {scene['scene_name']}")
    else:
        if not os.path.exists(args.source):
            raise FileNotFoundError(f"Source file not found: {args.source}")
        if not os.path.exists(args.reference):
            raise FileNotFoundError(f"Reference file not found: {args.reference}")
        src_input = args.source
        ref_input = args.reference
        print(f"[INFO] Source Image: {args.source}")
        print(f"[INFO] Reference Map: {args.reference}")

    pipeline = LunarRegistrationPipeline(
        feature_method=args.feature_method,
        clip_limit=args.clip_limit,
        tile_grid_size=(8, 8),
        ratio_threshold=0.75,
        ransac_threshold=args.ransac_threshold,
        grid_size=(args.grid_size, args.grid_size),
        refine_subpixel=True,
        use_optical_flow=True,
        warp_method=args.warp_method
    )

    print(f"\n[INFO] Executing 6-stage registration pipeline ({args.feature_method.upper()} + {args.warp_method.upper()})...")
    output = pipeline.run(src_input, ref_input)
    metrics = output.metrics

    print("\n" + "=" * 70)
    print("📊 REGISTRATION PERFORMANCE METRICS DASHBOARD")
    print("=" * 70)
    status_icon = "✅ PASSED" if metrics.subpixel_precision_achieved else "⚠️ FAILED"
    print(f"Sub-Pixel Precision Target (<0.5 px):    {status_icon}")
    print(f"Reprojection RMSE:                      {metrics.rmse_reprojection:.4f} pixels")
    print(f"Mean Absolute Error (MAE):              {metrics.mae_reprojection:.4f} pixels")
    print(f"Max Reprojection Error:                 {metrics.max_reprojection_error:.4f} pixels")
    print(f"Total Verified Inliers:                 {metrics.inlier_count} matches")
    print(f"Inlier Match Ratio:                     {metrics.inlier_ratio_pct:.2f}%")
    print(f"Grid Uniformity Coverage:               {metrics.grid_uniformity_score_pct:.1f}%")
    print(f"Spatial Shannon Entropy:                {metrics.spatial_entropy:.3f}")
    print(f"Structural Similarity (SSIM):           {metrics.ssim_overlap:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR):      {metrics.psnr_overlap:.2f} dB")
    print(f"Normalized Mutual Information:          {metrics.mutual_information:.4f}")
    print(f"Total Execution Time:                   {output.execution_time_sec:.3f} seconds")
    print("=" * 70)

    # Save output artifacts
    reg_path = os.path.join(args.output_dir, "registered_moving_image.png")
    diff_path = os.path.join(args.output_dir, "difference_heatmap.png")
    matches_path = os.path.join(args.output_dir, "keypoint_matches.png")
    checker_path = os.path.join(args.output_dir, "checkerboard_overlay.png")
    grid_path = os.path.join(args.output_dir, "grid_uniformity_map.png")
    metrics_path = os.path.join(args.output_dir, "metrics_report.json")

    cv2.imwrite(reg_path, output.registration_result.registered_u8)
    cv2.imwrite(diff_path, output.registration_result.difference_map)
    cv2.imwrite(matches_path, output.get_match_visualization())
    cv2.imwrite(checker_path, output.get_checkerboard_blend())
    cv2.imwrite(grid_path, output.get_grid_distribution_overlay())

    metrics_dict = {
        "rmse_reprojection_px": metrics.rmse_reprojection,
        "mae_reprojection_px": metrics.mae_reprojection,
        "max_reprojection_error_px": metrics.max_reprojection_error,
        "inlier_count": metrics.inlier_count,
        "raw_match_count": metrics.raw_match_count,
        "inlier_ratio_pct": metrics.inlier_ratio_pct,
        "grid_uniformity_score_pct": metrics.grid_uniformity_score_pct,
        "spatial_entropy": metrics.spatial_entropy,
        "ssim_overlap": metrics.ssim_overlap,
        "psnr_overlap_db": metrics.psnr_overlap,
        "mutual_information": metrics.mutual_information,
        "subpixel_precision_achieved": metrics.subpixel_precision_achieved,
        "warp_method": metrics.warp_method,
        "execution_time_sec": output.execution_time_sec,
        "pipeline_parameters": output.pipeline_params
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"\n[SUCCESS] All outputs saved to: {os.path.abspath(args.output_dir)}")
    print(f"  ├── Registered Image:     {reg_path}")
    print(f"  ├── Match Visuals:        {matches_path}")
    print(f"  ├── Difference Heatmap:   {diff_path}")
    print(f"  ├── Checkerboard Overlay: {checker_path}")
    print(f"  ├── Grid Uniformity Map:  {grid_path}")
    print(f"  └── Metrics Report:       {metrics_path}\n")


if __name__ == "__main__":
    main()
