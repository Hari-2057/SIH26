"""
MODULE 1: Image Ingestion & Metadata Parsing (PDS4 XML, GeoTIFF, GeoJSON Standardization)
Supports high-bit depth geospatial formats (.tif, .img, .cub), PDS4 XML labels, standard formats (.png, .jpg),
and in-memory buffers with preservation of geospatial metadata (CRS, transform, resolution, solar angles).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union, List
import os
import io
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2

# Optional geospatial imports with graceful fallbacks
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    Affine = None

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from PIL import Image


@dataclass
class LunarImageData:
    """Container for lunar image raster data and associated geospatial metadata."""
    raw_array: np.ndarray             # Standardized float32 array in [0.0, 1.0]
    display_uint8: np.ndarray         # Normalized uint8 array [0, 255] for OpenCV & UI
    shape: Tuple[int, int]            # (Height, Width)
    channels: int                     # 1 (grayscale) or 3 (RGB)
    dtype_original: str               # e.g., 'uint16', 'float32', 'uint8'
    filepath: Optional[str] = None
    affine_transform: Optional[Any] = None  # GeoTransform or 3x3 Affine matrix
    crs: Optional[str] = None         # CRS identifier (e.g. 'IAU2000:30100', 'EPSG:4326', etc.)
    resolution: Optional[Tuple[float, float]] = None # Pixel resolution (res_x, res_y) in meters/px
    bounds: Optional[Tuple[float, float, float, float]] = None # (min_x, min_y, max_x, max_y)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_grayscale_uint8(self) -> np.ndarray:
        """Returns a single-channel 8-bit grayscale image for computer vision routines."""
        if self.display_uint8.ndim == 3 and self.display_uint8.shape[2] == 3:
            return cv2.cvtColor(self.display_uint8, cv2.COLOR_RGB2GRAY)
        return self.display_uint8


def _normalize_to_float32(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Standardize image array to float32 in [0.0, 1.0] and uint8 in [0, 255]
    with robust NaN/Inf handling and percentile dynamic range scaling for lunar imagery.
    """
    orig_dtype = str(arr.dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if np.issubdtype(arr.dtype, np.floating):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val > min_val:
            norm_f32 = ((arr - min_val) / (max_val - min_val)).astype(np.float32)
        else:
            norm_f32 = np.zeros_like(arr, dtype=np.float32)
    elif arr.dtype == np.uint8:
        norm_f32 = (arr.astype(np.float32) / 255.0)
    elif arr.dtype == np.uint16:
        p1, p99 = np.percentile(arr, (0.5, 99.5))
        if p99 > p1:
            clipped = np.clip(arr, p1, p99)
            norm_f32 = ((clipped - p1) / (p99 - p1)).astype(np.float32)
        else:
            norm_f32 = (arr.astype(np.float32) / 65535.0)
    elif arr.dtype in (np.int16, np.int32, np.uint32):
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val > min_val:
            norm_f32 = ((arr - min_val) / (max_val - min_val)).astype(np.float32)
        else:
            norm_f32 = np.zeros_like(arr, dtype=np.float32)
    else:
        norm_f32 = arr.astype(np.float32)
        if norm_f32.max() > 1.0:
            norm_f32 /= norm_f32.max()

    norm_f32 = np.clip(norm_f32, 0.0, 1.0)
    display_uint8 = np.clip(norm_f32 * 255.0, 0, 255).astype(np.uint8)

    return norm_f32, display_uint8, orig_dtype


# =============================================================================
# PDS4 XML METADATA PARSER & GEOJSON STANDARDIZATION
# =============================================================================

def parse_pds4_xml_metadata(xml_input: Union[str, bytes, io.BytesIO]) -> Dict[str, Any]:
    """
    Parses PDS4 XML observation labels for Chandrayaan-2 (OHRC, TMC-2, IIRS) and NASA LRO (LROC NAC).
    Extracts Product ID, UTC Timestamp, Center Coordinates, Solar Azimuth/Elevation, and Spacecraft Telemetry.
    """
    meta: Dict[str, Any] = {
        "product_id": "CH2_OHRC_GENERIC_L2",
        "instrument_name": "OHRC",
        "utc_timestamp": "2021-04-15T06:22:14.500Z",
        "center_lat": -89.9,
        "center_lon": 0.0,
        "sun_azimuth_deg": 45.0,
        "sun_elevation_deg": 28.0,
        "phase_angle_deg": 62.0,
        "spacecraft_altitude_km": 100.2,
        "resolution_m_per_px": 1.25,
        "crs": "Moon 2000 Stereographic (IAU_2000:30100)",
        "target_name": "Moon"
    }

    try:
        if isinstance(xml_input, (bytes, io.BytesIO)):
            content = xml_input.getvalue() if isinstance(xml_input, io.BytesIO) else xml_input
            root = ET.fromstring(content)
        elif os.path.exists(xml_input):
            tree = ET.parse(xml_input)
            root = tree.getroot()
        else:
            root = ET.fromstring(xml_input)

        # Iterate elements matching tag suffixes ignoring namespaces
        for elem in root.iter():
            tag = elem.tag.split('}')[-1].lower()
            text = (elem.text or "").strip()
            if not text:
                continue

            if "logical_identifier" in tag or "product_id" in tag:
                meta["product_id"] = text
            elif "start_date_time" in tag or "observation_time" in tag:
                meta["utc_timestamp"] = text
            elif "instrument_name" in tag or "instrument_id" in tag:
                meta["instrument_name"] = text
            elif "center_latitude" in tag or "latitude" in tag:
                try: meta["center_lat"] = float(text)
                except ValueError: pass
            elif "center_longitude" in tag or "longitude" in tag:
                try: meta["center_lon"] = float(text)
                except ValueError: pass
            elif "solar_azimuth" in tag or "sub_solar_azimuth" in tag:
                try: meta["sun_azimuth_deg"] = float(text)
                except ValueError: pass
            elif "solar_elevation" in tag or "sun_elevation" in tag:
                try: meta["sun_elevation_deg"] = float(text)
                except ValueError: pass
            elif "phase_angle" in tag:
                try: meta["phase_angle_deg"] = float(text)
                except ValueError: pass
            elif "spacecraft_altitude" in tag or "altitude" in tag:
                try: meta["spacecraft_altitude_km"] = float(text)
                except ValueError: pass
            elif "pixel_resolution" in tag or "ground_sample_distance" in tag:
                try: meta["resolution_m_per_px"] = float(text)
                except ValueError: pass
    except Exception:
        pass

    return meta


def generate_planetary_geojson(
    metadata_dict: Dict[str, Any],
    footprint_corners: Optional[List[Tuple[float, float]]] = None
) -> str:
    """
    Standardizes planetary observation metadata into GeoJSON format for PostGIS and web maps.
    """
    lat = metadata_dict.get("center_lat", -89.9)
    lon = metadata_dict.get("center_lon", 0.0)
    delta = 0.05

    if footprint_corners is None:
        coords = [
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta]
        ]
    else:
        coords = [[pt[0], pt[1]] for pt in footprint_corners]
        if coords[0] != coords[-1]:
            coords.append(coords[0])

    geojson_obj = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:IAU_2000:30100"}
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                },
                "properties": metadata_dict
            }
        ]
    }
    return json.dumps(geojson_obj, indent=2)


# =============================================================================
# MASTER GEOSPATIAL IMAGE INGESTION
# =============================================================================

def load_geospatial_image(
    source: Union[str, bytes, io.BytesIO, np.ndarray],
    filename_hint: Optional[str] = None
) -> LunarImageData:
    raw_array = None
    orig_dtype = "unknown"
    filepath = source if isinstance(source, str) else None
    meta_dict: Dict[str, Any] = {}
    affine_transform = None
    crs = None
    resolution = None
    bounds = None

    if isinstance(source, np.ndarray):
        raw_array = source
        orig_dtype = str(source.dtype)
    else:
        # PDS raster format inspection
        if raw_array is None:
            hint = filename_hint
            if hint is None and isinstance(source, str):
                hint = source
            raw_array = _try_load_pds_image(source, filename_hint=hint)
            if raw_array is not None:
                orig_dtype = str(raw_array.dtype)

        # Rasterio geospatial reader
        if raw_array is None and HAS_RASTERIO and isinstance(source, (str, bytes, io.BytesIO)):
            try:
                stream = source
                if isinstance(source, bytes):
                    stream = io.BytesIO(source)
                elif isinstance(source, io.BytesIO):
                    source.seek(0)
                    stream = source
                with rasterio.open(stream) as dataset:
                    if dataset.count == 1:
                        raw_array = dataset.read(1)
                    else:
                        raw_array = np.transpose(dataset.read(), (1, 2, 0))
                    orig_dtype = str(raw_array.dtype)
                    meta_dict = dict(dataset.meta)
                    affine_transform = dataset.transform
                    crs = str(dataset.crs) if dataset.crs else None
                    resolution = dataset.res if dataset.res else None
                    bounds = dataset.bounds
            except Exception:
                pass

        # Tifffile reader
        if raw_array is None and HAS_TIFFFILE and isinstance(source, (str, bytes, io.BytesIO)):
            try:
                stream = source
                if isinstance(source, io.BytesIO):
                    source.seek(0)
                    stream = source
                elif isinstance(source, bytes):
                    stream = io.BytesIO(source)
                raw_array = tifffile.imread(stream)
                orig_dtype = str(raw_array.dtype)
            except Exception:
                pass

        # OpenCV / Pillow reader
        if raw_array is None:
            try:
                if isinstance(source, str):
                    raw_array = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                elif isinstance(source, bytes):
                    raw_array = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_UNCHANGED)
                elif isinstance(source, io.BytesIO):
                    source.seek(0)
                    raw_array = cv2.imdecode(np.frombuffer(source.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                if raw_array is not None:
                    orig_dtype = str(raw_array.dtype)
            except Exception:
                pass

        if raw_array is None:
            try:
                stream = io.BytesIO(source) if isinstance(source, bytes) else source
                pil_img = Image.open(stream)
                raw_array = np.array(pil_img)
                orig_dtype = str(raw_array.dtype)
            except Exception as err:
                raise ValueError(f"Failed to ingest lunar image from source: {err}")

    if raw_array is None:
        raise ValueError("Image array could not be loaded from provided source.")

    if raw_array.ndim == 2:
        h, w = raw_array.shape
        channels = 1
    elif raw_array.ndim == 3:
        h, w, channels = raw_array.shape
    else:
        raise ValueError(f"Unsupported image array dimensions: {raw_array.shape}")

    norm_f32, display_uint8, _ = _normalize_to_float32(raw_array)

    if crs is None:
        crs = "IAU_2000:30100 (Lunar Moon 2000 Polar Stereographic)"
    if resolution is None:
        resolution = (1.25, 1.25)
    if bounds is None:
        bounds = (0.0, 0.0, float(w) * resolution[0], float(h) * resolution[1])

    meta_dict.update({
        "height": h,
        "width": w,
        "channels": channels,
        "resolution_m_per_px": resolution[0],
        "crs": crs
    })

    return LunarImageData(
        raw_array=norm_f32,
        display_uint8=display_uint8,
        shape=(h, w),
        channels=channels,
        dtype_original=orig_dtype,
        filepath=filepath,
        affine_transform=affine_transform,
        crs=crs,
        resolution=resolution,
        bounds=bounds,
        metadata=meta_dict
    )


def load_lunar_image(
    file_input: Union[str, bytes, io.BytesIO, np.ndarray],
    filename_hint: Optional[str] = None
) -> LunarImageData:
    return load_geospatial_image(source=file_input, filename_hint=filename_hint)


def _try_load_pds_image(
    source: Union[str, bytes, io.BytesIO],
    filename_hint: Optional[str] = None
) -> Optional[np.ndarray]:
    ext = None
    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower()
    elif filename_hint:
        ext = os.path.splitext(filename_hint)[1].lower()

    if ext not in (".img", ".cub"):
        return None

    if HAS_RASTERIO:
        try:
            stream = source
            if isinstance(source, bytes):
                stream = io.BytesIO(source)
            elif isinstance(source, io.BytesIO):
                source.seek(0)
                stream = source
            with rasterio.open(stream) as dataset:
                if dataset.count == 1:
                    return dataset.read(1)
                return np.transpose(dataset.read(), (1, 2, 0))
        except Exception:
            pass

    try:
        if isinstance(source, (bytes, io.BytesIO)):
            data = source.getvalue() if isinstance(source, io.BytesIO) else source
            arr = np.frombuffer(data, dtype=np.uint16)
            side = int(np.sqrt(len(arr)))
            if side * side == len(arr):
                return arr.reshape(side, side)
    except Exception:
        pass
    return None
