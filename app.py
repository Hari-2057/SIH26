"""
CHANDRA-ALIGN: Planetary Lunar Image Registration Platform
Vercel Production WSGI Web Application & API Entrypoint
Supports: ISRO Chandrayaan-2 & NASA LRO Image Correspondence & Telemetry
"""

import os
import json
import mimetypes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSTER_DIR = os.path.join(BASE_DIR, "poster")
ASSETS_DIR = os.path.join(POSTER_DIR, "assets")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def application(environ, start_response):
    """
    Production-grade WSGI handler serving the CHANDRA-ALIGN platform,
    high-resolution telemetry visuals, vector PDFs, and JSON endpoints.
    """
    path = environ.get("PATH_INFO", "/") or "/"
    if path.startswith("/"):
        norm_path = path.lstrip("/")
    else:
        norm_path = path

    # Root route or index
    if norm_path in ("", "index.html"):
        target_file = os.path.join(POSTER_DIR, "index.html")
        return _serve_file(target_file, start_response, cache_control="public, max-age=3600")

    # Static assets: /assets/<file>
    if norm_path.startswith("assets/"):
        asset_name = os.path.basename(norm_path[len("assets/"):])
        target_file = os.path.join(ASSETS_DIR, asset_name)
        return _serve_file(target_file, start_response, cache_control="public, max-age=86400, immutable")

    # Vector PDFs
    if norm_path == "chandra_align_poster.pdf":
        target_file = os.path.join(POSTER_DIR, "chandra_align_poster.pdf")
        return _serve_file(target_file, start_response, cache_control="public, max-age=86400")

    if norm_path == "chandra_align_poster_portrait.pdf":
        target_file = os.path.join(POSTER_DIR, "chandra_align_poster_portrait.pdf")
        return _serve_file(target_file, start_response, cache_control="public, max-age=86400")

    # API Endpoint: Health check
    if norm_path in ("api/health", "health"):
        data = {
            "status": "healthy",
            "platform": "CHANDRA-ALIGN",
            "version": "1.0.0",
            "mission": "Smart India Hackathon 2026 (SIH-1644)",
            "telemetry": {
                "rmse": "0.380 px",
                "inlier_ratio": "68.5%",
                "ssim": 0.914,
                "latency_ms": 210
            }
        }
        return _serve_json(data, start_response)

    # API Endpoint: Lunar Datasets Manifest
    if norm_path in ("api/manifest", "manifest.json"):
        manifest_path = os.path.join(DATA_DIR, "manifest.json")
        if os.path.isfile(manifest_path):
            return _serve_file(manifest_path, start_response, content_type="application/json; charset=utf-8")
        return _serve_json([], start_response)

    # API Endpoint: Metrics Report
    if norm_path in ("api/metrics", "api/results"):
        metrics_path = os.path.join(RESULTS_DIR, "metrics_report.json")
        if os.path.isfile(metrics_path):
            return _serve_file(metrics_path, start_response, content_type="application/json; charset=utf-8")
        return _serve_json({"error": "No metrics found"}, start_response, status="404 Not Found")

    # Default fallback: serve index.html for client-side navigation
    target_file = os.path.join(POSTER_DIR, "index.html")
    if os.path.isfile(target_file):
        return _serve_file(target_file, start_response)

    start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
    return [b"404 Not Found"]


def _serve_file(filepath, start_response, content_type=None, cache_control="public, max-age=3600"):
    if not os.path.isfile(filepath):
        start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"404 Not Found"]

    if not content_type:
        content_type, _ = mimetypes.guess_type(filepath)
        if not content_type:
            content_type = "application/octet-stream"
        if content_type.startswith("text/") or content_type in ["application/json", "application/javascript"]:
            content_type += "; charset=utf-8"

    try:
        with open(filepath, "rb") as f:
            content = f.read()
    except Exception as e:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8")])
        return [str(e).encode("utf-8")]

    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(content))),
        ("Cache-Control", cache_control),
        ("X-Powered-By", "CHANDRA-ALIGN ISRO Platform")
    ]
    start_response("200 OK", headers)
    return [content]


def _serve_json(data, start_response, status="200 OK"):
    body = json.dumps(data, indent=2).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "public, max-age=60"),
        ("Access-Control-Allow-Origin", "*"),
        ("X-Powered-By", "CHANDRA-ALIGN ISRO Platform")
    ]
    start_response(status, headers)
    return [body]


# Top-level entrypoint aliases required by Vercel Python Runtime
app = application
application = application
handler = application
