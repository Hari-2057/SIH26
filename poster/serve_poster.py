"""
CHANDRA-ALIGN: Poster Exporter & Viewer Utility
Allows one-click opening, local previewing, or headless PDF generation for academic conferences (A1/A0/48x36").
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import subprocess
from pathlib import Path

POSTER_DIR = Path(__file__).parent.resolve()
INDEX_HTML = POSTER_DIR / "index.html"

def open_in_browser():
    """Open the interactive poster directly in the default web browser."""
    print(f"🚀 Opening Poster in default browser: file://{INDEX_HTML}")
    webbrowser.open(f"file://{INDEX_HTML}")

def serve_poster(port=8502):
    """Serve the poster directory locally over HTTP."""
    os.chdir(POSTER_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"📡 Serving Chandra-Align Project Poster at http://localhost:{port}")
        print("💡 Press Ctrl+C to stop the server.")
        webbrowser.open(f"http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

def export_pdf():
    """Attempt automated PDF export using system Chrome / Edge / Playwright if installed."""
    pdf_out = POSTER_DIR / "chandra_align_poster.pdf"
    print(f"🖨️ Generating PDF: {pdf_out}")

    # Check for google-chrome or chromium or edge
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser"
    ]
    chrome_bin = next((p for p in chrome_paths if os.path.exists(p)), None)

    if chrome_bin:
        cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_out}",
            "--virtual-time-budget=5000",
            f"file://{INDEX_HTML}"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully exported high-resolution PDF to: {pdf_out}")
            return
        except Exception as e:
            print(f"⚠️ Chrome headless export failed ({e}). Falling back to browser print.")
    else:
        print("ℹ️ Headless Chrome not found. Opening browser for direct Print to PDF (Cmd+P / Ctrl+P)...")
        open_in_browser()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8502
        serve_poster(port)
    elif len(sys.argv) > 1 and sys.argv[1] == "--pdf":
        export_pdf()
    else:
        open_in_browser()
