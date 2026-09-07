# 🌖 CHANDRA-ALIGN: Project Poster Package

> **Smart India Hackathon 2026 (SIH-1644) | ISRO & Department of Space**  
> *Automated Multi-Modal Lunar Image Registration & 3D Planetary Surface Reconstruction Platform*

---

## 📂 Included Poster Deliverables

| Deliverable | File Path | Format & Specs |
| :--- | :--- | :--- |
| **Print-Ready Vector PDF** | [`poster/chandra_align_poster.pdf`](file:///Users/haripreethp/SIH26/poster/chandra_align_poster.pdf) | **300 DPI Vector PDF** (Standard 48"×36" / A1 Landscape Print) |
| **Interactive Web Poster** | [`poster/index.html`](file:///Users/haripreethp/SIH26/poster/index.html) | Standalone HTML5 + KaTeX + Dual Theme (Dark / Light) |
| **Python CLI Poster Tool** | [`poster/serve_poster.py`](file:///Users/haripreethp/SIH26/poster/serve_poster.py) | Automated Browser Preview & Headless PDF Exporter |
| **High-Res Visual Assets** | [`poster/assets/`](file:///Users/haripreethp/SIH26/poster/assets/) | Space Orbiter Hero Graphic, MAGSAC++ Matches, Checkerboard, Heatmaps |

---

## 🚀 How to View & Present the Poster

### 1. View Interactive Poster in Browser
Double-click [`poster/index.html`](file:///Users/haripreethp/SIH26/poster/index.html) or run:
```bash
python3 poster/serve_poster.py
```

### 2. Export / Print Directly to PDF
Click the **"Print / Save as PDF"** button on the top right toolbar of the web poster, or run:
```bash
python3 poster/serve_poster.py --pdf
```

### 3. Switch Themes for Printing
- **ISRO Deep Space Dark**: Ideal for OLED monitors, projectors, TV screens, and digital presentations.
- **Academic Conference Light**: Clean white background optimized for crisp vinyl standee / flex banner / A1 paper printing with zero dark toner waste.

---

## 📐 Poster Structure & Key Scientific Highlights

1. **Header & Badges**:
   - Sub-Pixel Reprojection Error: **0.380 px RMSE**
   - Sun-Angle Invariance: **$\Delta 75^\circ$ Solar Azimuth Shifts**
   - Non-Rigid Warping: **Thin-Plate Spline (TPS)** with Radial Basis Kernel $U(r) = r^2 \ln r$
   - Real-Time Inference: **210 ms** Execution Latency
   - Multi-Sensor Fusion: **Chandrayaan-2 OHRC (0.25m) + TMC-2 (5.0m) + IIRS + NASA LRO NAC (0.5m)**

2. **Column 1 — Challenges & Mathematical Formulations**:
   - Polar crater shadow inversions & low radiometric contrast
   - RIFT Phase Congruency Maximum Moment ($M_\psi$) formulation
   - Thin-Plate Spline Bending Energy Minimization
   - Adaptive Non-Maximal Suppression (ANMS) spatial radius $r_i$

3. **Column 2 (Center) — Architecture & High-Resolution Visuals**:
   - 3D Chandrayaan-2 polar mapping mission visualization over Shackleton Crater
   - 6-Stage Autonomous Registration Architecture Flowchart
   - Multi-scale Keypoint Correspondence Matching Map (MAGSAC++)
   - Continuous Seam Checkerboard Blend & Difference Error Heatmap (PSNR: 28.4 dB)
   - ANMS Spatial Grid Coverage (Spatial Entropy: 0.675)

4. **Column 3 — Benchmarks, 3D Elevation Platform & Mission Impact**:
   - SOTA Benchmark Comparison Matrix (SIFT vs ORB vs SuperPoint vs Chandra-Align)
   - 360° Interactive 3D Terrain Mesh & DEM Elevation Cross-Section Transects
   - Natural Language "Text-to-Spatial" AI Query Engine
   - Chandrayaan-3/4 Landing Site Hazard Avoidance & South Pole Water-Ice Prospecting
