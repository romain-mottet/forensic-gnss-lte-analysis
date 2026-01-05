# Forensic Analysis of Smartphone Geolocation Quality

**Master Thesis Implementation**  
*Quality of Geolocation in Smartphones: Measurement, Analysis, and Limitations in Real-World Scenarios*  
**Author:** Romain Mottet  
**Institution:** École Polytechnique de Louvain (UCLouvain)  
**Year:** 2025-2026  

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## 📖 Overview

This repository contains the complete analytical framework and source code developed for the Master's thesis **"Quality of Geolocation in Smartphones"**. The project evaluates the accuracy, precision, and forensic reliability of smartphone positioning systems (GNSS and LTE) across varying Belgian environments (Urban, Suburban, Rural).

The codebase implements a three-stage pipeline designed to:
1.  **Establish Ground Truth:** Synchronize and compare smartphone GPS against a Garmin reference watch and forensically extracted EXIF waypoints.
2.  **Model Signal Propagation:** Create empirical LTE distance models using signal strength (RSRP) and quality (RSRQ).
3.  **Analyze Network Positioning:** Perform LTE-based trilateration and validate it against the established ground truth.

Additionally, the **full LaTeX source code** for the thesis document is included in the `thesis_latex/` directory.

## 📂 Repository Structure

The project is organized into three distinct analytical modules corresponding to the thesis methodology chapters, plus the thesis document source.

```text
smartphone-geolocation-forensics/
├── 01_gps_comparison/          # [Stage 1] GPS Accuracy & Ground Truth Pipeline
│   ├── src/                    # Parsers for G-NetTrack & Garmin GPX
│   └── main.py                 # Time synchronization and accuracy metrics
├── 02_distance_modeling/       # [Stage 2] LTE Signal Distance Regression
│   ├── src/                    # Feature expansion & LOFO Cross-Validation
│   └── formulas/               # JSON outputs of empirical distance models
├── 03_network_analysis/        # [Stage 3] LTE Trilateration & Validation
│   ├── src/                    # Tower DB parsing, path-loss models, trilateration
│   └── tower_db/               # Corrected cell tower location databases
├── thesis_latex/               # [Thesis] Full LaTeX source code
│   ├── chapters/               # Individual chapter files (.tex)
│   ├── figures/                # Generated plots and diagrams
│   └── main.tex                # Main document file
├── data/                       # Input logs (G-NetTrack, Garmin) and Output CSVs
```

## 🚀 Analytical Pipelines

### Stage 1: GPS Comparison Pipeline (`01_gps_comparison`)
**Goal:** Establish forensic ground truth and quantify GNSS accuracy.
*   **Inputs:** Smartphone logs (G-NetTrack), Reference tracks (Garmin Venu Sq 2), Forensic Waypoints (Autopsy/EXIF).
*   **Key Operations:**
    *   Parses and standardizes GPX and CSV logs.
    *   **Time Synchronization:** Detects optimal time windows to align smartphone and watch clocks, correcting for offsets.
    *   **Validation:** Calculates distance errors (RMSE, CEP, R95) against physical checkpoints.
*   **Output:** `unified_gps_dataset.csv` (Quality-flagged ground truth).

### Stage 2: Distance Calculation Algorithm (`02_distance_modeling`)
**Goal:** Develop environment-specific models to estimate distance to cell towers.
*   **Inputs:** Parsed LTE logs from Stage 3 prep + Tower Databases.
*   **Key Operations:**
    *   **Feature Engineering:** Creates polynomial and interaction terms from `RSRP`, `RSRQ`, and `EARFCN`.
    *   **LOFO Cross-Validation:** Uses "Leave-One-File-Out" strategy to ensure models generalize to new sessions.
    *   **Regression:** Trains models to predict `log(distance)`.
*   **Output:** `formula_{context}.json` (Production-ready distance formulas).

### Stage 3: Network Position Analysis (`03_network_analysis`)
**Goal:** Reconstruct device location using ONLY LTE network data.
*   **Inputs:** Raw LTE logs + Tower DB + Formulas from Stage 2.
*   **Key Operations:**
    *   **Parsing:** Extracts serving and neighbor cell measurements.
    *   **Distance Estimation:** Compares standard Path-Loss models vs. Machine Learning formulas.
    *   **Trilateration:** Solves for position using weighted least-squares optimization.
*   **Output:** Position error metrics, coverage reports, and comparison datasets.

## 🛠️ Prerequisites & Setup

### Environment
*   **Python 3.10+**
*   **Dependencies:** `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `gpxpy`, `scikit-learn`

### Installation
```bash
git clone https://github.com/yourusername/smartphone-geolocation-forensics.git
cd smartphone-geolocation-forensics
pip install -r requirements.txt
```

## 📱 Experimental Setup
This code is tailored for data collected using the specific thesis protocol:
*   **Target Device:** OnePlus 6T (LineageOS 22.2 / Android 15, Rooted).
*   **Logging Tool:** G-NetTrack Pro (Log interval: 6s).
*   **Reference Device:** Garmin Venu Sq 2.
*   **Forensic Extraction:** Autopsy & aLEAPP (for verifying EXIF timestamps/coords).

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🎓 Citation
If you use this code or methodology in your research, please cite the master thesis:

> **Mottet, R. (2025).** *Quality of Geolocation in Smartphones: Measurement, Analysis, and Limitations in Real-World Scenarios* (Master's Thesis). École Polytechnique de Louvain, Université catholique de Louvain.
