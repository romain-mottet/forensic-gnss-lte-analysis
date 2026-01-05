# LTE-Based Geolocation Analysis Pipeline

**Evaluate LTE-based geolocation accuracy across urban/rural environments**

A comprehensive mobile geolocation research pipeline combining LTE signal analysis, distance estimation (path loss and machine learning), bearing calculations, and weighted trilateration for smartphone positioning validation across city/town/village environments.

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Guide](#usage-guide)
6. [Data Provenance](#data-provenance)
7. [Pipeline Architecture](#pipeline-architecture)
8. [Input/Output Specifications](#inputoutput-specifications)
9. [Parameter Reference](#parameter-reference)
10. [Scientific Methodology](#scientific-methodology)
11. [Reproducibility Checklist](#reproducibility-checklist)
12. [Known Limitations](#known-limitations)
13. [Related Tools](#related-tools)

---

## Overview

This pipeline evaluates LTE-based smartphone positioning accuracy by processing cellular signal logs through a structured multi-phase analysis framework. It supports three distinct distance estimation approaches (path loss with context-specific calibration, and regression-based formulas), and outputs comprehensive accuracy metrics (CEP, R95, RMSE) against ground truth GPS coordinates.

### Key Features

- **9-phase pipeline** spanning data extraction, signal analysis, distance estimation, and geolocation validation
- **Environment context support** (city, town, village, default) for model calibration
- **Dual distance methods**: Path loss models (context-tuned) + machine learning regression formulas
- **Weighted trilateration solver** with uncertainty propagation and GDOP filtering
- **Ground truth validation** with comprehensive accuracy reporting (CEP, R95, RMSE, per-cell analysis)
- **Batch processing** with multiple sessions and contexts
- **Reproducibility tracking**: Parameter logging, metadata generation, coverage reporting
- **Tower database validation** with multi-source reconciliation (CellMapper + official permits)

### Thesis Context

This pipeline was developed to evaluate LTE signal-based geolocation accuracy across Belgian urban and rural environments, comparing path loss models against machine-learned regression formulas. The work validates tower location quality (reconciling CellMapper with official telecom permits) and quantifies positioning errors under various signal conditions.

---

## System Requirements

### Python Environment

- **Python 3.14** (confirmed version; earlier 3.x versions may work but untested)
- **Windows** (primary development platform; Linux/macOS requires testing)

### Dependencies

```
pandas>=1.0.0
openpyxl>=3.0.0
scipy<=1.16.3
```

Install via:
```bash
pip install -r requirements.txt
```

### Disk Space

Expect ~10-50 MB per analysis depending on log file size and context combinations. Example from lln_5.txt + 5 subcontexts: ~20 MB output files.

### Execution Time

Typical single-run timing:
- Log parsing: 0.5-1.0s
- Signal extraction: 0.3-0.7s
- Distance estimation: 0.8-2.0s
- Bearing calculation: 1.0-2.5s
- Trilateration: 0.5-1.5s
- Validation: 0.2-0.5s
- **Total per run**: 3-8s (batch runs: 2-3s per context after initial parse)

---

## Installation

### 1. Clone/Download Repository

```bash
git clone <repository-url>
cd lte-geolocation-pipeline
```

### 2. Create Virtual Environment (Recommended)

```bash
python3.14 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Setup

```bash
python main.py
# Should output usage information with examples
```

### 5. Check Data Directory Structure

Ensure the following directories and files exist (pipeline will fail with helpful error messages if missing):

```
data/
  raw_logs/
    *.txt          # G-NetTrack Pro logs (e.g., lln_1.txt, ixelle_4.txt)
  formulas/
    *_formula.json # Regression model coefficients (e.g., lln_formula.json)
  tower_data/
    towers.json    # Tower location database (detailed format below)
    pci.json       # PCI to coordinate mapping (simplified lookup)
```

---

## Quick Start

### Single Run Example

```bash
# Default context (path loss, standard parameters)
python main.py lln_1.txt

# City context (optimized for urban signal propagation)
python main.py lln_2.txt city

# Formula mode (machine-learned distance estimation)
python main.py lln_5.txt formula

# Formula + specific bearing context
python main.py waha_1.txt formula town
```

### Batch Processing

```bash
# All logs, all contexts (4 runs × 4 contexts = 16 total)
python batch_run.py

# Specific prefix only
python batch_run.py --prefix lln

# Specific prefix + context
python batch_run.py --prefix ixelle --context city

# With formula variants (creates 20 runs: 4 regular + 4×4 formula)
python batch_run.py --prefix lln --include-formula

# Formula-only mode
python batch_run.py --prefix waha --formula-only --subcontext city
```

### Output Location

Results are organized by log file:

```
result/
  lln_data/
    lln_1_data/
      city/                    # Context-specific results
        parsed_lln_1_city.csv
        distance_estimates_lln_1_city.csv
        bearing_estimates_lln_1_city.csv
        trilateration_results_lln_1_city.csv
        validation_results_lln_1_city.csv
      formula/
        city/                  # Formula + bearing context
          distance_estimates_lln_1_formula_city.csv
          ...

reports/
  lln_reports/
    lln_1_reports/
      city/
        accuracy_summary_lln_1_city.txt
        coverage_report_lln_1_city.txt
      formula/
        city/
          accuracy_summary_lln_1_formula_city.txt
          ...
  batch_report/
    batch_run_lln_with_formula_city_ALL_YYYYMMDD_HHMMSS.txt
```

---

## Usage Guide

### Command Line Interface

#### `main.py` – Single Log Analysis

```bash
python main.py <log_file> [context] [subcontext]
```

**Arguments:**

| Argument | Options | Default | Purpose |
|----------|---------|---------|---------|
| `log_file` | `{prefix}_{n}.txt` | Required | Input log file (e.g., `lln_1.txt`) |
| `context` | `default`, `city`, `town`, `village`, `formula` | `default` | Environment model calibration |
| `subcontext` | `default`, `city`, `town`, `village` | N/A (formula only) | Bearing/trilateration context for formula mode |

**Examples:**

```bash
# Standard contexts (path loss models)
python main.py lln_1.txt               # → Default path loss model
python main.py lln_2.txt city          # → City path loss model
python main.py waha_1.txt village      # → Village path loss model

# Formula context (machine-learned distance)
python main.py lln_5.txt formula           # → Formula distance + default bearing
python main.py lln_5.txt formula town      # → Formula distance + town bearing
python main.py waha_2.txt formula city     # → Formula distance + city bearing
```

**Output:**
- CSV files in `result/{prefix}_data/{basename}_{context}/`
- Reports in `reports/{prefix}_reports/{basename}_{context}/`
- Parameter metadata in `analysis_metadata_{context}.json`

---

#### `batch_run.py` – Multi-Log Batch Processing

```bash
python batch_run.py [--prefix PREFIX] [--context CONTEXT] [--log LOGFILE] 
                    [--formula-only] [--include-formula] [--subcontext SC] [--dry-run]
```

**Options:**

| Option | Values | Purpose |
|--------|--------|---------|
| `--prefix` | `lln`, `ixelle`, `waha` | Filter logs by prefix (run only matching logs) |
| `--log` | `{prefix}_{n}.txt` | Run single specific log file |
| `--context` | `default`, `city`, `town`, `village` | Run only this context (no formula) |
| `--formula-only` | flag | Skip regular contexts; run only formula variants |
| `--include-formula` | flag | Run both regular contexts AND formula variants |
| `--subcontext` | `default`, `city`, `town`, `village` | Filter formula subcontexts |
| `--dry-run` | flag | Show configuration without executing |

**Examples:**

```bash
# All logs, all regular contexts (4 contexts × log count)
python batch_run.py

# Specific prefix
python batch_run.py --prefix lln

# Prefix + specific context
python batch_run.py --prefix ixelle --context city

# Single log, all contexts
python batch_run.py --log lln_5.txt

# Regular + formula (double runs)
python batch_run.py --prefix lln --include-formula

# Formula only
python batch_run.py --prefix waha --formula-only

# Formula-only + specific subcontext
python batch_run.py --prefix lln --formula-only --subcontext city

# Dry run (preview without executing)
python batch_run.py --prefix lln --dry-run
```

**Output:**
- All phase outputs for each log × context combination
- Aggregated summary: `reports/batch_report/batch_run_{session}_{mode}_{context}_{YYYYMMDD_HHMMSS}.txt`
- Accuracy aggregation across all contexts

---

### Report Interpretation

#### Accuracy Summary (`accuracy_summary_{context}.txt`)

Key metrics explained:

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **CEP (50th %)** | Circular error probability | Radius containing 50% of errors; primary accuracy measure |
| **R95 (95th %)** | 95th percentile error | Radius containing 95% of estimates (worst-case typical error) |
| **RMSE** | Root mean square error | Sensitive to large outliers; useful for model comparison |
| **Std Dev** | Standard deviation of errors | Measure of consistency |
| **GDOP < 8** | Geometric dilution of precision | % of solutions with acceptable geometry; higher is better |
| **Within Uncertainty** | % of errors ≤ 200m | Fraction of estimates satisfying uncertainty bounds |

**Quality Interpretation:**

```
CEP < 100m     → Excellent positioning
CEP 100-300m   → Good positioning
CEP > 300m     → Fair positioning (distance model calibration issues likely)
CEP > 500m     → Poor positioning (verify tower coordinates and signal quality)
```

#### Coverage Report (`coverage_report_{context}.txt`)

Verifies database completeness:

| Section | Meaning |
|---------|---------|
| **SERVING CELLS** | Primary cellular connection(s) in log |
| **NEIGHBOR CELLS** | Secondary cells (triangulation inputs) |
| **Coverage %** | Fraction of cells found in tower database |

**Recommended:** Coverage ≥ 80% for reliable trilateration.

#### Distance Validation (`Accuracy_distance_cell_summary_{context}.txt`)

Per-cell distance estimation accuracy (Phase 2.2.5):

- MAE/RMSE in meters (path loss) or dB (formula log-space regression)
- Cell-specific performance identifies problem towers
- High RMSE indicates poor signal model fit or tower location errors

---

## Data Provenance

### Input Data: Raw Logs (`data/raw_logs/*.txt`)

**Format:** Tab-separated with 100+ fields per record

**Collection Method:**
- **App:** G-NetTrack Pro v375 (Android)
- **Configuration:** See [G-NetTrack Pro Setup](#g-nettrack-pro-configuration) below
- **Device:** OnePlus A6013 
- **OS:** Android (BP1A.250505.005 variant)

**Key Fields (Parsed):**

| Field | Type | Description |
|-------|------|-------------|
| `Timestamp` | ISO 8601 | `YYYY.MM.DD_HH.MM.SS` format |
| `Latitude` | Float (WGS84) | Device GPS latitude (EXIF ground truth) |
| `Longitude` | Float (WGS84) | Device GPS longitude (EXIF ground truth) |
| `ARFCN` / `EARFCN` | Integer | E-UTRA Absolute Radio Frequency Channel Number |
| `CellID` / `RawCellID` | Integer | 28-bit eNB-Cell ID |
| `Level` / `RSRP` | Integer (dBm) | Reference Signal Received Power (-140 to -50 dBm typical) |
| `Qual` / `RSRQ` | Integer (dB) | Reference Signal Received Quality (-20 to 0 dB typical) |
| `PSC` / `PCI` | Integer (0-503) | Physical Cell Identity |
| `NodeID` / `eNB_ID` | Integer | Evolved Node-B identifier |
| Neighbor fields (N1-N18) | Repeated | `NTech`, `NCellName`, `NCellID`, `NARFCN`, `NRxLev`, `NQual`, etc. |

**Data Quality Notes:**

- Timestamps span multiple days; no post-processing (raw device output)
- Ground truth (Lat/Lon) from smartphone GPS with typical CEP ~5-10m accuracy
- RSRP/RSRQ vary ±3-5 dB between consecutive measurements (propagation fading)
- Some records may have RSRP < -120 dBm (very weak signal); filtered during analysis if below threshold
- Neighbor cells limited to 18 visible (hardware constraint)

---

### Tower Database (`data/tower_data/towers.json`)

**Format:** JSON, keyed by Cell ID (RAWCELLID)

**Construction Method (Multi-Source):**

1. **Initial Data:** CellMapper open database
2. **Quality Check:** Haversine distance validation against 50 manually geolocated antennas (Google Street View + on-site inspection)
   - Initial CellMapper RMSE: 1166 m (median: 471 m, max: 3153 m)
   - Outliers > 1000 m flagged for remediation
3. **Official Source Integration:** Belgian/Walloon telecom antenna permits
   - Brussels: `geodata.environnement.brussels` (urban coordinates)
   - Wallonia: `geoportail.wallonie.be` (regional coordinates)
4. **Matching Process:** CellMapper eNBs matched to official permits via:
   - Frequency band alignment
   - Azimuth consistency (±30°)
   - Installation date alignment (CellMapper first-seen ≥ official approval date)
   - Address/municipality confirmation
5. **Population:** Corrected (lat, lon) assigned; other fields (PCI, EARFCN, azimuth) retained from CellMapper

**Data Structure:**

```json
{
  "103059557": {
    "enb_id": 402576,
    "enb_type": "Macro",
    "system_type": "LTE",
    "mcc": 206,
    "mnc": 4,
    "region": 4210,
    "latitude": 50.669576891,
    "longitude": 4.616264081,
    "pci": 277,
    "pci_structure": "?/?",
    "band": 7,
    "earfcn": 2850,
    "maximum_signal_rsrp": "-95 dBm",
    "direction": "NE",
    "azimuth": 65,
    "max_rsrq": "-8 dB",
    "bandwidth_mhz": 20,
    "uplink_frequency": "2510 MHz",
    "downlink_frequency": "2630 MHz",
    "frequency_band": "IMT-E (B7 FDD)",
    "first_seen": "17/11/2023",
    "last_seen": "07/08/2025"
  }
}
```

**Guaranteed Fields:** `latitude`, `longitude`, `pci`, `band`, `earfcn`, `azimuth`

**Data Quality:**
- **Position accuracy:** Official permits typically ±50-200m precision (building/rooftop level)
- **Residual errors:** Antenna panel offsets (±50m), permit coordinate imprecision
- **Coverage:** ~87.5% of cells in analysis have coordinates; missing cells flagged in coverage reports
- **Validation:** Manual review of > 1000m error cases; confirmed improvements over raw CellMapper

---

### PCI Mapping Database (`data/tower_data/pci.json`)

**Format:** JSON, keyed by PCI (0-503 global identifier)

**Purpose:** Lightweight lookup for cells without full tower data; derived from `towers.json`

**Data Structure:**

```json
{
  "277": {
    "pci": "277",
    "latitude": 50.669576891,
    "longitude": 4.616264081,
    "band": 7,
    "operator": 4,
    "frequency_mhz": "2630 MHz"
  }
}
```

**Usage:** Pipeline prefers `towers.json`; falls back to `pci.json` if cell not found in full database.

---

### Formula Models (`data/formulas/*_formula.json`)

**Format:** JSON regression coefficients and performance metrics

**Source:** Generated by separate `formula_finder` training pipeline (see [Related Tools](#related-tools))

**Training Data:** `parsed_*.csv` files from initial pipeline runs (nonlinear regression on log-distance targets)

**Data Structure:**

```json
{
  "context": "lln",
  "model": "I2 logd ~ rsrp + rsrq + earfcn_k + rsrp_x_rsrq",
  "description": "Nonlinear regression model with interaction term rsrp * rsrq",
  "features": ["rsrp", "rsrq", "earfcn_k", "rsrp_x_rsrq"],
  "coefficients": {
    "intercept": 2.6682636537666475,
    "coef_rsrp": 0.0013441275505445324,
    "coef_rsrq": 0.007880858746604984,
    "coef_earfcn_k": -0.00031936501931168984,
    "coef_rsrp_x_rsrq": 7.146324117282298e-05
  },
  "performance_metrics": {
    "mae_mean_m": 137.495623153659,
    "rmse_mean_m": 160.59738245958192,
    "mape_mean": 0.8158741723822512,
    "folds": 5,
    "train_samples": 1071
  },
  "notes": "earfcn_k = EARFCN / 1000. Interaction term: rsrp_x_rsrq = rsrp * rsrq"
}
```

**Key Metrics:**
- **MAE:** Mean absolute error in meters (linear distance space)
- **RMSE:** Root mean square error in meters
- **MAPE:** Mean absolute percentage error (%)
- **Folds:** K-fold cross-validation folds (5 typical)
- **Train Samples:** Number of observations in training set

**Interpretation:**
- Training MAE ~137m, RMSE ~161m indicates reasonable formula fit
- Compare validation CEP against training RMSE to assess generalization
- MAPE > 1.0 suggests heavy-tailed error distribution (outliers)

---

### G-NetTrack Pro Configuration

**Mobile Data Collection Setup (Android):**

1. Install `gnettrackpro.375.apk` (bypass Google Play Protect warning if required)
2. Launch application; grant **Always Allow** location permission
3. Navigate to **Settings > Log Parameters:**
   - Timer Interval: **1s** (one record per second)
   - Include IMSI: **Enabled** (SIM card info)
   - Include IMEI: **Enabled** (device identifier)
   - Include MSISDN: **Enabled** (phone number)
   - Verbose Log: **Enabled** (detailed signal fields)
   - Write Cell Info in Log: **Enabled** (serving + neighbor cells)
   - Ask for Log Name: **Enabled** (custom filename per session)

4. Start logging at collection location; walk designated route (15-30 min typical)
5. Export log file as `.txt` (tab-separated format) to `data/raw_logs/{prefix}_{n}.txt`

**Log Format Notes:**
- Timestamps in `YYYY.MM.DD_HH.MM.SS` format (not ISO 8601; pipeline auto-converts)
- 100+ tab-separated columns; pipeline extracts only essential fields
- Neighbor cells: Up to 18 visible cells (LTE limitation)
- GPS coordinates: From device receiver (5-50m CEP typical, phone-dependent)

---

## Pipeline Architecture

The pipeline orchestrates 9 sequential phases spanning data extraction, signal analysis, and validation:

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA EXTRACTION                                        │
├─────────────────────────────────────────────────────────────────┤
│ 1.1  Parse network logs (g-nettrack format)                     │
│ 1.2  Verify tower database coverage (serving + neighbor cells)  │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: SIGNAL ANALYSIS                                        │
├─────────────────────────────────────────────────────────────────┤
│ 2.1  Extract signal strength data (RSRP, RSRQ, per-cell)        │
│ 2.2  Estimate distances from signal (path loss or formula)      │
│ 2.2.5 Validate distance estimates vs ground truth (NEW!)        │
│ 2.3  Estimate bearing angles from tower positions               │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: GEOLOCATION SOLVING & VALIDATION                       │
├─────────────────────────────────────────────────────────────────┤
│ 3.1  Prepare trilateration inputs (cell filtering, weighting)   │
│ 3.2  Solve device positions (weighted trilateration, GDOP)      │
│ 3.3  Validate against ground truth (CEP, R95, RMSE, per-cell)   │
└─────────────────────────────────────────────────────────────────┘
```

### Phase Details

#### Phase 1.1: Parse Network Logs

**Input:** `data/raw_logs/{log_file}.txt` (G-NetTrack Pro tab-separated format)

**Processing:**
- Extract timestamp, device position (GPS), serving cell, neighbor cells
- Normalize EARFCN/ARFCN field names
- Convert timestamp to ISO 8601 format
- Filter rows with invalid coordinates or missing serving cell

**Output:** `result/{prefix}_data/{basename}_{context}/parsed_{basename}_{context}.csv`

**Columns:** `timestamp`, `latitude`, `longitude`, `earfcn`, `serving_pci`, `serving_level`, `serving_qual`, `serving_cgi`, `serving_node`, ...

---

#### Phase 1.2: Verify Database Coverage

**Input:**
- `result/parsed_*.csv` (parsed log)
- `data/tower_data/towers.json` (full tower database)
- `data/tower_data/pci.json` (PCI lookup)

**Processing:**
- Count serving cells found in towers.json
- Count neighbor cells found in towers.json + pci.json
- Flag missing cells for data collection
- Calculate coverage percentage

**Output:**
- `reports/{prefix}_reports/{basename}_{context}/coverage_report_{basename}_{context}.txt`
- `reports/{prefix}_reports/{basename}_{context}/towers_status_{basename}_{context}.json`

**Typical Result:** 80-100% serving cell coverage; 60-90% neighbor coverage

---

#### Phase 2.1: Extract Signal Strength Data

**Input:** `result/parsed_*.csv`

**Processing:**
- Group by timestamp
- Extract serving cell RSRP (signal strength) → Level (dBm)
- Extract serving cell RSRQ (signal quality) → Qual (dB)
- Extract neighbor cell RSRP for trilateration
- Create per-cell metadata (EARFCN, PCI, band)

**Output:**
- `result/{prefix}_data/{basename}_{context}/signal_data_{basename}_{context}.csv`
- `result/{prefix}_data/{basename}_{context}/signal_metadata_{basename}_{context}.csv`
- `result/{prefix}_data/{basename}_{context}/neighbor_signals_{basename}_{context}.csv`

**Columns (signal_data):** `timestamp`, `serving_pci`, `rsrp_dbm`, `rsrq_db`, `earfcn`, `band`, `operator`, ...

---

#### Phase 2.2: Estimate Distances from Signal

**Input:**
- `result/signal_data_*.csv`
- `data/tower_data/towers.json`, `pci.json`

**Method A: Path Loss Model (default, city, town, village contexts)**

Distance estimated from RSRP using context-specific path loss formula:

```
d (m) = 10^((RSRP_0dBm - RSRP_measured) / (10 * n) + h(RSRQ))
```

Where:
- **RSRP_0dBm:** Reference power at 1m (~-40 dBm typical)
- **n:** Path loss exponent (2-4, context-dependent)
- **h(RSRQ):** RSRQ-based offset adjustment (uncertainty calibration)

**Uncertainty Assignment (Phase 2.2 Enhancement):**

| RSRP Level | Category | Max Uncertainty |
|-----------|----------|-----------------|
| > -85 dBm | EXCELLENT | 80 m |
| -85 to -100 dBm | HIGH | 100 m |
| -100 to -110 dBm | GOOD | 120 m |
| -110 to -120 dBm | MEDIUM | 150 m |
| -120 to -130 dBm | WEAK | 250 m |
| < -130 dBm | VERY_WEAK | 350 m |

**Method B: Formula-Based Regression (formula context)**

Distance estimated from nonlinear regression on log-space targets:

```
log10(d) = intercept + coef_rsrp * rsrp + coef_rsrq * rsrq + 
           coef_earfcn_k * (earfcn/1000) + coef_rsrp_x_rsrq * (rsrp * rsrq)
d = 10^(log10_d)
```

Coefficients loaded from `data/formulas/{prefix}_formula.json` during formula mode execution.

**Output:** `result/{prefix}_data/{basename}_{context}/distance_estimates_{basename}_{context}.csv`

**Columns:** `timestamp`, `serving_pci`, `distance_m`, `distance_uncertainty_m`, `quality_level`, `method`, ...

---

#### Phase 2.2.5: Distance Ground Truth Validation (NEW!)

**Input:**
- `result/distance_estimates_*.csv`
- `result/ground_truth_*.csv` (device GPS positions)
- Tower coordinate database

**Processing:**
- For each distance estimate: compute Haversine distance between tower and device
- Calculate error: |estimated_distance - true_distance|
- Report MAE, RMSE per cell
- Identify problematic towers (high RMSE)

**Output:** `reports/{prefix}_reports/{basename}_{context}/Accuracy_distance_cell_summary_{basename}_{context}.txt`

**Purpose:** Validate distance model quality before trilateration; identify systematic biases or tower location errors.

---

#### Phase 2.3: Estimate Bearing Angles

**Input:**
- `result/distance_estimates_*.csv`
- Tower database (latitude, longitude, azimuth)

**Processing:**
- For each cell: compute azimuth from device to tower
- Adjust for tower antenna azimuth (directional gain)
- Score bearing confidence based on RSRQ (higher RSRQ → higher confidence)
- Filter unreliable bearings (RSRQ < threshold)

**Output:** `result/{prefix}_data/{basename}_{context}/bearing_estimates_{basename}_{context}.csv`

**Columns:** `timestamp`, `pci`, `azimuth`, `confidence_score`, `rsrq_db`, ...

---

#### Phase 3.1: Prepare Trilateration Input

**Input:**
- `result/distance_estimates_*.csv`
- `result/bearing_estimates_*.csv`
- Algorithm parameters (min/max cells, RSRP threshold)

**Processing:**
1. Group by timestamp
2. Filter cells by RSRP threshold (default: -110 dBm minimum)
3. Rank cells by signal strength (RSRP descending)
4. Keep top N cells (default: 4-8) for trilateration
5. Calculate per-cell weight: `weight = 1 / (uncertainty_m)^2`
6. Validate minimum cell count (default: 4 cells required)

**Output:** `result/{prefix}_data/{basename}_{context}/trilateration_input_{basename}_{context}.csv`

**Columns:** `timestamp`, `cell_count`, `pci_1`, `distance_1_m`, `uncertainty_1_m`, `weight_1`, `azimuth_1`, ...

---

#### Phase 3.2: Solve Device Positions (Weighted Trilateration)

**Input:** `result/trilateration_input_*.csv` + tower coordinates

**Algorithm:** Weighted iterative trilateration

1. **Initialization:** Start with centroid of tower positions
2. **Weighting:** `w_i = 1 / u_i^2` where u_i = distance uncertainty
3. **Cost Function:** Minimize residual:
   ```
   cost = Σ w_i * (||P - T_i|| - d_i)^2
   ```
4. **Solver:** Scipy least-squares optimizer
5. **Iteration:** Repeat until convergence (< 0.5m change) or max 100 iterations
6. **GDOP Filtering:** Reject solutions with GDOP > 8.0 (geometry quality)
7. **Residual:** Compute final position error vs. distances

**Output:** `result/{prefix}_data/{basename}_{context}/trilateration_results_{basename}_{context}.csv`

**Columns:** `timestamp`, `est_latitude`, `est_longitude`, `num_cells`, `gdop`, `residual_m`, ...

---

#### Phase 3.3: Ground Truth Validation

**Input:**
- `result/trilateration_results_*.csv` (estimated positions)
- `result/ground_truth_*.csv` (device GPS positions)

**Processing:**
- Compute Haversine distance: error_m = distance(estimated, ground_truth)
- Calculate percentile statistics:
  - **CEP:** 50th percentile (median error)
  - **R95:** 95th percentile (worst-case typical)
  - **RMSE:** Root mean square error
  - **Mean/Std:** Bias and consistency
- Per-cell accuracy: errors grouped by contributing cell
- Per-timestamp accuracy: errors vs. timestamp patterns
- Quality categorization: EXCELLENT (CEP < 100m), GOOD (100-300m), FAIR (300-500m), POOR (> 500m)

**Output:**
- `result/{prefix}_data/{basename}_{context}/validation_results_{basename}_{context}.csv`
- `reports/{prefix}_reports/{basename}_{context}/accuracy_summary_{basename}_{context}.txt` (human-readable)

**Columns (validation_results):** `timestamp`, `est_lat`, `est_lon`, `truth_lat`, `truth_lon`, `error_m`, `within_uncertainty`, `num_cells`, `gdop`, `quality_flag`, ...

---

## Input/Output Specifications

### Input Files

| File | Format | Required | Key Fields |
|------|--------|----------|-----------|
| `data/raw_logs/{prefix}_{n}.txt` | Tab-separated (G-NetTrack Pro) | ✓ | Timestamp, Lat, Lon, EARFCN, Level, Qual, PSC, Neighbor cells (N1-N18) |
| `data/tower_data/towers.json` | JSON (Cell ID keyed) | ✓ | `latitude`, `longitude`, `pci`, `band`, `earfcn`, `azimuth`, `enb_id` |
| `data/tower_data/pci.json` | JSON (PCI keyed) | ✓ | `latitude`, `longitude`, `pci`, `band`, `operator` |
| `data/formulas/{prefix}_formula.json` | JSON (regression coefficients) | ✓ (formula mode only) | `coefficients`, `features`, `performance_metrics` |

### Output Files (per log × context)

| Phase | File | Rows | Key Columns |
|-------|------|------|-------------|
| 1.1 | `parsed_{basename}_{context}.csv` | 1 per timestamp | `timestamp`, `latitude`, `longitude`, `serving_pci`, `serving_level`, `serving_qual` |
| 2.1 | `signal_data_{basename}_{context}.csv` | N × cells | `timestamp`, `pci`, `rsrp_dbm`, `rsrq_db`, `earfcn`, `band` |
| 2.1 | `signal_metadata_{basename}_{context}.csv` | N × cells | `pci`, `latitude`, `longitude`, `band`, `frequency_mhz` |
| 2.1 | `neighbor_signals_{basename}_{context}.csv` | N × neighbor cells | `timestamp`, `pci`, `rsrp_dbm`, `rsrq_db` |
| 2.2 | `distance_estimates_{basename}_{context}.csv` | N × cells | `timestamp`, `pci`, `distance_m`, `uncertainty_m`, `quality_level` |
| 2.3 | `bearing_estimates_{basename}_{context}.csv` | N × cells | `timestamp`, `pci`, `azimuth`, `confidence_score`, `rsrq_db` |
| 3.1 | `trilateration_input_{basename}_{context}.csv` | N (timestamps) | `timestamp`, `cell_count`, `pci_1..8`, `distance_1..8_m`, `uncertainty_1..8_m` |
| 3.2 | `trilateration_results_{basename}_{context}.csv` | N (timestamps) | `timestamp`, `est_latitude`, `est_longitude`, `num_cells`, `gdop`, `residual_m` |
| 3.3 | `validation_results_{basename}_{context}.csv` | N (timestamps) | `timestamp`, `est_lat`, `est_lon`, `truth_lat`, `truth_lon`, `error_m`, `gdop`, `quality_flag` |

### Report Files

| Report | Content | Audience |
|--------|---------|----------|
| `coverage_report_{context}.txt` | Cell database coverage (%) + missing cell list | Data QA; identifies collection gaps |
| `towers_status_{context}.json` | Per-cell status (found/missing, data source) | Machine-readable coverage metadata |
| `Accuracy_distance_cell_summary_{context}.txt` | Per-cell distance MAE/RMSE + problem towers | Model validation; tower location QA |
| `accuracy_summary_{context}.txt` | CEP, R95, RMSE, per-cell accuracy, parameters | Main results; thesis reporting |
| `analysis_metadata_{context}.json` | Algorithm parameters (machine-readable) | Reproducibility; methodology traceability |
| `batch_run_*.txt` | Batch execution log (timing, success/failure) | Execution tracking |

---

## Parameter Reference

### Algorithm Parameters (Context-Specific)

All parameters are context-specific and loaded from `src/config/algorithm_params.py`. Key parameters:

#### Phase 2.2: Distance Estimation

| Parameter | Default (City) | Default (Village) | Default (Town) | Justification |
|-----------|---|---|---|---|
| `RSRP_QUALITY_THRESHOLD_DBM` | -110 | -110 | -110 | Exclude very weak signals (RSRP < -110 introduces high estimation error) |
| `MIN_UNCERTAINTY_M` | 30 | 30 | 30 | Minimum bound ensures uncertainty never artificially small |
| `MAX_UNCERTAINTY_EXCELLENT` | 80 | 100 | 90 | RSRP > -85 dBm: near-tower excellent signal (low variability) |
| `MAX_UNCERTAINTY_HIGH` | 100 | 120 | 110 | RSRP -85 to -100: good urban signal (moderate path loss variance) |
| `MAX_UNCERTAINTY_GOOD` | 120 | 140 | 130 | RSRP -100 to -110: degraded signal (higher variance) |
| `MAX_UNCERTAINTY_MEDIUM` | 150 | 180 | 170 | RSRP -110 to -120: weak signal (multipath, NLOS effects) |
| `MAX_UNCERTAINTY_WEAK` | 250 | 280 | 270 | RSRP -120 to -130: very weak (edge of coverage, unreliable) |
| `MAX_UNCERTAINTY_VERY_WEAK` | 350 | 400 | 380 | RSRP < -130: extreme edge (shadowing, far field) |

**Rationale:**
- **City parameters:** Tighter (more aggressive) uncertainty due to dense tower deployment → shorter distances, less propagation variance
- **Village parameters:** Relaxed (conservative) uncertainty due to sparse towers → longer distances, greater multipath/NLOS effects
- **Town parameters:** Intermediate between city and village

#### Phase 2.3: Bearing Estimation

| Parameter | Default | Justification |
|-----------|---------|---|
| `RSRQ_SCORE_THRESHOLD_10DB` | 50 points | RSRQ ≥ 10 dB: excellent signal quality (low fading, reliable angle) |
| `RSRQ_SCORE_THRESHOLD_5DB` | 30 points | RSRQ 5-10 dB: good quality (some fading, acceptable) |
| `RSRQ_SCORE_THRESHOLD_NEG5DB` | 20 points | RSRQ -5 to 5 dB: fair quality (multipath, variable) |
| `BEARING_CONFIDENCE_THRESHOLD` | Configured | RSRQ < -5: exclude from trilateration (bearing unreliable) |

**Note:** Bearing confidence not directly exposed; controlled via RSRQ filtering in Phase 3.1.

#### Phase 3.1: Trilateration Input Preparation

| Parameter | Default | Justification |
|-----------|---------|---|
| `MIN_CELLS_REQUIRED` | 4 | Minimum cells for 2D trilateration (3 theoretically sufficient, 4 provides redundancy) |
| `MAX_CELLS_TO_KEEP` | 8 | Cap on cells to limit computational cost + reduce weak signal weighting (keep strongest signals) |
| `RSRP_QUALITY_THRESHOLD_DBM` | -110 | Exclude very weak cells (see Phase 2.2) |

#### Phase 3.2: Trilateration Solver

| Parameter | Default | Justification |
|-----------|---------|---|
| `WEIGHT_BY_UNCERTAINTY_POWER` | 2.0 | Weight = 1 / uncertainty^2; power=2 gives inverse-squared weighting (stronger signal more trusted) |
| `CONVERGENCE_THRESHOLD_M` | 0.5 | Stop iteration when change < 0.5m (meter-level precision) |
| `MAX_ITERATIONS` | 100 | Safety limit on iterations (typically converges in 3-5 iterations) |
| `MAX_GDOP_ACCEPTED` | 8.0 | Reject solutions with GDOP > 8 (geometry quality too poor; geometry degenerate) |
| `RESIDUAL_WEIGHT` | 0.1 | Weight given to final distance residual in cost function (avoid over-fitting to outlier cells) |

**GDOP Interpretation:**
- GDOP < 5: Excellent geometry (tight constraint)
- 5-8: Good geometry (acceptable)
- > 8: Poor geometry (satellite/cell configuration unfavorable; solution unreliable)

#### Formula Mode Parameters

**Distance Method:** `formula` (instead of `path_loss`)

**Distance Calculation:**
```
log10(d_m) = intercept + Σ coef_i * feature_i
d_m = 10^(log10_d)
```

**Features:**
- `rsrp`: RSRP in dBm
- `rsrq`: RSRQ in dB
- `earfcn_k`: EARFCN / 1000
- `rsrp_x_rsrq`: RSRP × RSRQ interaction term

**Bearing/Trilateration Context:** Specified via `[subcontext]` argument; uses context-specific parameters from Phase 2.3, 3.1, 3.2

---

### Validation Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `UNCERTAINTY_THRESHOLD_M` | 200 | Distance within this threshold considered "within uncertainty" |
| `CEP_QUALITY_EXCELLENT` | 100 | CEP < 100m = excellent |
| `CEP_QUALITY_GOOD` | 300 | 100 ≤ CEP < 300m = good |
| `CEP_QUALITY_FAIR` | 500 | 300 ≤ CEP < 500m = fair |
| `CEP_QUALITY_POOR` | ∞ | CEP ≥ 500m = poor |

---

## Scientific Methodology

### Assumptions & Constraints

#### Signal Propagation Model

1. **Path Loss Exponent (n):** Assumes 2.0 (urban) to 4.0 (rural)
   - Derived from empirical measurements; does not account for diffraction/reflection
   - Invalid in deep urban canyons or tunnels (NLOS > 50%)

2. **Antenna Gain:** Assumes omnidirectional reception at phone
   - Reality: smartphone antennas have 3-5 dB directional gain variations
   - Impact: ±15% distance estimation error under worst-case orientation

3. **Multipath/Fading:** Averaged over 1-second windows
   - Rayleigh fading at ±3 dB typical; outliers (>10 dB swings) not filtered
   - Impacts: RMSE inflated by ~20-30% relative to line-of-sight baseline

#### Ground Truth Validation

1. **GPS Accuracy:** Device receiver, typical CEP 5-15m
   - Reported accuracy field not used (unreliable)
   - Impact: Introduces baseline error floor (~5m) into validation metrics

2. **Time Synchronization:** Assumed synchronized within 1s
   - Device clock may drift ±1-2s over multi-hour session
   - Impact: Potential ±1-2 cell association errors in dense coverage

3. **Stationary Assumption:** Each timestamp treated as independent position
   - Reality: Device moves ~1-2 m/s during collection
   - Impact: Estimated positions may lag ground truth by ±2m; temporal correlation ignored

#### Tower Location Quality

1. **Coordinate Source:** Official permits preferred; CellMapper fallback
   - Official permits: ±50-200m precision (building-level, not antenna-specific)
   - CellMapper: RMSE 471m initially; reconciled to <300m after manual validation
   - Impact: ~100-150m systematic error in tower location is expected residual

2. **Azimuth Accuracy:** CellMapper + validation data; ±30° typical accuracy
   - Impact: Bearing estimates have ±15-20° uncertainty; non-negligible in tight geometry

#### Bearing Estimation

1. **Linear Bearing:** Straight-line azimuth from device to tower
   - Ignores: antenna gain pattern, multipath, Fresnel zone blockage
   - Impact: Bearing errors ±10-30° common in urban areas

2. **RSRQ Weighting:** Assumes RSRQ monotonically correlates with bearing reliability
   - Reality: RSRQ affected by interference, not just bearing accuracy
   - Impact: Some high-RSRQ signals may have poor bearing due to reflected path dominance

### Threats to Validity

| Threat | Severity | Mitigation |
|--------|----------|-----------|
| Tower location errors (±100-200m) | **HIGH** | Manual validation + official source reconciliation; residual documented |
| GPS ground truth accuracy (±5-15m) | **MEDIUM** | Use CEP/R95 (robust to outliers); report mean ± std |
| NLOS propagation (non-random error) | **HIGH** | Compare urban vs. rural; context-specific models account for bias |
| Multipath fading (measurement noise) | **MEDIUM** | 1-second averaging + RSRQ filtering; residual ~20-30% inflation |
| Device antenna orientation | **MEDIUM** | Treat as noise; report variability; recommend averaging over routes |
| Sample size bias | **LOW** | Collect 5+ sessions per location; aggregate across 15+ timestamps/session |
| Temporal correlation | **LOW** | Treat timestamps as independent; check autocorrelation in residuals |
| Formula generalization | **MEDIUM** | Cross-validate on held-out test set; report MAPE; compare to path loss |

### Reproducibility Checklist

- [ ] **Environment Setup**
  - [ ] Python 3.14 installed (`python3.14 --version`)
  - [ ] Virtual environment created and activated
  - [ ] Dependencies installed (`pip install -r requirements.txt` exits 0)

- [ ] **Data Preparation**
  - [ ] `data/raw_logs/` contains at least one `{prefix}_{n}.txt` file
  - [ ] `data/tower_data/towers.json` exists and is valid JSON
  - [ ] `data/tower_data/pci.json` exists and is valid JSON
  - [ ] For formula mode: `data/formulas/{prefix}_formula.json` exists for each prefix

- [ ] **Single Run Verification**
  - [ ] Execute: `python main.py {prefix}_{n}.txt` (no arguments)
  - [ ] Check exit code: should be 0 (success)
  - [ ] Verify output files exist:
    - `result/{prefix}_data/{basename}_default/parsed_*.csv`
    - `result/{prefix}_data/{basename}_default/distance_estimates_*.csv`
    - `reports/{prefix}_reports/{basename}_default/accuracy_summary_*.txt`
  - [ ] Check accuracy_summary.txt contains CEP/R95/RMSE metrics

- [ ] **Context-Specific Runs**
  - [ ] Execute each context: `python main.py {log} city`, `...town`, `...village`
  - [ ] Verify output directories created correctly
  - [ ] Compare CEP across contexts (expect variance due to model tuning)

- [ ] **Formula Mode Verification**
  - [ ] Execute: `python main.py {log} formula`
  - [ ] Check for formula loading message: `"Loaded formula coefficients..."`
  - [ ] Verify output path: `result/.../formula_default/`
  - [ ] Execute with subcontext: `python main.py {log} formula city`
  - [ ] Verify output path: `result/.../formula_city/`

- [ ] **Batch Runs**
  - [ ] Dry-run: `python batch_run.py --prefix {prefix} --dry-run`
  - [ ] Verify configuration matches intention
  - [ ] Execute batch: `python batch_run.py --prefix {prefix}`
  - [ ] Check batch report: `reports/batch_report/batch_run_*.txt`
  - [ ] Verify aggregation: `reports/{prefix}_reports/{basename}_reports/batch_accuracy_summary_*.txt`

- [ ] **Data Integrity**
  - [ ] All CSV outputs: row counts > 0, no silent failures
  - [ ] Coverage report: serving cell coverage ≥ 80%
  - [ ] Accuracy summary: CEP, R95, RMSE all finite (no NaN/Inf)
  - [ ] No exceptions or ERROR messages in console output

- [ ] **Reproducibility Validation**
  - [ ] Re-run same log × context; compare outputs
  - [ ] Verify binary reproducibility: checksums match (within float precision)
  - [ ] Check parameter metadata: `analysis_metadata_{context}.json` contains all algorithm parameters
  - [ ] Document Git commit SHA in results for thesis methodology section

---

## Known Limitations

### Data Limitations

1. **Tower Database Incompleteness**
   - ~12-15% of cells lack full coordinate data
   - Mitigation: `coverage_report_*.txt` flags missing cells; can supplement with CellMapper
   - Impact: Trilateration may fail if >1 contributing cell is missing

2. **Neighbor Cell Visibility**
   - LTE hardware limits to ~18 visible cells; nearby strong cells may not be reported
   - Mitigation: Use only top cells by RSRP (strongest signals most reliable)
   - Impact: May under-constrain trilateration; results more sensitive to noise

3. **GPS Ground Truth Accuracy**
   - Device receiver: ±5-15m CEP typical (smartphone-dependent)
   - Mitigation: Use relative accuracy (CEP, R95) rather than absolute; collect multiple sessions
   - Impact: Validation metrics have ~5m floor; cannot resolve <5m estimation errors

### Algorithm Limitations

4. **Path Loss Model Simplicity**
   - Assumes free-space propagation with fixed exponent per context
   - Invalid in NLOS (>50% shadowing), tunnels, dense urban canyons
   - Mitigation: Use formula mode (learns empirical patterns); apply context-specific calibration
   - Impact: NLOS areas show 200-500m CEP errors; urban bias systematic

5. **Formula Generalization**
   - Trained on specific dataset (location-time); may not generalize to new areas/times
   - Mitigation: Cross-validate; compare formula vs. path loss CEP; report MAPE
   - Impact: Formula may perform worse than path loss in unseen environments

6. **Bearing Estimation Reliability**
   - Ignores multipath + fading; RSRQ not perfect proxy for angle accuracy
   - Mitigation: Filter low-RSRQ signals (< -5 dB); use only 4+ strong cells
   - Impact: Bearing errors ±10-30° common; tight geometry (>50° angular spread) required

7. **Trilateration Instability**
   - Underdetermined geometry (GDOP > 8) leads to non-unique solutions
   - Mitigation: Apply GDOP filter; require ≥4 cells; weight by uncertainty
   - Impact: ~5-10% of solutions rejected; those with poor geometry unreliable

### Operational Limitations

8. **Computational Performance**
   - Single run: 3-8s (varies by log size); batch: 2-3s per context
   - Large datasets (>10,000 timestamps): may require >1 hour
   - Mitigation: Parallelize batch runs; pre-filter logs by location
   - Impact: Exploratory analysis limited to moderate dataset sizes

9. **Hard Dependencies**
   - Windows compatibility uncertain (developed on Linux/macOS)
   - Mitigation: Use Docker or WSL; test on target platform early
   - Impact: May require dev environment setup time

---

## Related Tools

### formula_finder (Regression Model Training)

**Repository:** [Same GitHub, separate tool]

**Purpose:** Generate `*_formula.json` regression coefficients for new datasets

**Usage:**

```bash
# Train regression on parsed_*.csv outputs from pipeline
python formula_finder.py --input result/lln_data/lln_1_data/parsed_lln_1_default.csv \
                          --output data/formulas/lln_formula.json \
                          --model "logd ~ rsrp + rsrq + earfcn_k + rsrp_x_rsrq"
```

**Features:**
- K-fold cross-validation
- Automatic feature engineering (interaction terms, scaling)
- Performance metrics (MAE, RMSE, MAPE, per-fold breakdown)
- Serialization to JSON for pipeline loading

**Inputs:** `parsed_*.csv` from Phase 1.1 (any context; aggregatable)

**Outputs:** `*_formula.json` (ready for `python main.py {log} formula`)

**See:** `formula_finder/README.md` for full documentation

---

## Support & Troubleshooting

### Common Errors

**Error: "No log files found in data/raw_logs/"**
- Cause: Missing `.txt` files or incorrect directory
- Fix: Ensure `data/raw_logs/{prefix}_{n}.txt` exists; check file permissions

**Error: "No formula found for prefix 'xyz'"**
- Cause: `data/formulas/{prefix}_formula.json` missing
- Fix: Either omit formula mode, or train formula using `formula_finder` tool

**Error: "GDOP > 8: rejecting solution"**
- Cause: Poor trilateration geometry (cells in line or too far apart)
- Fix: Check coverage report for cell distribution; may indicate data quality issue

**Error: "ValidationError: ... coordinates must be finite"**
- Cause: Tower database has NaN or invalid coordinates
- Fix: Verify `towers.json` and `pci.json` integrity; check for malformed JSON

**Error: "Insufficient cells for trilateration (n < 4)"**
- Cause: Too few cells with RSRP > -110 dBm at this timestamp
- Fix: Check coverage report; may need to relax RSRP threshold or collect in better coverage area

## License & Disclaimer

This tool is provided for research and educational purposes. Verify all outputs before use in critical applications. Tower data sourced from public databases; respect data usage terms.

**No warranty:** Outputs are estimates; not suitable for emergency services or safety-critical positioning.

---

## Version History

- **v1.0 (2025-12-31):** Initial release with formula support + distance validation phase
  - Features: 9-phase pipeline, context-specific models, formula regression, batch processing
  - Outputs: CEP/R95/RMSE validation, parameter tracking, coverage reporting

---

**Questions?** Refer to individual phase source files (`src/*.py`) for implementation details, or contact thesis advisor.
