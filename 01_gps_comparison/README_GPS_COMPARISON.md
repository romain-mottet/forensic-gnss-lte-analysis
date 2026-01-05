# GPS Ground Truth Comparison Pipeline

Complete end-to-end pipeline for comparing smartphone and Garmin watch GPS tracks against ground-truth waypoints. Built for master thesis research on GPS geolocation quality in forensic contexts.

## Overview

This pipeline processes GPS data from two acquisition sources (smartphone and watch), aligns them in time, validates against ground-truth waypoints, and generates CSV summaries plus thesis-ready PDF visualizations.

Running `python main.py` executes 12 automated steps: GPX conversion, optimal time-window detection, timestamp-based merging, quality analysis, unified dataset creation, waypoint distance evaluation, accuracy metrics computation (CEP/R95/RMSE), and figure generation.

## Project Structure

```
project_root/
├── main.py                          # Pipeline orchestrator
├── src/
│   ├── config/
│   │   └── parameters.py            # Central configuration
│   ├── *.py                         # Step implementations
│   └── helpers/
│       └── *.py                     # Reusable utilities
├── data/
│   ├── SMARTPHONE/
│   │   └── <location>/
│   │       └── ground_truth_<location>_<session>.csv
│   ├── WATCH/
│   │   └── <location>/
│   │       └── *.gpx
│   └── ground_truth_waypoints.csv
├── results/
│   ├── <location>/
│   │   ├── comparison_<location><session>.csv
│   │   └── comparison_flag_<location>_<session>.csv
│   ├── unified_gps_dataset.csv
│   ├── device_comparison.csv
│   ├── location_comparison.csv
│   ├── cloud_coverage_comparison.csv
│   ├── phone_watch_agreement.csv
│   └── visualization/
│       └── figure_01-06.pdf
└── reports/
    └── <location>/
        └── measurement_accuracy_<location>_<session>.txt
```

## Installation

### Requirements

- **Python 3.14.0** (tested version)
- Virtual environment (recommended)

### Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)

```
gpxpy
pandas
numpy
matplotlib
seaborn
```

The pipeline uses `gpxpy` for GPX parsing, `pandas`/`numpy` for data processing, and `matplotlib`/`seaborn` for visualization.

## Configuration

All parameters are centralized in `src/config/parameters.py`:

### Dataset Configuration
- `LOCATION_NAMES`: List of location identifiers (e.g., `['ixelle', 'lln', 'waha']`)
- `SESSION_NUMBERS`: Session identifiers (e.g., `[1, 2, 3, 4, 5]`)

**To add new locations/sessions**: Edit these lists in `parameters.py`.

### Time Matching Parameters
- `TEST_WINDOWS_PARAM`: Candidate windows tested (0.5s–8.0s in 0.5s increments)
- `TIME_WINDOW_SECONDS_PARAM`: Default matching tolerance (4 seconds)
- `ALLOWED_OFFSET_SECONDS_PARAM`: Maximum device clock offset (3600s = ±1 hour)

**Rationale**: The ±1 hour offset limit handles timestamp errors from DST/clock changes. Time windows are optimized per file-pair to balance match rate and accuracy.

### Quality Thresholds
- `GAP_THRESHOLD_SECONDS_PARAM`: Flag gaps ≥12s
- `MAJOR_GAP_THRESHOLD_SECONDS_PARAM`: Major gap detection (60s)
- `NEAR_GAP_WINDOW_SECONDS_PARAM`: Flag records within 30s of major gaps

### Waypoint Matching
- `WAYPOINT_TIME_WINDOW_SECONDS_PARAM`: Maximum time difference for waypoint-GPS matching (15s)
- `PHONE_WATCH_DISTANCE_THRESHOLD`: Agreement threshold (10.0m)

## Data Contracts

### Input: Smartphone CSV
**Location**: `data/SMARTPHONE/<location>/ground_truth_<location>_<session>.csv`

Required columns:
- `timestamp` (format: `YYYY.MM.DD_HH.MM.SS`)
- `gps_longitude`, `gps_latitude` (numeric)
- `gps_accuracy_m` (numeric, optional)

### Input: Watch GPX
**Location**: `data/WATCH/<location>/*.gpx`

Step 1 converts GPX to CSV format `gps_ground_truth_*.csv`. Accuracy (`gps_accuracy_m`) is extracted only if present in GPX (HDOP/VDOP or extensions).

### Input: Waypoints CSV
**Location**: `data/ground_truth_waypoints.csv`

Required columns:
- `timestamp`, `location`, `session_number`
- `theoretical_latitude`, `theoretical_longitude`
- `cloud_coverage` (0=clear, 5=overcast)
- `waypoint_name`, `waypoint_id`

Step 6 uses this to auto-populate cloud coverage for all measurements in a session.

### Output: Unified Dataset
**Location**: `results/unified_gps_dataset.csv`

Combined schema:
- `timestamp`, `location`, `session_number`, `source_file`
- `smartphone_longitude`, `smartphone_latitude`, `smartphone_accuracy_m`
- `watch_longitude`, `watch_latitude`, `watch_accuracy_m`
- `cloud_coverage` (auto-populated)
- `is_cache_duplicate`, `near_major_gap_30s` (quality flags)

## Pipeline Steps

### Step 1: GPX → CSV Conversion
Recursively finds `.gpx` files under `WATCH_DIR` and converts to CSV.

**Output**: `gps_ground_truth_<stem>.csv` files in watch subdirectories

### Step 2: Find Optimal Match Window
Tests candidate windows (0.5–8.0s) per smartphone-watch pair and selects optimal window.

**Output**: Per-pair optimal windows used in Step 3

### Step 3: Merge Smartphone + Watch
Merges rows by timestamp within optimal window, applying offset correction.

**Output**: `results/<location>/comparison_<location><session>.csv`

**Rejection**: Pairs with offset exceeding ±1 hour are rejected.

### Step 4: Quality Analysis
Scans comparison files for duplicate timestamps and time gaps.

**Output**: `reports/<location>/comparison_<location><session>_quality_report.txt`

### Step 5: Flag Comparison Data
Adds quality flags (`is_cache_duplicate`, `near_major_gap_30s`).

**Output**: `results/<location>/comparison_flag_<location>_<session>.csv`

### Step 6: Create Unified Dataset
Concatenates all flagged files, adds metadata, auto-populates cloud coverage.

**Output**: `results/unified_gps_dataset.csv`

### Step 7: Waypoint Distance Analysis
Matches waypoints to GPS measurements within time window, computes Haversine distances.

**Output**: `reports/<location>/measurement_accuracy_<location>_<session>.txt`

### Step 8: Global Device Accuracy
Computes CEP (50th percentile), R95 (95th percentile), RMSE per device.

**Output**: `results/device_comparison.csv`

### Step 9: Accuracy by Location
Groups metrics by location.

**Output**: `results/location_comparison.csv`

### Step 10: Accuracy by Cloud Coverage
Groups metrics by cloud coverage category.

**Output**: `results/cloud_coverage_comparison.csv`

### Step 11: Phone-Watch Agreement
Analyzes agreement between devices at aligned timestamps.

**Output**: `results/phone_watch_agreement.csv`

### Step 12: Thesis Visualizations
Generates six publication-ready PDF figures:
1. GPS accuracy by location (CEP)
2. Global device comparison (CEP/R95/RMSE)
3. Phone-watch error agreement scatter
4. GPS accuracy vs cloud coverage trends
5. Error distributions (boxplots)
6. Location × device heatmap

**Output**: `results/visualization/figure_01-06.pdf`

## Running the Pipeline

```bash
python main.py
```

The pipeline runs Steps 1–12 sequentially. Each step depends on previous outputs. A final summary shows success/failure status for each step.

## Troubleshooting

### "No GPX files found"
- Verify GPX files exist under `data/WATCH/`
- Check that `gpxpy` is installed

### "Waypoints file not found"
- Ensure `data/ground_truth_waypoints.csv` exists
- Verify path matches `DATA_DIR` in `parameters.py`

### "Offset exceeds 1 hour limit"
- Device clock offset is too large (>3600s)
- Adjust `ALLOWED_OFFSET_SECONDS_PARAM` if needed

### "Missing sessions/locations"
- Update `LOCATION_NAMES` and `SESSION_NUMBERS` in `src/config/parameters.py`
- Pipeline only processes configured locations/sessions

### "Step 6 fails with empty results"
- Ensure Step 5 completed successfully and created `comparison_flag_*.csv` files
- Check that location directory structure matches expected format

## Methodology Notes

### Time Window Optimization
Windows are tested from 0.5–8.0s to find the optimal balance between match rate and accuracy. This avoids arbitrary tolerance choices that either under-match (too strict) or over-match (too permissive).

### Offset Enforcement
The ±1 hour limit prevents accidental alignment of unrelated timestamps. This threshold was set based on observed timestamp errors from DST/clock changes in the thesis dataset.

### Quality Flagging
Records near large gaps (≥60s) are flagged because they're more likely affected by device sleep, acquisition issues, or GPS re-lock events.

## Cloud Coverage Scale

Cloud coverage is recorded by human observation on a 0–5 scale:
- **0** = clear
- **5** = overcast

Values are assumed constant within each session.

## Expected Outputs

After successful execution:
- **results/**: Comparison CSVs, unified dataset, analysis summaries, PDF figures
- **reports/**: Per-session waypoint reports and quality analysis reports

Use included example outputs as validation reference.

## License & Citation

Developed for master thesis research on GPS geolocation quality in forensic contexts.

---

**Last updated**: December 2025
