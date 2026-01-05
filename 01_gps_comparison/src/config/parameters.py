"""
Configuration Parameters for GPS Ground Truth Pipeline

Centralized parameter management for the entire pipeline.

"""

from pathlib import Path

# DIRECTORY CONFIGURATION

# Base directory (project root)
# parameters.py is at: src/config/parameters.py
# Going up 3 levels: config → src → project_root
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SMARTPHONE_DIR = DATA_DIR / "SMARTPHONE"
WATCH_DIR = DATA_DIR / "WATCH"


# Output directory
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATION_DIR = RESULTS_DIR / "visualization"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
LOCATION_NAMES = ['ixelle', 'lln', 'waha']
SESSION_NUMBERS = [1, 2, 3, 4, 5]

# ============================================================================
# TIMESTAMP CONFIGURATION
# ============================================================================

TIMESTAMP_FORMAT_PARAM = '%Y.%m.%d_%H.%M.%S'

# ============================================================================
# GPS MATCHING CONFIGURATION
# ============================================================================

# Time window for matching smartphone and watch GPS records
TIME_WINDOW_SECONDS_PARAM = 4

# Maximum allowed time offset between devices (in seconds)
ALLOWED_OFFSET_SECONDS_PARAM = 3600

# ============================================================================
# OPTIMAL WINDOW DETECTION CONFIGURATION
# ============================================================================

# Test windows for optimal time window detection (in seconds)
TEST_WINDOWS_PARAM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

# Quality thresholds for optimal window selection
MINIMUM_MATCH_RATE_PARAM = 85.0

# Maximum mean error: 10 meters (strict accuracy for geolocation thesis)
MAXIMUM_MEAN_ERROR_PARAM = 10.0

# ============================================================================
# QUALITY ANALYSIS CONFIGURATION
# ============================================================================

# Gap threshold for quality analysis (in seconds)
GAP_THRESHOLD_SECONDS_PARAM = 12

# ============================================================================
# FLAG COMPARISON CONFIGURATION (Step 5)
# ============================================================================

# Major gap threshold: gaps >= this value trigger near_major_gap flagging
MAJOR_GAP_THRESHOLD_SECONDS_PARAM = 60

# Time window around major gaps to flag as unreliable
NEAR_GAP_WINDOW_SECONDS_PARAM = 30

# Prefix for flagged comparison files
# Example: comparison_lln_5.csv → comparison_flag_lln_5.csv
COMPARISON_FLAG_PREFIX = "comparison_flag_"

# ============================================================================
# WAYPOINT ANALYSIS CONFIGURATION (Step 7)
# ============================================================================
# Maximum time difference for waypoint-GPS matching (seconds)
WAYPOINT_TIME_WINDOW_SECONDS_PARAM = 10

PHONE_WATCH_DISTANCE_THRESHOLD = 10.0  # Threshold for phone-watch agreement analysis (meters)