from pathlib import Path
import sys


# ============================================================
# EXPERIMENT
# ============================================================
EXPERIMENT_ID = "illusion_or_confusion"
PROTOCOL_ID = "right_90deg_IoC"

# ============================================================
# DATA PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_ROOT = "../data"
RESULTS_ROOT = "../results"


PROTOCOL_PATH = ( PROJECT_ROOT / "experiments" / EXPERIMENT_ID / PROTOCOL_ID / "protocol.json")
DATA_PATH = Path(DATA_ROOT) / EXPERIMENT_ID / PROTOCOL_ID / "data_all_subjects.xlsx"

# ============================================================
# RESULTS PATH
# ============================================================
RESULTS_PATH = Path(RESULTS_ROOT) / EXPERIMENT_ID / PROTOCOL_ID
DATA_PROCESSED_PATH = RESULTS_PATH / "data_processed"
HEATMAPS_RESULTS_PATH = RESULTS_PATH / "heatmaps"
MODEL_PARAMETERS_PATH = RESULTS_PATH / "model_parameters"
MODEL_FITTING_PATH = RESULTS_PATH / "model_fitting"
REGRESSIONS_RESULTS_PATH = RESULTS_PATH / "regressions"
LOSOCV_PATH = RESULTS_PATH / "losocv"

ALL_PATHS = [RESULTS_PATH, DATA_PROCESSED_PATH, HEATMAPS_RESULTS_PATH, 
             MODEL_PARAMETERS_PATH, MODEL_FITTING_PATH, REGRESSIONS_RESULTS_PATH, LOSOCV_PATH]

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

DO_HEATMAPS                              = False
DO_SINGLE_SUBJECT_HEATMAPS               = False
DO_EXTRACT_MODEL_PARAMETERS              = True
DO_REGRESSIONS                           = True
DO_MODEL_FITTING                         = True
DO_LOSOCV                                = True

# ============================================================
# OPTIONAL
# ============================================================
USE_VIVIDNESS_WEIGHTS = False # If True, regression weights are computed from the vividness values.
RECALC_SUBJECT        = True  # If True, heatmaps of a subject for which the file already exists is recalculated. 
UPDATE_GROUP_AVERAGE  = True  
SAVE_PLOTS            = True


# ============================================================
# ANALYSIS 
# ============================================================
SUBJECTS_TO_PROCESS = None
PURE_PATTERNS = {"001_000": "b", "000_001": "t"}
PATTERNS_TO_PROCESS = ["001_000", "000_001"]
FLEXION_PATTERN_FOR_QUADRATIC_INTERP = "000_001"
EXTENSION_PATTERN_FOR_QUADRATIC_INTERP = "001_000"

# Saturation point used for plots
SATURATION_THRESHOLD = 0.95

# Thresholds used for saturation sensitivity analysis in Excel
SATURATION_THRESHOLDS_SENSITIVITY = [0.85, 0.90, 0.95]

# Regression analysis and plots
INCLUDE_GROUP_LEVEL   = True
INCLUDE_SUBJECT_LEVEL = False
INCLUDE_TRIAL_LEVEL   = True

# Plot font
PLOT_FONT_FAMILY = "Times New Roman"
