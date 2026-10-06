 # config.py

from pathlib import Path
import sys

# ============================================================
# EXPERIMENT / PROTOCOL
# ============================================================
EXPERIMENT_ID = "left_arm"
PROTOCOL_ID = "left_90deg"

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("../data")
RESULTS_ROOT = Path("../results")

# ============================================================
# DATA PATHS
# ============================================================
PROTOCOL_PATH = (PROJECT_ROOT / "experiments" / EXPERIMENT_ID / PROTOCOL_ID / "protocol.json")
DATA_PATH = (DATA_ROOT / EXPERIMENT_ID / PROTOCOL_ID / "data_all_subjects.xlsx")

# ============================================================
# RESULTS PATHS
# ============================================================
RESULTS_PATH = RESULTS_ROOT / EXPERIMENT_ID / PROTOCOL_ID

DATA_PROCESSED_PATH = RESULTS_PATH / "data_processed"
HEATMAPS_RESULTS_PATH = RESULTS_PATH / "heatmaps"
MODEL_PARAMETERS_PATH = RESULTS_PATH / "model_parameters"
MODEL_FITTING_PATH = RESULTS_PATH / "model_fitting"
REGRESSIONS_RESULTS_PATH = RESULTS_PATH / "regressions"
LOSOCV_PATH = RESULTS_PATH / "losocv"

ALL_PATHS = [
    RESULTS_PATH,
    DATA_PROCESSED_PATH,
    HEATMAPS_RESULTS_PATH,
    MODEL_PARAMETERS_PATH,
    MODEL_FITTING_PATH,
    REGRESSIONS_RESULTS_PATH,
    LOSOCV_PATH,
]

# ============================================================
# ANALYSIS PIPELINE
# ============================================================
DO_HEATMAPS                 = True
DO_SINGLE_SUBJECT_HEATMAPS  = False
DO_EXTRACT_MODEL_PARAMETERS = False
DO_REGRESSIONS              = False
DO_MODEL_FITTING            = False
DO_LOSOCV                   = False


# ============================================================
# DATA / SUBJECT SELECTION
# ============================================================
SUBJECTS_TO_PROCESS         = None

RECALC_SUBJECT              = True
UPDATE_GROUP_AVERAGE        = True


# ============================================================
# MODEL
# ============================================================
USE_VIVIDNESS_WEIGHTS = True

PURE_PATTERNS = {
    "100_000": "b",
    "000_100": "t",
}

# ============================================================
# REGRESSION
# ============================================================
PATTERNS_TO_PROCESS = [
    "100_000",
    "000_100",
]

INCLUDE_GROUP_LEVEL = True
INCLUDE_SUBJECT_LEVEL = False
INCLUDE_TRIAL_LEVEL = True

# ============================================================
# QUADRATIC INTERPOLATION
# ============================================================
FLEXION_PATTERN_FOR_QUADRATIC_INTERP = "000_111"
EXTENSION_PATTERN_FOR_QUADRATIC_INTERP = "111_000"

# ============================================================
# SATURATION ANALYSIS
# ============================================================
SATURATION_THRESHOLD = 0.95

SATURATION_THRESHOLDS_SENSITIVITY = [0.85, 0.90, 0.95]

# ============================================================
# PLOTS
# ============================================================
SAVE_PLOTS = True
PLOT_FONT_FAMILY = "Times New Roman"