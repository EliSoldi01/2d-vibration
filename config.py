from pathlib import Path
import sys


# ============================================================
# EXPERIMENT
# ============================================================
EXPERIMENT_ID = "baseline"
PROTOCOL_ID = "right_90deg"

# ============================================================
# DATA PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_ROOT = "../data"
RESULTS_ROOT = "../results"


PROTOCOL_PATH = ( PROJECT_ROOT / "experiments" / EXPERIMENT_ID / PROTOCOL_ID / "protocol.json"
)
DATA_PATH = Path(DATA_ROOT) / EXPERIMENT_ID / PROTOCOL_ID / "data_all_subjects.xlsx"


# ============================================================
# RESULTS PATH
# ============================================================
RESULTS_PATH = Path(RESULTS_ROOT) / EXPERIMENT_ID / PROTOCOL_ID
DATA_PROCESSED_PATH = RESULTS_PATH / "data_processed"
HEATMAPS_RESULTS_PATH = RESULTS_PATH / "heatmaps"
MODEL_PARAMETERS_PATH = RESULTS_PATH / "model_parameters"
MODEL_FITTING_PATH = RESULTS_PATH / "model_fitting"

ALL_PATHS = [RESULTS_PATH, DATA_PROCESSED_PATH, HEATMAPS_RESULTS_PATH, MODEL_PARAMETERS_PATH]

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

DO_HEATMAPS                              = False
DO_SINGLE_SUBJECT_HEATMAPS               = False
DO_REGRESSIONS                           = False
DO_REGRESSION_ORIGINAL_DURATIONS_SUBJ    = False
DO_REGRESSION_ORDERED_DURATIONS_SUBJ     = False
DO_REGRESSION_ORIGINAL_DURATIONS_GLOBAL  = False
DO_REGRESSION_ORDERED_DURATIONS_GLOBAL   = False
DO_STD_EXCEL                             = False
DO_EXTRACT_MODEL_PARAMETERS              = True
DO_MODEL_FITTING                         = True
DO_CROSS_VALIDATION                      = False
DO_SINGLE_STIMULATION_FIT                = False

# ============================================================
# OPTIONAL
# ============================================================
USE_VIVIDNESS_WEIGHTS = True
RECALC_SUBJECT        = True
UPDATE_GROUP_AVERAGE  = True
SAVE_PLOTS            = True


# ============================================================
# ANALYSIS 
# ============================================================
SUBJECTS_TO_PROCESS = None
PURE_PATTERNS = {
    "100_000": "b",
    "000_100": "t"
}
PATTERNS_TO_PROCESS = ["100_000", "000_100"]
SATURATION_THRESHOLD = 0.95