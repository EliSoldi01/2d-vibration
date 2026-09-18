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
RESULTS_PATH = Path(RESULTS_ROOT) / EXPERIMENT_ID / PROTOCOL_ID

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

DO_HEATMAPS                              = True
DO_SINGLE_SUBJECT_HEATMAPS               = False
DO_REGRESSIONS                           = False
DO_REGRESSION_ORIGINAL_DURATIONS_SUBJ    = False
DO_REGRESSION_ORDERED_DURATIONS_SUBJ     = False
DO_REGRESSION_ORIGINAL_DURATIONS_GLOBAL  = False
DO_REGRESSION_ORDERED_DURATIONS_GLOBAL   = False
DO_STD_EXCEL                             = False
DO_MODEL_PARAMETERS                      = True
DO_SIGMOID_FIT                           = True
DO_CROSS_VALIDATION                      = False
DO_SINGLE_STIMULATION_FIT                = True

# ============================================================
# OPTIONAL
# ============================================================
RECALC_SUBJECT      = True
UPDATE_GROUP_AVERAGE = True
SAVE_PLOTS          = True

# ============================================================
# DA SPOSTARE NEL MAIN
# ============================================================
SUBJECTS_TO_PROCESS = None
PATTERNS_TO_PROCESS = ["001_000", "000_001"]
