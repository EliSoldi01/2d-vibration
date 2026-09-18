from pathlib import Path

# ============================================================
# EXPERIMENT
# ============================================================
EXPERIMENT_ID = "baseline"
PROTOCOL_ID = "right_90deg"

# ============================================================
# DATA PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Determines the root directory of the project
DATA_ROOT = "../data"
RESULTS_ROOT = "../results"


PROTOCOL_PATH = ( PROJECT_ROOT / "experiments" / EXPERIMENT_ID / PROTOCOL_ID / "protocol.json"
)
DATA_PATH = Path(DATA_ROOT) / EXPERIMENT_ID / PROTOCOL_ID / "data_all_subjects.xlsx"
RESULTS_PATH = Path(RESULTS_ROOT) / EXPERIMENT_ID / PROTOCOL_ID

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
SUBJECTS_TO_PROCESS = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16","S18", "S19", "S20", "S21", "S22"]
PATTERNS_TO_PROCESS = ["001_000", "000_001"]
