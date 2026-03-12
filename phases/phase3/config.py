ARM        = "left"
START_CELL = [11, 5]

DATA_PATH     = "C:/Users/Utente/Desktop/Elisa/Research/2D-vibration/Phase3-Left/Results/data_all_subjects_P3_with0.xlsx"
SUBJECTS_PATH = "C:/Users/Utente/Desktop/Elisa/Research/2D-vibration/Phase3-Left/Results/Subjects/Subjects_list_3.xlsx"
PROTOCOL_PATH = "phases/phase3/protocol2.json"
OUTPUT_ROOT   = "results/phase3"

# --- Cosa eseguire ---
DO_HEATMAPS                              = True
DO_REGRESSIONS                           = True
DO_REGRESSION_ORIGINAL_DURATIONS_SUBJ   = True
DO_REGRESSION_ORDERED_DURATIONS_SUBJ    = False
DO_REGRESSION_ORIGINAL_DURATIONS_GLOBAL = True
DO_REGRESSION_ORDERED_DURATIONS_GLOBAL  = False
DO_STD_EXCEL                             = False
DO_MODEL_PARAMETERS                      = True
DO_SIGMOID_FIT                           = True
DO_CROSS_VALIDATION                      = True

# --- Opzioni aggiuntive ---
RECALC_SUBJECT      = True
UPDATE_GROUP_AVERAGE = True
SAVE_PLOTS          = True

# --- Filtri ---
SUBJECTS_TO_PROCESS = ["S01", "S02"]
PATTERNS_TO_PROCESS = ["001_000", "000_001"]
