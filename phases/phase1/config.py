ARM        = "right"
START_CELL = [11, 5]

DATA_PATH     = "C:/Users/Utente/Desktop/Elisa/Research/2D-vibration/Phase1-Right/Results/data_all_subjects_P1.xlsx"
SUBJECTS_PATH = "C:/Users/Utente/Desktop/Elisa/Research/2D-vibration/Phase1-Right/Results/Subjects/Subjects_list_P1.xlsx"
PROTOCOL_PATH = "phases/phase1/protocol2.json"
OUTPUT_ROOT   = "results/phase1"

# --- Cosa eseguire ---
DO_HEATMAPS                              = True
DO_SINGLE_SUBJECT_HEATMAPS               = False
DO_REGRESSIONS                           = False
DO_REGRESSION_ORIGINAL_DURATIONS_SUBJ   = False
DO_REGRESSION_ORDERED_DURATIONS_SUBJ    = False
DO_REGRESSION_ORIGINAL_DURATIONS_GLOBAL = False
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
SUBJECTS_TO_PROCESS = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17","S18", "S19", "S20", "S21", "S22"]
PATTERNS_TO_PROCESS = ["001_000", "000_001"]
