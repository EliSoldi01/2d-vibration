ARM        = "right"
INITIAL_ANGLE = 120 # 70 o 120
START_CELL = [11, 5] if INITIAL_ANGLE == 120 else [11, 7]


DATA_PATH     = "C:/Users/Utente/Desktop/Elisa/Research/2D-vibration/Phase4-Right-" + str(INITIAL_ANGLE) + "/Results/data_all_subjects_P4_" + str(INITIAL_ANGLE) + ".xlsx"
SUBJECTS_PATH = "C:/Users/Utente/Desktop/Elisa/Research/2D-vibration/Phase4-Right-" + str(INITIAL_ANGLE) + "/Results/Subjects/Subjects_list_P4_" + str(INITIAL_ANGLE) + ".xlsx"
PROTOCOL_PATH = "phases/phase4/protocol4_" + str(INITIAL_ANGLE) + ".json"
OUTPUT_ROOT   = "results/phase4/" + str(INITIAL_ANGLE)

# --- Cosa eseguire ---
DO_HEATMAPS                              = True
DO_SINGLE_SUBJECT_HEATMAPS               = False 
DO_REGRESSIONS                           = False
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
SUBJECTS_TO_PROCESS = None # Es. ["S01", "S02"] o None per tutti
PATTERNS_TO_PROCESS = ["001_000", "000_001"]
