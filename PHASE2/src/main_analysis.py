import data_io.load_data as load_data
import data_io.validate_data as validate_data
import visualization.heatmaps as heatmaps

print("RUNNING MAIN ANALYSIS SCRIPT")
print("----------------------------------------------------------------------------------------")

# --------------------------------------------
# DATA LOADING, PREPROCESSING & VALIDATION
# --------------------------------------------

# Load protocol
protocol = load_data.load_protocol("config\\protocol1.json")

# Load subject data
maindata = load_data.load_main_data("C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Phase1\\Subjects\\data_all_subjects - RIDOTTO.xlsx")
subject_info = load_data.load_data_file("C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Phase1\\Subjects\\RIDOTTO_results\\Subjects_list.xlsx")

print("Data loaded.")
print("----------------------------------------------------------------------------------------")

# Preprocess data
df = load_data.merge_subject_data(maindata, subject_info)
df = load_data.preprocess_data(df)

print("Data preprocessed.")
print("----------------------------------------------------------------------------------------")

# Validate protocol
if not validate_data.validate_protocol(protocol):
    raise ValueError("Protocol JSON not valid!")

# Validate Excel data
if not validate_data.validate_subject_data(df, protocol):
    raise ValueError("Subject data not valid!")

print("Data validated successfully.")
print("----------------------------------------------------------------------------------------")

# --------------------
# ANALYSIS
# --------------------

# --- CONFIGURATION ---
OUTPUT_FOLDER = "Results_"
subjects_to_process = ["S01"]  # None = all subjects; else list e.g., ["subj1", "subj2"]
recalc_subject = True       # False = skip existing subject folder
update_group_average = True # True = recalculate ALL SUBJECTS average heatmaps
# ---------------------

subjects_list = df["subject"].unique() if subjects_to_process is None else subjects_to_process

for subj in subjects_list:
    print(f"[INFO] Processing subject: {subj}")
    heatmaps.save_subject_heatmaps(df, subj, protocol, output_folder=OUTPUT_FOLDER, metric="vividness", recalc_subject=recalc_subject, start_pos=(11,5))

if update_group_average:
    print("[INFO] Processing ALL SUBJECTS average heatmaps")
    heatmaps.save_all_subjects_heatmaps(df, protocol, output_folder=OUTPUT_FOLDER, metric="vividness", start_pos=(11,5))

