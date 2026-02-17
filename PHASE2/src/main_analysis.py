import os
import warnings
from analysis.sigmoid_fit import fit_group_sigmoid
from utils.create_std_excel import create_std_normalized_excel
from utils.extract_model_parameters import create_subject_model_and_global_excel_weighted

# ----------------------------
# Data I/O
# ----------------------------
import data_io.load_data as load_data
import data_io.validate_data as validate_data

# ----------------------------
# Analysis
# ----------------------------
from utils import geometry, ordering

# ----------------------------
# Visualization
# ----------------------------
import visualization.heatmaps as heatmaps
import visualization.plot_regressions as plot_regression

warnings.simplefilter(action='ignore', category=FutureWarning)

print("RUNNING MAIN ANALYSIS SCRIPT")
print("----------------------------------------------------------------------------------------")

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
subjects_to_process = None #None for all subjects, otherwise write subjects to analyze e.g. ["S01", "S02"]
pattern_to_process = ["001_000", "000_001"] # None for all patterns, otherwise write patterns to analyze e.g. ["001_000", "000_001"]

do_heatmaps = False
do_regressions = False
do_regression_original_durations = False
do_regression_ordered_durations = False
do_std_excel = True
do_model_parameters = True
do_sigmoid_fit = True
recalc_subject = True
update_group_average = True
save_plots = True

# --------------------------------------------
# DATA LOADING
# --------------------------------------------
protocol = load_data.load_protocol("config\\protocol2.json")
maindata = load_data.load_main_data(
    "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Phase2\\Results\\data_all_subjects_P2.xlsx"
)
subject_info = load_data.load_data_file(
    "C:\\Users\\Utente\\Desktop\\Elisa\\Research\\2D-vibration\\Phase2\\Results\\Subjects\\Subjects_list.xlsx"
)

print("Data loaded.")

# Output paths
OUTPUT_FOLDER = "Results_"+protocol["name"]
OUTPUT_FOLDER_REG = os.path.join(OUTPUT_FOLDER, "regressions")
OUTPUT_FOLDER_STD = os.path.join(OUTPUT_FOLDER, "stats")
subject_output_folder = os.path.join(OUTPUT_FOLDER, "subject_models")
global_output_path = os.path.join(OUTPUT_FOLDER, "Global_Model_Parameters.xlsx")

# Crea tutte le cartelle necessarie
for folder in [OUTPUT_FOLDER, OUTPUT_FOLDER_REG, OUTPUT_FOLDER_STD]:
    os.makedirs(folder, exist_ok=True)

# Merge and preprocess
df = load_data.merge_subject_data(maindata, subject_info)
df = load_data.preprocess_data(df)
print("Data preprocessed.")

# Validate protocol and data
if not validate_data.validate_protocol(protocol):
    raise ValueError("Protocol JSON not valid!")
if not validate_data.validate_subject_data(df, protocol):
    raise ValueError("Subject data not valid!")
print("Data validated successfully.")

# --------------------------------------------
# ADD angle_deg & presentation info
# --------------------------------------------
df = geometry.add_angle_column(df, protocol["grid"]["start_cell"])
subject_orders = subject_info.set_index("subject")["block_order"].astype(str).to_dict()
df = ordering.add_presentation_order(df, subject_orders, duration_col="duration")
df = ordering.add_presentation_pos(df, subject_orders, duration_col="duration")
print("Presentation order and positions added.")
print("----------------------------------------------------------------------------------------")

# --------------------------------------------
# ANALYSIS & PLOTTING
# --------------------------------------------
subjects_list = df["subject"].unique() if subjects_to_process is None else subjects_to_process
patterns_list = df["pattern_pair"].unique() if pattern_to_process is None else pattern_to_process
metrics_df = df.copy()

print(f"Subjects to analyze: {subjects_list}")
print(f"Patterns to analyze: {patterns_list}")
print("------------------------------------------------------------------------------------------")

# -------------------------
# REGRESSION PLOTS
# -------------------------
if do_regressions:
    print("Doing regression plots...")
    for subj in subjects_list:
        print(f"-- Analyzing subject {subj}")
        subj_df = df[df["subject"] == subj].copy()
        for pat in patterns_list:

            if do_regression_ordered_durations:
                # Durations in ascending order
                plot_regression.plot_subject_pattern(subj_df, subj, pat, protocol, output_folder=os.path.join(OUTPUT_FOLDER_REG, "duration"))
                plot_regression.plot_group_average(df, patterns_list, protocol, output_folder=os.path.join(OUTPUT_FOLDER_REG, "duration"))

            if do_regression_original_durations:
                # Durations in presentation order
                plot_regression.plot_subject_pattern(
                    subj_df,
                    subj,
                    pat,
                    protocol,
                    per_reps_plot=False,
                    output_folder=os.path.join(OUTPUT_FOLDER_REG, "block_order"),
                    x_col="presentation_order",
                    x_label="Duration (s) in presentation order"
                )
                plot_regression.plot_group_average(
                    df,
                    patterns_list,
                    protocol,
                    per_reps_plot=False,
                    output_folder=os.path.join(OUTPUT_FOLDER_REG, "block_order"),
                    x_col="presentation_order",
                    x_label="Duration (s) in presentation order"
                )

else:
    print("SKIPPING regression plots")

# -------------------------
# HEATMAPS
# -------------------------
if do_heatmaps:
    print("Doing heatmaps...")
    for subj in subjects_list:
        heatmaps.save_subject_heatmaps(
            df,
            subj,
            protocol,
            output_folder=OUTPUT_FOLDER,
            metric="vividness",
            recalc_subject=recalc_subject,
            start_pos=(11, 5)
        )
    if update_group_average:
        heatmaps.save_all_subjects_heatmaps(
            df,
            protocol,
            output_folder=OUTPUT_FOLDER,
            metric="vividness",
            start_pos=(11, 5)
        )
else:
    print("SKIPPING heatmaps")

# -------------------------
# STD NORMALIZED EXCEL
# -------------------------
if do_std_excel:
    print("Doing std excel...")
    output_excel = os.path.join(OUTPUT_FOLDER_STD, "Std_Normalized_Phase2.xlsx")
    create_std_normalized_excel(
        metrics_df,
        output_excel,
        subjects=subjects_list,
        patterns=patterns_list
    )
    print(f"[INFO] Std normalized Excel created at {output_excel}")
else:
    print("SKIPPING creation std excel")

# -------------------------
# MODEL PARAMETERS EXCEL
# -------------------------
if do_model_parameters:
    print("Doing model parameters extraction and validation...")
    global_validation_df =create_subject_model_and_global_excel_weighted(df, protocol, subject_output_folder, global_output_path)
else:
    print("SKIPPING subject model files")

if do_sigmoid_fit:
    print("Doing sigmoid fit...")
    fit_group_sigmoid(global_validation_df, True)
else:
    print("SKIPPING sigmoid fit")


print("MAIN ANALYSIS COMPLETED.")
