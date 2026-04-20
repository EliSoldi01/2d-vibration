import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import warnings
import phases.phase4.config as cfg

from core.data_io         import load_data, validate_data
from core.utils           import geometry, ordering
from core.utils.create_std_excel                import create_std_normalized_excel
from core.analysis.extract_model_parameters     import create_subject_and_global_excel
from core.analysis.model_fit                    import fit_group_sigmoid
from core.analysis.cross_validation             import loocv_leave_one_subject
from core.visualization   import heatmaps
from core.visualization   import plot_regressions as plot_regression

warnings.simplefilter(action="ignore", category=FutureWarning)

print("RUNNING PHASE 4 ANALYSIS")
print("-" * 90)

# --- Output folders ---
OUT          = cfg.OUTPUT_ROOT
OUT_HEATMAPS = os.path.join(OUT, "heatmaps")
OUT_REG      = os.path.join(OUT, "regressions")
OUT_STD      = os.path.join(OUT, "stats")
OUT_MODEL    = os.path.join(OUT, "model")
OUT_SUBJ_MOD = os.path.join(OUT_MODEL, "subject_models")
OUT_VAL      = os.path.join(OUT_MODEL, "model_validation")
for f in [OUT, OUT_REG, OUT_STD, OUT_MODEL]: os.makedirs(f, exist_ok=True)

# --- Caricamento ---
protocol     = load_data.load_protocol(cfg.PROTOCOL_PATH)
maindata     = load_data.load_main_data(cfg.DATA_PATH)
subject_info = load_data.load_data_file(cfg.SUBJECTS_PATH)
print("Data loaded.")

df = load_data.merge_subject_data(maindata, subject_info)
df = load_data.preprocess_data(df)
print("Data preprocessed.")

if not validate_data.validate_protocol(protocol):
    raise ValueError("Protocol JSON not valid!")
if not validate_data.validate_subject_data(df, protocol):
    raise ValueError("Subject data not valid!")
print("Data validated successfully.")

# --- Adding angle column and presentation order ---
df["duration"] = df["duration"].astype(int)
df = geometry.add_angle_column(df, cfg.START_CELL, arm=cfg.ARM)
subject_orders = subject_info.set_index("subject")["block_order"].astype(str).to_dict()
df = ordering.add_presentation_order(df, subject_orders, duration_col="duration")
df = ordering.add_presentation_pos(df, subject_orders, duration_col="duration")
print("Presentation order and positions added.")

enriched_path = os.path.join(OUT, "data_with_angles_and_presentation.xlsx")
df.to_excel(enriched_path, index=False, engine="openpyxl")
print(f"✅ DataFrame saved in: {enriched_path}")
print("-" * 90)

subjects_list = df["subject"].unique() if cfg.SUBJECTS_TO_PROCESS is None else cfg.SUBJECTS_TO_PROCESS
patterns_list = df["pattern_pair"].unique() if cfg.PATTERNS_TO_PROCESS is None else cfg.PATTERNS_TO_PROCESS

# df_analysis è il sottoinsieme filtrato usato da tutte le analisi
# df originale resta intatto nel caso serva per confronti
df_analysis = df[df["subject"].isin(subjects_list)].copy()
metrics_df  = df_analysis.copy()

print(f"Subjects to analyze: {list(subjects_list)}")
print(f"Patterns to analyze: {list(patterns_list)}")
print("-" * 90)

# --- Heatmaps ---
if cfg.DO_HEATMAPS:
    print("Generating heatmaps...")
    for subj in subjects_list:
        heatmaps.save_subject_heatmaps(
            df, subj, protocol,
            output_folder=OUT_HEATMAPS,
            metric="vividness",
            recalc_subject=cfg.RECALC_SUBJECT,
            start_pos=(11,5)
        )
    if cfg.UPDATE_GROUP_AVERAGE:
        heatmaps.save_all_subjects_heatmaps(df_analysis, protocol, output_folder=OUT_HEATMAPS, metric="vividness",start_pos=(12,3))
else:
    print("SKIPPING heatmaps")

# --- Model parameters ---
if cfg.DO_MODEL_PARAMETERS:
    print("Doing model parameters extraction and global validation...")
    all_params, validation_dfs, Kb_global, Kt_global, group_validation_df = create_subject_and_global_excel(
        df_analysis, protocol, OUT_SUBJ_MOD
    )
    print(f"Global Kb={Kb_global:.3f}, Kt={Kt_global:.3f}")
else:
    print("SKIPPING model parameters")

# --- Regressions ---
if cfg.DO_REGRESSIONS:
    print("Doing regression plots...")
    for pat in patterns_list:
        for subj in subjects_list:
            subj_df = df[df["subject"] == subj].copy()

            if cfg.DO_REGRESSION_ORIGINAL_DURATIONS_SUBJ:
                print(f"  Plotting {subj} | {pat} | duration...")
                plot_regression.plot_subject_pattern(
                    subj_df, subj, pat, protocol,
                    output_folder=os.path.join(OUT_REG, "duration"),
                    x_col="duration", x_label="Duration (s)"
                )

            if cfg.DO_REGRESSION_ORDERED_DURATIONS_SUBJ:
                print(f"  Plotting {subj} | {pat} | presentation_order...")
                plot_regression.plot_subject_pattern(
                    subj_df, subj, pat, protocol,
                    output_folder=os.path.join(OUT_REG, "presentation_order"),
                    x_col="presentation_order", x_label="Presentation Order"
                )

    if cfg.DO_REGRESSION_ORIGINAL_DURATIONS_GLOBAL:
        print("  Plotting ALL_SUBJECTS | duration...")
        plot_regression.plot_group_average(
            df_analysis, patterns_list, protocol,
            output_folder=os.path.join(OUT_REG, "duration", "ALL_SUBJECTS"),
            x_col="duration", x_label="Duration (s)"
        )

    if cfg.DO_REGRESSION_ORDERED_DURATIONS_GLOBAL:
        print("  Plotting ALL_SUBJECTS | presentation_order...")
        plot_regression.plot_group_average(
            df_analysis, patterns_list, protocol,
            output_folder=os.path.join(OUT_REG, "presentation_order", "ALL_SUBJECTS"),
            x_col="presentation_order", x_label="Presentation Order"
        )
else:
    print("SKIPPING regression plots")

# --- Std Excel ---
if cfg.DO_STD_EXCEL:
    print("Generating std normalized Excel...")
    create_std_normalized_excel(
        metrics_df,
        os.path.join(OUT_STD, "Std_Normalized.xlsx"),
        subjects=subjects_list,
        patterns=patterns_list
    )
else:
    print("SKIPPING std excel")

# --- Sigmoid fit ---
if cfg.DO_SIGMOID_FIT:
    if "group_validation_df" not in locals():
        print("SKIP sigmoid: esegui prima con DO_MODEL_PARAMETERS=True")
    else:
        print("Fitting sigmoid to group validation data...")
        print(group_validation_df.head())
        fit_group_sigmoid(group_validation_df, plot=True, output_folder=OUT_VAL)
else:
    print("SKIPPING sigmoid fit")

# --- Cross validation ---
if cfg.DO_CROSS_VALIDATION:
    print("Running LOOCV...")
    loocv_df = loocv_leave_one_subject(df_analysis, os.path.join(OUT_VAL, "LOOCV_sklearn.xlsx"))
else:
    print("SKIPPING cross validation")

print("-" * 90)
print("PHASE 2 ANALYSIS COMPLETED.")
