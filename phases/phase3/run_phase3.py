import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import warnings
import phases.phase3.config as cfg

from core.data_io         import load_data, validate_data
from core.utils           import geometry, ordering
from core.analysis.extract_model_parameters import create_subject_and_global_excel
from core.analysis.model_fit                import fit_group_sigmoid
from core.analysis.cross_validation         import loocv_leave_one_subject
from core.visualization   import heatmaps
from core.visualization   import plot_regressions as plot_regression

warnings.simplefilter("ignore", category=FutureWarning)

print("RUNNING PHASE 3 ANALYSIS")
print("-" * 60)

OUT          = cfg.OUTPUT_ROOT
OUT_REG      = os.path.join(OUT, "regressions")
OUT_MODEL    = os.path.join(OUT, "model")
OUT_SUBJ_MOD = os.path.join(OUT_MODEL, "subject_models")
OUT_VAL      = os.path.join(OUT_MODEL, "model_validation")
for f in [OUT, OUT_REG, OUT_MODEL]: os.makedirs(f, exist_ok=True)

# --- Caricamento ---
protocol     = load_data.load_protocol(cfg.PROTOCOL_PATH)
maindata     = load_data.load_main_data(cfg.DATA_PATH)
subject_info = load_data.load_data_file(cfg.SUBJECTS_PATH)

df = load_data.merge_subject_data(maindata, subject_info)
df = load_data.preprocess_data(df)
print("Dati caricati e preprocessati.")

if not validate_data.validate_protocol(protocol):
    raise ValueError("Protocol non valido")
if not validate_data.validate_subject_data(df, protocol):
    raise ValueError("Dati non validi")

# --- Arricchimento ---
df = geometry.add_angle_column(df, cfg.START_CELL, arm=cfg.ARM)
subject_orders = subject_info.set_index("subject")["block_order"].astype(str).to_dict()
df = ordering.add_presentation_order(df, subject_orders)
df = ordering.add_presentation_pos(df, subject_orders)
df.to_excel(os.path.join(OUT, "data_with_angles.xlsx"), index=False)
print(f"DataFrame salvato in: {OUT}")

subjects_list = df["subject"].unique() if cfg.SUBJECTS_TO_PROCESS is None else cfg.SUBJECTS_TO_PROCESS
patterns_list = df["pattern_pair"].unique() if cfg.PATTERNS_TO_PROCESS is None else cfg.PATTERNS_TO_PROCESS

print(f"Soggetti: {list(subjects_list)}")
print(f"Pattern:  {list(patterns_list)}")
print("-" * 60)

# --- Heatmaps ---
if cfg.DO_HEATMAPS:
    print("Generando heatmaps...")
    for subj in subjects_list:
        heatmaps.save_subject_heatmaps(
            df, subj, protocol,
            output_folder=OUT,
            metric="vividness",
            recalc_subject=True
        )
    heatmaps.save_all_subjects_heatmaps(
        df, protocol,
        output_folder=OUT,
        metric="vividness"
    )
    print("Heatmaps completate.")
else:
    print("SKIP heatmaps")

# --- Regressions ---
if cfg.DO_REGRESSIONS:
    print("Generando regression plots...")
    for pat in patterns_list:
        for subj in subjects_list:
            subj_df = df[df["subject"] == subj].copy()
            plot_regression.plot_subject_pattern(
                subj_df, subj, pat, protocol,
                output_folder=os.path.join(OUT_REG, "duration"),
                x_col="duration", x_label="Duration (s)"
            )
    plot_regression.plot_group_average(
        df, patterns_list, protocol,
        output_folder=os.path.join(OUT_REG, "duration", "ALL_SUBJECTS"),
        x_col="duration", x_label="Duration (s)"
    )
    print("Regression plots completati.")
else:
    print("SKIP regressions")

# --- Model parameters ---
if cfg.DO_MODEL_PARAMETERS:
    print("Estraendo parametri modello...")
    _, _, Kb, Kt, group_val_df = create_subject_and_global_excel(
        df, protocol, OUT_SUBJ_MOD
    )
    print(f"Kb={Kb:.3f}, Kt={Kt:.3f}")
else:
    print("SKIP model parameters")

# --- Sigmoid fit ---
if cfg.DO_SIGMOID_FIT:
    if "group_val_df" not in locals():
        print("SKIP sigmoid: esegui prima con DO_MODEL_PARAMETERS=True")
    else:
        print("Fitting sigmoid...")
        fit_group_sigmoid(group_val_df, True, output_folder=OUT_VAL)
else:
    print("SKIP sigmoid fit")

# --- Cross validation ---
if cfg.DO_CROSS_VALIDATION:
    print("Eseguendo LOOCV...")
    loocv_leave_one_subject(df, os.path.join(OUT_VAL, "LOOCV.xlsx"))
else:
    print("SKIP cross validation")

print("=" * 60)
print("PHASE 3 COMPLETED.")
