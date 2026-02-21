import os
import numpy as np
import pandas as pd

# ----------------------------
# Weighted regression slope through origin
# ----------------------------
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def compute_weighted_slope(x, y, vividness):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    # vividness 1–3 → weight 1/3, 2/3, 1
    weights = np.asarray(vividness) / 3.0

    model = LinearRegression(fit_intercept=False) # fit_intercept=False to make it pass through the origin (D = 0, means no illusion -> DeltaAngle = 0)
    model.fit(x, y, sample_weight=weights)

    slope = model.coef_[0] 
    y_pred = model.predict(x)

    # weighted R² 
    #r2 = model.score(x, y, sample_weight=weights)
    ss_res = np.sum(weights * (y - y_pred)**2)
    ss_tot_uncentered = np.sum(weights * (y**2)) # Riferimento a ZERO
    
    r2 = 1 - (ss_res / ss_tot_uncentered)

    return {
        "slope": slope,
        "r_squared": r2
    }


# ----------------------------
# Extract weighted Ab/At/Kb/Kt for a subject
# ----------------------------
def extract_subject_parameters_weighted(subj_df, durations):
    params = {}

    for pattern, label in [("001_000", "b"), ("000_001", "t")]:
        pat_df = subj_df[subj_df["pattern_pair"] == pattern] # Subset 

        if pat_df.empty:
            params[f"K{label}"] = np.nan
            continue

        # Regression slope K (weighted)
        result = compute_weighted_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values
        )

        K = result["slope"]

        params[f"K{label}"] = K

    return params

# ----------------------------
# Build validation table (ideal vs real)
# ----------------------------
def build_validation_table_weighted(subj_df, params, durations, patterns):
    Kb = params["Kb"]
    Kt = params["Kt"]
    rows = []
    for pattern in patterns:
        b_part, t_part = pattern.split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)

        for D in durations:
            # Ideal (weighted)
            ideal = (
                (Kb * pb_sum * D if not np.isnan(Kb) else 0)
                +
                (Kt * pt_sum * D if not np.isnan(Kt) else 0)
            )

            # Real weighted mean across repetitions
            real_subset = subj_df[(subj_df["pattern_pair"] == pattern) & (subj_df["duration"] == D)]
            if not real_subset.empty:
                real = np.average(real_subset["angle_deg"], weights=real_subset["vividness"]/3)
            else:
                real = np.nan

            error = ideal - real if not np.isnan(real) else np.nan
            abs_error = abs(error) if not np.isnan(error) else np.nan

            rows.append({
                "pattern": pattern,
                "duration": D,
                "ideal_angle": ideal,
                "real_angle": real,
                "error": error,
                "abs_error": abs_error
            })

    return pd.DataFrame(rows)

def compute_global_fit(df):
    """R² del modello LINEARMENTE addestrato su TUTTI i dati"""
    
    # Ricostruisci X, y, weights
    X, y, weights = [], [], []
    for _, row in df.iterrows():
        b_part, t_part = row["pattern_pair"].split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)
        
        X.append([pb_sum * row["duration"], pt_sum * row["duration"]])
        y.append(row["angle_deg"])
        weights.append(row["vividness"] / 3.0)
    
    X, y, weights = np.array(X), np.array(y), np.array(weights)
    
    model_global = LinearRegression(fit_intercept=False)
    model_global.fit(X, y, sample_weight=weights)

    print(f"X shape: {X.shape}, y shape: {y.shape}, weights shape: {weights.shape}")
    

    y_pred = model_global.predict(X)
    # R² PESATO (già ce l'hai): 0.337
    r2_weighted = 1 - np.sum(weights * (y - y_pred)**2) / np.sum(weights * (y)**2)

    # R² NON PESATO (nuovo!)
    r2_unweighted = r2_score(y, y_pred)  # ← UNA RIGA!

    print(f"R² globale PESATO:   {r2_weighted:.3f}")
    print(f"R² globale NON pesato: {r2_unweighted:.3f}")
    
    return {
        "r2_global_weighted": r2_weighted,
        "r2_global_unweighted": r2_unweighted,
        "Kb_global": model_global.coef_[0],
        "Kt_global": model_global.coef_[1]
    }

def create_subject_model_and_global_excel_weighted(df, protocol, subject_output_folder, global_output_path):
    os.makedirs(subject_output_folder, exist_ok=True)
    subjects = df["subject"].unique()
    durations = sorted(protocol["blocks"]["durations"])
    patterns = sorted(df["pattern_pair"].unique())

    parameters_rows = []
    all_params = []
    all_subject_validation = []

    # ================================
    # Aggiungi punto zero per tutti i dati globali
    # ================================
    #df = df.copy()
    #df = add_zero_point_for_model(df, patterns=["001_000","000_001"])

    # ================================
    # Estrai parametri per soggetto
    # ================================
    all_params = {}
    for subj in subjects:
        subj_df = df[df["subject"] == subj].copy()
        params = extract_subject_parameters_weighted(subj_df, durations)
        all_params[subj] = params

    # ================================
    # Parametri medi (Kb/Kt) per gruppo
    # ================================
    mean_params = {
        "Kb": np.nanmean([p["Kb"] for p in all_params.values()]),
        "Kt": np.nanmean([p["Kt"] for p in all_params.values()])
    }

    # ================================
    # Calcolo R² globale (con dati zero inclusi)
    # ================================
    global_fit = compute_global_fit(df)
    global_r2_weighted = global_fit["r2_global_weighted"]
    global_r2_unweighted = global_fit["r2_global_unweighted"]

    # ================================
    # Validation per soggetto
    # ================================
    for subj in subjects:
        print(f"[MODEL WEIGHTED] Processing {subj}")
        subj_df = df[df["subject"] == subj].copy()
        params = all_params[subj]

        # Build subject validation file
        validation_df = build_validation_table_weighted(subj_df, params, durations, patterns)
        group_mean_df = build_validation_table_weighted(subj_df, mean_params, durations, patterns)

        subj_file = os.path.join(subject_output_folder, f"{subj}_model_validation_weighted.xlsx")
        with pd.ExcelWriter(subj_file, engine="openpyxl") as writer:
            validation_df.to_excel(writer, sheet_name="Subject_Model", index=False)
            group_mean_df.to_excel(writer, sheet_name="Group_Mean_Model", index=False)
        print(f"[MODEL WEIGHTED] Saved subject validation file with group mean: {subj_file}")

        parameters_rows.append({"subject": subj, "Kb": params["Kb"], "Kt": params["Kt"]})
        all_subject_validation.append(validation_df)

    # ================================
    # Salva Excel globale
    # ================================
    parameters_out = pd.DataFrame(parameters_rows)
    parameters_mean_row = {
        "subject": "Mean",
        "Kb": np.nanmean(parameters_out["Kb"]),
        "Kt": np.nanmean(parameters_out["Kt"]),
        "Global_R2_weighted": global_r2_weighted,
        "Global_R2_unweighted": global_r2_unweighted
    }
    parameters_out = pd.concat([parameters_out, pd.DataFrame([parameters_mean_row])], ignore_index=True)

    # ================================
    # Costruisci MODEL VALIDATION globale
    # ================================
    rows = []
    for pattern in patterns:
        b_part, t_part = pattern.split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)

        for D in durations:
            row = {"pattern": pattern, "duration": D}
            subject_values = []
            subject_errors = []

            for subj in subjects:
                subj_df_sub = df[df["subject"] == subj]
                subset = subj_df_sub[
                    (subj_df_sub["pattern_pair"] == pattern) &
                    (subj_df_sub["duration"] == D)
                ]
                if not subset.empty:
                    real = np.average(subset["angle_deg"], weights=subset["vividness"] / 3)
                else:
                    real = np.nan
                row[subj] = real
                subject_values.append(real)
                ideal = mean_params["Kb"] * pb_sum * D + mean_params["Kt"] * pt_sum * D
                subject_errors.append(ideal - real if not np.isnan(real) else np.nan)

            subject_values = np.array(subject_values, dtype=float)
            subject_errors = np.array(subject_errors, dtype=float)
            row["ideal_mean"] = mean_params["Kb"] * pb_sum * D + mean_params["Kt"] * pt_sum * D
            row["real_mean"] = np.nanmean(subject_values)
            row["dev_std_mean"] = np.nanstd(subject_values, ddof=1)
            row["error_std_mean"] = np.nanstd(subject_errors, ddof=1)

            rows.append(row)

    group_validation_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(global_output_path), exist_ok=True)
    with pd.ExcelWriter(global_output_path, engine="openpyxl") as writer:
        parameters_out.to_excel(writer, sheet_name="Parameters", index=False)
        group_validation_df.to_excel(writer, sheet_name="Model_validation", index=False)

    print(f"[MODEL WEIGHTED] Global model parameters + group validation saved at {global_output_path}")

    return group_validation_df, parameters_out
# ----------------------------
# Main function
# ----------------------------
"""
def create_subject_model_and_global_excel_weighted(df, protocol, subject_output_folder, global_output_path):
    os.makedirs(subject_output_folder, exist_ok=True)
    subjects = df["subject"].unique()
    durations = sorted(protocol["blocks"]["durations"])
    patterns = sorted(df["pattern_pair"].unique())

    parameters_rows = []
    all_params = {}
    all_subject_validation = []

    # ----------------------------
    # Extract parameters per subject (Ab, Kb, At, Kt)
    # ----------------------------
    for subj in subjects:
        subj_df = df[df["subject"] == subj].copy()
        subj_df = add_zero_point_for_model(subj_df, patterns=["001_000","000_001"], include_reps=True)
        params = extract_subject_parameters_weighted(subj_df, durations)
        all_params[subj] = params

    #  Mean parameters across subjects (for group validation) 
    mean_params = {
        "Kb": np.nanmean([p["Kb"] for p in all_params.values()]),
        "Kt": np.nanmean([p["Kt"] for p in all_params.values()])
    }

    global_r2_weighted = compute_global_fit(df)["r2_global_weighted"]
    global_r2_unweighted = compute_global_fit(df)["r2_global_unweighted"]

    # ----------------------------
    # Build validation tables per subject and save individual Excels
    # ----------------------------
    for subj in subjects:
        print(f"[MODEL WEIGHTED] Processing {subj}")
        subj_df = df[df["subject"] == subj].copy()
        params = all_params[subj] 

        Kb = params["Kb"]
        Kt = params["Kt"]

        # Build subject validation file
        validation_df = build_validation_table_weighted(subj_df, params, durations, patterns)
        group_mean_df = build_validation_table_weighted(subj_df, mean_params, durations, patterns)

        subj_file = os.path.join(subject_output_folder, f"{subj}_model_validation_weighted.xlsx")
        with pd.ExcelWriter(subj_file, engine="openpyxl") as writer:
            validation_df.to_excel(writer, sheet_name="Subject_Model", index=False)
            group_mean_df.to_excel(writer, sheet_name="Group_Mean_Model", index=False)
        print(f"[MODEL WEIGHTED] Saved subject validation file with group mean: {subj_file}")

        parameters_rows.append({"subject": subj, "Kb": Kb, "Kt": Kt})
        all_subject_validation.append(validation_df)

    # ----------------------------
    # Save global Excel with all parameters and group validation
    # ----------------------------
    parameters_out = pd.DataFrame(parameters_rows)
    parameters_mean_row = {
        "subject": "Mean",
        "Kb": np.nanmean(parameters_out["Kb"]),
        "Kt": np.nanmean(parameters_out["Kt"]),
        "Global_R2_weighted": global_r2_weighted,
        "Global_R2_unweighted": global_r2_unweighted
    }
    parameters_out = pd.concat([parameters_out, pd.DataFrame([parameters_mean_row])], ignore_index=True)

    # ----------------------------
    # Build MODEL VALIDATION table (matrix structure)
    # ----------------------------
    rows = []

    for pattern in patterns:
        b_part, t_part = pattern.split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)

        for D in durations:

            row = {
                "pattern": pattern,
                "duration": D
            }

            subject_values = []
            subject_errors = []

            # ---- valori per soggetto ----
            for subj in subjects:
                subj_df = df[df["subject"] == subj]

                subset = subj_df[
                    (subj_df["pattern_pair"] == pattern) &
                    (subj_df["duration"] == D)
                ]

                if not subset.empty:
                    # media delle 3 rep (pesata)
                    real = np.average(
                        subset["angle_deg"],
                        weights=subset["vividness"] / 3
                    )
                else:
                    real = np.nan

                row[subj] = real
                subject_values.append(real)

                # errore rispetto al modello medio
                ideal = (
                    mean_params["Kb"] * pb_sum * D +
                    mean_params["Kt"] * pt_sum * D
                )
                subject_errors.append(ideal - real if not np.isnan(real) else np.nan)

            # ---- statistiche di gruppo ----
            subject_values = np.array(subject_values, dtype=float)
            subject_errors = np.array(subject_errors, dtype=float)

            ideal_mean = (
                mean_params["Kb"] * pb_sum * D +
                mean_params["Kt"] * pt_sum * D
            )

            real_mean = np.nanmean(subject_values)
            dev_std_mean = np.nanstd(subject_values, ddof=1)
            error_std_mean = np.nanstd(subject_errors, ddof=1)

            row["ideal_mean"] = ideal_mean
            row["real_mean"] = real_mean
            row["dev_std_mean"] = dev_std_mean
            row["error_std_mean"] = error_std_mean

            rows.append(row)

    group_validation_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(global_output_path), exist_ok=True)
    with pd.ExcelWriter(global_output_path, engine="openpyxl") as writer:
        parameters_out.to_excel(writer, sheet_name="Parameters", index=False)
        group_validation_df.to_excel(writer, sheet_name="Model_validation", index=False)

    return group_validation_df, parameters_out

    print(f"[MODEL WEIGHTED] Global model parameters + group validation saved at {global_output_path}")
"""