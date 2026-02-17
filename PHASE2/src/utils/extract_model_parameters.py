import os
import numpy as np
import pandas as pd

# ----------------------------
# Weighted regression slope through origin
# ----------------------------
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_weighted_slope(x, y, vividness):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)

    # vividness 1–3 → weight 1/3, 2/3, 1
    weights = np.asarray(vividness) / 3.0

    model = LinearRegression(fit_intercept=False) # fit_intercept=False to make it pass through the origin (D = 0, means no illusion -> DeltaAngle = 0)
    model.fit(x, y, sample_weight=weights)

    slope = model.coef_[0] 

    # weighted R² 
    r2 = model.score(x, y, sample_weight=weights)

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
        pat_df = subj_df[subj_df["pattern_pair"] == pattern] # Subset for this pattern

        if pat_df.empty:
            params[f"A{label}"] = np.nan
            params[f"K{label}"] = np.nan
            continue

        # Ab / At = weighted mean at duration 1
        subset_1s = pat_df[pat_df["duration"] == 1]
        if not subset_1s.empty:
            A = np.average(
                subset_1s["angle_deg"],
                weights=subset_1s["vividness"] / 3
            )
        else:
            A = np.nan

        # Regression slope K (weighted)
        result = compute_weighted_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values
        )

        K = result["slope"]

        params[f"A{label}"] = A
        params[f"K{label}"] = K

    return params

# ----------------------------
# Build validation table (ideal vs real)
# ----------------------------
def build_validation_table_weighted(subj_df, params, durations, patterns):
    Ab, Kb = params["Ab"], params["Kb"]
    At, Kt = params["At"], params["Kt"]
    rows = []
    for pattern in patterns:
        b_part, t_part = pattern.split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)

        for D in durations:
            # Ideal (weighted)
            ideal = (
                (np.abs(Ab) * Kb * pb_sum * D if not np.isnan(Kb) and not np.isnan(Ab) else 0)
                +
                (np.abs(At) * Kt * pt_sum * D if not np.isnan(Kt) and not np.isnan(At) else 0)
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

# ----------------------------
# Main function
# ----------------------------
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
        params = extract_subject_parameters_weighted(subj_df, durations)
        all_params[subj] = params

    #  Mean parameters across subjects (for group validation) 
    mean_params = {
        "Ab": np.nanmean([p["Ab"] for p in all_params.values()]),
        "Kb": np.nanmean([p["Kb"] for p in all_params.values()]),
        "At": np.nanmean([p["At"] for p in all_params.values()]),
        "Kt": np.nanmean([p["Kt"] for p in all_params.values()])
    }

    # ----------------------------
    # Build validation tables per subject and save individual Excels
    # ----------------------------
    for subj in subjects:
        print(f"[MODEL WEIGHTED] Processing {subj}")
        subj_df = df[df["subject"] == subj].copy()
        params = all_params[subj] 

        Ab, Kb = params["Ab"], params["Kb"]
        At, Kt = params["At"], params["Kt"]

        # Build subject validation file
        validation_df = build_validation_table_weighted(subj_df, params, durations, patterns)
        group_mean_df = build_validation_table_weighted(subj_df, mean_params, durations, patterns)

        subj_file = os.path.join(subject_output_folder, f"{subj}_model_validation_weighted.xlsx")
        with pd.ExcelWriter(subj_file, engine="openpyxl") as writer:
            validation_df.to_excel(writer, sheet_name="Subject_Model", index=False)
            group_mean_df.to_excel(writer, sheet_name="Group_Mean_Model", index=False)
        print(f"[MODEL WEIGHTED] Saved subject validation file with group mean: {subj_file}")

        parameters_rows.append({"subject": subj, "Ab": Ab, "Kb": Kb, "At": At, "Kt": Kt})
        all_subject_validation.append(validation_df)

    # ----------------------------
    # Save global Excel with all parameters and group validation
    # ----------------------------
    parameters_out = pd.DataFrame(parameters_rows)
    parameters_mean_row = {
        "subject": "Mean",
        "Ab": np.nanmean(parameters_out["Ab"]),
        "Kb": np.nanmean(parameters_out["Kb"]),
        "At": np.nanmean(parameters_out["At"]),
        "Kt": np.nanmean(parameters_out["Kt"])
    }
    parameters_out = pd.concat([parameters_out, pd.DataFrame([parameters_mean_row])], ignore_index=True)

    # ----------------------------
    # Build global validation table (mean across subjects)
    # ----------------------------
    group_validation_rows = []

    for pattern in patterns:
        for D in durations:
            ideal_vals = []
            real_vals = []
            # Takes the ideal and real values for this pattern and duration from each subject's validation table
            for i, subj in enumerate(subjects):
                subj_val_df = all_subject_validation[i]  # validation table for this subject
                row = subj_val_df[(subj_val_df["pattern"] == pattern) & (subj_val_df["duration"] == D)]
                if not row.empty:
                    ideal_vals.append(row["ideal_angle"].values[0])
                    real_vals.append(row["real_angle"].values[0])
                else:
                    ideal_vals.append(np.nan)
                    real_vals.append(np.nan)

            # Media su soggetti
            ideal_mean = np.nanmean(ideal_vals)
            real_mean = np.nanmean(real_vals)
            error_mean = ideal_mean - real_mean
            abs_error_mean = abs(error_mean)

            group_validation_rows.append({
                "pattern": pattern,
                "duration": D,
                "ideal_mean": ideal_mean,
                "real_mean": real_mean,
                "error_mean": error_mean,
                "abs_error_mean": abs_error_mean
            })

    group_validation_df = pd.DataFrame(group_validation_rows)

    os.makedirs(os.path.dirname(global_output_path), exist_ok=True)
    with pd.ExcelWriter(global_output_path, engine="openpyxl") as writer:
        parameters_out.to_excel(writer, sheet_name="Parameters", index=False)
        group_validation_df.to_excel(writer, sheet_name="Group_Validation", index=False)
    
    return group_validation_df

    print(f"[MODEL WEIGHTED] Global model parameters + group validation saved at {global_output_path}")
