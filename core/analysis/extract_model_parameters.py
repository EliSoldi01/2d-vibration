import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from core.utils.r2_metrics import compute_r2_metrics as r2_metrics

# ----------------------------
# HELPERS
# ----------------------------
def _pattern_sums(pattern):
    """Return (pb_sum, pt_sum) for a pattern string like '011_100'."""
    b, t = pattern.split("_")
    return sum(int(x) for x in b), sum(int(x) for x in t)


def _predict_raw(raw_df, Kb, Kt):
    """
    For every trial in raw_df, compute the model prediction and return
    (y_true, y_pred, weights) as numpy arrays.

    Model:    ΔA = Kb · pb_sum · D  +  Kt · pt_sum · D
    Weights:  vividness / 3.0  (consistent with Kb/Kt estimation)
    """
    pb_map = {p: _pattern_sums(p)[0] for p in raw_df["pattern_pair"].unique()}
    pt_map = {p: _pattern_sums(p)[1] for p in raw_df["pattern_pair"].unique()}

    y_true  = raw_df["angle_deg"].values
    y_pred  = np.array([
        Kb * pb_map[p] * d + Kt * pt_map[p] * d
        for p, d in zip(raw_df["pattern_pair"], raw_df["duration"])
    ])
    weights = raw_df["vividness"].values / 3.0
    return y_true, y_pred, weights


# ----------------------------
# REGRESSION PESATA PER Kb/Kt
# ----------------------------
def compute_weighted_slope(x, y, vividness):
    """
    Linear regression forced through origin with weights based on vividness.
    Returns (slope, y_pred, R²_zero).

    R²_zero is appropriate because:
      - fit_intercept=False  (algebraic constraint)
      - D=0 is observed in data  (empirical constraint)
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    weights = np.asarray(vividness) / 3.0 if vividness is not None else np.ones_like(y)

    model = LinearRegression(fit_intercept=False)
    model.fit(x, y, sample_weight=weights)

    slope  = model.coef_[0]
    y_pred = model.predict(x)
    r2     = r2_metrics(y, y_pred, weights=weights)["R2_zero"]

    return slope, y_pred, r2


# ----------------------------
# ESTRAZIONE PARAMETRI PER SOGGETTO
# ----------------------------
def extract_subject_parameters(subj_df):
    """
    Compute Kb, Kt and their R²_zero for a subject using pure patterns.
      001_000 -> Kb  (biceps only)
      000_001 -> Kt  (triceps only)
    """
    params = {}
    for pattern, label in [("001_000", "b"), ("000_001", "t")]:
        pat_df = subj_df[subj_df["pattern_pair"] == pattern]
        if pat_df.empty:
            params[f"K{label}"]    = np.nan
            params[f"R2_K{label}"] = np.nan
            continue
        slope, _, r2 = compute_weighted_slope(
            x=pat_df["duration"].values,
            y=pat_df["angle_deg"].values,
            vividness=pat_df["vividness"].values,
        )
        params[f"K{label}"]    = slope
        params[f"R2_K{label}"] = r2
    return params


# ----------------------------
# METRICHE SUL MODELLO COMPLETO — RAW DATA
# ----------------------------
def _compute_model_metrics(raw_df, Kb, Kt):
    """
    Compute R²_zero, MAE and RMSE (all weighted by vividness) of the full model
        ΔA = Kb · pb_sum · D  +  Kt · pt_sum · D
    evaluated on raw trial data.

    Why raw data and not means:
        Kb/Kt are estimated on raw data via weighted OLS. Evaluating on means
        would be inconsistent and would optimistically inflate all metrics by
        removing within-cell variance that the model never tried to explain.

    Why R²_zero:
        D=0 is present in every subject's raw data, so the physical constraint
        ΔA=0 at zero stimulus is empirically observed, not just assumed.

    Why weighted MAE / RMSE:
        Vividness weights reflect perceptual reliability. Consistent with
        the estimation step.

    ─── HOW TO INTERPRET MAE ────────────────────────────────────────────────
    MAE = weighted mean of |angolo_reale − angolo_predetto|    [degrees]

    It answers: "on average, by how many degrees does the model miss?"

    - MAE is in the same unit as your outcome (degrees), so it is immediately
      interpretable without any statistical background.
    - It treats all errors equally: a 2° miss counts twice as much as a 1°
      miss, nothing more.
    - It is robust to occasional large errors (outlier trials).

    Practical reference for your task (grid step ≈ 4 cm ≈ few degrees):
      MAE < 5°    excellent  — sub-actuator precision
      MAE 5–10°   acceptable — correct direction and rough magnitude
      MAE > 10°   poor       — model misses a substantial part of the range

    ─── HOW TO INTERPRET RMSE ───────────────────────────────────────────────
    RMSE = sqrt(weighted mean of (angolo_reale − angolo_predetto)²)  [degrees]

    It answers: "what is the typical size of errors, penalising large ones?"

    - RMSE is always ≥ MAE because squaring amplifies large errors.
    - The ratio RMSE / MAE reveals the error distribution:
        RMSE ≈ MAE   → errors are uniform across trials (no dominant outliers)
        RMSE >> MAE  → a few trials have very large errors
    - Use RMSE when you care more about avoiding large misses than about
      average performance (e.g. safety-critical applications).

    Key comparison — in-sample vs LOOCV:
        gap = RMSE_LOOCV − RMSE_insample
        A small gap (< 1–2°) means the model generalises well to new subjects.
        A large gap means Kb/Kt are subject-specific and the global model
        loses important information when applied to a held-out subject.
    ─────────────────────────────────────────────────────────────────────────

    Returns dict with keys: R2_zero, MAE, RMSE  (floats, NaN if empty df)
    """
    if raw_df.empty:
        return {"R2_zero": np.nan, "MAE": np.nan, "RMSE": np.nan}

    y_true, y_pred, weights = _predict_raw(raw_df, Kb, Kt)

    r2   = r2_metrics(y_true, y_pred, weights=weights)["R2_zero"]
    mae  = float(np.average(np.abs(y_true - y_pred),      weights=weights))
    rmse = float(np.sqrt(np.average((y_true - y_pred)**2, weights=weights)))

    return {"R2_zero": r2, "MAE": mae, "RMSE": rmse}


# ----------------------------
# VALIDATION TABLE (for sigmoid fit — means only)
# ----------------------------
def build_validation_table(subj_df, Kb, Kt, patterns, durations):
    """
    Build table comparing ideal vs weighted-mean real angles.
    Used downstream for the sigmoid fit (ideal vs real mean).
    NOT used for performance metrics — use _compute_model_metrics for that.
    """
    rows = []
    for pattern in patterns:
        pb, pt = _pattern_sums(pattern)
        for D in durations:
            ideal  = Kb * pb * D + Kt * pt * D
            subset = subj_df[
                (subj_df["pattern_pair"] == pattern) & (subj_df["duration"] == D)
            ]
            real = (
                np.average(subset["angle_deg"], weights=subset["vividness"] / 3)
                if not subset.empty else np.nan
            )
            rows.append({
                "pattern":     pattern,
                "duration":    D,
                "ideal_angle": ideal,
                "real_angle":  real,
                "abs_error":   np.abs(ideal - real) if not np.isnan(real) else np.nan,
            })
    return pd.DataFrame(rows)


# ----------------------------
# FUNZIONE PRINCIPALE
# ----------------------------
def create_subject_and_global_excel(df, protocol, subject_output_folder):
    """
    Compute Kb/Kt per soggetto e globali, evaluate full model with R²_zero,
    MAE and RMSE on raw data, build validation tables for sigmoid fit.

    Parameters sheet columns:
      Kb                     — biceps slope  (pure pattern 001_000)
      R²_zero (Kb fit)       — goodness of pure-pattern Kb regression
      Kt                     — triceps slope  (pure pattern 000_001)
      R²_zero (Kt fit)       — goodness of pure-pattern Kt regression
      R²_zero (full model)   — full model on all raw trials  [adimensional]
      MAE (full model) [°]   — mean absolute error, weighted  [degrees]
      RMSE (full model) [°]  — root mean squared error, weighted  [degrees]

    For individual subjects: all metrics computed on that subject's own data.
    For GLOBAL row: Kb/Kt estimated on all subjects pooled; metrics on all
    raw trials of all subjects.
    """
    os.makedirs(subject_output_folder, exist_ok=True)
    subjects  = df["subject"].unique()
    durations = sorted(protocol["blocks"]["durations"])
    patterns  = sorted(df["pattern_pair"].unique())

    # ── Per-subject ───────────────────────────────────────────
    all_params     = {}
    validation_dfs = {}

    for subj in subjects:
        subj_df = df[df["subject"] == subj].copy()

        # Step 1: Kb, Kt, R²_zero on pure patterns
        params = extract_subject_parameters(subj_df)

        # Step 2: full model metrics on subject's own raw data
        metrics = _compute_model_metrics(subj_df, params["Kb"], params["Kt"])
        params["R2_full"]   = metrics["R2_zero"]
        params["MAE_full"]  = metrics["MAE"]
        params["RMSE_full"] = metrics["RMSE"]

        all_params[subj]     = params
        validation_dfs[subj] = build_validation_table(
            subj_df, params["Kb"], params["Kt"], patterns, durations
        )

        subj_file = os.path.join(subject_output_folder, f"{subj}_validation.xlsx")
        validation_dfs[subj].to_excel(subj_file, index=False)

    # ── Global Kb / Kt (estimated on all raw data pooled) ─────
    df_all = df.copy()

    Kb_global, _, R2_Kb_global = compute_weighted_slope(
        x=df_all[df_all["pattern_pair"] == "001_000"]["duration"].values,
        y=df_all[df_all["pattern_pair"] == "001_000"]["angle_deg"].values,
        vividness=df_all[df_all["pattern_pair"] == "001_000"]["vividness"].values,
    )
    Kt_global, _, R2_Kt_global = compute_weighted_slope(
        x=df_all[df_all["pattern_pair"] == "000_001"]["duration"].values,
        y=df_all[df_all["pattern_pair"] == "000_001"]["angle_deg"].values,
        vividness=df_all[df_all["pattern_pair"] == "000_001"]["vividness"].values,
    )

    global_metrics = _compute_model_metrics(df_all, Kb_global, Kt_global)

    # ── Global validation table (means — for sigmoid fit) ─────
    group_rows = []
    for pattern in patterns:
        pb, pt = _pattern_sums(pattern)
        for D in durations:
            ideal       = Kb_global * pb * D + Kt_global * pt * D
            real_values = []
            vividness_values = []
            for subj in subjects:
                subset = df[
                    (df["subject"] == subj) &
                    (df["pattern_pair"] == pattern) &
                    (df["duration"] == D)
                ]
                if not subset.empty:
                    real_values.append(
                        np.average(subset["angle_deg"], weights=subset["vividness"] / 3)
                    )
                    vividness_values.append(np.average(subset["vividness"]))
            group_rows.append({
                "trial": subset["trial"].iloc[0] if not subset.empty else np.nan,
                "pattern":     pattern,
                "duration":    D,
                "ideal_angle": ideal,
                "real_mean":   np.nanmean(real_values) if real_values else np.nan,
                "vividness_mean": np.nanmean(vividness_values) if vividness_values else np.nan
            })

    group_validation_df = pd.DataFrame(group_rows)

    print(f"[INFO] Global Kb={Kb_global:.3f} (R²={R2_Kb_global:.3f}), "
          f"Kt={Kt_global:.3f} (R²={R2_Kt_global:.3f})")
    print(f"[INFO] Global full model — "
          f"R²_zero={global_metrics['R2_zero']:.3f}, "
          f"MAE={global_metrics['MAE']:.2f}°, "
          f"RMSE={global_metrics['RMSE']:.2f}°")

    # ── Excel ─────────────────────────────────────────────────
    group_file = os.path.join(subject_output_folder, "group_validation.xlsx")
    with pd.ExcelWriter(group_file, engine="openpyxl") as writer:

        # Sheet 1: Parameters
        params_df = pd.DataFrame(all_params).T
        params_df = params_df[[
            "Kb", "R2_Kb",
            "Kt", "R2_Kt",
            "R2_full", "MAE_full", "RMSE_full",
        ]]
        params_df.columns = [
            "Kb",
            "R²_zero (Kb fit)",
            "Kt",
            "R²_zero (Kt fit)",
            "R²_zero (full model)",
            "MAE (full model) [°]",
            "RMSE (full model) [°]",
        ]
        params_df.loc["GLOBAL"] = {
            "Kb":                    Kb_global,
            "R²_zero (Kb fit)":      R2_Kb_global,
            "Kt":                    Kt_global,
            "R²_zero (Kt fit)":      R2_Kt_global,
            "R²_zero (full model)":  global_metrics["R2_zero"],
            "MAE (full model) [°]":  global_metrics["MAE"],
            "RMSE (full model) [°]": global_metrics["RMSE"],
        }
        params_df.round(3).to_excel(writer, sheet_name="Parameters", index=True)

        # Sheet 2: Global validation table (means — input for sigmoid fit)
        group_validation_df.round(3).to_excel(
            writer, sheet_name="Validation", index=False
        )

    import scipy.stats as stats
        # --- INIZIO BLOCCO TEST STATISTICO ---

    # 1. Estrazione dei coefficienti (usiamo il valore assoluto per Kt perché è negativo)
    # Confrontiamo la "sensibilità" o "guadagno" dei due muscoli
    kb_samples = []
    kt_abs_samples = []

    for subj, p in all_params.items():
        if not np.isnan(p['Kb']) and not np.isnan(p['Kt']):
            kb_samples.append(p['Kb'])
            kt_abs_samples.append(abs(p['Kt']))

    # 2. Esecuzione del test (Wilcoxon è l'alternativa non-parametrica al paired t-test)
    # È più robusto se i soggetti sono pochi o se i dati non sono perfettamente normali
    stat, p_value = stats.wilcoxon(kb_samples, kt_abs_samples)

    # 3. Preparazione dei risultati per Excel
    stats_results = pd.DataFrame({
        "Metric": ["Mean Kb", "Std Kb", "Mean |Kt|", "Std |Kt|", "Difference (Kb - |Kt|)", "Wilcoxon Stat", "p-value"],
        "Value": [
            np.mean(kb_samples), 
            np.std(kb_samples), 
            np.mean(kt_abs_samples), 
            np.std(kt_abs_samples),
            np.mean(kb_samples) - np.mean(kt_abs_samples),
            stat, 
            p_value
        ]
    })

    # Aggiungi un'interpretazione rapida
    stats_results["Significance"] = stats_results["Value"].apply(
        lambda x: "p < 0.05 (*)" if p_value < 0.05 else "n.s." 
        if isinstance(x, float) and x == p_value else ""
    )

    # 4. Salvataggio in una nuova pagina del file Excel
    with pd.ExcelWriter(group_file, engine="openpyxl", mode='a') as writer:
        stats_results.to_excel(writer, sheet_name="Kb_vs_Kt_Test", index=False)

    print(f"\n[STAT TEST] Confronto Kb vs |Kt|: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("-> Esiste una differenza significativa tra l'efficacia di Bicipite e Tricipite.")
    else:
        print("-> Non sono state trovate differenze significative tra i due muscoli.")

    # --- FINE BLOCCO TEST STATISTICO ---



    return all_params, validation_dfs, Kb_global, Kt_global, group_validation_df