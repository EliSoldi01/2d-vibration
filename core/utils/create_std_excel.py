import numpy as np
import pandas as pd
import os

def std_normalized(x, eps=1e-8):
    """
    Normalized standard deviation.

    Parameters
    ----------
    x : array-like
        Input values (e.g. angle, vividness).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        std(x) / (max(x) - min(x))
        Returns np.nan if not computable.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        return np.nan

    denom = np.nanmax(x) - np.nanmin(x)

    if denom < eps:
        return np.nan

    return np.nanstd(x, ddof=1) / denom

def create_std_normalized_excel(metrics_df, output_excel, subjects=None, patterns=None):
    """
    Crea il file Excel con media, std e std normalized
    per ogni soggetto, pattern e duration.
    
    metrics_df: DataFrame con almeno ['subject', 'pattern_pair', 'duration', 'angle_deg', 'vividness']
    output_excel: path completo del file Excel da salvare
    subjects: lista opzionale di soggetti da includere
    patterns: lista opzionale di pattern da includere
    """
    if subjects is not None:
        metrics_df = metrics_df[metrics_df["subject"].isin(subjects)]
    if patterns is not None:
        metrics_df = metrics_df[metrics_df["pattern_pair"].isin(patterns)]
    
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:

        for pattern in metrics_df["pattern_pair"].unique():
            df_pat = metrics_df[metrics_df["pattern_pair"] == pattern]
            rows = []

            for subj in df_pat["subject"].unique():
                df_subj = df_pat[df_pat["subject"] == subj]

                for dur, group in df_subj.groupby("duration"):
                    rows.append({
                        "Subject": subj,
                        "Duration": dur,
                        "Mean_Vividness": round(group["vividness"].mean(),2),
                        "Std_Vividness": round(group["vividness"].std(),2),
                        "StdNorm_Vividness": round(std_normalized(group["vividness"]),2),
                        "Mean_Angle": round(group["angle_deg"].mean(),2),
                        "Std_Angle": round(group["angle_deg"].std(),2),
                        "StdNorm_Angle": round(std_normalized(group["angle_deg"]),2)
                    })

            df_sheet = pd.DataFrame(rows).sort_values("Subject")
            df_sheet.to_excel(writer, sheet_name=pattern, index=False)

    print(f"[INFO] Std normalized Excel saved at: {output_excel}")
