import pandas as pd
def add_zero_point_for_model(df, patterns=None):
    """
    Aggiunge righe con duration=0 e angle_deg=0 per ogni soggetto, pattern,
    replicando trial e rep esistenti per avere coerenza con la struttura del dataframe.
    """
    df_zero = df.copy()
    subjects = df["subject"].unique()
    if patterns is None:
        patterns = df["pattern_pair"].unique()
    
    zero_rows = []
    
    for subj in subjects:
        for pat in patterns:
            # Prendi le combinazioni uniche di trial/rep dal dataframe originale di quel soggetto e pattern
            subset = df[(df["subject"] == subj) & (df["pattern_pair"] == pat)]
            if subset.empty:
                continue
            trial_rep_combos = subset[["trial", "rep"]].drop_duplicates()
            
            for _, row in trial_rep_combos.iterrows():
                new_row = row.to_dict()  # copia trial e rep
                # Copia anche tutte le altre colonne ma setta angle_deg e duration
                for col in df.columns:
                    if col not in new_row:
                        new_row[col] = subset.iloc[0][col]  # copia un valore di riferimento
                new_row["duration"] = 0
                new_row["angle_deg"] = 0.0
                new_row["vividness"] = 3  # peso massimo
                zero_rows.append(new_row)
    
    df_zero = pd.concat([df_zero, pd.DataFrame(zero_rows)], ignore_index=True)
    return df_zero
