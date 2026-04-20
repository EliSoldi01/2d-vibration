import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE PERCORSI E PARAMETRI ---
file_dati = r'C:\Users\Utente\Github\Research\2d-vibration\results\phase1\data_with_angles_and_presentation.xlsx'
file_coeff = r'C:\Users\Utente\Github\Research\2d-vibration\results\phase1\model\subject_models\group_validation.xlsx'

# Mappatura Colori dal tuo JSON
colors_dict = {
    "001_000": "#ffe8bf", "100_011": "#749bce", "011_100": "#ffd080",
    "100_001": "#b7b2b8", "000_111": "#032e64", "011_000": "#ffa100",
    "001_001": "#c9c9c9", "111_000": "#b37100", "000_011": "#3164a5",
    "001_100": "#b7b2b8", "000_001": "#c3d5ea"
}

# Mappatura Marker per Durata
marker_map = {0: 'x', 1: 'o', 3: '^', 6: 's', 9: 'D', 12: 'v'}

# --- 2. DEFINIZIONE MODELLO ---
def saturation_model(E, L, k_gain):
    """
    E: Eccitazione ideale (D * [pb*kb + pt*kt])
    L: Valore di saturazione massimo (asintoto)
    k_gain: Pendenza iniziale (guadagno)
    """
    return L * np.tanh((k_gain / L) * E)

# --- 3. CARICAMENTO E PRE-PROCESSING ---

# Caricamento parametri Kb e Kt (Sheet 'Parameters')
df_params = pd.read_excel(file_coeff, sheet_name='Parameters')
df_params['subject'] = df_params['subject'].astype(str).str.strip()
df_params = df_params.set_index('subject')

# Caricamento dati sperimentali
df = pd.read_excel(file_dati)
df['subject'] = df['subject'].astype(str).str.strip()

def process_data(df, df_params):
    results = []
    for subj in df['subject'].unique():
        if subj in df_params.index:
            subj_df = df[df['subject'] == subj].copy()
            kb, kt = df_params.loc[subj, 'Kb'], df_params.loc[subj, 'Kt']
            
            # Calcolo intensità pattern (somma dei bit attivi)
            def get_int(p): return str(p).strip().count('1')
            pb = subj_df['pattern_biceps'].apply(get_int)
            pt = subj_df['pattern_triceps'].apply(get_int)
            
            # STADIO 1: Calcolo E (Eccitazione Ideale Lineare)
            subj_df['E_ideal'] = subj_df['duration'] * (pb * kb + pt * kt)
            
            # Angolo Reale
            subj_df['real_angle'] = subj_df['angle_deg']
            results.append(subj_df)
    return pd.concat(results)

df_total = process_data(df, df_params)

# --- 4. RAGGRUPPAMENTO (MEDIE DI GRUPPO) ---
df_grouped = df_total.groupby(['pattern_pair', 'duration']).agg({
    'E_ideal': 'mean',
    'real_angle': 'mean',
    'vividness': 'mean'
}).reset_index()

# --- 5. FITTING DELLE COSTANTI DI SATURAZIONE (L, k_gain) ---
# Filtriamo durata 0 per il fit
df_fit = df_grouped[df_grouped['duration'] > 0]
popt, _ = curve_fit(saturation_model, df_fit['E_ideal'], df_fit['real_angle'], 
                     p0=[15, 1.0], sigma=1/df_fit['vividness'])

L_opt, k_gain_opt = popt

# Calcolo della predizione finale corretta
df_grouped['pred_angle'] = saturation_model(df_grouped['E_ideal'], L_opt, k_gain_opt)

# --- 6. GRAFICO FINALE: PREDICTED VS OBSERVED ---
fig, ax = plt.subplots(figsize=(10, 8))

for _, row in df_grouped.iterrows():
    pat = row['pattern_pair']
    dur = row['duration']
    
    ax.scatter(
        row['pred_angle'], # Asse X: Predizione Non-Lineare
        row['real_angle'], # Asse Y: Reale osservato
        color=colors_dict.get(pat, 'black'),
        marker=marker_map.get(dur, 'o'),
        s=150,
        alpha=0.9,
        edgecolors='k',
        label=f"{pat} ({dur}s)" if dur > 0 else None
    )

# Linea di Identità (Y = X)
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Identità (Modello Perfetto)')

ax.set_xlabel('Angolo PREDETTO (Modello con Saturazione) [°]', fontsize=12)
ax.set_ylabel('Angolo OSSERVATO (Media Gruppo) [°]', fontsize=12)
ax.set_title(f'Validazione Modello: L={L_opt:.2f}, k_gain={k_gain_opt:.2f}', fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)

# Legenda pulita
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

print(f"Modello validato.\nSaturazione (L): {L_opt:.2f}\nGuadagno (k): {k_gain_opt:.2f}")