from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os
from typing import Dict

def loocv_sklearn(df, output_path: str = None):
    """
    Leave-One-Group-Out CV per il modello [pb_sum*D, pt_sum*D] -> angle_deg
    Con pesatura per vividness.
    """

    print("[LOOCV] Costruzione feature matrix...")

    # Feature engineering
    X = []
    y = []
    groups = []
    weights = []
    
    for _, row in df.iterrows():
        b_part, t_part = row["pattern_pair"].split("_")
        pb_sum = sum(int(bit) for bit in b_part)
        pt_sum = sum(int(bit) for bit in t_part)
        
        X.append([pb_sum * row["duration"], pt_sum * row["duration"]])
        y.append(row["angle_deg"])
        weights.append(row["vividness"] / 3.0)
        groups.append(row["subject"])
    
    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    groups = np.array(groups)
    
    # LOOCV per gruppi (soggetti)
    logo = LeaveOneGroupOut()
    r2_scores = []
    r2_weighted_scores = []
    rmse_scores = []
    mae_scores = []
    subjects_tested = []
    
    print(f"[LOOCV] {len(np.unique(groups))} folds (uno per soggetto)")
    
    # All'inizio
    Kb_list = []
    Kt_list = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subject = np.unique(groups[test_idx])[0]
        subjects_tested.append(test_subject)
        
        # Training model
        model = LinearRegression(fit_intercept=False)
        model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        
        # Salva coefficienti per questo fold
        Kb_list.append(model.coef_[0])
        Kt_list.append(model.coef_[1])
        
        # Test predictions
        y_pred = model.predict(X[test_idx])
        y_true = y[test_idx]
        w_test = weights[test_idx]
        
        # Calcola metriche
        #r2 = r2_score(y_true, y_pred)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot_uncentered = np.sum((y_true**2)) # Riferimento a ZERO
        r2 = 1 - (ss_res / ss_tot_uncentered)

        ss_res_weighted = np.sum(w_test * (y_true - y_pred)**2)
        ss_tot_uncentered_weighted = np.sum(w_test * (y_true**2)) # Riferimento a ZERO
        r2_weighted = 1 - (ss_res_weighted / ss_tot_uncentered_weighted)

        #r2_weighted = r2_score(y_true, y_pred, sample_weight=w_test)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=w_test))
        mae = mean_absolute_error(y_true, y_pred, sample_weight=w_test)
        
        r2_scores.append(r2)
        r2_weighted_scores.append(r2_weighted)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"  Fold {fold+1}: {test_subject} | R²={r2:.3f}, R²_weighted={r2_weighted:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    # Poi usa queste liste nel DataFrame
    results_df = pd.DataFrame({
        "subject": subjects_tested,
        "Kb": Kb_list,
        "Kt": Kt_list,
        "R2": r2_scores,
        "R2_weighted": r2_weighted_scores,
        "RMSE_weighted": rmse_scores,
        "MAE_weighted": mae_scores
    })

    # Metriche aggregate
    print("\n" + "="*60)
    print("RISULTATI LOOCV AGGREGATI")
    print("="*60)
    print(f"Mean R² (unweighted):   {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"Mean R² (weighted):   {np.mean(r2_weighted_scores):.3f} ± {np.std(r2_weighted_scores):.3f}")
    print(f"Mean MAE (weighted):  {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    
    # Salva Excel se richiesto
    if output_path:
        if not output_path or not isinstance(output_path, str):
            print("[LOOCV] WARNING: output_path non valido, salto salvataggio Excel")
        else:
            try:
                # Crea cartella SOLO se necessario
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                    results_df.to_excel(writer, sheet_name="LOOCV_per_subject", index=False)
                    
                    summary = pd.DataFrame({
                        'Metric': ['Mean_R2', 'Std_R2', 'Mean_R2_weighted', 'Std_R2_weighted', 'Mean_RMSE_weighted', 'Std_RMSE_weighted', 'Mean_MAE_weighted', 'Std_MAE_weighted'],
                        'Value': [
                            np.mean(r2_scores), np.std(r2_scores),
                            np.mean(r2_weighted_scores), np.std(r2_weighted_scores),
                            np.mean(rmse_scores), np.std(rmse_scores),
                            np.mean(mae_scores), np.std(mae_scores)
                        ]
                    })
                    summary.to_excel(writer, sheet_name="Summary", index=False)
                
                print(f"[LOOCV] ✅ Excel salvato: {output_path}")
                
            except Exception as e:
                print(f"[LOOCV] ❌ Errore salvataggio Excel: {e}")
                print("[LOOCV] Risultati disponibili in 'loocv_results'")

    return results_df

# FUNZIONE PER IL MAIN (compatibilità)
def run_loocv_analysis(df: pd.DataFrame, protocol: Dict, output_filename: str):
    """Wrapper per il main - VERSIONE ROBUSTA"""
    
    print(f"[LOOCV] Saving in: {output_filename}")
    
    return plot_loocv_results(loocv_sklearn(df, output_filename), os.path.dirname(output_filename))

def plot_loocv_results(loocv_df, output_folder):
    """Plot diagnostico LOOCV"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    plt.scatter(loocv_df['MAE_weighted'], loocv_df['R2_weighted'], s=150, alpha=0.7, c='red')
    plt.xlabel('MAE (°)', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.title('LOOCV Performance: MAE vs R² per subject', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Annota outlier
    for i, subj in enumerate(loocv_df['subject']):
        plt.annotate(subj, (loocv_df['MAE_weighted'].iloc[i], loocv_df['R2_weighted'].iloc[i]), 
                    xytext=(5,5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'LOOCV_scatter.png'), dpi=300, bbox_inches='tight')
    plt.show()

