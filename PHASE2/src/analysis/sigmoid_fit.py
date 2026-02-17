import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# MODELLI
# ----------------------------
def linear_model(x, slope):
    """Regressione lineare passante per origine"""
    return slope * x

def sigmoid_model(x, L_min, L_max, k, D0):
    """Sigmoide logistica"""
    return L_min + (L_max - L_min) / (1 + np.exp(-k * (x - D0)))

# ----------------------------
# FIT MODELLI
# ----------------------------
def fit_models(x, y, weights=None, plot=True):
    """
    Confronta modelli: lineare, potenza, sigmoide
    Args:
        x, y: dati
        weights: pesi opzionali
        plot: se True, mostra i grafici
    Returns:
        dict con parametri stimati e R² per ciascun modello
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if weights is None:
        weights = np.ones_like(y)

    results = {}

    # ---------------- Lineare ----------------
    lin_model = LinearRegression(fit_intercept=False)
    lin_model.fit(x.reshape(-1,1), y, sample_weight=weights)
    y_lin_pred = lin_model.predict(x.reshape(-1,1))
    r2_lin = lin_model.score(x.reshape(-1,1), y, sample_weight=weights)
    results['linear'] = {'params': lin_model.coef_[0], 'r2': r2_lin, 'y_pred': y_lin_pred}

    # ---------------- Sigmoide ----------------
    try:
        L0 = max(abs(y))           # plateau positivo stimato
        k0 = 1.0                   # steepness iniziale
        D0 = np.median(x)          # centro iniziale
        p0 = [-L0, L0, k0, D0]

        # limiti


        popt_sig, _ = curve_fit(sigmoid_model, x, y, p0=p0, sigma=1/weights, absolute_sigma=True, maxfev=5000)
        y_sig_pred = sigmoid_model(x, *popt_sig)
        r2_sig = r2_score(y, y_sig_pred, sample_weight=weights)
        results['sigmoid'] = {'params': popt_sig, 'r2': r2_sig, 'y_pred': y_sig_pred}
    except:
        results['sigmoid'] = {'params':[np.nan,np.nan,np.nan], 'r2': np.nan, 'y_pred': np.full_like(y, np.nan)}

    # ---------------- Plot ----------------
    if plot:
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, s=weights*50+10, color='black', label='Data (weights)')
        x_smooth = np.linspace(np.min(x), np.max(x), 200)

        # linee modello
        plt.plot(x_smooth, linear_model(x_smooth, results['linear']['params']), 'b--', label=f'Linear R²={results["linear"]["r2"]:.2f}')
        plt.plot(x_smooth, sigmoid_model(x_smooth, *results['sigmoid']['params']), 'r-', label=f'Sigmoid R²={results["sigmoid"]["r2"]:.2f}')

        plt.xlabel("Ideal mean")
        plt.ylabel("Real mean")
        plt.title("Model comparison")
        plt.legend()
        plt.show()

    return results

def fit_group_sigmoid(df, plot=True):
    x = df["ideal_mean"].values
    y = df["real_mean"].values

    fit_models(x, y, plot=plot)

    
