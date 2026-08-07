from cProfile import label

import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from core.utils.r2_metrics import compute_r2_metrics as r2_score

logger = logging.getLogger(__name__)

# ============================================================
# MODELLI
# ============================================================

def linear_model(x, slope):
    """Regressione lineare passante per origine: f(x) = slope * x"""
    return slope * x


def tanh_sigmoid(x, L, k):
    """
    Tangent sigmoid with 2 parameters:
    - L : float - max angle (asymptote)
    - k : float - steepness

        f(x) = L * tanh(k * x)

    It is forced to pass through the origin (f(0)=0) and is antisymmetric.
    """
    return L * np.tanh(k * x)


def logistic_sigmoid(x, L_min, L_max, k, D0):
    """
    Sigmoide logistica generica a 4 parametri.

    f(x) = L_min + (L_max - L_min) / (1 + exp(-k * (x - D0)))

    Parametri (4):
        L_min : float - min asymptote 
        L_max : float - max asymptote
        k     : float - steepness
        D0    : float - inflection point (x value at midpoint)
    """
    return L_min + (L_max - L_min) / (1 + np.exp(-k * (x - D0)))


# ============================================================
# AIC
# ============================================================

def aic(y_true, y_pred, k_params, weights=None):
    """
    Akaike Information Criterion (AIC). It balances model fit and complexity:

    AIC = n * log(SSE / n) + 2 * k_params

    Args:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values
    - k_params: int, number of parameters in the model
    - weights: array-like or None, optional weights for each observation

    Returns:
    - AIC value (float)

    Note:
    - Lower AIC indicates a better model fit, balancing goodness of fit and model complexity.
    - AIC is used for model comparison; absolute values are not interpretable on their own.

    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights)
    sse = np.sum(weights * (y_true - y_pred) ** 2)
    return n * np.log(sse / n) + 2 * k_params


def _fit_one(fn, x, y, weights, p0, bounds, k_params):
    """
    Helper: fitta un modello con curve_fit e restituisce il dict risultato.
    Restituisce NaN su tutti i campi in caso di fallimento.
    """
    try:
        popt, pcov = curve_fit(
            fn, x, y,
            p0=p0,
            bounds=bounds,
            sigma=1.0 / np.clip(weights, 1e-6, None),
            absolute_sigma=True,
            maxfev=10000,
        )
        y_pred = fn(x, *popt)
        r2 = r2_score(y, y_pred, weights=weights)
        return {
            "params": popt,
            "params_std": np.sqrt(np.diag(pcov)),
            "r2_zero": r2["R2_zero"],
            "r2_mean": r2["R2_mean"],
            "aic": aic(y, y_pred, k_params=k_params, weights=weights),
            "y_pred": y_pred,
            "ok": True,
        }
    except Exception as exc:
        logger.warning(f"[sigmoid_fit] fit failed for {fn.__name__}: {exc}")
        return {
            "params": np.full(k_params, np.nan),
            "params_std": np.full(k_params, np.nan),
            "r2_zero": np.nan,
            "r2_mean": np.nan,
            "aic": np.nan,
            "y_pred": np.full_like(y, np.nan),
            "ok": False,
        }


# ============================================================
# FIT MODELLI
# ============================================================

def fit_models(x, y, weights=None, plot=True, output_folder=None, labels=None):
    """
    Compare linear, tanh sigmoid (2p) and logistic sigmoid (4p) fits to the data.

    Args:
        x, y         : array
        weights      : array or None
        plot         : bool - if True, saves a comparison plot in output_folder
        output_folder: str - where to save the plot (if plot=True)
    Returns:
        dict with keys 'linear', 'tanh', 'logistic4'
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float)

    results = {}

    # 1. Linear (1 param)
    lin_model = LinearRegression(fit_intercept=False)
    lin_model.fit(x.reshape(-1, 1), y, sample_weight=weights)
    y_lin = lin_model.predict(x.reshape(-1, 1))
    r2_lin = r2_score(y, y_lin, weights=weights)
    results["linear"] = {
        "params": np.array([lin_model.coef_[0]]),
        "r2_zero": r2_lin["R2_zero"],
        "r2_mean": r2_lin["R2_mean"],
        "aic": aic(y, y_lin, k_params=1, weights=weights),
        "y_pred": y_lin,
        "ok": True,
    }

    # 2. Tanh (2 param, pass through origin, antisymmetric)
    L0 = float(np.max(np.abs(y)))
    results["tanh"] = _fit_one(
        tanh_sigmoid, x, y, weights,
        p0=[L0, 0.1],
        bounds=([0, 1e-4], [np.inf, np.inf]),
        k_params=2,
    )

    # 3. Logistic 4 param (no constraints)
    results["logistic4"] = _fit_one(
        logistic_sigmoid, x, y, weights,
        p0=[-L0, L0, 0.1, np.median(x)],
        bounds=([-np.inf, -np.inf, 1e-4, -np.inf],
                [np.inf,  np.inf,  np.inf,  np.inf]),
        k_params=4,
    )

    _print_summary(results)

    if plot and labels is None:
        _plot_comparison(x, y, weights, results, output_folder)
        
    elif plot and labels is not None:
        _plot_comparison_with_labels(x, y, weights, results, output_folder, all_models = False, labels=labels)

    return results


# ============================================================
# SUMMARY & PLOT
# ============================================================

def _print_summary(results):
    """Stampa tabella di confronto modelli con DELTA_AIC rispetto al migliore."""
    valid_aics = {k: r["aic"] for k, r in results.items() if not np.isnan(r["aic"])}
    best_aic = min(valid_aics.values())

    print("\n== Model comparison =============================================")
    print(f"  {'Model':<16} {'k':>4} {'R2_zero':>9} {'R2_mean':>9} {'AIC':>10} {'DELTA':>8}")
    print(f"  {'-'*60}")

    labels = {"linear": "Linear (1p)", "tanh": "Tanh (2p)", "logistic4": "Logistic (4p)"}
    k_map  = {"linear": 1, "tanh": 2, "logistic4": 4}

    for key, label in labels.items():
        r = results[key]
        delta = r["aic"] - best_aic if not np.isnan(r["aic"]) else np.nan
        marker = " <- best" if abs(delta) < 1e-6 else ""
        print(f" {r['r2_zero']:>9.3f} {r['r2_mean']:>9.3f} "
              f"{r['aic']:>10.2f} {delta:>8.2f}{marker}")

    print(f"\n  AIC: lower = better | DELTA > 2 significativo | > 6 sostanziale")
    print("=================================================================\n")

def get_best_model(results):
    """
    Return the best fitted model according to AIC.
    """
    valid_models = {
        k: v for k, v in results.items()
        if v.get("ok", False) and not np.isnan(v["aic"])
    }

    if not valid_models:
        return None, None

    best_name = min(valid_models, key=lambda k: valid_models[k]["aic"])
    return best_name, valid_models[best_name]

def plot_grouped_scatter(
    df,
    selected_patterns=None,
    protocol_path=None,
    output_folder=None,
    results=None
):
    """
    Scatter plot avanzato con overlay Tanh e legenda ordinata.
    
    Args:
        df               : DataFrame con colonne 'ideal_angle', 'real_mean', 'pattern', 'duration'
        selected_patterns: list di pattern da plottare (default tutti)
        protocol_path    : path al protocol.json (per colori e ordine legenda)
        output_folder    : cartella dove salvare il plot
        tanh_params      : tuple (L, k) della tanh da sovrapporre
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import json

    # 2. Estrai parametri tanh
    tanh_params = results["tanh"]["params"] if results.get("tanh", {}).get("ok") else None
    valid_aics = {k: r["aic"] for k, r in results.items() if not np.isnan(r["aic"])}
    best_aic = min(valid_aics.values())
    x_smooth = np.linspace(df['ideal_angle'].min(), df['ideal_angle'].max(), 300)

    # --- filtro durata ---
    df = df[df['duration'] != 0]

    # --- filtro pattern ---
    if selected_patterns is not None:
        df = df[df['pattern'].isin(selected_patterns)]

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- colori da protocol.json ---
    colors = {}
    legend_order = {}
    if protocol_path is not None:
        with open(protocol_path, 'r') as f:
            protocol = json.load(f)

        definitions = protocol.get("patterns", {}).get("definitions", [])
        for d in definitions:
            text = d.get("text")
            color = d.get("color")
            pos = d.get("legend_position", 999)
            if text and color:
                colors[text] = color
                legend_order[text] = pos

    # --- marker per durata ---
    marker_map = {}

    if protocol:
        duration_markers = protocol.get("blocks", {}).get("duration_markers", {})

        for duration, marker in duration_markers.items():
            marker_map[int(duration)] = marker

    # fallback
    if not marker_map:
        marker_map = {
            3: 'o',
            6: '^',
            9: 's',
            15: 'D'
        }

    # fallback colori automatici
    missing_patterns = set(df['pattern'].unique()) - set(colors.keys())
    if missing_patterns:
        cmap = plt.cm.tab20
        for i, p in enumerate(sorted(missing_patterns)):
            colors[p] = cmap(i / max(1, len(missing_patterns)))
            legend_order[p] = 999

    # --- plot scatter ---
    for _, row in df.iterrows():
        x = row['ideal_angle']
        y = row['real_mean']
        pat = row['pattern']
        dur = row['duration']
        ax.scatter(
            x, y,
            color=colors.get(pat, 'black'),
            marker=marker_map.get(dur, 'o'),
            s=100,
            alpha=0.9,
            label=f"{pat} {dur}"
        )

    # --- TANH overlay ---
    # tanh = results["tanh"]
    #d_tanh = tanh["aic"] - best_aic
    #if tanh_params is not None:
    #    L, k = tanh_params
    #    x_smooth = np.linspace(df['ideal_angle'].min(), df['ideal_angle'].max(), 300)
    #    y_tanh = L * np.tanh(k * x_smooth)
    #    ax.plot(x_smooth,
    #    tanh_sigmoid(x_smooth, tanh["params"][0], tanh["params"][1]),
    #    "r-", 
    #    linewidth=2,
    #    label=f"Tanh  AIC={tanh['aic']:.1f}  \n dAIC={d_tanh:.1f}  \n R^2={tanh['r2_mean']:.3f}")

    # --- BEST MODEL overlay ---
    best_name, best_model = get_best_model(results)

    if best_model is not None:

        d_best = best_model["aic"] - best_aic

        if best_name == "linear":

            y_fit = linear_model(
                x_smooth,
                best_model["params"][0]
            )

            style = "b--"


        elif best_name == "tanh":

            y_fit = tanh_sigmoid(
                x_smooth,
                best_model["params"][0],
                best_model["params"][1]
            )

            style = "r-"


        elif best_name == "logistic4":

            p = best_model["params"]

            y_fit = logistic_sigmoid(
                x_smooth,
                p[0],
                p[1],
                p[2],
                p[3]
            )

            style = "g-."


        ax.plot(
            x_smooth,
            y_fit,
            style,
            linewidth=2,
            label=(
                f"{best_name} (best AIC)\n"
                f"AIC={best_model['aic']:.1f}\n"
                f"ΔAIC={d_best:.1f}\n"
                f"R²={best_model['r2_mean']:.3f}"
            )
        )
        
        # --- gestione legenda ---
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))

    # ordina solo secondo legend_position
    def sort_key(label):
        if "(best AIC)" in label:
            return 9999
    
        pat = label.split()[0]
        return legend_order.get(pat, 999)

    sorted_items = sorted(unique.items(), key=lambda x: sort_key(x[0]))
    sorted_labels, sorted_handles = zip(*sorted_items)

    # --- assi centrali ---
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_ylim(-max(abs(df['real_mean'].min()), df['real_mean'].max()) * 1.1,
                max(abs(df['real_mean'].min()), df['real_mean'].max()) * 1.1)
    ax.set_xlim(df['ideal_angle'].min()*1.1, df['ideal_angle'].max()*1.1)

    # --- labels ---
    ax.set_xlabel("Ideal mean Δθ (°)", fontsize=20)
    ax.set_ylabel("Real mean Δθ (°)", fontsize=20)

    # --- legenda ---
    ax.legend(
        sorted_handles,
        sorted_labels,
        bbox_to_anchor=(1.02,1),
        loc="best",
        fontsize=16,
        frameon = False,
        ncol=2  # se vuoi più colonne, aumentale
    )

    ax.grid(True, alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # lascia spazio sotto per la legenda

    # --- save ---
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "grouped_scatter.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[plot_grouped_scatter] Saved: {out_path}")

    plt.show()
    plt.close(fig)

def _plot_comparison(x, y, weights, results, output_folder):
    """Comparison plot of the three models with AIC in legend."""
    x_smooth = np.linspace(np.min(x), np.max(x), 300)

    valid_aics = {k: r["aic"] for k, r in results.items() if not np.isnan(r["aic"])}
    best_aic = min(valid_aics.values())

    fig, ax = plt.subplots(figsize=(9, 6))

    plt.rcParams['font.family'] = 'Times New Roman'

    # Scatter dati
    ax.scatter(x, y, s=weights * 50 + 10, color="black",
               alpha=0.7, label="Data", zorder=5)

    lin  = results["linear"]
    tanh = results["tanh"]
    log4 = results["logistic4"]
    
    # Lineare
    d_lin = lin["aic"] - best_aic
    ax.plot(x_smooth, linear_model(x_smooth, lin["params"][0]),
            "b--", linewidth=2,
            #label=f"Linear  slope={lin['params'][0]:.2f}  "
            #      f"AIC={lin['aic']:.1f}  dAIC={d_lin:.1f}  R^2={lin['r2_zero']:.3f}")        
            label=f"Linear  AIC={lin['aic']:.1f}  dAIC={d_lin:.1f}  R^2={lin['r2_zero']:.3f}")

    # Tanh
    if tanh["ok"]:
        d_tanh = tanh["aic"] - best_aic
        ax.plot(x_smooth,
                tanh_sigmoid(x_smooth, tanh["params"][0], tanh["params"][1]),
                "r-", linewidth=2,
                #label=f"Tanh  L={tanh['params'][0]:.2f}  k={tanh['params'][1]:.3f}  "
                #      f"AIC={tanh['aic']:.1f}  dAIC={d_tanh:.1f}  R^2={tanh['r2_mean']:.3f}")
                label=f"Tanh  AIC={tanh['aic']:.1f}  dAIC={d_tanh:.1f}  R^2={tanh['r2_mean']:.3f}")

    # Logistica 4p
    if log4["ok"]:
        d_log4 = log4["aic"] - best_aic
        p = log4["params"]
        ax.plot(x_smooth,
                logistic_sigmoid(x_smooth, p[0], p[1], p[2], p[3]),
                "g-.", linewidth=2,
                #label=f"Logistic4  [{p[0]:.1f},{p[1]:.1f}]  k={p[2]:.3f}  D0={p[3]:.1f}  "
                #      f"AIC={log4['aic']:.1f}  dAIC={d_log4:.1f} R^2={log4['r2_mean']:.3f}")
                label = f"Logistic4  AIC={log4['aic']:.1f}  dAIC={d_log4:.1f} R^2={log4['r2_mean']:.3f}")

    ax.set_xlabel("Ideal mean Δθ (°)", fontsize=20)
    ax.set_ylabel("Real mean Δθ (°)", fontsize=20)
    ax.legend(fontsize=14, loc="upper left")
    ax.grid(False, alpha=0.3)
    fig.tight_layout()

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "Figure2B_v2.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[sigmoid_fit] Plot saved: {out_path}")
    else:
        logger.warning("[sigmoid_fit] output_folder not provided -- plot will not be saved.")

    plt.show()
    plt.close(fig)

def _plot_comparison_with_labels(x, y, weights, results, output_folder, all_models = False, labels=None):
    """Comparison plot of the three models with AIC in legend."""
    x_smooth = np.linspace(np.min(x), np.max(x), 300)

    valid_aics = {k: r["aic"] for k, r in results.items() if not np.isnan(r["aic"])}
    best_aic = min(valid_aics.values())

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter dati
    ax.scatter(x, y, s=weights * 50 + 10, color="black",
               alpha=0.7, label="Data", zorder=5)

    # --- aggiungi etichette se presenti ---
    if labels is not None:
        for i, lbl in enumerate(labels):
            if lbl:
                ax.annotate(
                    lbl,
                    xy=(x[i], y[i]),
                    xytext=(-5,5),          # 5 punti in x e y
                    textcoords='offset points',
                    fontsize=10,
                    ha='left',
                    va='bottom',
                    color='black'
                )

    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.4)

    tanh = results["tanh"]

    # Tanh
    if tanh["ok"]:
        d_tanh = tanh["aic"] - best_aic
        ax.plot(x_smooth,
                tanh_sigmoid(x_smooth, tanh["params"][0], tanh["params"][1]),
                "r-", linewidth=2,
                #label=f"Tanh  L={tanh['params'][0]:.2f}  k={tanh['params'][1]:.3f}  "
                #      f"AIC={tanh['aic']:.1f}  dAIC={d_tanh:.1f}  R^2={tanh['r2_mean']:.3f}")
                label=f"Tanh  AIC={tanh['aic']:.1f}  dAIC={d_tanh:.1f}  R^2={tanh['r2_mean']:.3f}")

    if all_models:
        lin  = results["linear"]
        log4 = results["logistic4"]

        # Lineare
        d_lin = lin["aic"] - best_aic
        ax.plot(x_smooth, linear_model(x_smooth, lin["params"][0]),
                "b--", linewidth=2,
                #label=f"Linear  slope={lin['params'][0]:.2f}  "
                #      f"AIC={lin['aic']:.1f}  dAIC={d_lin:.1f}  R^2={lin['r2_zero']:.3f}")        
                label=f"Linear  AIC={lin['aic']:.1f}  dAIC={d_lin:.1f}  R^2={lin['r2_zero']:.3f}")

        # Logistica 4p
        if log4["ok"]:
            d_log4 = log4["aic"] - best_aic
            p = log4["params"]
            ax.plot(x_smooth,
                    logistic_sigmoid(x_smooth, p[0], p[1], p[2], p[3]),
                    "g-.", linewidth=2,
                    #label=f"Logistic4  [{p[0]:.1f},{p[1]:.1f}]  k={p[2]:.3f}  D0={p[3]:.1f}  "
                    #      f"AIC={log4['aic']:.1f}  dAIC={d_log4:.1f} R^2={log4['r2_mean']:.3f}")
                    label = f"Logistic4  AIC={log4['aic']:.1f}  dAIC={d_log4:.1f} R^2={log4['r2_mean']:.3f}")

    ax.set_xlabel("Ideal mean Δθ (deg)", fontsize=20)
    ax.set_ylabel("Real mean Δθ (deg)", fontsize=20)
    ax.legend(fontsize=14, loc="upper left")
    ax.grid(False, alpha=0.3)
    fig.tight_layout()

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, "Figure2B_withLabels.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[sigmoid_fit] Plot saved: {out_path}")
    else:
        logger.warning("[sigmoid_fit] output_folder not provided -- plot will not be saved.")

    plt.show()
    plt.close(fig)

# ============================================================
# ENTRY POINT
# ============================================================

def fit_group_sigmoid(df, plot=True, output_folder=None):
    """
    Wrapper per il fit sul DataFrame di validazione globale.

    Args:
        df           : DataFrame con colonne 'ideal_angle' e 'real_mean'
        plot         : bool
        output_folder: str
    Returns:
        dict - risultati dei modelli ('linear', 'tanh', 'logistic4')
    """
    x = df["ideal_angle"].values
    y = df["real_mean"].values

    mask = ~np.isnan(y)
    if mask.sum() < 5:
        logger.warning("[fit_group_sigmoid] Too few valid points for fitting.")
        return {}

    weights = np.ones(mask.sum())

    n_plateau = 3  # quanti punti alle estremità etichettare
    idx_left  = np.argsort(x)[:n_plateau]  # indice/i a sinistra
    idx_right = np.argsort(x)[-n_plateau:] # indice/i a destra
    labels = [""] * len(x)  # default vuoto
    for i in idx_left:
        labels[i] = f"{"P" + str(df['trial'].iloc[i])}, {df['duration'].iloc[i]}s"
    for i in idx_right:
        labels[i] = f"{"P" + str(df['trial'].iloc[i])}, {df['duration'].iloc[i]}s"

    return fit_models(x[mask], y[mask], weights=weights,
                      plot=plot, output_folder=output_folder, labels=None)