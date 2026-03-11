import numpy as np

# =========================
# PARAMETRI MATRICE
# =========================
N_ROWS = 9
N_COLS = 21
DEG_PER_ROW = 5  # 1 riga = 5°

# =========================
# MODELLO LINEARE
# =========================
def predict_delta_theta(Ab, At, Kb, Kt, pb, pt, D):
    return (abs(Ab) * Kb * pb * D) + (abs(At) * Kt * pt * D)

# =========================
# CONVERSIONE ANGLO → MATRICE
# =========================
def angle_to_row(delta_theta):
    """
    Converte delta angolo in spostamento righe.
    Flessione (negativa) → righe verso il basso
    Estensione (positiva) → righe verso l'alto
    """
    row_shift = int(round(delta_theta / DEG_PER_ROW))
    return row_shift

# =========================
# MAIN INTERFACCIA
# =========================
def main():

    print("\n--- PARAMETRI GLOBALI ---")
    Kb = float(input("Kb: "))
    Kt = float(input("Kt: "))

    print("\n--- PARAMETRI SOGGETTO ---")
    Ab = float(input("Ab (calibrato 1s): "))
    At = float(input("At (calibrato 1s): "))

    print("\n--- PATTERN E DURATA ---")
    pb = float(input("pb: "))
    pt = float(input("pt: "))
    D = float(input("Durata D (s): "))

    print("\n--- POSIZIONE INIZIALE ---")
    start_row = int(input("Riga iniziale (0-8): "))
    start_col = int(input("Colonna iniziale (0-20): "))
    start_angle = float(input("Angolo iniziale (default 90): ") or 90)

    # Calcolo delta angolo
    delta_theta = predict_delta_theta(Ab, At, Kb, Kt, pb, pt, D)

    # Angolo finale
    final_angle = start_angle + delta_theta

    # Calcolo nuova riga
    row_shift = angle_to_row(delta_theta)
    new_row = start_row - row_shift  # meno perché estensione va verso l'alto

    # Clipping nei limiti della matrice
    new_row = max(0, min(N_ROWS - 1, new_row))
    new_col = max(0, min(N_COLS - 1, start_col))

    print("\n==============================")
    print(f"Delta angolo previsto: {delta_theta:.2f}°")
    print(f"Angolo finale: {final_angle:.2f}°")
    print(f"Nuova posizione matrice: ({new_row}, {new_col})")
    print("==============================\n")


if __name__ == "__main__":
    main()
