"""
Model Predictor GUI
===================
Permette di:
  1. Caricare parametri del modello (Kb, Kt) da un file Excel globale
     OPPURE inserirli manualmente insieme a pattern/durate/cella iniziale.
  2. Inserire i parametri soggetto-specifici: Ab, At, forearm_length, forearm_angle_deg.
  3. Calcolare l'angolo previsto dal modello per ogni combinazione pattern x durata.
  4. Invertire la geometria: dall'angolo previsto risalire alla cella (x, y) della griglia.
  5. Esportare tutto in un file Excel formattato.

---
LOGICA MATEMATICA
-----------------

FORWARD (dati → angolo):
  Il modello lineare pesato calcola l'angolo previsto come:
      angle_predicted = Ab * Kb * pb_sum * D  +  At * Kt * pt_sum * D
  dove:
    Ab, At  = ampiezza del pattern puro (angolo a durata 1s) per bicipite/tricipite
    Kb, Kt  = slope della regressione lineare angolo~durata per il pattern puro
    pb_sum  = numero di attuatori attivi nel pattern bicipite (somma dei bit)
    pt_sum  = numero di attuatori attivi nel pattern tricipite
    D       = durata dello stimolo in secondi

INVERSE (angolo → cella):
  La pipeline diretta è:
    1. cell(x,y)  →  cm_coords  (moltiplicazione per cell_size)
    2. elbow_pos  =  start_cm + forearm_length * [cos(θ), sin(θ)]
    3. baseline   =  start_cm - elbow_pos   (vettore di riferimento: elbow→start)
    4. current    =  index_cm - elbow_pos   (vettore elbow→dito indice)
    5. angle      =  signed_angle(baseline, current)   [arctan2(det, dot)]

  Per invertire, dato un angolo target α (in gradi):
    - Il vettore baseline è fisso (dipende solo da forearm params e start_cell).
    - Il vettore current deve formare un angolo α con baseline.
    - Ruotiamo il vettore baseline di α per ottenere la direzione di current.
    - La cella cercata è quella della griglia che minimizza la distanza angolare
      tra l'angolo che produce e l'angolo target.
    - In pratica, iteriamo su tutte le celle (x, y) della griglia, calcoliamo
      l'angolo che produrrebbero, e scegliamo quella più vicina al target.
      Questo approccio è robusto e non richiede algebra vettoriale inversa complessa.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import os

# ─────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────

CELL_SIZE = 4  # cm

def cell_to_cm(x, y):
    return np.array([(x - 1) * CELL_SIZE + CELL_SIZE / 2,
                     (y - 1) * CELL_SIZE + CELL_SIZE / 2])

def compute_elbow(start_cm, forearm_length, forearm_angle_deg):
    theta = np.deg2rad(forearm_angle_deg)
    return start_cm + forearm_length * np.array([np.cos(theta), np.sin(theta)])

def signed_angle(v_ref, v_cur):
    dot = np.dot(v_ref, v_cur)
    det = v_ref[0] * v_cur[1] - v_ref[1] * v_cur[0]
    return np.degrees(np.arctan2(det, dot))

def compute_angle_from_cell(start_cell, index_cell, forearm_length, forearm_angle_deg):
    """Calcola l'angolo (deg) che la cella index_cell produce."""
    start_cm  = cell_to_cm(*start_cell)
    elbow_cm  = compute_elbow(start_cm, forearm_length, forearm_angle_deg)
    baseline  = start_cm - elbow_cm
    idx_cm    = cell_to_cm(*index_cell)
    current   = idx_cm - elbow_cm
    return signed_angle(baseline, current)

def angle_to_cell(target_angle_deg, start_cell, grid_x, grid_y,
                  forearm_length, forearm_angle_deg):
    """
    Inverso della geometria diretta tramite brute-force vettorizzato (numpy).

    Perche' non e' possibile un inverso analitico semplice:
      - La geometria diretta misura l'angolo tra il vettore gomito->start e il
        vettore gomito->cella. La lunghezza gomito->cella NON e' fissa (dipende
        dalla cella scelta), quindi non esiste un cerchio di raggio costante su
        cui cercare il punto. La rotazione del vettore baseline produrrebbe un
        punto a distanza uguale a forearm_length dal gomito, ma le celle reali
        sono a distanze diverse.

    Soluzione: calcolo vettorizzato degli angoli di tutte le 189 celle (21x9)
    simultaneamente con numpy, poi selezione della cella con angolo piu' vicino
    al target. Veloce (~0.1ms) e concettualmente corretto.

    Procedura:
      1. Calcola start_cm ed elbow_cm.
      2. Per ogni cella (xi, yi) della griglia, calcola il vettore gomito->cella.
      3. Calcola l'angolo signed tra baseline e ogni vettore (vettorizzato).
      4. Trova la cella che minimizza |angolo - target|.
      5. Restituisce quella cella + l'angolo effettivo che produce + il delta
         residuo (dovuto alla discretizzazione della griglia).
    """
    start_cm = cell_to_cm(*start_cell)
    elbow_cm = compute_elbow(start_cm, forearm_length, forearm_angle_deg)
    baseline = start_cm - elbow_cm  # vettore gomito->posizione neutra dito

    # Griglia completa di celle come matrici numpy
    xs = np.arange(1, grid_x + 1)
    ys = np.arange(1, grid_y + 1)
    XX, YY = np.meshgrid(xs, ys)  # shape (grid_y, grid_x)

    # Posizioni cm di tutte le celle
    cx_cm = (XX - 1) * CELL_SIZE + CELL_SIZE / 2
    cy_cm = (YY - 1) * CELL_SIZE + CELL_SIZE / 2

    # Vettori gomito->ogni cella
    vx = cx_cm - elbow_cm[0]
    vy = cy_cm - elbow_cm[1]

    # Angolo signed: arctan2(det, dot) con baseline
    bx, by = baseline
    det    = bx * vy - by * vx
    dot    = bx * vx + by * vy
    angles = np.degrees(np.arctan2(det, dot))  # shape (grid_y, grid_x)

    # Cella con angolo piu' vicino al target
    diff = np.abs(angles - target_angle_deg)
    idx  = np.unravel_index(np.argmin(diff), diff.shape)

    best_x     = int(XX[idx])
    best_y     = int(YY[idx])
    actual_angle = float(angles[idx])
    delta      = float(diff[idx])

    return best_x, best_y, actual_angle, delta

# ─────────────────────────────────────────────
# MODEL PREDICTION
# ─────────────────────────────────────────────

def predict_angle(pattern_text, duration, Ab, Kb, At, Kt):
    """
    Calcola l'angolo previsto dal modello per un dato pattern e durata.

    Formula:
        angle = (|Ab| * Kb * pb_sum + |At| * Kt * pt_sum) * D

    Il segno di Ab/At è già incorporato in Kb/Kt tramite la regressione,
    per coerenza con extract_model_parameters.py usiamo |Ab| e |At|.
    """
    b_part, t_part = pattern_text.split("_")
    pb_sum = sum(int(c) for c in b_part)
    pt_sum = sum(int(c) for c in t_part)

    angle = (abs(Ab) * Kb * pb_sum * duration +
             abs(At) * Kt * pt_sum * duration)
    return angle

# ─────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────

def export_to_excel(rows, output_path, params_summary):
    """
    rows: lista di dict con chiavi:
        pattern, duration, predicted_angle, best_x, best_y, actual_angle, delta_deg
    params_summary: dict con Ab, At, Kb, Kt, forearm_length, forearm_angle_deg, start_cell
    """
    wb = openpyxl.Workbook()

    # ── Foglio 1: Parametri ──────────────────────────────────────────────────
    ws_p = wb.active
    ws_p.title = "Parameters"

    header_fill = PatternFill("solid", start_color="2E4057")
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=11)
    val_font    = Font(name="Arial", size=10)
    center      = Alignment(horizontal="center", vertical="center")

    ws_p["A1"] = "Model & Subject Parameters"
    ws_p["A1"].font = Font(bold=True, name="Arial", size=13, color="2E4057")
    ws_p.merge_cells("A1:B1")

    param_data = [
        ("Ab (biceps amplitude at 1s)", params_summary["Ab"]),
        ("At (triceps amplitude at 1s)", params_summary["At"]),
        ("Kb (biceps slope)", params_summary["Kb"]),
        ("Kt (triceps slope)", params_summary["Kt"]),
        ("Forearm length (cm)", params_summary["forearm_length"]),
        ("Forearm angle (deg)", params_summary["forearm_angle_deg"]),
        ("Start cell X", params_summary["start_cell"][0]),
        ("Start cell Y", params_summary["start_cell"][1]),
        ("Grid X", params_summary["grid_x"]),
        ("Grid Y", params_summary["grid_y"]),
    ]

    for i, (label, val) in enumerate(param_data, start=3):
        ws_p.cell(i, 1, label).font = Font(bold=True, name="Arial", size=10)
        c = ws_p.cell(i, 2, round(val, 4) if isinstance(val, float) else val)
        c.font = val_font
        c.alignment = center

    ws_p.column_dimensions["A"].width = 35
    ws_p.column_dimensions["B"].width = 18

    # ── Foglio 2: Predictions ─────────────────────────────────────────────────
    ws = wb.create_sheet("Predictions")

    col_headers = ["Pattern", "Duration (s)", "Predicted Angle (deg)",
                   "Best Cell X", "Best Cell Y", "Actual Angle (deg)", "Delta (deg)"]

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for ci, h in enumerate(col_headers, 1):
        cell = ws.cell(1, ci, h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    alt_fill = PatternFill("solid", start_color="EBF0F7")

    for ri, row in enumerate(rows, 2):
        fill = alt_fill if ri % 2 == 0 else PatternFill("solid", start_color="FFFFFF")
        vals = [row["pattern"], row["duration"],
                round(row["predicted_angle"], 3),
                row["best_x"], row["best_y"],
                round(row["actual_angle"], 3), round(row["delta_deg"], 3)]
        for ci, v in enumerate(vals, 1):
            c = ws.cell(ri, ci, v)
            c.font = val_font
            c.alignment = center
            c.fill = fill
            c.border = border

    col_widths = [18, 14, 22, 14, 14, 22, 14]
    for ci, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(ci)].width = w

    # Auto-freeze header row
    ws.freeze_panes = "A2"

    wb.save(output_path)

# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────

class ModelPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Predictor – Vibrotactile Illusion")
        self.resizable(True, True)
        self.configure(bg="#F5F7FA")

        # Stato interno
        self.protocol_data  = None   # dict JSON protocol
        self.model_params   = {}     # Kb, Kt loaded from Excel
        self.manual_patterns = []    # lista di pattern aggiunti manualmente

        self._build_ui()
        self.geometry("760x700")
        self.minsize(600, 400)

    # ── UI BUILDING ──────────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabelframe.Label", font=("Arial", 10, "bold"), foreground="#2E4057")
        style.configure("TButton", font=("Arial", 9), padding=4)
        style.configure("Accent.TButton", font=("Arial", 9, "bold"), foreground="white",
                        background="#2E75B6")
        style.configure("TEntry", font=("Arial", 10))
        style.configure("TLabel", font=("Arial", 10), background="#F5F7FA")

        # ── Scrollable canvas wrapper ────────────────────────────────────────
        outer = tk.Frame(self, bg="#F5F7FA")
        outer.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(outer, bg="#F5F7FA", highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame lives inside the canvas
        main = ttk.Frame(self._canvas, padding=12)
        self._canvas_window = self._canvas.create_window((0, 0), window=main, anchor="nw")

        # Resize canvas scroll region when inner frame changes size
        def _on_frame_configure(event):
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        main.bind("<Configure>", _on_frame_configure)

        # Stretch inner frame to canvas width when window is resized
        def _on_canvas_configure(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)
        self._canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse-wheel scrolling (cross-platform)
        def _on_mousewheel(event):
            if event.num == 4:       # Linux scroll up
                self._canvas.yview_scroll(-1, "units")
            elif event.num == 5:     # Linux scroll down
                self._canvas.yview_scroll(1, "units")
            else:                    # Windows / macOS
                self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._canvas.bind_all("<Button-4>", _on_mousewheel)
        self._canvas.bind_all("<Button-5>", _on_mousewheel)

        # Section 1 – Load model
        self._build_model_section(main)
        # Section 2 – Subject params
        self._build_subject_section(main)
        # Section 3 – Manual patterns (optional)
        self._build_manual_section(main)
        # Section 4 – Grid & start cell
        self._build_grid_section(main)
        # Section 5 – Run & export
        self._build_run_section(main)
        # Log area
        self._build_log(main)

    def _section(self, parent, title):
        lf = ttk.LabelFrame(parent, text=title, padding=8)
        lf.pack(fill=tk.X, pady=5)
        return lf

    def _row(self, parent, label, row, col=0, width=12, default=""):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
        var = tk.StringVar(value=default)
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(row=row, column=col+1, sticky="w", padx=4, pady=3)
        return var

    # ── Section: Model ───────────────────────────────────────────────────────

    def _build_model_section(self, parent):
        lf = self._section(parent, "1 · Parametri Modello (Kb, Kt) e Protocollo")

        # Load from Excel
        f1 = ttk.Frame(lf)
        f1.pack(fill=tk.X, pady=2)
        ttk.Label(f1, text="Carica Global_Model_Parameters.xlsx:").pack(side=tk.LEFT)
        self.model_excel_path = tk.StringVar()
        ttk.Entry(f1, textvariable=self.model_excel_path, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(f1, text="Sfoglia…", command=self._load_model_excel).pack(side=tk.LEFT)

        # Load protocol JSON
        f2 = ttk.Frame(lf)
        f2.pack(fill=tk.X, pady=2)
        ttk.Label(f2, text="Carica protocol.json:          ").pack(side=tk.LEFT)
        self.protocol_path = tk.StringVar()
        ttk.Entry(f2, textvariable=self.protocol_path, width=40).pack(side=tk.LEFT, padx=4)
        ttk.Button(f2, text="Sfoglia…", command=self._load_protocol).pack(side=tk.LEFT)

        ttk.Separator(lf, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(lf, text="Oppure inserisci manualmente:").pack(anchor="w")

        fg = ttk.Frame(lf)
        fg.pack(fill=tk.X)
        self.v_Kb = self._row(fg, "Kb", 0, 0)
        self.v_Kt = self._row(fg, "Kt", 0, 2)

    def _load_model_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not path:
            return
        self.model_excel_path.set(path)
        try:
            df = pd.read_excel(path, sheet_name="Parameters")
            # Riga "Mean"
            mean_row = df[df["subject"] == "Mean"].iloc[0]
            self.v_Kb.set(round(float(mean_row["Kb"]), 5))
            self.v_Kt.set(round(float(mean_row["Kt"]), 5))
            self._log(f"✓ Kb={mean_row['Kb']:.4f}, Kt={mean_row['Kt']:.4f} caricati dal file Excel.")
        except Exception as ex:
            messagebox.showerror("Errore", f"Impossibile leggere il file modello:\n{ex}")

    def _load_protocol(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:
            return
        self.protocol_path.set(path)
        try:
            with open(path) as f:
                self.protocol_data = json.load(f)
            patterns = [p["text"] for p in self.protocol_data["patterns"]["definitions"]]
            durations = self.protocol_data["blocks"]["durations"]
            grid = self.protocol_data["grid"]
            self.v_grid_x.set(grid["x"])
            self.v_grid_y.set(grid["y"])
            self.v_start_x.set(grid["start_cell"][0])
            self.v_start_y.set(grid["start_cell"][1])
            self._log(f"✓ Protocollo caricato: {len(patterns)} pattern, durate {durations}, "
                      f"griglia {grid['x']}×{grid['y']}.")
        except Exception as ex:
            messagebox.showerror("Errore", f"Impossibile leggere il protocollo:\n{ex}")

    # ── Section: Subject ─────────────────────────────────────────────────────

    def _build_subject_section(self, parent):
        lf = self._section(parent, "2 · Parametri Soggetto")
        fg = ttk.Frame(lf)
        fg.pack(fill=tk.X)
        self.v_Ab            = self._row(fg, "Ab (deg)",          0, 0, default="0.0")
        self.v_At            = self._row(fg, "At (deg)",          0, 2, default="0.0")
        self.v_forearm_len   = self._row(fg, "Forearm length (cm)", 1, 0, default="30.0")
        self.v_forearm_angle = self._row(fg, "Forearm angle (deg)", 1, 2, default="0.0")

    # ── Section: Manual patterns ─────────────────────────────────────────────

    def _build_manual_section(self, parent):
        lf = self._section(parent, "3 · Pattern & Durate aggiuntivi (opzionale)")

        info = ttk.Label(lf, text="Aggiungi pattern non presenti nel protocollo "
                                  "(es. pattern personalizzati o durate extra).",
                         foreground="#555555")
        info.pack(anchor="w")

        fg = ttk.Frame(lf)
        fg.pack(fill=tk.X, pady=4)
        ttk.Label(fg, text="Pattern (es. 100_000):").grid(row=0, column=0, padx=4, sticky="w")
        self.v_man_pattern = tk.StringVar()
        ttk.Entry(fg, textvariable=self.v_man_pattern, width=14).grid(row=0, column=1, padx=4)
        ttk.Label(fg, text="Durata (s):").grid(row=0, column=2, padx=4, sticky="w")
        self.v_man_duration = tk.StringVar()
        ttk.Entry(fg, textvariable=self.v_man_duration, width=8).grid(row=0, column=3, padx=4)
        ttk.Button(fg, text="Aggiungi", command=self._add_manual).grid(row=0, column=4, padx=4)

        # List box
        self.lb_manual = tk.Listbox(lf, height=4, font=("Courier", 9),
                                    selectmode=tk.SINGLE, bg="white")
        self.lb_manual.pack(fill=tk.X, pady=2)
        ttk.Button(lf, text="Rimuovi selezionato", command=self._remove_manual).pack(anchor="e")

    def _add_manual(self):
        pat = self.v_man_pattern.get().strip()
        dur_str = self.v_man_duration.get().strip()
        if not pat or "_" not in pat:
            messagebox.showwarning("Attenzione", "Formato pattern non valido. Usa es. 100_000")
            return
        try:
            dur = float(dur_str)
        except ValueError:
            messagebox.showwarning("Attenzione", "Durata non valida.")
            return
        entry = (pat, dur)
        self.manual_patterns.append(entry)
        self.lb_manual.insert(tk.END, f"  {pat}   @  {dur} s")

    def _remove_manual(self):
        sel = self.lb_manual.curselection()
        if sel:
            idx = sel[0]
            self.lb_manual.delete(idx)
            self.manual_patterns.pop(idx)

    # ── Section: Grid ─────────────────────────────────────────────────────────

    def _build_grid_section(self, parent):
        lf = self._section(parent, "4 · Griglia e cella iniziale")
        fg = ttk.Frame(lf)
        fg.pack(fill=tk.X)
        self.v_grid_x   = self._row(fg, "Grid X (colonne)", 0, 0, default="21")
        self.v_grid_y   = self._row(fg, "Grid Y (righe)",   0, 2, default="9")
        self.v_start_x  = self._row(fg, "Start cell X",     1, 0, default="11")
        self.v_start_y  = self._row(fg, "Start cell Y",     1, 2, default="5")

    # ── Section: Run & export ─────────────────────────────────────────────────

    def _build_run_section(self, parent):
        lf = self._section(parent, "5 · Esegui e Esporta")
        f = ttk.Frame(lf)
        f.pack(fill=tk.X)
        ttk.Label(f, text="File di output Excel:").pack(side=tk.LEFT)
        self.v_output = tk.StringVar(value="model_predictions.xlsx")
        ttk.Entry(f, textvariable=self.v_output, width=38).pack(side=tk.LEFT, padx=4)
        ttk.Button(f, text="Sfoglia…", command=self._choose_output).pack(side=tk.LEFT)

        ttk.Button(lf, text="▶  Calcola e Salva Excel", style="Accent.TButton",
                   command=self._run).pack(pady=8)

    def _choose_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.v_output.set(path)

    # ── Log ──────────────────────────────────────────────────────────────────

    def _build_log(self, parent):
        lf = ttk.LabelFrame(parent, text="Log", padding=6)
        lf.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(lf, height=7, font=("Courier", 9), bg="#1E1E1E",
                                fg="#AAFFAA", state=tk.DISABLED, relief=tk.FLAT)
        sc = ttk.Scrollbar(lf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sc.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sc.pack(side=tk.RIGHT, fill=tk.Y)

    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ── MAIN RUN LOGIC ───────────────────────────────────────────────────────

    def _run(self):
        # Raccoglie parametri
        try:
            Kb = float(self.v_Kb.get())
            Kt = float(self.v_Kt.get())
            Ab = float(self.v_Ab.get())
            At = float(self.v_At.get())
            fl = float(self.v_forearm_len.get())
            fa = float(self.v_forearm_angle.get())
            gx = int(self.v_grid_x.get())
            gy = int(self.v_grid_y.get())
            sx = int(self.v_start_x.get())
            sy = int(self.v_start_y.get())
        except ValueError as e:
            messagebox.showerror("Errore input", f"Valore non valido: {e}")
            return

        start_cell = (sx, sy)
        output_path = self.v_output.get().strip()
        if not output_path:
            messagebox.showwarning("Attenzione", "Specifica il percorso del file di output.")
            return

        # Costruisce lista (pattern, duration) da protocollo + manuali
        combos = []
        if self.protocol_data:
            patterns = [p["text"] for p in self.protocol_data["patterns"]["definitions"]]
            durations = self.protocol_data["blocks"]["durations"]
            for pat in patterns:
                for dur in durations:
                    combos.append((pat, float(dur)))
        combos += [(pat, dur) for pat, dur in self.manual_patterns]

        if not combos:
            messagebox.showwarning("Attenzione",
                                   "Nessun pattern/durata specificato. Carica un protocollo "
                                   "o aggiungi entry manuali.")
            return

        # Calcola predizioni
        rows = []
        self._log("─" * 60)
        self._log(f"Kb={Kb:.4f}, Kt={Kt:.4f}, Ab={Ab:.4f}, At={At:.4f}")
        self._log(f"Forearm: {fl}cm @ {fa}°  |  Start cell: {start_cell}  |  Grid: {gx}×{gy}")
        self._log("─" * 60)

        for pat, dur in combos:
            try:
                angle = predict_angle(pat, dur, Ab, Kb, At, Kt)
                bx, by, actual_a, delta = angle_to_cell(
                    angle, start_cell, gx, gy, fl, fa)
                rows.append({
                    "pattern": pat,
                    "duration": dur,
                    "predicted_angle": angle,
                    "best_x": bx,
                    "best_y": by,
                    "actual_angle": actual_a,
                    "delta_deg": delta,
                })
                self._log(f"  {pat:12s} | D={dur:4.1f}s | "
                          f"angle={angle:7.2f}° → cell=({bx:2.0f},{by:2.0f}) "
                          f"[actual={actual_a:7.2f}°, Δ={delta:.3f}°]")
            except Exception as ex:
                self._log(f"  ERRORE per {pat} @ {dur}s: {ex}")

        # Esporta
        params_summary = {
            "Ab": Ab, "At": At, "Kb": Kb, "Kt": Kt,
            "forearm_length": fl, "forearm_angle_deg": fa,
            "start_cell": start_cell, "grid_x": gx, "grid_y": gy
        }
        try:
            export_to_excel(rows, output_path, params_summary)
            self._log(f"\n✓ Excel salvato in: {output_path}")
            messagebox.showinfo("Completato",
                                f"Predizioni salvate con successo in:\n{output_path}")
        except Exception as ex:
            messagebox.showerror("Errore salvataggio", str(ex))
            self._log(f"✗ Errore: {ex}")


if __name__ == "__main__":
    app = ModelPredictorApp()
    app.mainloop()
