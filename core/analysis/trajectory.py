"""
core/analysis/trajectory.py
============================
Trajectory analysis for Phase 3.

Geometric model
---------------
The subject holds their forearm with the elbow as a fixed pivot.
The finger/wrist traces an arc of radius = forearm_cm.

Key coordinate chain:
    start_cell  → convert to cm  → start_cm
    start_cm, forearm_cm, forearm_angle_deg  → elbow_cm
    every clicked point  → cm  → angle from elbow  → arc projection

Definitions
-----------
    start_cm        : position of the finger at rest  [cm]
    elbow_cm        : pivot point, derived as:
                        elbow_cm = start_cm - forearm_cm * [cos(θ), sin(θ)]
                      where θ = forearm_angle_deg in radians
    R               : forearm_cm  (radius of the ideal arc)
    raw_angles      : signed angle of each clicked point as seen from elbow
    ideal_arc_pts   : points on the circle of radius R at each raw_angle
                      → this is the "projected" trajectory on the true arc
    angle_deg       : signed angular displacement from the start angle
                      positive = extension, negative = flexion  (same sign
                      convention as Phase2 geometry.py)

Public API
----------
    compute_trajectory(step_list, elbow_cm, forearm_cm, cell_size)
        Core geometry — returns a dict per step.

    build_trajectory_df(df_long, subject_info, protocol)
        Builds a tidy DataFrame of trajectory geometry from the long-format
        data produced by load_trajectory_data().

    plot_trajectory(traj_df, subject, trial, rep, protocol,
                    output_folder, show_ideal_arc=True)
        One figure per (subject, trial, rep): clicked points connected by
        arc segments, ideal arc overlay, elbow marker.

    plot_all_trajectories(traj_df, protocol, output_folder)
        Iterates over all (subject, trial, rep) combinations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from pathlib import Path
from core.utils import geometry as geom


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

CELL_SIZE_DEFAULT = 4  # cm per grid cell (from protocol grid.cell_size_cm)


# ─────────────────────────────────────────────────────────────────
# LOW-LEVEL GEOMETRY
# ─────────────────────────────────────────────────────────────────

def _project_on_arc(point_cm, elbow_cm, radius):
    """
    Project point_cm onto the circle of given radius centred on elbow_cm.
    Returns the projected point in cm.
    """
    vec  = point_cm - elbow_cm
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return elbow_cm + np.array([radius, 0.0])
    return elbow_cm + (vec / norm) * radius


# ─────────────────────────────────────────────────────────────────
# CORE PER-STEP COMPUTATION
# ─────────────────────────────────────────────────────────────────

def compute_trajectory(step_list, elbow_cm, forearm_cm, cell_size=CELL_SIZE_DEFAULT):
    """
    Compute geometry for each step of a single trajectory.

    Parameters
    ----------
    step_list   : list of dicts with keys 'step', 'x' (col), 'y' (row)
    elbow_cm    : np.array [x, y]  — elbow position in cm
    forearm_cm  : float            — ideal arc radius
    cell_size   : float            — cm per grid cell

    Returns
    -------
    list of dicts, one per step, with keys:
        step            int    step index (1-based)
        col, row        int    original grid coords
        x_cm, y_cm      float  clicked point in cm
        ideal_x, ideal_y float  projection on ideal arc  [cm]
        angle_deg       float  signed angular displacement from step 1
        radius          float  distance from elbow  [cm]  (≈ forearm_cm ideally)
        radial_err      float  radius - forearm_cm  [cm]  (deviation from arc)
    """
    if not step_list:
        return []

    # Convert all steps to cm
    pts_cm = [geom.cell_to_cm(s['x'], s['y'], cell_size) for s in step_list]

    # Reference direction: elbow → first point
    ref_cm = pts_cm[0]

    results = []
    for i, (s, pt) in enumerate(zip(step_list, pts_cm)):
        # Project onto ideal arc
        ideal_pt = _project_on_arc(pt, elbow_cm, forearm_cm)

        # Signed angle from first step
        angle = geom.signed_angle_from_elbow(pt, elbow_cm, ref_cm)

        # Radial distance and error
        radius     = float(np.linalg.norm(pt - elbow_cm))
        radial_err = radius - forearm_cm

        results.append({
            'step':      s['step'],
            'col':       s['x'],
            'row':       s['y'],
            'x_cm':      float(pt[0]),
            'y_cm':      float(pt[1]),
            'ideal_x':   float(ideal_pt[0]),
            'ideal_y':   float(ideal_pt[1]),
            'angle_deg': float(angle),
            'radius':    radius,
            'radial_err': radial_err,
        })

    return results


# ─────────────────────────────────────────────────────────────────
# BUILD TIDY DATAFRAME
# ─────────────────────────────────────────────────────────────────

def build_trajectory_df(df_long, subject_info, protocol):
    """
    Build a tidy trajectory DataFrame with geometry columns.

    Parameters
    ----------
    df_long      : DataFrame from load_trajectory_data()
                   Required columns: subject, trial, rep, duration,
                                     pattern_pair, direction, step, x, y
    subject_info : DataFrame with columns: subject, forearm_cm,
                   forearm_angle_deg
    protocol     : dict — protocol JSON (needs grid.start_cell,
                   grid.cell_size_cm)

    Returns
    -------
    DataFrame with original columns plus:
        elbow_x, elbow_y   float  elbow position [cm]
        x_cm, y_cm         float  clicked point [cm]
        ideal_x, ideal_y   float  arc projection [cm]
        angle_deg          float  signed angular displacement from step 1
        radius             float  distance from elbow [cm]
        radial_err         float  deviation from ideal arc [cm]
    """
    cell_size = protocol.get('grid', {}).get('cell_size_cm', CELL_SIZE_DEFAULT)
    start_col, start_row = protocol['grid']['start_cell']
    start_cm = geom.cell_to_cm(start_col, start_row, cell_size)

    # Build per-subject lookup
    subj_map = subject_info.set_index('subject')[
        ['forearm_cm', 'forearm_angle_deg']
    ].to_dict('index')

    rows = []
    for (subj, trial, rep, direction), grp in df_long.groupby(
            ['subject', 'trial', 'rep', 'direction'], sort=False):

        info = subj_map.get(subj)
        if info is None:
            print(f"[WARN] No subject info for {subj}, skipping.")
            continue

        forearm_cm         = float(info['forearm_cm'])
        forearm_angle_deg  = float(info['forearm_angle_deg'])
        elbow_cm           = geom.compute_elbow(start_cm, forearm_cm,
                                               forearm_angle_deg, arm="right")

        # Only process forward direction for now (can extend to return)
        if direction != 'forward':
            # Still include raw data but skip geometry
            for _, row in grp.iterrows():
                r = row.to_dict()
                r.update(dict(elbow_x=float(elbow_cm[0]),
                              elbow_y=float(elbow_cm[1]),
                              x_cm=float(geom.cell_to_cm(row['x'], row['y'], cell_size)[0]),
                              y_cm=float(geom.cell_to_cm(row['x'], row['y'], cell_size)[1]),
                              ideal_x=np.nan, ideal_y=np.nan,
                              angle_deg=np.nan, radius=np.nan,
                              radial_err=np.nan))
                rows.append(r)
            continue

        step_list = grp.sort_values('step')[['step', 'x', 'y']].to_dict('records')
        geo       = compute_trajectory(step_list, elbow_cm, forearm_cm, cell_size)
        geo_map   = {g['step']: g for g in geo}

        for _, row in grp.sort_values('step').iterrows():
            r = row.to_dict()
            g = geo_map.get(int(row['step']), {})
            r.update(dict(
                elbow_x    = float(elbow_cm[0]),
                elbow_y    = float(elbow_cm[1]),
                x_cm       = g.get('x_cm',       np.nan),
                y_cm       = g.get('y_cm',       np.nan),
                ideal_x    = g.get('ideal_x',    np.nan),
                ideal_y    = g.get('ideal_y',    np.nan),
                angle_deg  = g.get('angle_deg',  np.nan),
                radius     = g.get('radius',     np.nan),
                radial_err = g.get('radial_err', np.nan),
            ))
            rows.append(r)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# PLOT — SINGLE (subject, trial, rep)
# ─────────────────────────────────────────────────────────────────

def plot_trajectory(traj_df, subject, trial, rep, protocol,
                    output_folder=None, show_ideal_arc=True):
    """
    Plot the forward trajectory for one (subject, trial, rep).

    Layout
    ------
    - Grey grid (cell boundaries)
    - Red cross at start_cell
    - (Elbow marker (×) outside the grid)
    - Clicked points connected by straight segments (raw path)
    - Ideal arc overlay: thin dashed circle centred on elbow
    - Arc segments between consecutive clicked points projected on the
      ideal arc (interpolated, solid coloured line)
    - Step numbers annotated on each dot

    Parameters
    ----------
    traj_df      : output of build_trajectory_df()
    subject      : str
    trial        : int
    rep          : int
    protocol     : dict
    output_folder: str or None — if None, plt.show() is called
    show_ideal_arc: bool — whether to draw the full dashed circle
    """
    mask = (
        (traj_df['subject']   == subject) &
        (traj_df['trial']     == trial)   &
        (traj_df['rep']       == rep)     &
        (traj_df['direction'] == 'forward')
    )
    df = traj_df[mask].sort_values('step')
    if df.empty:
        print(f"[WARN] No data for {subject} T{trial} R{rep}")
        return

    cell_size  = protocol.get('grid', {}).get('cell_size_cm', CELL_SIZE_DEFAULT)
    n_cols     = protocol['grid']['x']
    n_rows     = protocol['grid']['y']
    start_col, start_row = protocol['grid']['start_cell']
    start_cm   = geom.cell_to_cm(start_col, start_row, cell_size)

    # Grid extent in cm
    grid_w = n_cols * cell_size
    grid_h = n_rows * cell_size

    # Pattern info for title/colour
    pat_text   = df['pattern_pair'].iloc[0]
    pat_defs   = {p['text']: p for p in protocol['patterns']['definitions']}
    pat        = pat_defs.get(pat_text, {})
    pat_color  = pat.get('color', '#aaaaaa')
    pat_effect = pat.get('effect', '')

    elbow_x    = df['elbow_x'].iloc[0]
    elbow_y    = df['elbow_y'].iloc[0]
    # Use the stored radius (forearm_cm) directly — more reliable than
    # recomputing distance from potentially noisy first clicked point
    forearm_cm = df['radius'].dropna().median() if not df['radius'].isna().all() else None

    # ── Figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect('equal')
    ax.set_facecolor('#f7f7f7')

    # Grid lines
    for c in range(n_cols + 1):
        ax.axvline(c * cell_size, color='#cccccc', lw=0.5, zorder=0)
    for r in range(n_rows + 1):
        ax.axhline(r * cell_size, color='#cccccc', lw=0.5, zorder=0)

    ax.set_xlim(-cell_size * 2, grid_w + cell_size)
    ax.set_ylim(-cell_size * 2, grid_h + cell_size)
    ax.invert_yaxis()   # row 1 at top (screen convention)
    x_ticks = [(c - 0.5) * cell_size for c in range(1, n_cols + 1)]
    y_ticks = [(r - 0.5) * cell_size for r in range(1, n_rows + 1)]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(range(1, n_cols + 1))
    ax.set_yticklabels(range(1, n_rows + 1))    

    # Start cell
    ax.scatter(*start_cm, marker='+', s=200, color='#2ecc71',
               linewidths=2, zorder=5, label='Start cell')

    # Elbow marker (may be outside grid)
    #ax.scatter(elbow_x, elbow_y, marker='x', s=150, color='#e74c3c',
    #           linewidths=2, zorder=5, label='Elbow (pivot)')

    # Ideal arc (dashed circle)
    if show_ideal_arc and forearm_cm is not None:
        ideal_circle = plt.Circle(
            (elbow_x, elbow_y), forearm_cm,
            fill=False, linestyle='--', linewidth=1,
            color='#aaaaaa', zorder=1, label='Ideal arc'
        )
        ax.add_patch(ideal_circle)

    pts_cm = list(zip(df['x_cm'], df['y_cm']))
    ideal_pts = list(zip(df['ideal_x'], df['ideal_y']))
    steps     = df['step'].tolist()
    angles    = df['angle_deg'].tolist()

    # Raw path (straight segments between clicked points)
    xs = [p[0] for p in pts_cm]
    ys = [p[1] for p in pts_cm]
    ax.plot(xs, ys, color='#999999', linewidth=1.2,
            linestyle=':', zorder=2, label='Raw path (straight)')

    # Arc path (interpolated arc segments between projected points)
    _draw_arc_path(ax, ideal_pts, elbow_x, elbow_y, forearm_cm,
                   color=pat_color, lw=2.5)

    # Dots and step labels on clicked points
    for i, (pt, step, ang) in enumerate(zip(pts_cm, steps, angles)):
        color = '#2ecc71' if i == 0 else pat_color
        ax.scatter(*pt, s=60, color=color, zorder=6, edgecolors='white',
                   linewidths=0.8)
        ax.annotate(
            f"S{step}\n{ang:+.1f}°" if not np.isnan(ang) else f"S{step}",
            xy=pt,
            xytext=(5, -10), textcoords='offset points',
            fontsize=7.5, color='#333333',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none')
        )

    # Arm line from elbow to start (context)
    #ax.annotate('', xy=start_cm, xytext=(elbow_x, elbow_y),
    #            arrowprops=dict(arrowstyle='->', color='#e74c3c',
    #                            lw=1.5, alpha=0.5))

    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title(
        f"{subject}  |  Trial {trial} ({pat_text} · {pat_effect})  |  Rep {rep}\n"
        f"Forward trajectory  —  arc radius ≈ {forearm_cm:.1f} cm",
        fontsize=11
    )
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(output_folder,
                             f"{subject}_T{trial}_R{rep}_forward.png")
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[trajectory] saved: {fname}")
    else:
        plt.show()
        plt.close(fig)

def _draw_arc_path(ax, ideal_pts, elbow_x, elbow_y, radius, color, lw):
    """
    Draw arc segments between consecutive projected points.
    Each segment is a true circular arc (interpolated via N intermediate
    points on the circle), giving a smooth curved path.
    """
    if not ideal_pts or any(np.isnan(p[0]) for p in ideal_pts):
        return

    elbow = np.array([elbow_x, elbow_y])
    N_INTERP = 30   # points per arc segment

    for i in range(len(ideal_pts) - 1):
        p_a = np.array(ideal_pts[i])
        p_b = np.array(ideal_pts[i + 1])

        # Angles of the two endpoints as seen from elbow
        a1 = np.arctan2(p_a[1] - elbow_y, p_a[0] - elbow_x)
        a2 = np.arctan2(p_b[1] - elbow_y, p_b[0] - elbow_x)

        # Always take the short arc
        diff = a2 - a1
        if diff >  np.pi: diff -= 2 * np.pi
        if diff < -np.pi: diff += 2 * np.pi

        alphas = np.linspace(0, 1, N_INTERP)
        arc_x  = elbow_x + radius * np.cos(a1 + alphas * diff)
        arc_y  = elbow_y + radius * np.sin(a1 + alphas * diff)

        alpha_val = 0.4 + 0.6 * (i / max(len(ideal_pts) - 2, 1))
        ax.plot(arc_x, arc_y, color=color, lw=lw,
                alpha=alpha_val, zorder=3,
                solid_capstyle='round')


# ─────────────────────────────────────────────────────────────────
# PLOT — ALL TRAJECTORIES
# ─────────────────────────────────────────────────────────────────

def plot_all_trajectories(traj_df, protocol, output_folder,
                          show_ideal_arc=True):
    """
    Generate one plot per (subject, trial, rep) combination.

    Output structure:
        output_folder/
          {subject}/
            {subject}_T{trial}_R{rep}_forward.png
    """
    groups = (traj_df[traj_df['direction'] == 'forward']
              .groupby(['subject', 'trial', 'rep']))

    print(f"[trajectory] Plotting {len(groups)} trajectories...")
    for (subject, trial, rep), _ in groups:
        subj_folder = os.path.join(output_folder, str(subject))
        plot_trajectory(
            traj_df, subject, int(trial), int(rep),
            protocol, output_folder=subj_folder,
            show_ideal_arc=show_ideal_arc
        )
    print("[trajectory] Done.")


# ─────────────────────────────────────────────────────────────────
# PLOT — ALL REPS OF ONE PATTERN ON THE SAME FIGURE
# ─────────────────────────────────────────────────────────────────

def plot_pattern_reps(traj_df, subject, trial, protocol,
                      output_folder=None, show_ideal_arc=True):
    """
    Plot all repetitions of a given pattern (trial) for one subject
    on a single figure, each rep with a distinct colour.

    Layout
    ------
    - Grey grid, start cell, elbow marker, ideal arc (shared)
    - One arc path per rep, colour-coded (Rep1=blue, Rep2=orange, Rep3=green)
    - Dots on clicked points; step numbers on Rep1 only (to avoid clutter)
    - Legend showing rep colours

    Parameters
    ----------
    traj_df      : output of build_trajectory_df()
    subject      : str
    trial        : int
    protocol     : dict
    output_folder: str or None
    show_ideal_arc: bool
    """

    mask = (
        (traj_df['subject'] == subject) &
        (traj_df['trial'] == trial) &
        (traj_df['direction'] == 'forward')
    )

    df_trial = traj_df[mask].copy()
    if df_trial.empty:
        print(f"[WARN] No data for {subject} T{trial}")
        return

    reps = sorted(df_trial['rep'].unique())

    cell_size = protocol.get('grid', {}).get('cell_size_cm', CELL_SIZE_DEFAULT)
    n_cols = protocol['grid']['x']
    n_rows = protocol['grid']['y']

    start_col, start_row = protocol['grid']['start_cell']

    # convert cm → cell coordinates
    def cm_to_cell(x, y):
        return x / cell_size + 0.5, y / cell_size + 0.5

    df_trial["x_cell"], df_trial["y_cell"] = cm_to_cell(
        df_trial["x_cm"], df_trial["y_cm"]
    )

    df_trial["ideal_x_cell"], df_trial["ideal_y_cell"] = cm_to_cell(
        df_trial["ideal_x"], df_trial["ideal_y"]
    )

    elbow_x, elbow_y = cm_to_cell(
        df_trial['elbow_x'].iloc[0],
        df_trial['elbow_y'].iloc[0]
    )

    forearm_cm = df_trial['radius'].dropna().median()

    forearm_cells = forearm_cm / cell_size if forearm_cm else None

    pat_text = df_trial['pattern_pair'].iloc[0]
    pat_defs = {p['text']: p for p in protocol['patterns']['definitions']}
    pat = pat_defs.get(pat_text, {})
    pat_effect = pat.get('effect', '')

    rep_colors = ['#2196f3', '#ff9800', '#4caf50', '#e91e63', '#9c27b0']

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect('equal')
    ax.set_facecolor('#f7f7f7')

    ax.set_xlim(0.5, n_cols + 0.5)
    ax.set_ylim(0.5, n_rows + 0.5)
    ax.invert_yaxis()

    # grid
    for c in range(1, n_cols + 1):
        ax.axvline(c - 0.5, color='#cccccc', lw=0.6)

    for r in range(1, n_rows + 1):
        ax.axhline(r - 0.5, color='#cccccc', lw=0.6)

    ax.set_xticks(range(1, n_cols + 1))
    ax.set_yticks(range(1, n_rows + 1))

    # start cell
    ax.scatter(start_col, start_row,
               marker='x', s=200, color="#f43e31",
               linewidths=2, zorder=5, label='Start cell')

     # Elbow
    #ax.scatter(elbow_x, elbow_y, marker='x', s=150, color='#e74c3c',
    #           linewidths=2, zorder=5, label='Elbow (pivot)')

    # Arm line
    #ax.annotate('', xy=start_cm, xytext=(elbow_x, elbow_y),
    #            arrowprops=dict(arrowstyle='->', color='#e74c3c',
    # 
    
    # ideal arc
    if show_ideal_arc and forearm_cells:
        circle = plt.Circle(
            (elbow_x, elbow_y),
            forearm_cells,
            fill=False,
            linestyle='--',
            linewidth=1,
            color='#aaaaaa'
        )
        ax.add_patch(circle)

    for rep, color in zip(reps, rep_colors):

        df_rep = df_trial[df_trial['rep'] == rep].sort_values('step')

        pts = list(zip(df_rep['x_cell'], df_rep['y_cell']))
        ideal_pts = list(zip(df_rep['ideal_x_cell'], df_rep['ideal_y_cell']))
        steps = df_rep['step'].tolist()
        angles = df_rep['angle_deg'].tolist()

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        ax.plot(xs, ys, linestyle=':', lw=0.8, color=color, alpha=0.4)

        _draw_arc_path(ax, ideal_pts, elbow_x, elbow_y,
                       forearm_cells, color=color, lw=2.2)

        for i, (pt, step, ang) in enumerate(zip(pts, steps, angles)):

            dot_color = '#2ecc71' if i == 0 else color

            ax.scatter(*pt, s=45,
                       color=dot_color,
                       edgecolors='white',
                       linewidths=0.6,
                       zorder=6)

            if rep == reps[0]:

                label = f"S{step}"
                if not np.isnan(ang):
                    label += f"\n{ang:+.1f}°"

                ax.annotate(label, xy=pt,
                            xytext=(5, -10),
                            textcoords='offset points',
                            fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7, ec='none'))

        ax.plot([], [], color=color, lw=2.2, label=f"Rep {rep}")

    ax.set_xlabel("X (cell)")
    ax.set_ylabel("Y (cell)")

    ax.set_title(
        f"{subject} | Trial {trial} ({pat_text} · {pat_effect})\n"
        f"All reps — forward trajectory"
    )

    ax.legend(fontsize=8)
    fig.tight_layout()

    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(output_folder,
                             f"{subject}_T{trial}_allreps_forward.png")
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[trajectory] saved: {fname}")
    else:
        plt.show()
        plt.close(fig)

def plot_all_patterns_reps(traj_df, protocol, output_folder,
                           show_ideal_arc=True):
    """
    Call plot_pattern_reps for every (subject, trial) combination.

    Output structure:
        output_folder/
          {subject}/
            {subject}_T{trial}_allreps_forward.png
    """
    groups = (traj_df[traj_df['direction'] == 'forward']
              .groupby(['subject', 'trial']))

    print(f"[trajectory] Plotting {len(groups)} pattern×subject combinations...")
    for (subject, trial), _ in groups:
        subj_folder = os.path.join(output_folder, str(subject))
        plot_pattern_reps(
            traj_df, subject, int(trial),
            protocol, output_folder=subj_folder,
            show_ideal_arc=show_ideal_arc
        )
    print("[trajectory] Done.")


# ─────────────────────────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────────────────────────

def compute_trajectory_metrics(traj_df):
    """
    Compute per-(subject, trial, rep) summary metrics for the forward
    trajectory.

    Returns
    -------
    DataFrame with columns:
        subject, trial, rep, pattern_pair
        n_steps         int    number of forward steps
        total_angle     float  total signed angular displacement [°]
                               (last step angle_deg)
        mean_radial_err float  mean deviation from ideal arc [cm]
        std_radial_err  float  std of radial error [cm]
        arc_length_cm   float  total arc length along ideal arc [cm]
    """
    fwd = traj_df[traj_df['direction'] == 'forward'].copy()
    rows = []
    for (subj, trial, rep), grp in fwd.groupby(['subject', 'trial', 'rep']):
        grp = grp.sort_values('step')
        angles = grp['angle_deg'].dropna().values
        rerr   = grp['radial_err'].dropna().values

        # Arc length = R * |Δθ_total|  (sum of absolute angular steps)
        forearm_cm = grp['radius'].iloc[0] if not grp['radius'].isna().all() else np.nan
        if len(angles) > 1:
            diffs      = np.diff(angles)
            arc_length = float(np.abs(np.deg2rad(diffs)).sum() * forearm_cm) \
                         if not np.isnan(forearm_cm) else np.nan
        else:
            arc_length = 0.0

        rows.append({
            'subject':        subj,
            'trial':          trial,
            'rep':            rep,
            'pattern_pair':   grp['pattern_pair'].iloc[0],
            'n_steps':        len(grp),
            'total_angle':    float(angles[-1]) if len(angles) else np.nan,
            'mean_radial_err': float(np.mean(rerr)) if len(rerr) else np.nan,
            'std_radial_err':  float(np.std(rerr))  if len(rerr) else np.nan,
            'arc_length_cm':  arc_length,
        })

    return pd.DataFrame(rows)

