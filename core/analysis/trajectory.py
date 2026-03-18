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
    arc_radius      : ‖start_cm - elbow_cm‖  — the ACTUAL distance from elbow
                      to the start cell in cm.  Used as the radius of the ideal
                      arc everywhere.  This is guaranteed to pass through
                      start_cm by construction, unlike forearm_cm which may
                      differ slightly due to grid discretisation.
    R               : arc_radius  (replaces forearm_cm in all arc geometry)
    raw_angles      : signed angle of each clicked point as seen from elbow
    ideal_arc_pts   : points on the circle of radius R at each raw_angle
                      → this is the "projected" trajectory on the true arc
    angle_deg       : signed angular displacement from the start angle
                      positive = extension, negative = flexion  (same sign
                      convention as Phase2 geometry.py)
 
    NOTE ON arc_radius vs forearm_cm
    ---------------------------------
    forearm_cm is the anatomical measurement of the subject's forearm.
    arc_radius = ‖start_cm - elbow_cm‖ is derived from the same measurement
    plus the grid position of start_cell.  They are *almost* identical, but
    arc_radius is guaranteed to pass through start_cm, whereas using
    forearm_cm directly can cause the ideal arc circle to miss start_cm by
    a few pixels in the plot.  arc_radius is the geometrically correct choice
    for all visual and computational purposes.
 
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
 
    plot_mean_trajectory(traj_df, subject, trial, protocol,
                         output_folder, show_ideal_arc=True)
        Mean ± 1 SD trajectory across reps, with directional arrows.
 
    plot_mean_angle_timeseries(traj_df, subject, trial, protocol,
                               output_folder)
        angle_deg vs normalised time, mean ± 1 SD across reps.
 
    plot_all_mean_trajectories / plot_all_mean_timeseries
        Batch wrappers for the two new plot types.
 
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
from ..utils import geometry as geom


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
    for (subj, trial, pattern, rep, direction), grp in df_long.groupby(
            ['subject', 'trial', 'pattern_pair', 'rep', 'direction'], sort=False):

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
    - Green cross at start_cell
    - Elbow marker (×) outside the grid
    - Clicked points connected by straight segments (raw path)
    - Ideal arc overlay: thin dashed circle centred on elbow, radius =
      ‖start_cm - elbow_cm‖  (guaranteed to pass through start_cell)
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

    # Start cell
    ax.scatter(*start_cm, marker='x', s=50, color="#f66b6b",
               linewidths=2, zorder=2, label='Start cell')

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
                   color='#032e64', lw=2.5)

    # Dots and step labels on clicked points
    for i, (pt, step, ang) in enumerate(zip(pts_cm, steps, angles)):
        color = '#2ecc71' if i == 0 else '#032e64'
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

    ax.set_xlabel('X [cm]', fontsize=10)
    ax.set_ylabel('Y [cm]', fontsize=10)
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
                             f"R{rep}_forward.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[trajectory] saved: {fname}")
    else:
        plt.show()
        plt.close(fig)


def _draw_arc_path(ax, ideal_pts, elbow_x, elbow_y, radius, color, lw, not_on_ideal_path=False):
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

        if not_on_ideal_path:
            r_a = np.linalg.norm(p_a - elbow)
            r_b = np.linalg.norm(p_b - elbow)
            radius = np.linspace(r_a, r_b, N_INTERP)
        
        alphas = np.linspace(0, 1, N_INTERP)
        arc_x  = elbow_x + radius * np.cos(a1 + alphas * diff)
        arc_y  = elbow_y + radius * np.sin(a1 + alphas * diff)

        alpha_val = 0.3 + 0.6 * (i / max(len(ideal_pts) - 2, 1))
        ax.plot(arc_x, arc_y, color=color, lw=lw,
                alpha=alpha_val, zorder=2,
                solid_capstyle='round')

# ─────────────────────────────────────────────────────────────────
# PLOT — ALL TRAJECTORIES
# ─────────────────────────────────────────────────────────────────

def plot_trajectories_single_reps(traj_df, protocol, output_folder,
                          show_ideal_arc=True):
    """
    Generate one plot per (subject, trial, rep) combination.

    Output structure:
        output_folder/
          {subject}/single_reps
            {subject}_T{trial}_R{rep}_forward.png
    """
    groups = (traj_df[traj_df['direction'] == 'forward']
              .groupby(['subject', 'trial', 'pattern_pair','rep']))

    print(f"[trajectory] Plotting {len(groups)} trajectories...")
    for (subject, trial, pattern, rep), _ in groups:
        all_reps_dir = os.path.join(output_folder, "single_reps", str(pattern))
        plot_trajectory(
            traj_df, subject, int(trial), int(rep),
            protocol, output_folder=all_reps_dir,
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
        (traj_df['subject']   == subject) &
        (traj_df['trial']     == trial)   &
        (traj_df['direction'] == 'forward')
    )
    df_trial = traj_df[mask].copy()
    if df_trial.empty:
        print(f"[WARN] No data for {subject} T{trial}")
        return

    reps = sorted(df_trial['rep'].unique())

    # Colours for each rep
    rep_colors = ['#2196f3', '#ff9800', '#4caf50', '#e91e63', '#9c27b0']

    # Protocol geometry
    cell_size        = protocol.get('grid', {}).get('cell_size_cm', CELL_SIZE_DEFAULT)
    n_cols           = protocol['grid']['x']
    n_rows           = protocol['grid']['y']
    start_col, start_row = protocol['grid']['start_cell']
    start_cm         = geom.cell_to_cm(start_col, start_row, cell_size)
    grid_w           = n_cols * cell_size
    grid_h           = n_rows * cell_size

    # Pattern info
    pat_text   = df_trial['pattern_pair'].iloc[0]
    pat_defs   = {p['text']: p for p in protocol['patterns']['definitions']}
    pat        = pat_defs.get(pat_text, {})
    pat_effect = pat.get('effect', '')

    elbow_x    = df_trial['elbow_x'].iloc[0]
    elbow_y    = df_trial['elbow_y'].iloc[0]
    forearm_cm = df_trial['radius'].dropna().median() if not df_trial['radius'].isna().all() else None

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
    ax.invert_yaxis()

    # Start cell
    ax.scatter(*start_cm, marker='x', s=50, color="#f66b6b",
               linewidths=2, zorder=2, label='Start cell')

    # Elbow
    #ax.scatter(elbow_x, elbow_y, marker='x', s=150, color='#e74c3c',
    #           linewidths=2, zorder=5, label='Elbow (pivot)')

    # Arm line
    #ax.annotate('', xy=start_cm, xytext=(elbow_x, elbow_y),
    #            arrowprops=dict(arrowstyle='->', color='#e74c3c',
    #                            lw=1.5, alpha=0.4))

    # Ideal arc
    if show_ideal_arc and forearm_cm is not None:
        ideal_circle = plt.Circle(
            (elbow_x, elbow_y), forearm_cm,
            fill=False, linestyle='--', linewidth=1,
            color='#aaaaaa', zorder=1, label='Ideal arc'
        )
        ax.add_patch(ideal_circle)

    # ── One arc path per rep ──────────────────────────────────────
    for rep, color in zip(reps, rep_colors):
        df_rep = df_trial[df_trial['rep'] == rep].sort_values('step')
        if df_rep.empty:
            continue

        pts_cm    = list(zip(df_rep['x_cm'],    df_rep['y_cm']))
        ideal_pts = list(zip(df_rep['ideal_x'], df_rep['ideal_y']))
        steps     = df_rep['step'].tolist()
        angles    = df_rep['angle_deg'].tolist()

        # Raw path (dotted)
        xs = [p[0] for p in pts_cm]
        ys = [p[1] for p in pts_cm]
        ax.plot(xs, ys, color=color, linewidth=0.8,
                linestyle=':', alpha=0.4, zorder=2)

        # Arc path (solid, coloured)
        _draw_arc_path(ax, ideal_pts, elbow_x, elbow_y, forearm_cm,
                       color=color, lw=2.2)

        # Dots on clicked points
        for i, (pt, step, ang) in enumerate(zip(pts_cm, steps, angles)):
            dot_color = '#2ecc71' if i == 0 else color
            ax.scatter(*pt, s=45, color=dot_color, zorder=6,
                       edgecolors='white', linewidths=0.6)
            # Step labels only on first rep to avoid clutter
            if rep == reps[0]:
                ax.annotate(
                    f"S{step}\n{ang:+.1f}°" if not np.isnan(ang) else f"S{step}",
                    xy=pt,
                    xytext=(5, -10), textcoords='offset points',
                    fontsize=7, color='#444444',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              alpha=0.7, ec='none')
                )

        # Legend handle for this rep
        ax.plot([], [], color=color, lw=2.2, label=f'Rep {rep}')

    ax.set_xlabel('X [cm]', fontsize=10)
    ax.set_ylabel('Y [cm]', fontsize=10)
    ax.set_title(
        f"{subject}  |  Trial {trial} ({pat_text} · {pat_effect})\n"
        f"All reps — forward trajectory  |  arc radius ≈ "
        f"{forearm_cm:.1f} cm" if forearm_cm else
        f"{subject}  |  Trial {trial} ({pat_text} · {pat_effect})\nAll reps — forward trajectory",
        fontsize=11
    )
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(output_folder,
                             f"Allreps_forward.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[trajectory] saved: {fname}")
    else:
        plt.show()
        plt.close(fig)


def plot_all_reps_trajectories(traj_df, protocol, output_folder,
                           show_ideal_arc=True):
    """
    Call plot_pattern_reps for every (subject, trial) combination.

    Output structure:
        output_folder/
          {subject}/
            {subject}_T{trial}_allreps_forward.png
    """
    groups = (traj_df[traj_df['direction'] == 'forward']
              .groupby(['subject', 'trial', 'pattern_pair']))

    print(f"[trajectory] Plotting {len(groups)} pattern×subject combinations...")
    for (subject, trial, pattern), _ in groups:
        reps_folder = os.path.join(output_folder, "all_reps", str(pattern))
        plot_pattern_reps(
            traj_df, subject, int(trial),
            protocol, output_folder=reps_folder,
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

# ─────────────────────────────────────────────────────────────────
# HELPER — NORMALISED ANGLE INTERPOLATION
# ─────────────────────────────────────────────────────────────────

def _interpolate_rep_angles(angles, n_points=100):
    """
    Resample a sequence of angular values onto a fixed grid of n_points,
    using linear interpolation on a normalised [0, 1] time axis.

    Why normalise?
    --------------
    Each repetition can have a different number of steps (e.g., 4, 5 or 7).
    We cannot average step-1 of rep A with step-1 of rep B if they represent
    different fractions of the total movement.  By mapping every trajectory
    to the same 0→1 axis we align "beginning", "middle" and "end" across
    reps regardless of how many discrete steps each one has.

    Parameters
    ----------
    angles   : array-like of float — angle_deg values in step order
    n_points : int — resolution of the output grid (default 100)

    Returns
    -------
    np.ndarray of shape (n_points,)
    """
    angles = np.asarray(angles, dtype=float)
    t_orig = np.linspace(0, 1, len(angles))
    t_new  = np.linspace(0, 1, n_points)
    return np.interp(t_new, t_orig, angles)


# ─────────────────────────────────────────────────────────────────
# PLOT — MEAN SPATIAL TRAJECTORY (grid view)
# ─────────────────────────────────────────────────────────────────

def plot_mean_trajectory(traj_df, subject, trial, protocol,
                         output_folder=None, show_ideal_arc=True,
                         n_arrows=3, n_interp=200):
    """
    Plot the **mean trajectory** of all repetitions for one (subject, trial)
    on the spatial grid, with directional flow arrows.

    Mathematical approach
    ---------------------
    1. For each rep, extract the sequence of `angle_deg` values (one per step).
    2. Resample every rep onto a shared normalised time axis t ∈ [0, 1] with
       n_interp points via linear interpolation  →  angles_rep(t).
    3. Compute:
           mean_angle(t) = mean over reps of angles_rep(t)
           std_angle(t)  = std  over reps of angles_rep(t)
    4. Convert mean_angle(t) and mean_angle(t) ± std_angle(t) back to (x, y)
       coordinates on the ideal arc:
           x(t) = elbow_x + R · cos(θ_start + mean_angle(t) · π/180)
           y(t) = elbow_y + R · sin(θ_start + mean_angle(t) · π/180)
       where θ_start is the absolute angle of the start cell as seen from
       the elbow.
    5. Draw the mean curve as a thick coloured line, the ±1 SD band as a
       filled semi-transparent ribbon, and n_arrows evenly-spaced arrow
       annotations along the mean curve to show movement direction.

    Parameters
    ----------
    traj_df      : DataFrame — output of build_trajectory_df()
    subject      : str
    trial        : int
    protocol     : dict
    output_folder: str or None — if None calls plt.show()
    show_ideal_arc : bool — draw the full dashed arc circle
    n_arrows     : int  — how many directional arrows to place on the curve
    n_interp     : int  — resolution of the normalised time grid
    """
    mask = (
        (traj_df['subject']   == subject) &
        (traj_df['trial']     == trial)   &
        (traj_df['direction'] == 'forward')
    )
    df_trial = traj_df[mask].copy()
    if df_trial.empty:
        print(f"[WARN] No data for {subject} T{trial}")
        return

    reps = sorted(df_trial['rep'].unique())
    if len(reps) < 1:
        print(f"[WARN] No reps for {subject} T{trial}")
        return

    # ── Protocol geometry ────────────────────────────────────────
    cell_size        = protocol.get('grid', {}).get('cell_size_cm', CELL_SIZE_DEFAULT)
    n_cols           = protocol['grid']['x']
    n_rows           = protocol['grid']['y']
    start_col, start_row = protocol['grid']['start_cell']
    start_cm         = geom.cell_to_cm(start_col, start_row, cell_size)
    grid_w           = n_cols * cell_size
    grid_h           = n_rows * cell_size

    # Elbow and radius (consistent across reps for a given subject)
    elbow_x    = df_trial['elbow_x'].iloc[0]
    elbow_y    = df_trial['elbow_y'].iloc[0]
    forearm_cm = df_trial['radius'].dropna().median()

    # Absolute angle of the start cell as seen from the elbow.
    # This is the angular "zero" from which angle_deg is measured.
    ref_vec   = start_cm - np.array([elbow_x, elbow_y])
    theta_start = np.arctan2(ref_vec[1], ref_vec[0])   # radians

    # ── Interpolate each rep onto the normalised grid ────────────
    all_angles = []   # shape: (n_reps, n_interp)
    for rep in reps:
        df_rep = df_trial[df_trial['rep'] == rep].sort_values('step')
        angles = df_rep['angle_deg'].dropna().values
        if len(angles) < 2:
            continue
        all_angles.append(_interpolate_rep_angles(angles, n_interp))

    if not all_angles:
        print(f"[WARN] Not enough data to average for {subject} T{trial}")
        return

    all_angles  = np.array(all_angles)          # (n_reps, n_interp)
    mean_angles = np.mean(all_angles, axis=0)   # (n_interp,)
    std_angles  = np.std(all_angles,  axis=0)   # (n_interp,)

    # ── Convert mean ± std angles → (x, y) on the ideal arc ─────
    # angle_deg is the signed displacement FROM the start direction,
    # so the absolute arc angle = theta_start + mean_angle_rad
    mean_rad  = np.deg2rad(mean_angles)
    upper_rad = np.deg2rad(mean_angles + std_angles)
    lower_rad = np.deg2rad(mean_angles - std_angles)

    def angles_to_xy(rad_array):
        xs = elbow_x + forearm_cm * np.cos(theta_start + rad_array)
        ys = elbow_y + forearm_cm * np.sin(theta_start + rad_array)
        return xs, ys

    mean_x,  mean_y  = angles_to_xy(mean_rad)
    upper_x, upper_y = angles_to_xy(upper_rad)
    lower_x, lower_y = angles_to_xy(lower_rad)

    # ── Pattern info ─────────────────────────────────────────────
    pat_text   = df_trial['pattern_pair'].iloc[0]
    pat_defs   = {p['text']: p for p in protocol['patterns']['definitions']}
    pat        = pat_defs.get(pat_text, {})
    pat_color  = pat.get('color', '#aaaaaa')
    pat_effect = pat.get('effect', '')

    # ── Figure ───────────────────────────────────────────────────
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
    ax.invert_yaxis()

    # Ideal arc (full dashed circle)
    if show_ideal_arc:
        ideal_circle = plt.Circle(
            (elbow_x, elbow_y), forearm_cm,
            fill=False, linestyle='--', linewidth=1,
            color='#bbbbbb', zorder=1, label='Ideal arc'
        )
        ax.add_patch(ideal_circle)

    # Start cell
    ax.scatter(*start_cm, marker='x', s=50, color="#f66b6b",
               linewidths=2, zorder=2, label='Start cell')

    # ± 1 SD ribbon
    # We draw two filled polygons: one for upper boundary, one for lower.
    # fill_between does not work directly in axis-equal plots with
    # non-monotonic x, so we build the polygon manually.
    poly_x = np.concatenate([upper_x, lower_x[::-1]])
    poly_y = np.concatenate([upper_y, lower_y[::-1]])
    ax.fill(poly_x, poly_y, color='#032e64', alpha=0.18,
            zorder=2, label='±1 SD')

    # Mean trajectory curve
    ax.plot(mean_x, mean_y, color='#032e64', lw=3,
            zorder=4, solid_capstyle='round', label=f'Mean trajectory (n={len(all_angles)} reps)')

    # ── Directional arrows ───────────────────────────────────────
    # Place n_arrows arrows at evenly spaced positions along the mean curve.
    # Each arrow points from sample i to sample i+1, giving an unambiguous
    # sense of direction without cluttering the plot.
    arrow_indices = np.linspace(5, n_interp - 10, n_arrows, dtype=int)

    for i, idx in enumerate(arrow_indices):
        dx = mean_x[idx + 1] - mean_x[idx]
        dy = mean_y[idx + 1] - mean_y[idx]
        
        # Colore dinamico: prima verde, ultima rossa, le altre blu
        if i == 0:
            color = "#06F00E"      # prima freccia
        elif i == len(arrow_indices) - 1:
            color = "#F00606"        # ultima freccia
        else:
            color = '#032e64'    # le altre

        ax.annotate(
            '', 
            xy=(mean_x[idx + 1], mean_y[idx + 1]),
            xytext=(mean_x[idx], mean_y[idx]),
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=2.0, mutation_scale=18
            ),
            zorder=3
        )

    # Raw individual rep curves (very faint, for context)
    for i, rep in enumerate(reps):
        df_rep = df_trial[df_trial['rep'] == rep].sort_values('step')
        angles_rep = df_rep['angle_deg'].dropna().values
        if len(angles_rep) < 2:
            continue
        rep_interp = _interpolate_rep_angles(angles_rep, n_interp)
        rx, ry = angles_to_xy(np.deg2rad(rep_interp))
        ax.plot(rx, ry, color='#032e64', lw=0.8, alpha=0.25,
                linestyle=':', zorder=3)

    ax.set_xlabel('X [cm]', fontsize=10)
    ax.set_ylabel('Y [cm]', fontsize=10)
    ax.set_title(
        f"{subject}  |  Trial {trial} ({pat_text} · {pat_effect})\n"
        f"Mean trajectory ± 1 SD  —  {len(all_angles)} reps  "
        f"|  arc radius ≈ {forearm_cm:.1f} cm",
        fontsize=11
    )
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()

    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(output_folder, 'mean_trajectory.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[trajectory] saved: {fname}")
    else:
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# PLOT — ANGULAR TIME-SERIES  (angle_deg vs normalised time)
# ─────────────────────────────────────────────────────────────────

def plot_mean_angle_timeseries(traj_df, subject, trial, protocol,
                               output_folder=None, n_interp=100):
    """
    Plot `angle_deg` as a function of **normalised time** t ∈ [0, 1],
    showing mean ± 1 SD across repetitions.

    Why this view is useful
    -----------------------
    The spatial grid plot shows *where* the subject moved.  This plot shows
    *how fast* and *how smoothly* they moved through the angular range:
    - A straight diagonal line means constant angular velocity.
    - A curve that rises quickly then plateaus suggests rapid initial movement
      followed by hesitation or saturation.
    - Wide SD bands indicate high variability in timing between reps.

    This representation is standard in biomechanics (cf. SPM, joint angle
    time-series analyses) and is completely independent of the grid geometry,
    making it easier to compare across subjects or patterns.

    Parameters
    ----------
    traj_df      : DataFrame — output of build_trajectory_df()
    subject      : str
    trial        : int
    protocol     : dict
    output_folder: str or None
    n_interp     : int — normalised time grid resolution
    """
    mask = (
        (traj_df['subject']   == subject) &
        (traj_df['trial']     == trial)   &
        (traj_df['direction'] == 'forward')
    )
    df_trial = traj_df[mask].copy()
    if df_trial.empty:
        print(f"[WARN] No data for {subject} T{trial}")
        return

    reps = sorted(df_trial['rep'].unique())

    pat_text   = df_trial['pattern_pair'].iloc[0]
    pat_defs   = {p['text']: p for p in protocol['patterns']['definitions']}
    pat        = pat_defs.get(pat_text, {})
    pat_color  = pat.get('color', '#aaaaaa')
    pat_effect = pat.get('effect', '')

    t_norm = np.linspace(0, 1, n_interp)

    all_angles = []
    for rep in reps:
        df_rep = df_trial[df_trial['rep'] == rep].sort_values('step')
        angles = df_rep['angle_deg'].dropna().values
        if len(angles) < 2:
            continue
        all_angles.append(_interpolate_rep_angles(angles, n_interp))

    if not all_angles:
        print(f"[WARN] Not enough angle data for {subject} T{trial}")
        return

    all_angles  = np.array(all_angles)
    mean_angles = np.mean(all_angles, axis=0)
    std_angles  = np.std(all_angles,  axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Individual rep traces (faint)
    for i, (rep, angles_rep) in enumerate(zip(reps, all_angles)):
        ax.plot(t_norm, angles_rep, color='#032e64',
                lw=1.0, alpha=0.3, linestyle='--',
                label=f'Rep {rep}' if i == 0 else '_nolegend_')

    # SD ribbon
    ax.fill_between(t_norm,
                    mean_angles - std_angles,
                    mean_angles + std_angles,
                    color='#032e64', alpha=0.2, label='±1 SD')

    # Mean curve
    ax.plot(t_norm, mean_angles, color='#032e64',
            lw=2.5, label=f'Mean (n={len(all_angles)} reps)')

    # Reference zero line
    ax.axhline(0, color='#999999', lw=0.8, linestyle='-', alpha=0.5)

    # ── Directional arrows on the mean curve ─────────────────────
    # A small triangle marker every ~20% of the normalised time axis,
    # pointing in the direction of increasing t, makes it unambiguous
    # that the trajectory runs left-to-right.
    for t_pos in np.linspace(0.1, 0.85, 4):
        idx = int(t_pos * (n_interp - 2))
        dy  = mean_angles[idx + 1] - mean_angles[idx]
        ax.annotate(
            '',
            xy=(t_norm[idx + 1], mean_angles[idx + 1]),
            xytext=(t_norm[idx],  mean_angles[idx]),
            arrowprops=dict(arrowstyle='->', color='#032e64',
                            lw=1.8, mutation_scale=14),
            zorder=5
        )

    ax.set_xlabel('Normalised time  t  [0 = start,  1 = end]', fontsize=11)
    ax.set_ylabel('Angular displacement  Δθ  [°]', fontsize=11)
    ax.set_ylim(-25, +25)
    ax.set_title(
        f"{subject}  |  Trial {trial} ({pat_text} · {pat_effect})\n"
        f"Angle vs normalised time  —  {len(all_angles)} reps",
        fontsize=11
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if output_folder:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        fname = os.path.join(output_folder, 'mean_angle_timeseries.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[trajectory] saved: {fname}")
    else:
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# BATCH WRAPPERS
# ─────────────────────────────────────────────────────────────────

def plot_all_mean_trajectories(traj_df, protocol, output_folder,
                               show_ideal_arc=True):
    """
    Call plot_mean_trajectory for every (subject, trial) combination.

    Output structure:
        output_folder/mean_trajectory/{pattern}/mean_trajectory.png
    """
    groups = (traj_df[traj_df['direction'] == 'forward']
              .groupby(['subject', 'trial', 'pattern_pair']))

    print(f"[trajectory] Plotting {len(groups)} mean trajectories...")
    for (subject, trial, pattern), _ in groups:
        out = os.path.join(output_folder, 'mean_trajectory', str(pattern))
        plot_mean_trajectory(
            traj_df, subject, int(trial), protocol,
            output_folder=out, show_ideal_arc=show_ideal_arc
        )
    print("[trajectory] Done.")


def plot_all_mean_timeseries(traj_df, protocol, output_folder):
    """
    Call plot_mean_angle_timeseries for every (subject, trial) combination.

    Output structure:
        output_folder/mean_timeseries/{pattern}/mean_angle_timeseries.png
    """
    groups = (traj_df[traj_df['direction'] == 'forward']
              .groupby(['subject', 'trial', 'pattern_pair']))

    print(f"[trajectory] Plotting {len(groups)} angle time-series...")
    for (subject, trial, pattern), _ in groups:
        out = os.path.join(output_folder, 'mean_timeseries', str(pattern))
        plot_mean_angle_timeseries(
            traj_df, subject, int(trial), protocol,
            output_folder=out
        )
    print("[trajectory] Done.")