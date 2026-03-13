import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.analysis.trajectory import (
    build_trajectory_df,
    plot_all_trajectories,       # una figura per rep
    plot_all_patterns_reps,      # una figura per pattern con tutte le rep
    compute_trajectory_metrics,
    plot_mean_trajectory,
    compute_mean_trajectory
)

from core.data_io import load_data
import phases.phase3.config as cfg
import matplotlib.pyplot as plt

protocol     = load_data.load_protocol(cfg.PROTOCOL_PATH)
df = load_data.load_trajectory_data(cfg.DATA_PATH)
subject_info = load_data.load_data_file(cfg.SUBJECTS_PATH)

OUT = cfg.OUTPUT_ROOT
OUT_TRAJ      = os.path.join(OUT, "trajectories")
OUT_METRICS   = os.path.join(OUT, "traj_metrics")
for f in [OUT, OUT_TRAJ, OUT_METRICS]: os.makedirs(f, exist_ok=True)

subjects_list = df["subject"].unique() if cfg.SUBJECTS_TO_PROCESS is None else cfg.SUBJECTS_TO_PROCESS
df_analysis   = df[df["subject"].isin(subjects_list)].copy()

# dopo il caricamento dei dati e del subject_info:
traj_df = build_trajectory_df(df_analysis, subject_info, protocol)

# plot
#plot_all_trajectories(traj_df, protocol, output_folder=OUT_TRAJ, show_ideal_arc = False)
#plot_all_patterns_reps(traj_df, protocol, output_folder=OUT_TRAJ)
# metriche per analisi successiva
#metrics_df = compute_trajectory_metrics(traj_df)
#metrics_df.to_excel(os.path.join(OUT_METRICS, "trajectory_metrics.xlsx"), index=False)

