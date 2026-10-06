# main.py

import config as cfg
from core.utils import plot_config, paths
from core.data_io import load_data, validate_data
from core.preprocessing.prepare_data import prepare_data
from core.analysis import extract_model_parameters, model_fitting, regressions, leave_one_subject_out_cross_validation
from core.visualization import plot_heatmaps, plot_model_fitting, plot_regressions, plot_losocv
from core.excels import model_results, model_fitting_results, regressions_results, loso_results

def main():

    print("=" * 70)
    print(f"EXPERIMENT: {cfg.EXPERIMENT_ID}")
    print(f"PROTOCOL:   {cfg.PROTOCOL_ID}")
    print("=" * 70)

    # ========================================================
    # LOAD
    # ========================================================

    print("\n-> Loading data...")
    protocol = load_data.load_protocol(cfg.PROTOCOL_PATH)
    df_main = load_data.load_data_file(cfg.DATA_PATH,"Trials")
    df_subjects = load_data.load_data_file(cfg.DATA_PATH,"Subjects")
    print("Data loaded.")

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n-> Validating data...")
    if not validate_data.validate_protocol(protocol):
        raise ValueError("Protocol JSON is not valid.")

    if not validate_data.validate_main_data(df_main,protocol):
        raise ValueError("Main data is not valid.")

    if not validate_data.validate_subjects_data(df_subjects):
        raise ValueError("Subjects data is not valid.")
    
    print("Data validated successfully.")

    # ========================================================
    # CREATE RESULTS FOLDERS
    # ========================================================
    print("\n-> Creating results folder...")
    for results_path in cfg.ALL_PATHS:
        if results_path:
            paths.create_directory(results_path)
    print("All results folders created successfully.")

    # ========================================================
    # PREPARE DATA
    # ========================================================

    print("\n-> Preparing data...")
    df = prepare_data(df_main=df_main,df_subject=df_subjects,protocol=protocol,output_path=cfg.DATA_PROCESSED_PATH)
    print(f"  Data prepared successfully and saved to: {cfg.DATA_PROCESSED_PATH}")

    # ========================================================
    # SELECT SUBJECTS
    # ========================================================

    if cfg.SUBJECTS_TO_PROCESS is None:
        subjects_list = sorted(df["subject"].unique())
    else:
        available_subjects = set(df["subject"].unique())
        subjects_list = [subject for subject in cfg.SUBJECTS_TO_PROCESS if subject in available_subjects]

    if not subjects_list:
        raise ValueError("No valid subjects selected for analysis.")

    df_analysis = df[df["subject"].isin(subjects_list)].copy()

    print(f"\n-> Subjects selected for analysis: {subjects_list}")
    print(f"   Number of subjects: {len(subjects_list)}")

    # ========================================================
    # SET PLOT FONT FAMILY
    # ========================================================

    plot_config.configure_plot_style()

    # ========================================================
    # HEATMAPS
    # ========================================================

    if cfg.DO_HEATMAPS:

        print("\n-> Generating heatmaps...")

        # ----------------------------------------------------
        # Single-subject heatmaps
        # ----------------------------------------------------

        if cfg.DO_SINGLE_SUBJECT_HEATMAPS:

            print("   Generating subject heatmaps...")
            for subject in subjects_list:
                plot_heatmaps.save_subject_heatmaps(df=df_analysis, subject=subject, protocol=protocol, output_folder=cfg.HEATMAPS_RESULTS_PATH,
                                                    metric="vividness", recalc_subject=cfg.RECALC_SUBJECT)

        # ----------------------------------------------------
        # Group heatmaps
        # ----------------------------------------------------

        if cfg.UPDATE_GROUP_AVERAGE:

            print("   Generating group heatmaps...")
            plot_heatmaps.save_all_subjects_heatmaps(df=df_analysis, protocol=protocol, output_folder=cfg.HEATMAPS_RESULTS_PATH, metric="vividness",
                                                      plot_quadratic_interpolation = True, show_plot=False)

    else:

        print("\n-> SKIPPING heatmaps")

    # ========================================================
    # MODEL PARAMETERS
    # ========================================================
    max_vividness = max(protocol["scales"]["vividness"]["values"]) if cfg.USE_VIVIDNESS_WEIGHTS else None

    if cfg.DO_EXTRACT_MODEL_PARAMETERS: 
        print("\n-> Running model analysis...")
        analysis_results = extract_model_parameters.run_model_analysis(df_analysis, subjects_list, max_vividness=max_vividness) 
        print("Model analysis completed.") 
        print("\n-> Saving model results...") 
        model_results.save_results( analysis_results, df_analysis, subjects_list, protocol, cfg.MODEL_PARAMETERS_PATH, max_vividness=max_vividness) 
        print("Model results saved.") 

    else: 

        print("\n-> SKIPPING model parameters")

    # ========================================================
    # REGRESSIONS
    # ========================================================

    if cfg.DO_REGRESSIONS:

        print("\n-> Running regressions...")
        regression_analysis = regressions.run_regression_analysis(df=df_analysis, protocol=protocol, group_level=cfg.INCLUDE_GROUP_LEVEL,
                                                                subject_level=cfg.INCLUDE_SUBJECT_LEVEL,trial_level=cfg.INCLUDE_TRIAL_LEVEL, max_vividness=max_vividness)
        regression_results = regression_analysis["results"]
        regression_plot_data = regression_analysis["plot_data"]
        print("Regression analysis completed.")

        # ----------------------------------------------------
        # Save Excel results
        # ----------------------------------------------------

        print("\n-> Saving regression results...")
        regressions_results.save_regression_results(results=regression_results, output_path=cfg.REGRESSIONS_RESULTS_PATH / "regression_results.xlsx")
        print("Regression results saved.")

        # ----------------------------------------------------
        # Plots
        # ----------------------------------------------------

        if cfg.SAVE_PLOTS:

            print("\n-> Generating regression plots...")
            plot_regressions.run_regression_plots(plot_data=regression_plot_data, results=regression_results, protocol=protocol, 
                                                group_level=cfg.INCLUDE_GROUP_LEVEL, subject_level=cfg.INCLUDE_SUBJECT_LEVEL, trial_level=cfg.INCLUDE_TRIAL_LEVEL)
            print("Regression plots saved.")

    else:

        print("\n-> SKIPPING regressions")

    # ========================================================
    # MODEL FITTING
    # ========================================================

    if cfg.DO_MODEL_FITTING:

        print("\n-> Running model fitting...")

        # ----------------------------------------------------
        # Load group validation data
        # ----------------------------------------------------

        validation_path = (cfg.MODEL_PARAMETERS_PATH / "group_validation.xlsx")

        if not validation_path.exists():
            raise FileNotFoundError(f"Group validation file not found: {validation_path}")

        x, y, df_model_fitting, Kb, Kt = model_fitting.prepare_data_for_model_fitting(validation_path)

        # ----------------------------------------------------
        # Fit candidate models
        # ----------------------------------------------------

        fitting_results = model_fitting.fit_models(x=x, y=y, weights=None)
        print("Model fitting completed.")

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        best_model = model_fitting.get_best_model(fitting_results)
        print(f"   Best model according to AIC: {best_model}")

        # ----------------------------------------------------
        # Save Excel results
        # ----------------------------------------------------

        print("\n-> Saving model fitting results...")
        model_fitting_results.save_model_fitting_results(results=fitting_results, output_path=cfg.MODEL_FITTING_PATH / "model_fitting_results.xlsx", Kb = Kb, Kt = Kt)
        print(f"   Model fitting results saved to: {cfg.MODEL_FITTING_PATH}/model_fitting_results.xlsx")

        # ----------------------------------------------------
        # Plots
        # ----------------------------------------------------

        if cfg.SAVE_PLOTS:

            print("\n-> Generating model fitting plots...")
            # ------------------------------------------------
            # Model comparison
            # ------------------------------------------------

            plot_model_fitting.plot_comparison(x=x, y=y, results=fitting_results, output_folder=cfg.MODEL_FITTING_PATH)

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            plot_model_fitting.plot_best_model(df=df_model_fitting, protocol_path=cfg.PROTOCOL_PATH, output_folder=cfg.MODEL_FITTING_PATH, results=fitting_results,
                                                plot_confidence_band=True, plot_saturation_points=True, confidence=0.95, forced_best_model=None)
            print("Model fitting plots saved.")

    else:

        print("\n-> SKIPPING model fitting")

    # ========================================================
    # LEAVE ONE SUBJECT OUT CROSS VALIDATION
    # ========================================================
    
    if cfg.DO_LOSOCV:

        print("\n-> Running LOSO cross-validation...")
        losocv_results = leave_one_subject_out_cross_validation.run_leave_one_subject_out_cross_validation(df_analysis, protocol)
        print("LOSO cross-validation completed.")

        # ----------------------------------------------------
        # Save Excel results
        # ----------------------------------------------------
        print("\n-> Saving LOSO results...")
        loso_results.save_loso_results(results=losocv_results, output_path=cfg.LOSOCV_PATH / "loso_results.xlsx")
        print("LOSO results saved.")

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------
        print("\n-> Generating LOSO plot...")
        plot_losocv.plot_loso_performance(results=losocv_results, output_path=cfg.LOSOCV_PATH / "loso_performance.png")
        print("LOSO plot saved.")
        
    else:
        print("\n-> SKIPPING loso cross validation")

if __name__ == "__main__":
    main()
