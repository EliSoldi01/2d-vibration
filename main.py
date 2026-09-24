import config as cfg

from core.data_io import load_data, validate_data
from core.preprocessing.prepare_data import prepare_data
from core.analysis import extract_model_parameters, heatmaps
from core.excels import model_results




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
            results_path.mkdir(parents=True,exist_ok=True)
            print(f"{results_path} created.")

    print("All results folders created successfully.")

    # ========================================================
    # PREPARE DATA
    # ========================================================

    print("\n-> Preparing data...")

    df = prepare_data(df_main=df_main,df_subject=df_subjects,protocol=protocol,output_path=cfg.DATA_PROCESSED_PATH)

    print(
        f"  Data prepared successfully and saved to: "
        f"{cfg.DATA_PROCESSED_PATH}"
    )

    # ========================================================
    # SELECT SUBJECTS
    # ========================================================

    if cfg.SUBJECTS_TO_PROCESS is None:

        subjects_list = sorted(
            df["subject"].unique()
        )

    else:

        available_subjects = set(
            df["subject"].unique()
        )

        subjects_list = [
            subject
            for subject in cfg.SUBJECTS_TO_PROCESS
            if subject in available_subjects
        ]

    if not subjects_list:
        raise ValueError(
            "No valid subjects selected for analysis."
        )

    df_analysis = df[
        df["subject"].isin(subjects_list)
    ].copy()

    print(
        f"\n-> Subjects selected for analysis: "
        f"{subjects_list}"
    )

    print(
        f"   Number of subjects: "
        f"{len(subjects_list)}"
    )

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

                heatmaps.save_subject_heatmaps(
                    df=df_analysis,
                    subject=subject,
                    protocol=protocol,
                    output_folder=cfg.HEATMAPS_RESULTS_PATH,
                    metric="vividness",
                    recalc_subject=cfg.RECALC_SUBJECT
                )

        # ----------------------------------------------------
        # Group heatmaps
        # ----------------------------------------------------

        if cfg.UPDATE_GROUP_AVERAGE:

            print(
                "   Generating group heatmaps..."
            )

            heatmaps.save_all_subjects_heatmaps(
                df=df_analysis,
                protocol=protocol,
                output_folder=cfg.HEATMAPS_RESULTS_PATH,
                metric="vividness"
            )

    else:

        print("\n-> SKIPPING heatmaps")

    # ========================================================
    # MODEL PARAMETERS
    # ========================================================
    if cfg.DO_EXTRACT_MODEL_PARAMETERS: 
        print("\n-> Running model analysis...")

        analysis_results = extract_model_parameters.run_model_analysis( df_analysis, subjects_list, protocol ) 
        print("Model analysis completed.") 
        print("\n-> Saving model results...") 

        model_results.save_results( analysis_results, df_analysis, subjects_list, protocol, cfg.MODEL_PARAMETERS_PATH) 
        print("Model results saved.") 

    else: 

        print("\n-> SKIPPING model parameters")

if __name__ == "__main__":
    main()
