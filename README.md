# 2d-vibration

Analysis pipeline for kinesthetic-illusion experiments in which the biceps and triceps are stimulated with different patterns and durations, and participants indicate a position on a 2D grid and rate the vividness of the illusion.

Starting from the raw trial data, the pipeline computes the elbow-centered angular deviation of each trial, fits an additive model of the illusion (parameters `Kb` and `Kt`), compares alternative models, runs regressions and leave-one-subject-out cross-validation, and saves the results as Excel tables and figures.

## Pipeline

`main.py` runs the following steps. Each analysis step can be switched on or off in `config.py`.

1. **Load** the protocol (`protocol.json`) and the data workbook (sheets `Trials` and `Subjects`).
2. **Validate** the protocol, the trial data and the subject data.
3. **Create** the results folders.
4. **Prepare data**: build the pattern pair (BBB_TTT), merge subject information, add protocol metadata, add the expected kinesthetic illusion, compute angle_deg and add the presentation order. The result is saved as `data_processed.xlsx`.
5. **Select subjects** (all subjects, or the ones listed in `SUBJECTS_TO_PROCESS`).
6. **Heatmaps** (`DO_HEATMAPS`): per-subject and group heatmaps, with an optional quadratic interpolation between the flexion and extension patterns.
7. **Model parameters** (`DO_EXTRACT_MODEL_PARAMETERS`): estimate `Kb` and `Kt` for each subject and globally, evaluate the model and build the validation tables.
8. **Regressions** (`DO_REGRESSIONS`): through-origin linear regressions at group, subject and trial level.
9. **Model fitting** (`DO_MODEL_FITTING`): fit and compare linear, tanh and four-parameter logistic models, select the best one by AIC and compute saturation points.
10. **Leave-one-subject-out cross-validation** (`DO_LOSOCV`).

## Project structure

```
.
├── main.py                  # Entry point
├── config.py                # Paths and analysis options
├── experiments/
│   └── <EXPERIMENT_ID>/<PROTOCOL_ID>/protocol.json
└── core/
    ├── data_io/             # load_data.py, validate_data.py
    ├── preprocessing/       # prepare_data.py, geometry.py, ordering.py
    ├── analysis/            # extract_model_parameters.py, metrics.py, model_fitting.py,
    │                        # saturation.py, regressions.py,
    │                        # leave_one_subject_out_cross_validation.py
    ├── visualization/       # plot_heatmaps.py, plot_model_fitting.py,
    │                        # plot_regressions.py, plot_losocv.py
    ├── excels/              # model_results.py, model_fitting_results.py,
    │                        # regressions_results.py, loso_results.py
    └── utils/               # paths.py, patterns.py, plot_config.py
```

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/EliSoldi01/2d-vibration.git
cd 2d-vibration
pip install numpy pandas scipy scikit-learn matplotlib openpyxl pillow
```

Notes:

- NumPy 2.0 or later is required (`np.trapezoid` is used to compute the length of the quadratic curve).
- Plots use the font set in `PLOT_FONT_FAMILY` (Times New Roman by default). If it is not installed, Matplotlib falls back to its default font.

## Input data

Data and results are located relative to the **working directory**, so run the pipeline from the project root, with the `data` and `results` folders next to the project folder:

```
../data/<EXPERIMENT_ID>/<PROTOCOL_ID>/data_all_subjects.xlsx
../results/<EXPERIMENT_ID>/<PROTOCOL_ID>/
```

### Data workbook

`data_all_subjects.xlsx` must contain two sheets.

**`Trials`** (one row per trial):

| Column | Description |
| --- | --- |
| `experiment_id` | Experiment identifier |
| `protocol_id` | Protocol identifier |
| `arm` | Left or Right |
| `initial_angle` | Initial arm angle |
| `subject` | Subject identifier |
| `duration` | Stimulation duration |
| `rep` | Repetition number |
| `x`, `y` | Indicated grid cell (1-based) |
| `pattern_biceps`, `pattern_triceps` | Stimulation patterns (zero-padded to 3 characters) |
| `vividness` | Vividness rating |

**`Subjects`** (one row per subject):

| Column | Description |
| --- | --- |
| `subject` | Subject identifier (unique) |
| `forearm_cm` | Forearm length (cm) |
| `forearm_angle_deg` | Baseline forearm angle (degrees) |
| `upper_cm` | Upper-arm length (cm) |
| `block_order` | Order of the duration blocks, separated by `-` (e.g. `3-1-2`) |

### Protocol

`experiments/<EXPERIMENT_ID>/<PROTOCOL_ID>/protocol.json` must define:

- `experiment_id`, `protocol_id`, `arm` (`"right"` or `"left"`), `initial_angle`
- `blocks`: `durations`, `order` (optionally `duration_markers`, used in the model-fitting plot)
- `patterns`: `sequence_length`, `repetitions`, `order` and `definitions`. Each definition has `id`, `biceps`, `triceps`, `text`, `effect`, `color` (hex) and `legend_position`.
- `grid`: `x`, `y`, `cell_size_cm`, `start_cell`
- `scales`: e.g. `scales.vividness.values`, used to normalize vividness

The protocol and the data are validated before the analysis starts.

## Configuration

All options are in `config.py`.

| Option | Description |
| --- | --- |
| `EXPERIMENT_ID`, `PROTOCOL_ID` | Experiment and protocol to analyze (they select the protocol, data and results folders) |
| `DO_HEATMAPS`, `DO_SINGLE_SUBJECT_HEATMAPS`, `DO_EXTRACT_MODEL_PARAMETERS`, `DO_REGRESSIONS`, `DO_MODEL_FITTING`, `DO_LOSOCV` | Switch the analysis steps on or off |
| `SUBJECTS_TO_PROCESS` | List of subjects to analyze, or `None` for all |
| `RECALC_SUBJECT` | Recompute subject heatmaps even if their folder already exists |
| `UPDATE_GROUP_AVERAGE` | Regenerate the group heatmaps |
| `USE_VIVIDNESS_WEIGHTS` | Weight samples by `vividness / max_vividness` |
| `PURE_PATTERNS` | Pure patterns used to estimate `Kb` (label `b`) and `Kt` (label `t`) |
| `PATTERNS_TO_PROCESS` | Patterns included in the regressions (`None` for all) |
| `INCLUDE_GROUP_LEVEL`, `INCLUDE_SUBJECT_LEVEL`, `INCLUDE_TRIAL_LEVEL` | Regression levels to run |
| `FLEXION_PATTERN_FOR_QUADRATIC_INTERP`, `EXTENSION_PATTERN_FOR_QUADRATIC_INTERP` | Patterns used for the quadratic interpolation in the heatmaps |
| `SATURATION_THRESHOLD`, `SATURATION_THRESHOLDS_SENSITIVITY` | Saturation threshold for the plots, and thresholds for the sensitivity table |
| `SAVE_PLOTS` | Save the regression and model-fitting plots |
| `PLOT_FONT_FAMILY` | Font used in all plots |

## Usage

```bash
python main.py
```

The model-fitting step requires the previously generated `model_parameters/group_validation.xlsx`. If this file does not exist, run the model-parameters step (`DO_EXTRACT_MODEL_PARAMETERS = True`) first.

## Outputs

Results are saved in `../results/<EXPERIMENT_ID>/<PROTOCOL_ID>/`:

| Folder | Content |
| --- | --- |
| `data_processed/` | `data_processed.xlsx`: trial data with pattern pair, expected illusion, angle and presentation order |
| `heatmaps/` | Per-subject heatmaps (`<subject>/`) and group heatmaps (`ALL_SUBJECTS/`), including the multi-duration figure `<experiment_id>_<protocol_id>.png` |
| `model_parameters/` | `group_validation.xlsx` (sheets `Parameters`, `Validation`, `Model_Performance`) and one `<subject>_validation.xlsx` per subject |
| `regressions/` | `regression_results.xlsx` and plots in `group_level/`, `subject_level/`, `trial_level/` |
| `model_fitting/` | `model_fitting_results.xlsx` (sheets `Parameters`, `Performance`, `Saturation`), `model_comparison.png`, `best_model.png` |
| `losocv/` | `loso_results.xlsx` (sheets `LOSO_per_subject`, `Summary`, `Pooled`) and the `loso_performance*.png` figures |

## Methods

### Angle

For each trial, the angle is the elbow-centered angular deviation between the baseline forearm direction and the direction towards the indicated cell. Positive values correspond to extension and negative values to flexion. The same convention is used for both arms. Grid cells are converted to cm using the cell size defined in the protocol.

### Model

Stimulation patterns are encoded as `BBB_TTT` (biceps and triceps). The predicted angle is

```
angle = Kb * pb * D + Kt * pt * D
```

where `pb` and `pt` are the number of active biceps and triceps stimulations in the pattern and `D` is the stimulation duration.

`Kb` and `Kt` are estimated using linear regressions through the origin on the pure patterns, with sample weights given by `vividness / max_vividness` when `USE_VIVIDNESS_WEIGHTS` is enabled. They are estimated for each subject and globally. Performance is reported as R² referenced to zero (`R2_zero`), MAE and RMSE, on all patterns and on complex patterns only (all patterns except the pure ones).

### Model fitting

The group mean real angle is fitted as a function of the ideal (model-predicted) angle with three candidate models:

- linear, through the origin: `y = slope * x`
- tanh: `y = L * tanh(k * x)`
- four-parameter logistic: `y = L_min + (L_max - L_min) / (1 + exp(-k * (x - D0)))`

The best model is selected by AIC. Parameter standard errors, 95% confidence intervals and pointwise confidence bands (delta method) are provided. For the tanh and logistic models, saturation points are computed at the configured thresholds and converted into equivalent stimulation durations.

### Leave-one-subject-out cross-validation

For each subject, `Kb` and `Kt` are estimated from the pure-pattern trials of all other subjects and used to predict the held-out subject. The results include per-subject metrics, their mean and standard deviation, and pooled metrics across all held-out trials.

## License

Copyright (c) 2026. All rights reserved.
No permission is granted to use, copy, modify or distribute this code without explicit written permission from the author.

## Author

Elisa Soldi  
PhD student, DINOGMI / Università degli studi di Genova  
GitHub: [@EliSoldi01](https://github.com/EliSoldi01)

## Contact

For questions or permission requests, contact: elisa.soldi.unige@gmail.com
