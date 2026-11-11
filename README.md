# Do machine learning models for stunting in children under two years stay accurate across two Rwanda surveys? Code and results

Code and summary results for a paper on temporal validation of machine learning models for stunting in children aged 0 to 23 months. Each model is trained on one Rwanda Demographic and Health Survey and tested, once, on the other:

- **Forward:** train on 2019-20 (1,573 children, 440 stunted), test on 2024-25 (2,789 children, 713 stunted)
- **Reverse:** train on 2024-25, test on 2019-20

Seven models are compared: logistic regression, decision tree, random forest, gradient boosting, XGBoost, LightGBM and CatBoost. Nothing is tuned on the test survey. The paper also reports recalibration in the test survey, decision curves, subgroup and survey-weighted AUC, a comparison with models trained inside each survey, and a repeat with a second tuning seed.

A companion repository covers the same design for children aged 0 to 59 months.

## Data (not included)

The survey data are free to registered users from the DHS Program (https://dhsprogram.com). Request the Rwanda 2019-20 and 2024-25 children's recode files. No survey data, per-child predictions or Stata log files are shared here, and `.gitignore` blocks them.

The Python scripts read two cleaned files made with `stata/01_clean_both_surveys.do` (see `stata/README.md`). Put them in the same folder as the scripts:

| File | Survey |
|---|---|
| `rwanda_stunting_clean_2020_v4.dta` | 2019-20 |
| `rwanda_stunting_clean_2025_b_v4.dta` | 2024-25 |

The Stata file cleans all ages. The Python scripts keep children aged 0 to 23 months (`AGE_MAX = 24`).

## Folders

| Folder | Contents |
|---|---|
| `python/` | Analysis scripts, numbered in run order |
| `results/` | Aggregate summary tables from our runs. No per-child data |
| `stata/` | Stata cleaning file that builds the cleaned survey files |

## Run order

Scripts that depend on direction come as a pair: `..._forward.py` trains on 2019-20 and tests on 2024-25, `..._reverse.py` does the opposite. The direction is already set inside each file. Output files carry a tag such as `_train2020_test2025_v7_under24m`, so the two directions never overwrite each other.

| Step | Script | What it does | Time |
|---|---|---|---|
| 0 | `stata/01_clean_both_surveys.do` | Builds the cleaned `.dta` files (once) | minutes |
| 1 | `python/01_temporal_validation_forward.py` and `..._reverse.py` | Tunes on the training survey (4 cluster-grouped inner folds, 100 Optuna trials per model, seed 42), picks the F1-optimal threshold, scores the test survey once, cluster bootstrap intervals (1,000 resamples), calibration, SHAP comparison | hours |
| 2 | `python/02_descriptive_table.py` | Table of the 33 predictors in both surveys | about a minute |
| 3 | `python/03_temporal_addons_forward.py` and `..._reverse.py` | Recalibration in the test survey (new intercept, and intercept plus slope, cross-fitted by cluster), decision curves, survey-weighted AUC, AUC by subgroup. Reuses the models tuned in step 1 | minutes |
| 4 | `python/04_within_survey_comparison.py` | Models trained and tested inside each survey (grouped nested cross-validation, 50 Optuna trials), compared with the models trained on the other survey on the same test children | hours |
| 5 | `python/05_seed_repeat_forward.py 43` and `..._reverse.py 43` | Repeats step 1 with another seed (the number after the script name) | hours |
| 6 | `python/06_seed_repeat_summary.py` | One table of AUC by seed | seconds |

Steps 3 to 6 need the output of step 1 in the same folder. Run step 1 before anything else.

## Settings

Near the top of the main scripts:

| Setting | Values | Meaning |
|---|---|---|
| `AGE_MAX` | `24` | Keep children aged 0 to 23 months in both surveys. `None` keeps ages 0 to 59 |
| `HARMONISED` | `False` | `True` drops the 12 predictors whose missingness differs most between the surveys |

Use the same settings in every script you run together, so the file names match.

## Results folder

| File | Content |
|---|---|
| `temporal_validation_results_*` | Test-survey performance of each model (AUC, accuracy, F1, recall, precision, calibration, Brier score, with bootstrap intervals) |
| `temporal_validation_tuning_*` | Chosen hyperparameters and F1-optimal threshold of each model |
| `temporal_validation_calibration_groups_*` | Predicted and observed risk in ten risk groups |
| `temporal_validation_shap_xgboost_*` | Mean absolute SHAP value of each encoded feature in both surveys |
| `temporal_addon_recalibration_*`, `..._decision_curve_*`, `..._subgroup_auc_*`, `..._weighted_auc_*` | Outputs of step 3 |
| `within_survey_cv_results_*`, `within_vs_temporal_auc_*` | Outputs of step 4 |
| `descriptive_table_under24m_v7.csv` | Output of step 2 |
| `seed_repeat_summary_under24m_v7.csv` | Output of step 6 |

## Software

Python with scikit-learn 1.9.0, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, Optuna 4.9.0 and SHAP 0.52.0. Install with `pip install -r requirements.txt`.

## Notes

- The surveys name regions differently ("kigali" and "city of kigali"). The scripts map them to the same labels first.
- Missing values are imputed from the training survey only. A category in the test survey that never appears in training is set to the training mode.
- Mother's height and body mass index are missing for about half of the 2024-25 children, but not in 2019-20.
- Random seed 42 is used in the main run, 43 in the repeat. Results can differ slightly across package versions.

## Licence and citation

MIT (see `LICENSE`). See `CITATION.cff`.

Archived on Zenodo: [10.5281/zenodo.24134547](https://doi.org/10.5281/zenodo.24134547)
