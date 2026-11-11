#======================================================================
# Temporal validation of stunting models, children aged 0-23 months
# Trains seven models on one Rwanda DHS survey and tests them once on the other.
# Direction: train on 2024-25, test on 2019-20 (reverse).
# Tuning: 4 cluster-grouped inner folds, 100 Optuna trials per model, F1-optimal threshold.
# Nothing is tuned on the test survey, and the test survey is scored once.
# Only summary tables are written; no per-child predictions or SHAP values are saved.
#======================================================================
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')

# Data and results folder: the folder that holds this script (put the two cleaned .dta files there).
# To use another folder, set OUT_DIR = r"C:\path\to\folder" instead.
try:
    OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # running in a notebook
    OUT_DIR = os.getcwd()
TRAIN_FILE = "rwanda_stunting_clean_2025_b_v4.dta"  # train survey (REVERSE direction)
TEST_FILE  = "rwanda_stunting_clean_2020_v4.dta"    # test survey (scored once)
RUN_TAG    = "_train2025_test2020_v7"
N_BOOT     = 1000
SEED       = 42

ml_features = [
    'b4', 'bord', 'b0', 'hw1', 'first_birth', 'lbw', 'birth_weight',
    'bf_months', 'still_breastfeeding', 'v106', 'mother_bmi', 'mother_height', 'literacy',
    'anc_visits', 'anc_timing', 'delivery_facility', 'deworming',
    'v190', 'v025', 'v024', 'water_source_grp', 'water_time', 'toilet_grp', 'shared_toilet',
    'v481', 'distance_problem', 'v012', 'v136', 'v137', 'v404', 'iron_supp', 'm17', 'birth_size'
]
high_miss_vars = ['birth_weight', 'mother_bmi', 'mother_height', 'bf_months',
                  'anc_visits', 'anc_timing', 'delivery_facility', 'deworming']
categorical_vars = ['b4', 'b0', 'first_birth', 'v106', 'delivery_facility',
                    'v190', 'v025', 'v024', 'water_source_grp', 'toilet_grp',
                    'shared_toilet', 'v481', 'distance_problem', 'lbw',
                    'still_breastfeeding', 'deworming', 'v404', 'iron_supp',
                    'm17', 'birth_size']
continuous_vars = ['bord', 'hw1', 'birth_weight', 'bf_months', 'mother_bmi',
                   'mother_height', 'literacy', 'anc_visits', 'anc_timing',
                   'water_time', 'v012', 'v136', 'v137']

# ---- Optional sensitivity run -------------------------------------------------------------
# In 2019-20 the birth, antenatal, delivery and maternal-measurement items are recorded for nearly
# all children; in 2024-25 they are missing for 39-51 percent (children aged 36+ months, and a
# maternal anthropometry subsample). Set HARMONISED = True to drop the 12 predictors whose
# missingness differs by more than 15 percentage points between the surveys.
HARMONISED = False

# Optional age restriction (applied to BOTH surveys). None = all children aged 0-59 months.
# AGE_MAX = 24 keeps children aged 0-23 months; AGE_MAX = 36 keeps 0-35 months.
AGE_MAX = 24
HARMONISED_DROP = ['anc_timing', 'anc_visits', 'birth_size', 'birth_weight', 'delivery_facility',
                   'deworming', 'iron_supp', 'lbw', 'm17', 'mother_bmi', 'mother_height',
                   'still_breastfeeding']
if HARMONISED:
    ml_features      = [v for v in ml_features if v not in HARMONISED_DROP]
    high_miss_vars   = [v for v in high_miss_vars if v not in HARMONISED_DROP]
    categorical_vars = [v for v in categorical_vars if v not in HARMONISED_DROP]
    continuous_vars  = [v for v in continuous_vars if v not in HARMONISED_DROP]
    RUN_TAG = RUN_TAG + "_harmonised"
    print(f"HARMONISED run: {len(ml_features)} predictors kept ({len(HARMONISED_DROP)} dropped)")
if AGE_MAX is not None:
    RUN_TAG = RUN_TAG + f"_under{AGE_MAX}m"
    print(f"AGE restriction: children aged 0 to {AGE_MAX - 1} months only")


for _f in (TRAIN_FILE, TEST_FILE):
    if not os.path.exists(os.path.join(OUT_DIR, _f)):
        raise FileNotFoundError(f"Cannot find {_f} in {OUT_DIR}. Put the two .dta files in the same folder as this script.")
print("Using folder:", OUT_DIR)

def load(fname):
    # Read only the columns we use. Other columns (e.g. strata_id) can carry repeated value
    # labels, which makes pandas stop when it converts every column.
    d = pd.read_stata(os.path.join(OUT_DIR, fname), columns=list(dict.fromkeys(['cluster_id', 'stunted', 'hw1'] + ml_features)))
    if AGE_MAX is not None:
        d = d[pd.to_numeric(d['hw1'], errors='coerce') < AGE_MAX].copy()
        d = d[['cluster_id', 'stunted'] + ml_features]
    obs = set(d['stunted'].astype(str).str.strip().unique())
    if obs != {'Stunted', 'Not stunted'}:
        raise ValueError(f"{fname}: unexpected outcome labels {sorted(obs)}")
    y = (d['stunted'].astype(str).str.strip() == 'Stunted').astype(int).reset_index(drop=True)
    miss = [v for v in ml_features if v not in d.columns]
    if miss: raise KeyError(f"{fname}: missing predictors {miss}")
    empty = [v for v in ml_features if d[v].isna().all()]
    if empty: raise ValueError(f"{fname}: predictors entirely missing: {empty}")
    X = d[['cluster_id'] + ml_features].copy().reset_index(drop=True)
    for v in categorical_vars:                       # same text labels in both surveys
        s = X[v]
        X[v] = s.astype(object).where(s.notna(), None)
        X[v] = X[v].map(lambda z: None if (z is None or str(z).strip() in ('', 'nan', '.')) else str(z).strip().lower())
        if v == 'v024':   # 2019-20 and 2024-25 name the same five regions differently
            X[v] = X[v].replace({'city of kigali': 'kigali', 'southern province': 'south',
                                 'western province': 'west', 'northern province': 'north',
                                 'eastern province': 'east'})
    for v in continuous_vars:
        before = int(X[v].notna().sum())
        X[v] = pd.to_numeric(X[v], errors='coerce')
        if int(X[v].notna().sum()) != before:
            raise ValueError(f"{fname}: '{v}' lost values when converted to numbers; it may carry text labels.")
    for v in high_miss_vars:
        X[f'{v}_missing'] = X[v].isna().astype(int)
    return X, y

X_train_raw, y_train = load(TRAIN_FILE)
X_test_raw,  y_test  = load(TEST_FILE)
print(f"Train {TRAIN_FILE}: n={len(X_train_raw)}, stunted={int(y_train.sum())} ({y_train.mean():.3f})")
print(f"Test  {TEST_FILE}: n={len(X_test_raw)}, stunted={int(y_test.sum())} ({y_test.mean():.3f})")

# Fixed category lists taken from the TRAIN survey only, so train and test are encoded with
# exactly the same columns and the first category is always dropped. A test-survey category
# never seen in training is treated as missing and filled with the training mode.
CATS = {v: sorted(set(X_train_raw[v].dropna())) for v in categorical_vars}   # from the TRAIN survey only
for v in categorical_vars:
    only_tr = sorted(set(X_train_raw[v].dropna()) - set(X_test_raw[v].dropna()))
    only_te = sorted(set(X_test_raw[v].dropna()) - set(X_train_raw[v].dropna()))
    flag = ''
    if only_tr or only_te:
        flag = f"   <-- CHECK: only in train {only_tr}; only in test {only_te}"
    print(f"  {v}: {CATS[v]}{flag}")

# Stop if a categorical variable is coded differently in the two surveys: otherwise the test
# children are silently set to the training mode and the temporal result is biased.
for v in categorical_vars:
    share = float((~X_test_raw[v].dropna().isin(CATS[v])).mean()) if X_test_raw[v].notna().any() else 0.0
    if share > 0.02:
        raise ValueError(f"'{v}': {share:.0%} of test labels are not in the training labels. "
                         f"train={CATS[v]} test={sorted(set(X_test_raw[v].dropna()))}. "
                         f"Recode the labels so both surveys use the same text.")

def impute_and_encode(train_raw, test_raw):
    """Impute from the training data only, then one-hot with fixed categories."""
    tr, te = train_raw.copy(), test_raw.copy()
    for var in continuous_vars:
        med = tr[var].median()
        med = 0.0 if pd.isna(med) else med
        tr[var] = tr[var].fillna(med); te[var] = te[var].fillna(med)
    for var in categorical_vars:
        modes = tr[var].mode(dropna=True)
        mode = modes.iloc[0] if not modes.empty else CATS[var][0]
        unseen = te[var].notna() & ~te[var].isin(CATS[var])
        if unseen.any():
            print(f"  note: {int(unseen.sum())} test value(s) of {var} not seen in training; set to the training mode")
            te.loc[unseen, var] = None
        tr[var] = tr[var].fillna(mode); te[var] = te[var].fillna(mode)
    outs = []
    for d in (tr, te):
        d = d.drop(columns=['cluster_id']).copy()
        for var in categorical_vars:
            d[var] = pd.Categorical(d[var], categories=CATS[var])
        outs.append(pd.get_dummies(d, columns=categorical_vars, drop_first=True, dtype=float))
    X_tr, X_te = outs
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    if X_tr.isna().any().any() or X_te.isna().any().any():
        raise ValueError("NaN values remain after imputation/encoding.")
    return X_tr, X_te


# ---- Imports, helpers and the Optuna objective ----
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, precision_recall_curve,
                              confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve

N_TRIALS = 100
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest',
               'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost']

def make_objective(name, inner_splits):
    """Optuna objective: mean ROC AUC over every grouped inner fold."""
    def objective(trial):
        if name == 'Logistic Regression':
            params = {
                'C': trial.suggest_float('C', 1e-4, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
            }
        elif name == 'Decision Tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
            }
        elif name == 'Random Forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif name == 'Gradient Boosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
        elif name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True)
            }
        elif name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 12),
                'num_leaves': trial.suggest_int('num_leaves', 7, 127),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
            }
        elif name == 'CatBoost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'depth': trial.suggest_int('depth', 2, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True)
            }
        else:
            raise ValueError(f"Unknown model name: {name}")

        fold_aucs = []
        for X_tr, y_tr, X_val, y_val, spw in inner_splits:
            model = build_final_model(name, params)
            model = fit_with_weights(name, model, X_tr, y_tr, spw)
            probs = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, probs))
        return float(np.mean(fold_aucs))
    return objective

def build_final_model(name, params):
    if name == 'Logistic Regression':
        return LogisticRegression(**params, solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42)
    if name == 'Decision Tree':
        return DecisionTreeClassifier(**params, class_weight='balanced', random_state=42)
    if name == 'Random Forest':
        return RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
    if name == 'Gradient Boosting':
        return GradientBoostingClassifier(**params, random_state=42)
    if name == 'XGBoost':
        return XGBClassifier(**params, eval_metric='logloss', random_state=42, verbosity=0)
    if name == 'LightGBM':
        return LGBMClassifier(**params, class_weight='balanced', subsample_freq=1, random_state=42, verbose=-1)
    if name == 'CatBoost':
        return CatBoostClassifier(**params, auto_class_weights='Balanced', verbose=0, random_state=42)

def fit_with_weights(name, model, X_tr, y_tr, spw):
    """Fit a model, applying the model-appropriate imbalance handling."""
    if name == 'XGBoost':
        model.set_params(scale_pos_weight=spw)
        model.fit(X_tr, y_tr)
    elif name == 'Gradient Boosting':
        model.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
    else:
        model.fit(X_tr, y_tr)
    return model

def select_threshold(probs_val, y_val):
    """Pick the F1-maximizing threshold from VALIDATION predictions only.
    Never call this with test-fold probabilities/labels — doing so leaks
    the test set into the threshold choice and inflates F1/precision/
    recall/accuracy (AUC is unaffected since it's threshold-free)."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


#======================================================================
# TUNE ON THE TRAIN SURVEY ONLY: 4 grouped inner folds, 100 Optuna trials per model
#======================================================================
inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
inner_splits, inner_val_idx = [], []
for k, (itr, iva) in enumerate(inner.split(X_train_raw, y_train, groups=X_train_raw['cluster_id']), start=1):
    g_tr, g_va = set(X_train_raw['cluster_id'].iloc[itr]), set(X_train_raw['cluster_id'].iloc[iva])
    if g_tr & g_va: raise RuntimeError(f"Cluster leakage in inner fold {k}")
    y_itr, y_iva = y_train.iloc[itr].reset_index(drop=True), y_train.iloc[iva].reset_index(drop=True)
    X_itr, X_iva = impute_and_encode(X_train_raw.iloc[itr], X_train_raw.iloc[iva])
    inner_splits.append((X_itr, y_itr, X_iva, y_iva, float((y_itr == 0).sum() / (y_itr == 1).sum())))

best_params, thresholds = {}, {}
for name in MODEL_NAMES:
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=4200))
    study.optimize(make_objective(name, inner_splits), n_trials=N_TRIALS, show_progress_bar=False)
    best_params[name] = study.best_params
    print(f"  {name}: mean inner AUC={study.best_value:.4f}; {study.best_params}")
    probs, ys = [], []
    for X_itr, y_itr, X_iva, y_iva, spw in inner_splits:
        m = fit_with_weights(name, build_final_model(name, best_params[name]), X_itr, y_itr, spw)
        probs.extend(m.predict_proba(X_iva)[:, 1].tolist()); ys.extend(y_iva.tolist())
    thresholds[name] = select_threshold(np.asarray(probs), np.asarray(ys))

#======================================================================
# REFIT ON ALL TRAIN-SURVEY CHILDREN, SCORE THE TEST SURVEY ONCE
#======================================================================
X_tr_all, X_te_all = impute_and_encode(X_train_raw, X_test_raw)
spw_all = float((y_train == 0).sum() / (y_train == 1).sum())
test_clusters = X_test_raw['cluster_id'].to_numpy()
yt = y_test.to_numpy()
rng = np.random.default_rng(SEED)
uniq = np.unique(test_clusters)
members = {c: np.where(test_clusters == c)[0] for c in uniq}
boot_idx = []
while len(boot_idx) < N_BOOT:
    b = np.concatenate([members[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
    if 0 < yt[b].sum() < len(b):          # skip a resample with only one outcome class
        boot_idx.append(b)

def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(auc=roc_auc_score(y, p), accuracy=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0))

def cal(y, p):
    from sklearn.linear_model import LogisticRegression as LR
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LR(C=1e10, solver='lbfgs', max_iter=2000).fit(lp, y)
    return float(m.intercept_[0]), float(m.coef_[0][0]), float(brier_score_loss(y, p))

rows, preds_by_model, models = [], {}, {}
for name in MODEL_NAMES:
    model = fit_with_weights(name, build_final_model(name, best_params[name]), X_tr_all, y_train, spw_all)
    models[name] = model
    p = model.predict_proba(X_te_all)[:, 1]
    preds_by_model[name] = p
    thr = thresholds[name]
    base = metrics(yt, p, thr)
    boots = [metrics(yt[b], p[b], thr) for b in boot_idx]
    ci = {k: (np.percentile([b[k] for b in boots], 2.5), np.percentile([b[k] for b in boots], 97.5)) for k in base}
    intercept, slope, brier = cal(yt, p)
    row = {'Model': name, 'threshold_from_train': round(thr, 3)}
    for k in base: row[k] = f"{base[k]:.3f} [{ci[k][0]:.3f} to {ci[k][1]:.3f}]"
    row.update({'mean_predicted_risk': round(float(p.mean()), 3), 'observed_prevalence': round(float(yt.mean()), 3),
                'calibration_intercept': round(intercept, 2), 'calibration_slope': round(slope, 2), 'brier': round(brier, 4)})
    # Descriptive only: best F1 if the threshold were tuned on the test survey (optimistic; not a performance estimate)
    pr, rc, th = precision_recall_curve(yt, p)
    f1s = 2 * pr * rc / (pr + rc + 1e-10)
    row['best_possible_F1_on_test_descriptive_only'] = round(float(f1s[:-1].max()), 3) if len(th) else np.nan
    rows.append(row)

res = pd.DataFrame(rows)
print("\n" + "=" * 90)
print(f"TEMPORAL VALIDATION: train {TRAIN_FILE} -> test {TEST_FILE} (95% CI by cluster bootstrap)")
print("=" * 90)
print(res.to_string(index=False))
res.to_csv(os.path.join(OUT_DIR, f"temporal_validation_results{RUN_TAG}.csv"), index=False)

#======================================================================
# SHAP: XGBOOST TRAINED ON THE TRAIN SURVEY, EXPLAINED ON BOTH SURVEYS
# (summary only: mean |SHAP| per encoded feature; no per-child values are saved)
#======================================================================
import shap
from scipy.stats import spearmanr
xgb = models['XGBoost']
ex = shap.TreeExplainer(xgb)
imp_train = np.abs(ex.shap_values(X_tr_all)).mean(axis=0)
imp_test  = np.abs(ex.shap_values(X_te_all)).mean(axis=0)
shap_df = pd.DataFrame({'feature': X_tr_all.columns, 'mean_abs_shap_train_survey': imp_train, 'mean_abs_shap_test_survey': imp_test})
shap_df['rank_train'] = shap_df['mean_abs_shap_train_survey'].rank(ascending=False).astype(int)
shap_df['rank_test'] = shap_df['mean_abs_shap_test_survey'].rank(ascending=False).astype(int)
shap_df = shap_df.sort_values('mean_abs_shap_train_survey', ascending=False)
rho = spearmanr(shap_df['mean_abs_shap_train_survey'], shap_df['mean_abs_shap_test_survey'])[0]
print(f"\nSpearman correlation of XGBoost SHAP importance, train vs test survey: {rho:.3f}")
print(shap_df.head(15).to_string(index=False))
shap_df.to_csv(os.path.join(OUT_DIR, f"temporal_validation_shap_xgboost{RUN_TAG}.csv"), index=False)
pd.DataFrame([{'model': n, 'params': str(best_params[n]), 'threshold': thresholds[n]} for n in MODEL_NAMES]).to_csv(
    os.path.join(OUT_DIR, f"temporal_validation_tuning{RUN_TAG}.csv"), index=False)
#======================================================================
# FIGURES (aggregate only): calibration plot and SHAP comparison between the two surveys
#======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.titlesize': 15,
                     'axes.labelsize': 14, 'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
lab = lambda f: '2019-20' if '2020' in f else '2024-25'
TRAIN_LAB, TEST_LAB = lab(TRAIN_FILE), lab(TEST_FILE)
OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']

# 1. Calibration (ten equal-sized risk groups) for the best-AUC model, XGBoost and logistic regression
aucs = {n: roc_auc_score(yt, preds_by_model[n]) for n in MODEL_NAMES}
best = max(aucs, key=aucs.get)
show = list(dict.fromkeys([best, 'XGBoost', 'Logistic Regression']))
cal_rows = []
fig, ax = plt.subplots(figsize=(7.5, 7.5))
for color, n in zip(OKABE, show):
    p = preds_by_model[n]
    grp = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False)
    for g in range(10):
        idx = (grp == g).to_numpy()
        pm, ob, k = float(p[idx].mean()), float(yt[idx].mean()), int(idx.sum())
        se = (ob * (1 - ob) / k) ** 0.5
        cal_rows.append({'model': n, 'group': g + 1, 'n': k, 'mean_predicted': round(pm, 4),
                         'observed': round(ob, 4), 'ci_low': round(max(0, ob - 1.96 * se), 4),
                         'ci_high': round(min(1, ob + 1.96 * se), 4)})
    d = pd.DataFrame([r for r in cal_rows if r['model'] == n])
    ax.errorbar(d.mean_predicted, d.observed, yerr=[d.observed - d.ci_low, d.ci_high - d.observed],
                marker='o', ms=7, lw=2, capsize=3, color=color, label=f"{n} (AUC {aucs[n]:.3f})")
ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=1.5, label='Perfect calibration')
ax.axhline(float(yt.mean()), color='black', lw=1, ls=':', label=f'Observed prevalence ({yt.mean():.1%})')
hi = max(0.8, float(max(preds_by_model[n].max() for n in show)))
ax.set_xlim(0, min(1, hi)); ax.set_ylim(0, min(1, hi))
ax.set_xlabel('Mean predicted risk in the group'); ax.set_ylabel('Observed proportion stunted')
ax.set_title(f'Calibration: trained on {TRAIN_LAB}, tested on {TEST_LAB}')
ax.legend(loc='upper left', frameon=False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"temporal_validation_calibration{RUN_TAG}.png"), dpi=300); plt.close(fig)
pd.DataFrame(cal_rows).to_csv(os.path.join(OUT_DIR, f"temporal_validation_calibration_groups{RUN_TAG}.csv"), index=False)

# 2. SHAP comparison: top 15 encoded features of the XGBoost model trained on the train survey
top = shap_df.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 8))
yy = np.arange(len(top)); h = 0.38
ax.barh(yy + h / 2, top.mean_abs_shap_train_survey, h, color=OKABE[0], label=f'{TRAIN_LAB} (training survey)')
ax.barh(yy - h / 2, top.mean_abs_shap_test_survey, h, color=OKABE[1], label=f'{TEST_LAB} (test survey)')
ax.set_yticks(yy); ax.set_yticklabels(top.feature)
ax.set_xlabel('Mean |SHAP value| (log-odds)')
ax.set_title(f'XGBoost feature importance (Spearman r = {rho:.2f})')
ax.legend(loc='lower right', frameon=False); ax.grid(axis='x', alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"temporal_validation_shap_comparison{RUN_TAG}.png"), dpi=300); plt.close(fig)

print("\nDone. Results are summary tables only. Do not share the .dta files or any per-child output.")
