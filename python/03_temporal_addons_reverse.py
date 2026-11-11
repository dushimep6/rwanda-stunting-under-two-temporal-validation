#======================================================================
# Add-on analyses for the temporal validation: recalibration in the test survey,
# decision curves, survey-weighted AUC and subgroup AUC.
# Direction: train on 2024-25, test on 2019-20 (reverse).
# Run 01_temporal_validation_*.py first: this script reuses its tuned models, so the
# files, direction, AGE_MAX, HARMONISED and RUN_TAG settings must be the same.
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
TRAIN_FILE = "rwanda_stunting_clean_2025_b_v4.dta"   # train survey
TEST_FILE  = "rwanda_stunting_clean_2020_v4.dta" # test survey (scored once)
RUN_TAG    = "_train2025_test2020_v7"              # change when you swap the files
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

WT_STORE = {}
def load(fname):
    # Read only the columns we use. Other columns (e.g. strata_id) can carry repeated value
    # labels, which makes pandas stop when it converts every column.
    _cols = list(dict.fromkeys(['cluster_id', 'stunted', 'hw1'] + ml_features))
    try:
        d = pd.read_stata(os.path.join(OUT_DIR, fname), columns=_cols + ['wt'])
    except (ValueError, KeyError):
        d = pd.read_stata(os.path.join(OUT_DIR, fname), columns=_cols)   # no sample weights in this file
    if AGE_MAX is not None:
        d = d[pd.to_numeric(d['hw1'], errors='coerce') < AGE_MAX].copy()
        d = d[['cluster_id', 'stunted'] + ml_features + (['wt'] if 'wt' in d.columns else [])]
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
    WT_STORE[fname] = pd.to_numeric(d['wt'], errors='coerce').to_numpy(dtype=float) if 'wt' in d.columns else None
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
# TEMPORAL ADD-ON: REUSES THE TUNED MODELS OF THE TEMPORAL RUN (NO NEW TUNING)
#======================================================================
# Reads the best hyperparameters and thresholds that 01_temporal_validation_*.py saved
# (temporal_validation_tuning<RUN_TAG>.csv), refits every model on the TRAIN survey exactly as that
# script did and scores the TEST survey once. It then adds:
#   1. survey-weighted AUC in the test survey (if the weight column wt is in the file);
#   2. recalibration in the new survey: cross-fitted intercept-only and Platt recalibration using the
#      test-survey children (grouped 5-fold by cluster), with Brier score, calibration intercept and slope,
#      expected calibration error and cluster-bootstrap 95% CIs before and after;
#   3. decision curve analysis (net benefit) on the recalibrated risks, against treating everybody and
#      treating nobody, with cluster-bootstrap CIs for the advantage over treating everybody;
#   4. AUC by subgroup in the test survey (province, residence, wealth, sex, age group).
# Output: aggregate tables only. Nothing per child is written.

import ast
from scipy.special import expit, logit
from scipy.stats import norm

lab = lambda f: '2019-20' if '2020' in f else '2024-25'
TRAIN_LAB, TEST_LAB = lab(TRAIN_FILE), lab(TEST_FILE)
BASE_TAG = RUN_TAG
RUN_TAG = RUN_TAG + "_addons"
tun_path = os.path.join(OUT_DIR, f"temporal_validation_tuning{BASE_TAG}.csv")
res_path = os.path.join(OUT_DIR, f"temporal_validation_results{BASE_TAG}.csv")
for pth in (tun_path, res_path):
    if not os.path.exists(pth):
        raise FileNotFoundError(f"{pth} not found. Run the temporal validation script with the SAME settings "
                                f"(files, direction, AGE_MAX, HARMONISED) first; this add-on reuses its tuned models.")
tun = pd.read_csv(tun_path)
best_params = {r['model']: ast.literal_eval(r['params']) for _, r in tun.iterrows()}
for m in MODEL_NAMES:
    if m not in best_params:
        raise ValueError(f"No tuned parameters for {m} in {tun_path}")

X_tr_all, X_te_all = impute_and_encode(X_train_raw, X_test_raw)
spw_all = float((y_train == 0).sum() / (y_train == 1).sum())
yt = y_test.to_numpy().astype(int)
n = len(yt)
test_clusters = X_test_raw['cluster_id'].to_numpy()
w_raw = WT_STORE.get(TEST_FILE)
HAVE_W = w_raw is not None and np.isfinite(w_raw).all() and (w_raw > 0).all()
w = (w_raw / w_raw.mean()) if HAVE_W else np.ones(n)
print(f"\nTest survey {TEST_FILE}: n = {n}, stunted = {int(yt.sum())} ({yt.mean():.3f}); "
      f"survey weights {'found' if HAVE_W else 'NOT found (weighted results skipped)'}")

preds = {}
for name in MODEL_NAMES:
    mdl = fit_with_weights(name, build_final_model(name, best_params[name]), X_tr_all, y_train, spw_all)
    preds[name] = mdl.predict_proba(X_te_all)[:, 1]

# reproduction check against the temporal results file
prev_res = pd.read_csv(res_path)
print("\nReproduction check (test-survey AUC) against the temporal run:")
for name in MODEL_NAMES:
    old = float(str(prev_res.loc[prev_res['Model'] == name, 'auc'].iloc[0]).split()[0])
    new = roc_auc_score(yt, preds[name])
    flag = "" if abs(old - new) <= 0.003 else "   <-- DIFFERS, check that the settings match the temporal run"
    print(f"  {name:20s} {new:.3f}  (temporal run {old:.3f}){flag}")

# cluster bootstrap on the test survey
rng = np.random.default_rng(SEED)
uniq = np.unique(test_clusters)
members = {c: np.where(test_clusters == c)[0] for c in uniq}
boot = []
while len(boot) < max(N_BOOT, 200):
    b = np.concatenate([members[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
    if 0 < yt[b].sum() < len(b):
        boot.append(b)

EPS = 1e-6
def to_lp(p): return logit(np.clip(p, EPS, 1 - EPS))
def fit_intercept(y, lp):
    a = 0.0
    for _ in range(100):
        mu = expit(a + lp); step = np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-12); a += step
        if abs(step) < 1e-10: break
    return a
def fit_platt(y, lp):
    mm = LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000).fit(lp.reshape(-1, 1), y)
    return float(mm.intercept_[0]), float(mm.coef_[0][0])
def ece(y, p, bins=10):
    o = np.argsort(p); ch = np.array_split(o, bins)
    return float(sum(len(c) / len(y) * abs(y[c].mean() - p[c].mean()) for c in ch if len(c)))
def calib(y, p):
    lp = to_lp(p)
    return fit_intercept(y, lp), fit_platt(y, lp)[1], float(brier_score_loss(y, np.clip(p, EPS, 1 - EPS))), ece(y, p)

#======================================================================
# 1. SURVEY-WEIGHTED AUC IN THE TEST SURVEY
#======================================================================
if HAVE_W:
    rows = []
    for name in MODEL_NAMES:
        p = preds[name]
        est = roc_auc_score(yt, p, sample_weight=w)
        bs = [roc_auc_score(yt[b], p[b], sample_weight=w[b]) for b in boot]
        rows.append({'Model': name, 'AUC_unweighted': roc_auc_score(yt, p), 'AUC_weighted': est,
                     'AUC_weighted_lo': np.percentile(bs, 2.5), 'AUC_weighted_hi': np.percentile(bs, 97.5)})
    wauc = pd.DataFrame(rows)
    wauc.to_csv(os.path.join(OUT_DIR, f"temporal_addon_weighted_auc{RUN_TAG}.csv"), index=False)
    print("\n" + "=" * 100 + f"\nSURVEY-WEIGHTED AUC IN THE TEST SURVEY ({TEST_LAB})\n" + "=" * 100)
    print(wauc.round(3).to_string(index=False))

#======================================================================
# 2. RECALIBRATION IN THE NEW SURVEY (cross-fitted within the test survey, grouped by cluster)
#======================================================================
cvf = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(n), yt, groups=test_clusters))
variants = {}
for name in MODEL_NAMES:
    lp = to_lp(preds[name]); ri = np.full(n, np.nan); rp = np.full(n, np.nan)
    for tr, te in cvf:
        ri[te] = expit(fit_intercept(yt[tr], lp[tr]) + lp[te])
        a2, b2 = fit_platt(yt[tr], lp[tr]); rp[te] = expit(a2 + b2 * lp[te])
    variants[name] = {'Raw (trained on the other survey)': preds[name], 'Intercept recalibration': ri, 'Platt recalibration': rp}
rows = []
for name in MODEL_NAMES:
    for vn, p in variants[name].items():
        a, s, br, e = calib(yt, p)
        bb = np.array([calib(yt[b], p[b]) for b in boot])
        q = lambda j: (np.percentile(bb[:, j], 2.5), np.percentile(bb[:, j], 97.5))
        rows.append({'Model': name, 'Variant': vn, 'AUC': roc_auc_score(yt, p), 'Mean_predicted_risk': float(p.mean()),
                     'Observed_prevalence': float(yt.mean()), 'Calibration_intercept': a,
                     'Intercept_lo': q(0)[0], 'Intercept_hi': q(0)[1], 'Calibration_slope': s, 'Slope_lo': q(1)[0], 'Slope_hi': q(1)[1],
                     'Brier': br, 'Brier_lo': q(2)[0], 'Brier_hi': q(2)[1], 'ECE_10bins': e})
rec = pd.DataFrame(rows)
rec.to_csv(os.path.join(OUT_DIR, f"temporal_addon_recalibration{RUN_TAG}.csv"), index=False)
print("\n" + "=" * 100 + f"\nRECALIBRATION IN THE NEW SURVEY ({TEST_LAB}); cross-fitted within the test survey\n" + "=" * 100)
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    show = rec[['Model', 'Variant', 'AUC', 'Mean_predicted_risk', 'Calibration_intercept', 'Calibration_slope', 'Brier', 'ECE_10bins']].copy()
    print(show.round(3).to_string(index=False))
print(f"\nObserved prevalence in the test survey: {yt.mean():.3f}")

#======================================================================
# 3. DECISION CURVE ANALYSIS (net benefit) ON THE PLATT-RECALIBRATED RISKS
#======================================================================
pts = np.round(np.arange(0.05, 0.61, 0.05), 2)
def net_benefit(y, p, pt, ww):
    sw = ww.sum(); pred = p >= pt
    tp = ww[pred & (y == 1)].sum() / sw; fp = ww[pred & (y == 0)].sum() / sw
    return tp - fp * pt / (1 - pt)
def nb_all(y, pt, ww):
    sw = ww.sum(); prev = ww[y == 1].sum() / sw
    return prev - (1 - prev) * pt / (1 - pt)
rows = []
for name in MODEL_NAMES:
    p = variants[name]['Platt recalibration']
    for pt in pts:
        nb = net_benefit(yt, p, pt, w); na = nb_all(yt, pt, w)
        adv = np.array([net_benefit(yt[b], p[b], pt, w[b]) - nb_all(yt[b], pt, w[b]) for b in boot])
        nbs = np.array([net_benefit(yt[b], p[b], pt, w[b]) for b in boot])
        rows.append({'Model': name, 'Threshold_probability': pt, 'Net_benefit': nb, 'NB_lo': np.percentile(nbs, 2.5), 'NB_hi': np.percentile(nbs, 97.5),
                     'Net_benefit_treat_all': na, 'Advantage_over_treat_all': nb - na,
                     'Advantage_lo': np.percentile(adv, 2.5), 'Advantage_hi': np.percentile(adv, 97.5)})
dca = pd.DataFrame(rows)
dca.to_csv(os.path.join(OUT_DIR, f"temporal_addon_decision_curve{RUN_TAG}.csv"), index=False)
print("\n" + "=" * 100 + f"\nDECISION CURVE ANALYSIS in the test survey ({TEST_LAB}), Platt-recalibrated risks"
      + (", survey-weighted" if HAVE_W else "") + "\n" + "=" * 100)
print("Net benefit per child; 'advantage' = model minus treating every child (95% cluster-bootstrap CI). Treating nobody = 0.")
for name in ['CatBoost', 'XGBoost', 'Logistic Regression']:
    d = dca[dca.Model == name]
    print(f"\n{name}:")
    t = d.assign(Advantage=[f"{a:+.3f} [{lo:+.3f},{hi:+.3f}]" for a, lo, hi in zip(d.Advantage_over_treat_all, d.Advantage_lo, d.Advantage_hi)])
    print(t[['Threshold_probability', 'Net_benefit', 'Net_benefit_treat_all', 'Advantage']].round(4).to_string(index=False))
useful = {}
for name in MODEL_NAMES:
    d = dca[dca.Model == name]
    ok = d[d.Advantage_lo > 0].Threshold_probability
    useful[name] = f"{ok.min():.2f} to {ok.max():.2f}" if len(ok) else "none"
print("\nRange of threshold probabilities over which the model clearly beats treating everybody (lower CI above 0):")
for k, v in useful.items():
    print(f"  {k:20s} {v}")

#======================================================================
# 4. AUC BY SUBGROUP IN THE TEST SURVEY
#======================================================================
age_num = pd.to_numeric(X_test_raw['hw1'], errors='coerce')
age_grp = pd.cut(age_num, bins=[-0.1, 5.99, 11.99, 23.99],
                 labels=['0-5', '6-11', '12-23']).astype(str).values
SUB = {'Province (v024)': X_test_raw['v024'].astype(str).values, 'Residence (v025)': X_test_raw['v025'].astype(str).values,
       'Wealth quintile (v190)': X_test_raw['v190'].astype(str).values, 'Child sex (b4)': X_test_raw['b4'].astype(str).values,
       'Child age (months)': age_grp}
rows = []
for name in ['CatBoost', 'XGBoost', 'Logistic Regression']:
    p = preds[name]
    for var, lab in SUB.items():
        for lv in [l for l in pd.unique(lab) if l not in ('nan', 'None')]:
            m = (lab == lv)
            if m.sum() < 50 or len(np.unique(yt[m])) < 2:
                continue
            est = roc_auc_score(yt[m], p[m], sample_weight=w[m])
            bs = []
            for b in boot:
                sel = b[m[b]]
                if len(sel) >= 30 and 0 < yt[sel].sum() < len(sel):
                    bs.append(roc_auc_score(yt[sel], p[sel], sample_weight=w[sel]))
            rows.append({'Model': name, 'Variable': var, 'Level': lv, 'N': int(m.sum()), 'N_stunted': int(yt[m].sum()),
                         'AUC': est, 'AUC_lo': np.percentile(bs, 2.5) if len(bs) > 20 else np.nan,
                         'AUC_hi': np.percentile(bs, 97.5) if len(bs) > 20 else np.nan})
sub = pd.DataFrame(rows)
sub.to_csv(os.path.join(OUT_DIR, f"temporal_addon_subgroup_auc{RUN_TAG}.csv"), index=False)
print("\n" + "=" * 100 + f"\nAUC BY SUBGROUP IN THE TEST SURVEY ({TEST_LAB}); CatBoost shown\n" + "=" * 100)
print(sub[sub.Model == 'CatBoost'][['Variable', 'Level', 'N', 'N_stunted', 'AUC', 'AUC_lo', 'AUC_hi']].round(3).to_string(index=False))

print(f"\n{'='*70}\nTEMPORAL ADD-ON COMPLETE\n{'='*70}")
print("Files written (aggregate tables only):")
for f in (f"temporal_addon_weighted_auc{RUN_TAG}.csv", f"temporal_addon_recalibration{RUN_TAG}.csv",
          f"temporal_addon_decision_curve{RUN_TAG}.csv", f"temporal_addon_subgroup_auc{RUN_TAG}.csv"):
    print("  ", f)
