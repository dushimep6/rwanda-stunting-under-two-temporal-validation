#======================================================================
# Within-survey comparison, children aged 0-23 months
# Trains and tests the seven models inside each survey (grouped nested cross-validation)
# and compares them with the models trained on the other survey, on the same test children.
# Needs the tuning and results files written by 01_temporal_validation_*.py (same folder).
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
TRAIN_FILE = "rwanda_stunting_clean_2020_v4.dta"   # train survey
TEST_FILE  = "rwanda_stunting_clean_2025_b_v4.dta" # test survey (scored once)
RUN_TAG    = "_train2020_test2025_v7"              # change when you swap the files
N_BOOT     = 1000
N_TRIALS_WITHIN = 50   # Optuna trials per model per outer fold (the temporal runs used 100; 50 keeps the run time manageable)
INNER_K    = 4
OUTER_K    = 5
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
# WITHIN-SURVEY GROUPED NESTED CROSS-VALIDATION, ONE SURVEY AT A TIME
#======================================================================
import ast
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = N_TRIALS_WITHIN        # make_objective is reused unchanged; it only needs the inner splits

SURVEYS = {'2019-20': (TRAIN_FILE, X_train_raw, y_train, 'train2020_test2025'),
           '2024-25': (TEST_FILE,  X_test_raw,  y_test,  'train2025_test2020')}
# direction in which the OTHER survey is the training survey, for the paired comparison
TEMPORAL = {'2019-20': ('train2025_test2020', TEST_FILE, TRAIN_FILE),    # test 2019-20, train 2024-25
            '2024-25': ('train2020_test2025', TRAIN_FILE, TEST_FILE)}    # test 2024-25, train 2019-20

def cal_stats(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e10, solver='lbfgs', max_iter=2000).fit(lp, y)
    return float(m.intercept_[0]), float(m.coef_[0][0]), float(brier_score_loss(y, p))

oof_all, rows = {}, []
for sname, (fname, Xraw, yser, _) in SURVEYS.items():
    y = yser.to_numpy().astype(int); n = len(y); cl = Xraw['cluster_id'].to_numpy()
    # category lists from this survey (labels only, no outcome information), so the one-hot columns are the same in every fold
    CATS = {v: sorted(set(Xraw[v].dropna())) for v in categorical_vars}
    outer = StratifiedGroupKFold(n_splits=OUTER_K, shuffle=True, random_state=SEED)
    oof = {m: np.full(n, np.nan) for m in MODEL_NAMES}
    for k, (otr, ote) in enumerate(outer.split(Xraw, y, groups=cl), start=1):
        if set(cl[otr]) & set(cl[ote]): raise RuntimeError("Cluster leakage in outer fold")
        Xo_tr, Xo_te = Xraw.iloc[otr].reset_index(drop=True), Xraw.iloc[ote].reset_index(drop=True)
        yo_tr = y[otr]
        inner = StratifiedGroupKFold(n_splits=INNER_K, shuffle=True, random_state=SEED)
        isp = []
        for itr, iva in inner.split(Xo_tr, yo_tr, groups=Xo_tr['cluster_id']):
            Xi_tr, Xi_va = impute_and_encode(Xo_tr.iloc[itr], Xo_tr.iloc[iva])
            yi_tr = pd.Series(yo_tr[itr]).reset_index(drop=True); yi_va = pd.Series(yo_tr[iva]).reset_index(drop=True)
            isp.append((Xi_tr, yi_tr, Xi_va, yi_va, float((yi_tr == 0).sum() / (yi_tr == 1).sum())))
        Xa, Xb = impute_and_encode(Xo_tr, Xo_te)
        spw = float((yo_tr == 0).sum() / (yo_tr == 1).sum())
        for name in MODEL_NAMES:
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=4200))
            study.optimize(make_objective(name, isp), n_trials=N_TRIALS, show_progress_bar=False)
            mdl = fit_with_weights(name, build_final_model(name, study.best_params), Xa, pd.Series(yo_tr), spw)
            oof[name][ote] = mdl.predict_proba(Xb)[:, 1]
        print(f"  {sname}: outer fold {k}/{OUTER_K} done")
    oof_all[sname] = oof
    uniq = np.unique(cl); members = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(SEED); boot = []
    while len(boot) < N_BOOT:
        b = np.concatenate([members[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        if 0 < y[b].sum() < len(b): boot.append(b)
    oof_all[sname]['_boot'] = boot; oof_all[sname]['_y'] = y
    for name in MODEL_NAMES:
        p = oof[name]
        a, s, br = cal_stats(y, p)
        bs = [roc_auc_score(y[b], p[b]) for b in boot]
        rows.append({'Survey': sname, 'Model': name, 'n': n, 'n_stunted': int(y.sum()),
                     'AUC_within_survey': round(roc_auc_score(y, p), 3),
                     'AUC_lo': round(np.percentile(bs, 2.5), 3), 'AUC_hi': round(np.percentile(bs, 97.5), 3),
                     'mean_predicted_risk': round(float(p.mean()), 3), 'observed_prevalence': round(float(y.mean()), 3),
                     'calibration_intercept': round(a, 2), 'calibration_slope': round(s, 2), 'brier': round(br, 4)})
within = pd.DataFrame(rows)
within.to_csv(os.path.join(OUT_DIR, "within_survey_cv_results_under24m_v7.csv"), index=False)
print("\n" + "=" * 100 + "\nWITHIN-SURVEY GROUPED NESTED CROSS-VALIDATION (children aged 0-23 months)\n" + "=" * 100)
print(within.to_string(index=False))

#======================================================================
# PAIRED COMPARISON WITH THE MODELS TRAINED ON THE OTHER SURVEY (same test children)
#======================================================================
rows = []
for sname in SURVEYS:
    tag, trn, tst = TEMPORAL[sname]
    tun = pd.read_csv(os.path.join(OUT_DIR, f"temporal_validation_tuning_{tag}_v7_under24m.csv"))
    bp = {r['model']: ast.literal_eval(r['params']) for _, r in tun.iterrows()}
    Xtr_raw, ytr = load(trn); Xte_raw, yte = load(tst)
    CATS = {v: sorted(set(Xtr_raw[v].dropna())) for v in categorical_vars}      # as in the temporal run
    Xa, Xb = impute_and_encode(Xtr_raw, Xte_raw)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    y = oof_all[sname]['_y']; boot = oof_all[sname]['_boot']
    if not np.array_equal(y, yte.to_numpy().astype(int)): raise RuntimeError("Child order differs between runs")
    for name in MODEL_NAMES:
        mdl = fit_with_weights(name, build_final_model(name, bp[name]), Xa, ytr, spw)
        pt = mdl.predict_proba(Xb)[:, 1]; pw = oof_all[sname][name]
        d = roc_auc_score(y, pw) - roc_auc_score(y, pt)
        bd = [roc_auc_score(y[b], pw[b]) - roc_auc_score(y[b], pt[b]) for b in boot]
        rows.append({'Test_survey': sname, 'Model': name,
                     'AUC_trained_on_other_survey': round(roc_auc_score(y, pt), 3),
                     'AUC_trained_within_survey': round(roc_auc_score(y, pw), 3),
                     'Difference_within_minus_other': round(d, 3),
                     'Diff_lo': round(np.percentile(bd, 2.5), 3), 'Diff_hi': round(np.percentile(bd, 97.5), 3)})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(os.path.join(OUT_DIR, "within_vs_temporal_auc_under24m_v7.csv"), index=False)
print("\n" + "=" * 100 + "\nSAME TEST CHILDREN: model trained within the survey vs trained on the other survey\n" + "=" * 100)
print(cmp_df.to_string(index=False))
print("\nFiles written (aggregate tables only):\n   within_survey_cv_results_under24m_v7.csv\n   within_vs_temporal_auc_under24m_v7.csv")
