#======================================================================
# Descriptive table, children aged 0-23 months, both surveys
# For each of the 33 candidate predictors: median (IQR) or n (%) in each survey, plus the
# percentage of children with the value missing. Also the number of children and the
# stunting prevalence. Output is aggregate only.
# Put this script and the two cleaned .dta files in one folder and run it.
#======================================================================
import os, numpy as np, pandas as pd
try: OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: OUT_DIR = os.getcwd()
FILES = {'2019-20': "rwanda_stunting_clean_2020_v4.dta", '2024-25': "rwanda_stunting_clean_2025_b_v4.dta"}
AGE_MAX = 24
cont = ['hw1', 'bord', 'v012', 'v136', 'v137', 'birth_weight', 'mother_height', 'mother_bmi',
        'bf_months', 'anc_visits', 'anc_timing', 'water_time']
cat = ['literacy', 'b4', 'b0', 'first_birth', 'lbw', 'birth_size', 'still_breastfeeding', 'v106', 'v190', 'v025', 'v024',
       'delivery_facility', 'deworming', 'iron_supp', 'm17', 'v404', 'water_source_grp', 'toilet_grp',
       'shared_toilet', 'v481', 'distance_problem']
LABEL = {'hw1': "Child's age (months)", 'bord': 'Birth order', 'v012': "Mother's age (years)", 'v136': 'Household size',
         'v137': 'Children under 5 in household', 'literacy': "Mother's literacy category (0-2)", 'birth_weight': 'Birth weight (g)',
         'mother_height': "Mother's height (cm)", 'mother_bmi': "Mother's body mass index", 'bf_months': 'Months breastfed',
         'anc_visits': 'Antenatal visits (n)', 'anc_timing': 'Month of pregnancy at first antenatal visit', 'water_time': 'Minutes to water source',
         'b4': "Child's sex", 'b0': 'Multiple birth', 'first_birth': 'First birth', 'lbw': 'Low birth weight',
         'birth_size': 'Size at birth (mother report)', 'still_breastfeeding': 'Still breastfeeding (from months breastfed)', 'v106': "Mother's education",
         'v190': 'Wealth quintile', 'v025': 'Residence', 'v024': 'Region', 'delivery_facility': 'Place of delivery',
         'deworming': 'Dewormed', 'iron_supp': 'Iron supplements in pregnancy', 'm17': 'Delivery by caesarean section',
         'v404': 'Currently breastfeeding (DHS item v404)', 'water_source_grp': 'Drinking water source', 'toilet_grp': 'Toilet type',
         'shared_toilet': 'Toilet shared', 'v481': 'Health insurance', 'distance_problem': 'Distance to facility a problem'}
REGION = {'city of kigali': 'kigali', 'southern province': 'south', 'western province': 'west',
          'northern province': 'north', 'eastern province': 'east'}
cols = ['cluster_id', 'stunted'] + cont + cat
data = {}
for s, f in FILES.items():
    p = os.path.join(OUT_DIR, f)
    if not os.path.exists(p): raise FileNotFoundError(f"{f} not found in {OUT_DIR}")
    d = pd.read_stata(p, columns=list(dict.fromkeys(cols)))
    d = d[pd.to_numeric(d['hw1'], errors='coerce') < AGE_MAX].copy()
    d['stunted01'] = (d['stunted'].astype(str).str.strip() == 'Stunted').astype(int)
    d['literacy'] = pd.to_numeric(d['literacy'], errors='coerce').map(lambda z: None if pd.isna(z) else str(int(z)))
    for v in cat:
        x = d[v].astype(object); x = x.where(x.notna(), None)
        d[v] = x.map(lambda z: None if (z is None or str(z).strip() in ('', 'nan', '.')) else str(z).strip().lower())
    d['v024'] = d['v024'].replace(REGION)
    for v in cont: d[v] = pd.to_numeric(d[v], errors='coerce')
    data[s] = d
S1, S2 = list(FILES)
rows = [{'Variable': 'Children (n)', 'Level': '', S1: f"{len(data[S1]):,}", S2: f"{len(data[S2]):,}", f'Missing % {S1}': '', f'Missing % {S2}': ''},
        {'Variable': 'Stunted, n (%)', 'Level': '',
         S1: f"{data[S1]['stunted01'].sum():,} ({100*data[S1]['stunted01'].mean():.1f}%)",
         S2: f"{data[S2]['stunted01'].sum():,} ({100*data[S2]['stunted01'].mean():.1f}%)", f'Missing % {S1}': '', f'Missing % {S2}': ''}]
for v in cont:
    r = {'Variable': LABEL[v], 'Level': 'median (IQR)'}
    for s in FILES:
        x = data[s][v].dropna()
        r[s] = f"{x.median():.1f} ({x.quantile(.25):.1f} to {x.quantile(.75):.1f})" if len(x) else 'not recorded'
        r[f'Missing % {s}'] = f"{100*data[s][v].isna().mean():.1f}"
    rows.append(r)
for v in cat:
    levels = sorted(set(data[S1][v].dropna()) | set(data[S2][v].dropna()))
    for i, lv in enumerate(levels):
        r = {'Variable': LABEL[v] if i == 0 else '', 'Level': lv}
        for s in FILES:
            x = data[s][v]; nn = int(x.notna().sum()); k = int((x == lv).sum())
            r[s] = f"{k:,} ({100*k/nn:.1f}%)" if nn else '-'
            r[f'Missing % {s}'] = f"{100*x.isna().mean():.1f}" if i == 0 else ''
        rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT_DIR, "descriptive_table_under24m_v7.csv"), index=False)
print(tab.to_string(index=False))
print("\nWrote descriptive_table_under24m_v7.csv (percentages use children with a recorded value).")
