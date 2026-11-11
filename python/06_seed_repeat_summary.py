#======================================================================
# Summary of the seed repeats, children aged 0-23 months
# Reads the original results (seed 42) and every *_seed<N> results file in this folder and
# writes one table: AUC of each model by seed, with the range across seeds. Aggregate output only.
#======================================================================
import os, re, glob, pandas as pd
try: OUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: OUT_DIR = os.getcwd()
rows = []
for tag, lab in (("train2020_test2025", "Tested in 2024-25 (trained on 2019-20)"), ("train2025_test2020", "Tested in 2019-20 (trained on 2024-25)")):
    files = {42: f"temporal_validation_results_{tag}_v7_under24m.csv"}
    for f in glob.glob(os.path.join(OUT_DIR, f"temporal_validation_results_{tag}_v7_under24m_seed*.csv")):
        files[int(re.search(r"_seed(\d+)\.csv$", f).group(1))] = os.path.basename(f)
    res = {}
    for sd, f in sorted(files.items()):
        p = os.path.join(OUT_DIR, f)
        if not os.path.exists(p): raise FileNotFoundError(f"{f} not found in {OUT_DIR}")
        r = pd.read_csv(p).set_index("Model")
        res[sd] = {"auc": r["auc"].str.extract(r"([\d.]+)")[0].astype(float),
                   "slope": r["calibration_slope"].astype(float), "mean_risk": r["mean_predicted_risk"].astype(float)}
    for m in res[42]["auc"].index:
        a = [res[sd]["auc"][m] for sd in sorted(res)]
        row = {"Direction": lab, "Model": m, "Seeds": ", ".join(str(sd) for sd in sorted(res))}
        for sd in sorted(res): row[f"AUC_seed{sd}"] = round(res[sd]["auc"][m], 3)
        row["AUC_min"] = round(min(a), 3); row["AUC_max"] = round(max(a), 3); row["AUC_range"] = round(max(a) - min(a), 3)
        s_ = [res[sd]["slope"][m] for sd in sorted(res)]; mr = [res[sd]["mean_risk"][m] for sd in sorted(res)]
        row["Slope_min"] = round(min(s_), 2); row["Slope_max"] = round(max(s_), 2)
        row["MeanRisk_min"] = round(min(mr), 3); row["MeanRisk_max"] = round(max(mr), 3)
        rows.append(row)
out = pd.DataFrame(rows); out.to_csv(os.path.join(OUT_DIR, "seed_repeat_summary_under24m_v7.csv"), index=False)
print(out.to_string(index=False)); print("\nWrote seed_repeat_summary_under24m_v7.csv")
