"""
build_app_files.py
Run ONCE. It rebuilds the FINAL model from your notebook (Phase 10: train on Feb-Nov 2025,
test on Dec 2025 - Feb 2026) and saves the small files the Streamlit app needs.

It uses the same code and the same random seeds as your notebook, so the test result
printed at the end must match your notebook:
    MAE 30.282 | RMSE 58.267 | R2 0.361 | bias 0.134
"""
import os, shutil, time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

ROOT = os.environ.get("RAIL_ROOT", r"D:\Indian_Railway_Delay_Project")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "app_files")
N_FIT = int(os.environ.get("RAIL_NFIT", 3_000_000))
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


# ------------------------------------------------------------------ 1. load
log("Loading cleaned delay data (delay_clean_v2.pkl) ...")
clean2 = pd.read_pickle(os.path.join(ROOT, "delay_clean_v2.pkl"))
schedule = pd.read_csv(os.path.join(DATA, "combined_schedule.csv"))
stations = pd.read_csv(os.path.join(DATA, "station_full_names.csv"))
train_details = pd.read_csv(os.path.join(DATA, "train_details.csv"))
log(f"Loaded {len(clean2):,} rows")


# ------------------------------------------------------------------ 2. model table (Cell 7.1)
def to_minutes(t):
    hm = t.str.split(":", expand=True).astype(float)
    return hm[0] * 60 + hm[1]


arr = to_minutes(schedule["arrival_time"])
dep = to_minutes(schedule["departure_time"])
schedule["sched_hour"] = arr.fillna(dep) // 60
schedule["key"] = schedule["train_no"].astype("int64") * 1000 + schedule["station_no"].astype("int64")
assert schedule["key"].is_unique
look = schedule.set_index("key")

key = clean2["train_no"].astype("int64") * 1000 + clean2["station_no"].astype("int64")
base = pd.DataFrame({
    "date":         clean2["date"],
    "train_no":     clean2["train_no"].astype("int64"),
    "station_name": clean2["station_name"],
    "station_no":   clean2["station_no"].astype("int16"),
    "dist":         key.map(look["distance_from_origin"]).astype("float32"),
    "hour":         key.map(look["sched_hour"]).astype("float32"),
    "delay":        clean2["delay"].astype("float32"),
    "cap":          clean2["delay_capped"].astype("float32"),
})
del clean2, key
ok = base["dist"].notna() & base["hour"].notna()
base = base.loc[ok].reset_index(drop=True)
log(f"Modelling table rows: {len(base):,}   (notebook: 35,524,132)")

tr = base["date"] < "2025-11-01"
va = (base["date"] >= "2025-11-01") & (base["date"] < "2025-12-01")
te = base["date"] >= "2025-12-01"
tv = base["date"] < "2025-12-01"
log(f"Train / Validation / Test: {tr.sum():,} {va.sum():,} {te.sum():,}   (notebook: 25,879,636 2,971,571 6,672,925)")

# ------------------------------------------------------------------ 3. 7-day history (Cell 8.1)
days = pd.date_range(base["date"].min().normalize(), base["date"].max().normalize())
nD = len(days)
d_idx = (base["date"] - days[0]).dt.days.to_numpy().astype("int64")
y = base["cap"].to_numpy("float64")


def history_feature(codes, n_groups, k):
    """Mean of `cap` for the same group over the previous k days (today excluded)."""
    flat = codes.astype("int64") * nD + d_idx
    S = np.bincount(flat, weights=y, minlength=n_groups * nD).reshape(n_groups, nD)
    C = np.bincount(flat, minlength=n_groups * nD).reshape(n_groups, nD).astype("float64")
    zero = np.zeros((n_groups, 1))
    cS = np.hstack([zero, np.cumsum(S, axis=1)])
    cC = np.hstack([zero, np.cumsum(C, axis=1)])
    d = np.arange(nD)
    lo = np.maximum(d - k, 0)
    wS, wC = cS[:, d] - cS[:, lo], cC[:, d] - cC[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(wC > 0, wS / wC, np.nan)
    return M[codes, d_idx].astype("float32")


tcode, tuni = pd.factorize(base["train_no"])
scode, suni = pd.factorize(base["station_name"])
base["th7"] = history_feature(tcode, len(tuni), 7)
base["sh7"] = history_feature(scode, len(suni), 7)
base["g7"] = history_feature(np.zeros(len(base), dtype="int64"), 1, 7)
del tcode, scode, y
log("7-day history features built")

# ------------------------------------------------------------------ 4. encoders + features (Cells 7.2, 8.2, 10.2)
K = 20


def make_encoders_tv(target):
    d = base.loc[tv, ["train_no", "station_name", target]]
    mu = float(d[target].mean())
    enc = {}
    for col in ["train_no", "station_name"]:
        g = d.groupby(col, observed=True)[target].agg(["count", "mean"]).astype("float64")
        enc[col] = (g["count"] * g["mean"] + K * mu) / (g["count"] + K)
    return mu, enc


def build_X(rows, enc=None, mu=None):
    X = pd.DataFrame({
        "station_no": rows["station_no"].astype("float32"),
        "dist":       rows["dist"],
        "morning":    rows["hour"].between(6, 11).astype("float32"),
        "afternoon":  rows["hour"].between(12, 17).astype("float32"),
        "evening":    (rows["hour"] >= 18).astype("float32"),
    })
    if enc is not None:
        for col, name in [("train_no", "te_train"), ("station_name", "te_station")]:
            X[name] = (rows[col].map(enc[col]).astype("float64").fillna(mu).astype("float32"))
    return X


HIST = {"th7": "hist_train7", "sh7": "hist_station7", "g7": "hist_system7"}


def build_X3(rows, enc, mu, use):
    X = build_X(rows, enc, mu)
    fallback = {"th7": X["te_train"], "sh7": X["te_station"], "g7": mu}
    for c in use:
        X[HIST[c]] = rows[c].fillna(fallback[c]).astype("float32")
    return X


def metrics(y, p):
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    e = p - y
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), "bias": e.mean()}


# ------------------------------------------------------------------ 5. final model (Cell 10.2)
USE = ["th7", "sh7", "g7"]
mu_d, enc_d = make_encoders_tv("delay")
rng = np.random.RandomState(42)
idx = rng.choice(np.flatnonzero(tv.to_numpy()), min(N_FIT, int(tv.sum())), replace=False)
fr = base.iloc[idx]
log("Fitting final Linear Regression ...")
m3 = LinearRegression().fit(build_X3(fr, enc_d, mu_d, USE), fr["delay"])
cols = list(build_X3(fr.head(5), enc_d, mu_d, USE).columns)
print("Coefficients:", {c: round(float(v), 3) for c, v in zip(cols, m3.coef_)})
print("Notebook    : station_no .371, dist .008, morning 1.674, afternoon 1.243, evening 2.912, "
      "te_train .528, te_station -.302, hist_train7 .473, hist_station7 .438, hist_system7 .171")
print("Intercept:", float(m3.intercept_))

te_rows = base.loc[te]
p_te = m3.predict(build_X3(te_rows, enc_d, mu_d, USE))
m = metrics(te_rows["delay"], p_te)
print("\n=== FINAL TEST RESULT (must match notebook) ===")
print("This run :", {k: round(float(v), 3) for k, v in m.items()})
print("Notebook : {'MAE': 30.282, 'RMSE': 58.267, 'R2': 0.361, 'bias': 0.134}\n")

# ------------------------------------------------------------------ 6. save model + lookups
joblib.dump(m3, os.path.join(OUT, "model_m3.joblib"))
enc_d["train_no"].rename("te").to_csv(os.path.join(OUT, "enc_train.csv"))
enc_d["station_name"].rename("te").to_csv(os.path.join(OUT, "enc_station.csv"))
pd.Series({"mu": mu_d}).to_csv(os.path.join(OUT, "mu.csv"))
schedule[["train_no", "station_no", "station_name", "distance_from_origin", "sched_hour"]] \
    .to_csv(os.path.join(OUT, "schedule_lookup.csv"), index=False)

# latest 7-day history = last 7 days of data
last = base[base["date"] >= base["date"].max() - pd.Timedelta(days=6)]
hist_start, hist_end = last["date"].min().date(), last["date"].max().date()
last.groupby("train_no")["cap"].mean().rename("h").to_csv(os.path.join(OUT, "hist_train.csv"))
last.groupby("station_name", observed=True)["cap"].mean().rename("h").to_csv(os.path.join(OUT, "hist_station.csv"))
pd.Series({"g": last["cap"].mean(), "start": str(hist_start), "end": str(hist_end)}) \
    .to_csv(os.path.join(OUT, "hist_system.csv"))
log(f"History window saved: {hist_start} to {hist_end}")

# names
pr = {"T18-TRAINS": 1, "RAJ-TRAINS": 2, "SHT-TRAINS": 3, "GRB-TRAINS": 4,
      "EXP-TRAINS": 5, "SF-TRAINS": 6, "PASS-TRAINS": 7, "PRM-TRAINS": 8}
td = train_details.copy()
td["rank"] = td["type_code"].map(pr)
td = td.sort_values(["train_no", "rank"]).drop_duplicates("train_no")[["train_no", "train_name", "type_code"]]
td.to_csv(os.path.join(OUT, "train_names.csv"), index=False)
stations[["station_name", "station_full_name", "station_zone"]].to_csv(
    os.path.join(OUT, "station_names.csv"), index=False)

# ------------------------------------------------------------------ 7. test sample with predictions (for the app charts)
n_s = min(100_000, len(te_rows))
pos = np.random.RandomState(7).choice(len(te_rows), n_s, replace=False)
samp = te_rows.iloc[pos][["date", "train_no", "station_no", "station_name", "dist", "hour",
                          "delay", "th7", "sh7", "g7"]].copy()
samp["pred"] = np.asarray(p_te)[pos].astype("float32")
samp["station_name"] = samp["station_name"].astype(str)
samp.to_csv(os.path.join(OUT, "test_sample.csv"), index=False)
log("Test sample saved")

# ------------------------------------------------------------------ 8. summary tables for the Delay Explorer tab
ym = base["date"].dt.year * 100 + base["date"].dt.month
mon = base.groupby(ym)["delay"].agg(["count", "mean", "median"]).round(2)
mon.index = [f"{str(i)[:4]}-{str(i)[4:]}" for i in mon.index]
mon.index.name = "month"
mon.to_csv(os.path.join(OUT, "by_month.csv"))

tr_stat = base.groupby("train_no")["delay"].agg(["count", "mean"]).round(2)
tr_stat = tr_stat[tr_stat["count"] >= 1000].sort_values("mean", ascending=False).head(15).reset_index()
tr_stat = tr_stat.merge(td, on="train_no", how="left")
tr_stat.to_csv(os.path.join(OUT, "top_trains.csv"), index=False)

st_stat = base.groupby("station_name", observed=True)["delay"].agg(["count", "sum"])
st_stat = st_stat.reset_index()
st_stat["station_name"] = st_stat["station_name"].astype(str)
st_stat["mean"] = (st_stat["sum"] / st_stat["count"]).round(2)
top_st = st_stat[st_stat["count"] >= 1000].sort_values("mean", ascending=False).head(15)
top_st = top_st.merge(stations[["station_name", "station_full_name", "station_zone"]],
                      on="station_name", how="left")[["station_name", "station_full_name",
                                                       "station_zone", "count", "mean"]]
top_st.to_csv(os.path.join(OUT, "top_stations.csv"), index=False)

z = st_stat.merge(stations[["station_name", "station_zone"]], on="station_name", how="left")
z = z.dropna(subset=["station_zone"]).groupby("station_zone")[["count", "sum"]].sum()
z["mean"] = (z["sum"] / z["count"]).round(2)
z.sort_values("mean", ascending=False)[["count", "mean"]].to_csv(os.path.join(OUT, "by_zone.csv"))
log("Summary tables saved")

# ------------------------------------------------------------------ 9. copy your figures (if they exist)
fig_src = os.path.join(ROOT, "figures")
fig_dst = os.path.join(OUT, "figures")
os.makedirs(fig_dst, exist_ok=True)
for f in ["fig1_model_comparison.png", "fig2_actual_vs_pred.png", "fig3_error_share.png"]:
    if os.path.exists(os.path.join(fig_src, f)):
        shutil.copy(os.path.join(fig_src, f), fig_dst)
heat = os.path.join(ROOT, "graph_correlation_heatmap.png")
if os.path.exists(heat):
    shutil.copy(heat, fig_dst)

print("\nFiles in app_files:")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    if os.path.isfile(p):
        print(f"  {f:24s} {os.path.getsize(p) / 1024:10.1f} KB")
log("DONE. Now run:  streamlit run app.py")