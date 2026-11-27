"""
Indian Railway Train Delay Prediction - Streamlit app
Khushbu Sharma | Roll No GF202455346 | Machine Learning | Shoolini University
Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

DIR = Path(__file__).parent / "app_files"

st.set_page_config(page_title="Indian Railway Delay Prediction", page_icon="🚆", layout="wide")

# Numbers copied from the notebook outputs (Phase 10, test = Dec 2025 - Feb 2026)
TEST_MAE, TEST_RMSE, TEST_R2 = 30.28, 58.27, 0.361
BASE_MAE, BASE_RMSE, BASE_R2 = 38.18, 73.34, -0.012


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load(DIR / "model_m3.joblib")


@st.cache_data
def load_tables():
    t = {}
    t["enc_train"] = pd.read_csv(DIR / "enc_train.csv", index_col=0)["te"]
    t["enc_station"] = pd.read_csv(DIR / "enc_station.csv", index_col=0)["te"]
    t["mu"] = float(pd.read_csv(DIR / "mu.csv", index_col=0).iloc[0, 0])
    t["sched"] = pd.read_csv(DIR / "schedule_lookup.csv")
    t["h_train"] = pd.read_csv(DIR / "hist_train.csv", index_col=0)["h"]
    t["h_station"] = pd.read_csv(DIR / "hist_station.csv", index_col=0)["h"]
    hs = pd.read_csv(DIR / "hist_system.csv", index_col=0).iloc[:, 0]
    t["h_sys"] = float(hs["g"])
    t["h_start"], t["h_end"] = hs["start"], hs["end"]
    t["trains"] = pd.read_csv(DIR / "train_names.csv").set_index("train_no")
    t["stations"] = pd.read_csv(DIR / "station_names.csv").set_index("station_name")
    return t


@st.cache_data
def load_sample():
    s = pd.read_csv(DIR / "test_sample.csv", parse_dates=["date"])
    s["err"] = s["pred"] - s["delay"]
    return s


def read_csv_safe(name, **kw):
    p = DIR / name
    return pd.read_csv(p, **kw) if p.exists() else None


try:
    model = load_model()
    T = load_tables()
except Exception as e:  # files not built yet
    st.error("App files not found. Run  `python build_app_files.py`  first.\n\n" + str(e))
    st.stop()


def metrics(y, p):
    e = np.asarray(p, float) - np.asarray(y, float)
    y = np.asarray(y, float)
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), "bias": e.mean()}


def train_label(n):
    name = T["trains"]["train_name"].get(n, "")
    return f"{n} - {name}" if name else str(n)


def station_label(code):
    full = T["stations"]["station_full_name"].get(code, "")
    return f"{code} ({full})" if full else str(code)


# ----------------------------------------------------------------------------- header
st.title("🚆 Indian Railway Train Delay Prediction")
st.caption("Linear Regression on one year of Indian Railways delay data (Feb 2025 - Feb 2026) | "
           "Khushbu Sharma | GF202455346 | Machine Learning, Shoolini University")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Project summary", "🔮 Predict delay", "📊 Model performance", "🔁 Replay real records", "🚉 Delay explorer"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    st.subheader("What this project does")
    st.write(
        "The model predicts how many **minutes late** a train will be at a given stop, using only "
        "information that is known **before** the train runs: the stop's position and distance, the "
        "scheduled time of day, the long-run delay of the train and station, and the delay level of "
        "the **last 7 days**. The delay at the previous stop is deliberately **not** used, because "
        "it is only known after the train has started (that would be data leakage).")
    c = st.columns(4)
    c[0].metric("Records used", "35.5 M", "stops covered by the schedule")
    c[1].metric("Test MAE", f"{TEST_MAE} min", f"baseline {BASE_MAE}", delta_color="off")
    c[2].metric("Test RMSE", f"{TEST_RMSE} min", f"baseline {BASE_RMSE}", delta_color="off")
    c[3].metric("Test R²", f"{TEST_R2}", f"baseline {BASE_R2}", delta_color="off")
    st.markdown("""
**Method (Phases 1-12 of the notebook)**
1. Four files understood, cleaned (missing delays removed, 9 impossible ~365-day values removed, 4,084 journeys with impossible jumps removed).
2. EDA and correlation: delay is strongly right-skewed; train identity explains about 29% of variance, station 11%, weekday about 0%.
3. Time-based split: **train** Feb-Oct 2025, **validation** Nov 2025, **test** Dec 2025 - Feb 2026.
4. Features: station number, distance, time of day, train and station average delay, 7-day history of train, station and network.
5. Model ladder (test R²): 0.072 → 0.253 → **0.361**.
""")
    st.warning(
        "**Honest limits:** R² 0.361 means the model explains about 36% of the variation in delay. "
        "Delays above 300 minutes are only 1.43% of rows but cause 53.3% of the squared error and are "
        "under-predicted. The data has no weather, congestion or disruption information and covers "
        "only one year. This is a historical-data model, not a live forecast.")

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Predict the delay at a stop")
    st.caption(f"Uses the average delay of the last 7 days of data ({T['h_start']} to {T['h_end']}). "
               "It is **not** a live forecast.")

    sched = T["sched"]
    known = sorted(set(sched["train_no"]) & set(T["enc_train"].index))
    train_no = st.selectbox("Train", known, format_func=train_label)
    stops = sched[sched["train_no"] == train_no].sort_values("station_no")

    def stop_label(i):
        r = stops.loc[i]
        hh = int(r["sched_hour"]) if pd.notna(r["sched_hour"]) else 0
        return f"Stop {int(r['station_no'])}: {station_label(r['station_name'])} | {int(r['distance_from_origin'])} km | scheduled hour {hh:02d}:00"

    stop_idx = st.selectbox("Stop on this train's route", list(stops.index), format_func=stop_label)
    r = stops.loc[stop_idx]

    code = r["station_name"]
    station_no = float(r["station_no"])
    dist = float(r["distance_from_origin"])
    hour = float(r["sched_hour"]) if pd.notna(r["sched_hour"]) else 12.0

    mu = T["mu"]
    te_train = float(T["enc_train"].get(train_no, mu))
    te_station = float(T["enc_station"].get(code, mu))
    h_train = float(T["h_train"].get(train_no, te_train))
    h_station = float(T["h_station"].get(code, te_station))
    h_sys = T["h_sys"]

    feats = {"station_no": station_no, "dist": dist,
             "morning": float(6 <= hour <= 11), "afternoon": float(12 <= hour <= 17),
             "evening": float(hour >= 18), "te_train": te_train, "te_station": te_station,
             "hist_train7": h_train, "hist_station7": h_station, "hist_system7": h_sys}
    X = pd.DataFrame([feats])[list(model.feature_names_in_)]
    pred = float(model.predict(X)[0])

    a, b, c3 = st.columns(3)
    a.metric("Predicted delay", f"{pred:.0f} min")
    b.metric("Typical error (test MAE)", f"± {TEST_MAE:.0f} min")
    c3.metric("Naive guess (overall average)", f"{mu:.0f} min")
    if pred < 0:
        st.info("A negative value means the model expects the train to arrive early.")
    st.write(f"**Likely range:** about {max(pred - TEST_MAE, -120):.0f} to {pred + TEST_MAE:.0f} minutes "
             "(prediction ± typical error). Individual trains can be much later: extreme delays are hard to predict.")

    st.markdown("**How this prediction was built** (coefficient × value for each feature)")
    contrib = pd.DataFrame({"feature": list(model.feature_names_in_),
                            "value": [feats[f] for f in model.feature_names_in_],
                            "coefficient": model.coef_})
    contrib["contribution (min)"] = contrib["value"] * contrib["coefficient"]
    contrib = pd.concat([pd.DataFrame([{"feature": "intercept", "value": 1.0,
                                        "coefficient": model.intercept_,
                                        "contribution (min)": model.intercept_}]), contrib],
                        ignore_index=True)
    st.dataframe(contrib.round(3), hide_index=True)
    st.caption(f"Sum of contributions = {contrib['contribution (min)'].sum():.1f} min. "
               "Coefficients of overlapping features (for example the station average and the "
               "station 7-day history) should be read together, not one by one.")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    st.subheader("How good is the model? (test period Dec 2025 - Feb 2026)")
    res = pd.DataFrame({
        "Model": ["Always predict the average (baseline)", "M2: + train & station averages",
                  "M3 FINAL: + 7-day history"],
        "MAE (min)": [38.18, 30.55, 30.28], "RMSE (min)": [73.34, 62.63, 58.27],
        "R²": [-0.012, 0.262, 0.361]})
    st.dataframe(res, hide_index=True)
    st.caption("From notebook Phase 10 (all 6,672,925 test rows). The charts below use a random sample of "
               "100,000 test rows.")

    S = load_sample()
    m = metrics(S["delay"], S["pred"])
    st.write(f"Sample check: MAE {m['MAE']:.1f}, RMSE {m['RMSE']:.1f}, R² {m['R2']:.3f}, bias {m['bias']:+.1f}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sub = S[(S["delay"].between(-60, 600)) & (S["pred"].between(-60, 600))]
        hb = ax.hexbin(sub["delay"], sub["pred"], gridsize=45, bins="log", cmap="viridis", mincnt=1)
        ax.plot([-60, 600], [-60, 600], "r--", label="perfect prediction")
        ax.set_xlabel("Actual delay (min)"); ax.set_ylabel("Predicted delay (min)")
        ax.set_title("Actual vs Predicted (sample)"); ax.legend()
        fig.colorbar(hb, label="log10(number of records)")
        st.pyplot(fig)
        st.caption("Points below the red line are under-predicted. The model rarely predicts very large delays.")
    with c2:
        S["decile"] = pd.qcut(S["pred"], 10, duplicates="drop")
        cal = S.groupby("decile", observed=True)[["pred", "delay"]].mean()
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot(cal["pred"], cal["delay"], "o-", label="model")
        lim = [min(cal.min()), max(cal.max())]
        ax.plot(lim, lim, "k--", label="perfect")
        ax.set_xlabel("Mean predicted delay (min)"); ax.set_ylabel("Mean actual delay (min)")
        ax.set_title("Calibration by prediction decile"); ax.legend()
        st.pyplot(fig)
        st.caption("On average the predictions are close to the actual values, except for the very highest group.")

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(S["err"].clip(-150, 300), bins=100, color="steelblue")
        ax.axvline(0, color="red", ls="--")
        ax.set_xlabel("Error = predicted − actual (min, clipped)"); ax.set_title("Error distribution")
        st.pyplot(fig)
    with c4:
        bk = pd.cut(S["delay"], [-1e9, 0, 15, 30, 60, 120, 300, 1e9],
                    labels=["<=0", "0-15", "15-30", "30-60", "60-120", "120-300", ">300"])
        g = S.groupby(bk, observed=True)[["delay", "pred"]].mean()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        x = np.arange(len(g)); w = 0.4
        ax.bar(x - w / 2, g["delay"], w, label="Mean actual")
        ax.bar(x + w / 2, g["pred"], w, label="Mean predicted")
        ax.set_xticks(x); ax.set_xticklabels(g.index)
        ax.set_xlabel("Actual delay bucket (min)"); ax.set_title("Model hedges toward the middle"); ax.legend()
        st.pyplot(fig)
    st.info("Why R² is only 0.36: a few very large delays (above 300 min, 1.43% of rows) create 53.3% of the "
            "squared error and a straight-line model cannot predict them. Weather and disruption data are not in the dataset.")

    figs = [DIR / "figures" / f for f in
            ["fig1_model_comparison.png", "fig2_actual_vs_pred.png", "fig3_error_share.png",
             "graph_correlation_heatmap.png"]]
    shown = [f for f in figs if f.exists()]
    if shown:
        st.markdown("**Figures from the notebook**")
        cols = st.columns(2)
        for i, f in enumerate(shown):
            cols[i % 2].image(str(f))

# ----------------------------------------------------------------------------- tab 4
with tab4:
    st.subheader("Replay real test records")
    st.write("Pick a train and see what the model predicted against what actually happened "
             "(random sample of the test period).")
    S = load_sample()
    trains_in = sorted(S["train_no"].unique())
    cnt = S["train_no"].value_counts()
    default = int(cnt.index[0])
    pick = st.selectbox("Train", trains_in, index=trains_in.index(default), format_func=train_label)
    rows = S[S["train_no"] == pick].sort_values(["date", "station_no"]).head(60).copy()
    rows["station"] = rows["station_name"].map(station_label)
    rows["predicted"] = rows["pred"].round(0)
    rows["actual"] = rows["delay"].round(0)
    rows["error"] = (rows["predicted"] - rows["actual"]).round(0)
    st.dataframe(rows[["date", "station", "actual", "predicted", "error"]],
                 hide_index=True)
    mm = metrics(rows["delay"], rows["pred"])
    st.write(f"For these {len(rows)} records: MAE {mm['MAE']:.1f} min, bias {mm['bias']:+.1f} min.")
    st.line_chart(rows.reset_index(drop=True)[["actual", "predicted"]])

# ----------------------------------------------------------------------------- tab 5
with tab5:
    st.subheader("Historical delay patterns (Feb 2025 - Feb 2026)")
    st.caption("Historical analysis of stops covered by the schedule. It describes the past; it is not a prediction. "
               "A high average does not prove that a train or station *causes* delay.")
    bm = read_csv_safe("by_month.csv", index_col=0)
    if bm is not None:
        st.markdown("**Average delay by month** (Feb 2025 and Feb 2026 are partial months)")
        st.line_chart(bm["mean"])
        st.write("Winter months (Dec to Feb) show the highest average delay in this data. "
                 "The dataset has no weather column, so the cause cannot be confirmed.")
    bz = read_csv_safe("by_zone.csv", index_col=0)
    if bz is not None:
        st.markdown("**Average delay by railway zone**")
        st.bar_chart(bz["mean"])
    c1, c2 = st.columns(2)
    tt = read_csv_safe("top_trains.csv")
    if tt is not None:
        c1.markdown("**15 trains with the highest average delay** (at least 1,000 records)")
        c1.dataframe(tt, hide_index=True)
    ts = read_csv_safe("top_stations.csv")
    if ts is not None:
        c2.markdown("**15 stations with the highest average delay** (at least 1,000 records)")
        c2.dataframe(ts, hide_index=True)
    st.caption("Station averages are confounded with the trains that pass through them: a few very late trains "
               "can raise a station's average.")