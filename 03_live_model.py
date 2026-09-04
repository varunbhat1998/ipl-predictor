import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

Path("models").mkdir(exist_ok=True)
TEST_SEASONS = ["2023", "2024", "2025"]

print("=" * 55)
print("MODEL B: Live in-match (Calibrated Logistic Regression)")
print("=" * 55)

deliveries = pd.read_csv("data/deliveries.csv")
deliveries["date"]    = pd.to_datetime(deliveries["date"])
deliveries["season"]  = deliveries["date"].dt.year.astype(str)
deliveries["file_id"] = deliveries["file_id"].astype(str)

# 2nd innings only
inn2 = deliveries[deliveries["innings"] == 2].copy()

# 1st innings totals
inn1_totals = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")["runs_total"].sum()
    .rename("target").reset_index()
)
inn1_totals["target"]  = inn1_totals["target"] + 1
inn1_totals["file_id"] = inn1_totals["file_id"].astype(str)
inn2 = inn2.merge(inn1_totals, on="file_id", how="left")

# winner is already in deliveries — just drop nulls
inn2 = inn2[inn2["winner"].notna()].copy()
inn2["chasing_won"] = (inn2["winner"] == inn2["batting_team"]).astype(int)
print(f"Usable 2nd innings rows: {len(inn2)}")

inn2 = inn2.sort_values(["file_id", "over", "ball_in_over"]).copy()
inn2["ball_num"]        = inn2.groupby("file_id").cumcount() + 1
inn2["balls_remaining"] = (120 - inn2["ball_num"]).clip(0)
inn2["runs_needed"]     = (inn2["target"] - inn2["cum_runs"]).clip(0)
inn2["wickets_left"]    = 10 - inn2["cum_wickets"]
inn2["crr"]             = np.where(inn2["ball_num"] > 0, inn2["cum_runs"] / inn2["ball_num"] * 6, 0)
inn2["rrr"]             = np.where(inn2["balls_remaining"] > 0, inn2["runs_needed"] / inn2["balls_remaining"] * 6, 99.0)
inn2["rrr_diff"]        = inn2["crr"] - inn2["rrr"]
inn2["run_rate_ratio"]  = inn2["crr"] / inn2["rrr"].replace(0, 0.01)
inn2["balls_pct"]       = inn2["ball_num"] / 120
inn2["wickets_pct"]     = inn2["cum_wickets"] / 10

snapshots = inn2[inn2["ball_num"] % 6 == 0].copy()
snapshots = snapshots.dropna(subset=["chasing_won", "target"])

LIVE_FEATURES = [
    "ball_num", "balls_remaining", "balls_pct",
    "cum_runs", "runs_needed",
    "cum_wickets", "wickets_left", "wickets_pct",
    "crr", "rrr", "rrr_diff", "run_rate_ratio",
]

snap_train = snapshots[~snapshots["season"].isin(TEST_SEASONS)]
snap_test  = snapshots[ snapshots["season"].isin(TEST_SEASONS)]
print(f"Train snapshots: {len(snap_train)} | Test snapshots: {len(snap_test)}")

X_lt = snap_train[LIVE_FEATURES].fillna(0)
y_lt = snap_train["chasing_won"].astype(int)
X_lv = snap_test[LIVE_FEATURES].fillna(0)
y_lv = snap_test["chasing_won"].astype(int)

scaler = StandardScaler()
X_lt_s = scaler.fit_transform(X_lt)
X_lv_s = scaler.transform(X_lv)

base       = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
live_model = CalibratedClassifierCV(base, cv=5, method="isotonic")
live_model.fit(X_lt_s, y_lt)

probs_l = live_model.predict_proba(X_lv_s)[:, 1]
preds_l = (probs_l >= 0.5).astype(int)
print(f"Live model accuracy : {accuracy_score(y_lv, preds_l):.3f}")
print(f"Live model log-loss : {log_loss(y_lv, probs_l):.4f}")
print(f"Live model brier    : {brier_score_loss(y_lv, probs_l):.4f}")

live_bundle = {"model": live_model, "scaler": scaler, "features": LIVE_FEATURES}
with open("models/live_model.pkl", "wb") as f:
    pickle.dump(live_bundle, f)
print("Saved models/live_model.pkl")
print("\n✓ All models trained and saved. Ready to start the API.")
