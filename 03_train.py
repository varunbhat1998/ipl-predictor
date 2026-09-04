"""
03_train.py  —  Train both models + backtest on 2023/2024/2025
Pre-match: XGBoost with Optuna tuning (single model, all features)
Live:      LightGBM with momentum features + GroupKFold calibration
Saves:  models/prematch_model.pkl, models/live_model.pkl, data/backtest_results.csv
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
Path("models").mkdir(exist_ok=True)

# =====================================================================
# MODEL A — Pre-match win predictor (Optuna-tuned XGBoost)
# =====================================================================
print("=" * 55)
print("MODEL A: Pre-match (Optuna-tuned XGBoost)")
print("=" * 55)

df = pd.read_csv("data/match_features.csv", parse_dates=["date"])
df = df[df["team1_won"].notna()].copy()
df["season"] = df["season"].astype(str)

ALL_FEATURES = [
    "elo_diff",
    "team1_form", "team2_form", "form_diff",
    "form_3_diff", "form_10_diff", "form_weighted_diff",
    "h2h_win_rate_team1",
    "team1_won_toss", "toss_chose_bat",
    "venue_toss_win_rate", "venue_bat_first_win_rate",
    "venue_avg_first_innings", "venue_chase_win_rate",
    "team1_venue_win_rate", "team2_venue_win_rate",
    "match_num_in_season", "is_playoff",
    "bat_diff", "bowl_diff",
    "team1_bat_strength", "team2_bat_strength",
    "team1_bowl_strength", "team2_bowl_strength",
    # Post-toss features
    "team1_bats_second", "toss_venue_aligned",
    "team1_chase_wr", "team2_chase_wr", "chase_wr_diff",
    "team1_chase_advantage", "team2_chase_advantage", "chase_advantage_diff",
    "early_season",
    "venue_chase_batting_second",
    # Month / seasonality features (early_chase_boost removed — confirmed noisy)
    "is_march",
    "venue_month_chase_wr", "venue_month_chase_batting_second",
]
ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]
print(f"Using {len(ALL_FEATURES)} features")

TEST_SEASONS = ["2023", "2024", "2025"]
train = df[~df["season"].isin(TEST_SEASONS)].copy()
test  = df[df["season"].isin(TEST_SEASONS)].copy()
print(f"Train: {len(train)} matches | Test: {len(test)} matches")

train_median = train[ALL_FEATURES].median()
X_train = train[ALL_FEATURES].fillna(train_median)
y_train = train["team1_won"].astype(int)
X_test = test[ALL_FEATURES].fillna(train_median)
y_test = test["team1_won"].astype(int)

# ── Sample weights ────────────────────────────────────────────────────────
# 2026: 5x | 2024-2025: 4x | 2022-2023: 2.5x | 2021: 1.5x | 2019-2020: 1x | pre-2019: 0.5x
sample_weight_train = np.ones(len(y_train))
for i, s in enumerate(train["season"].values):
    s_int = int(s)
    if s_int >= 2026:
        sample_weight_train[i] = 5.0
    elif s_int >= 2024:
        sample_weight_train[i] = 4.0
    elif s_int >= 2022:
        sample_weight_train[i] = 2.5
    elif s_int == 2021:
        sample_weight_train[i] = 1.5
    elif s_int >= 2019:
        sample_weight_train[i] = 1.0
    else:
        sample_weight_train[i] = 0.5
w_dist = {w: int((sample_weight_train == w).sum()) for w in sorted(set(sample_weight_train))}
print(f"Pre-match sample weights: {w_dist}  (effective: {sample_weight_train.sum():.0f})")

# ── Optuna tuning ────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=False)

print("Tuning XGBoost with Optuna (60 trials)...")
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 2.0),
    }
    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
    oof = cross_val_predict(model, X_train, y_train, cv=kf, method="predict_proba",
                            params={"sample_weight": sample_weight_train})[:, 1]
    return log_loss(y_train, oof, sample_weight=sample_weight_train)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=60)
print(f"  Best CV log-loss: {study.best_value:.4f}")
bp = study.best_params
print(f"  Best params: depth={bp['max_depth']}, lr={bp['learning_rate']:.4f}, "
      f"n_est={bp['n_estimators']}, gamma={bp['gamma']:.2f}")

# ── Train final model + isotonic calibration (OOF-based) ────────────────
from sklearn.isotonic import IsotonicRegression

print("\nTraining final model...")
model = xgb.XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, verbosity=0)
model.fit(X_train, y_train, sample_weight=sample_weight_train)

# Fit isotonic calibrator on OOF predictions from training data
print("Fitting isotonic calibrator on OOF predictions...")
oof_prematch = cross_val_predict(
    xgb.XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, verbosity=0),
    X_train, y_train, cv=kf, method="predict_proba",
    params={"sample_weight": sample_weight_train}
)[:, 1]
prematch_calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
prematch_calibrator.fit(oof_prematch, y_train, sample_weight=sample_weight_train)

# Backtest: raw vs calibrated
probs_raw = model.predict_proba(X_test)[:, 1]
probs_a = prematch_calibrator.predict(probs_raw)
probs_a = np.clip(probs_a, 0.05, 0.95)
preds_a = (probs_a >= 0.5).astype(int)

acc = accuracy_score(y_test, preds_a)
ll = log_loss(y_test, probs_a)
brier = brier_score_loss(y_test, probs_a)
print(f"\nBacktest accuracy  : {acc:.3f}  (calibrated)")
print(f"Backtest log-loss  : {ll:.4f}")
print(f"Backtest brier     : {brier:.4f}")

# Compare with raw
preds_raw = (probs_raw >= 0.5).astype(int)
print(f"Raw accuracy       : {accuracy_score(y_test, preds_raw):.3f}")
print(f"Raw brier          : {brier_score_loss(y_test, probs_raw):.4f}")

per_season = test[["season"]].copy()
per_season["correct"] = (preds_a == y_test.values).astype(int)
print("\nPer-season accuracy:")
print(per_season.groupby("season")["correct"].agg(["mean", "count"]).to_string())

# Also test individual base models for comparison
print("\n--- Individual model comparison ---")
for name, clf in [
    ("XGBoost (tuned)", xgb.XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, verbosity=0)),
    ("LightGBM (default)", lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=-1)),
    ("LogReg (baseline)", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
]:
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    print(f"  {name:25s}: acc={accuracy_score(y_test, (p>=0.5).astype(int)):.3f}  "
          f"ll={log_loss(y_test, p):.4f}  brier={brier_score_loss(y_test, p):.4f}")

# Feature importance
print("\nTop features:")
fi = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
print(fi.head(12).to_string())

# ── Save model bundle ────────────────────────────────────────────────────
prematch_bundle = {
    "model": model,
    "calibrator": prematch_calibrator,
    "features": ALL_FEATURES,
    "train_median": train_median.to_dict(),
}
with open("models/prematch_model.pkl", "wb") as f:
    pickle.dump(prematch_bundle, f)
print("\nSaved models/prematch_model.pkl")

# Save backtest
bt = test[["file_id", "season", "date", "team1", "team2", "team1_won"]].copy()
bt["prob_team1_wins"] = probs_a
bt["predicted_winner"] = np.where(preds_a == 1, test["team1"].values, test["team2"].values)
bt["correct"] = (preds_a == y_test.values).astype(int)
bt.to_csv("data/backtest_results.csv", index=False)
print(f"Saved data/backtest_results.csv")

# =====================================================================
# MODEL B — Live in-match win probability (LightGBM + Momentum)
# =====================================================================
print("\n" + "=" * 55)
print("MODEL B: Live in-match (LightGBM + Momentum Features)")
print("=" * 55)

deliveries = pd.read_csv("data/deliveries.csv")
deliveries["date"]    = pd.to_datetime(deliveries["date"])
deliveries["season"]  = deliveries["date"].dt.year.astype(str)
deliveries["file_id"] = deliveries["file_id"].astype(str)

# 2nd innings only
inn2 = deliveries[deliveries["innings"] == 2].copy()

# Target totals from 1st innings
inn1_totals = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")["runs_total"].sum()
    .rename("target")
    .reset_index()
)
inn1_totals["target"]  = inn1_totals["target"] + 1
inn1_totals["file_id"] = inn1_totals["file_id"].astype(str)
inn2 = inn2.merge(inn1_totals, on="file_id", how="left")

# First innings wickets
inn1_wkts = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")["is_wicket"].sum()
    .rename("first_innings_wickets")
    .reset_index()
)
inn1_wkts["file_id"] = inn1_wkts["file_id"].astype(str)
inn2 = inn2.merge(inn1_wkts, on="file_id", how="left")

# Match winner — drop deliveries' winner column to avoid merge conflict
inn2 = inn2.drop(columns=["winner"], errors="ignore")
matches_for_winner = pd.read_csv("data/matches.csv")[["file_id", "winner"]].dropna(subset=["winner"])
matches_for_winner["file_id"] = matches_for_winner["file_id"].astype(str)
inn2 = inn2.merge(matches_for_winner, on="file_id", how="left")
inn2 = inn2[inn2["winner"].notna()].copy()
inn2["chasing_won"] = (inn2["winner"] == inn2["batting_team"]).astype(int)

# Build match state ball-by-ball
inn2 = inn2.sort_values(["file_id", "over", "ball_in_over"]).copy()
inn2["ball_num"] = inn2.groupby("file_id").cumcount() + 1
inn2["balls_remaining"] = (120 - inn2["ball_num"]).clip(0)
inn2["runs_needed"]     = (inn2["target"] - inn2["cum_runs"]).clip(0)
inn2["wickets_left"]    = 10 - inn2["cum_wickets"]

inn2["crr"] = np.where(inn2["ball_num"] > 0, inn2["cum_runs"] / inn2["ball_num"] * 6, 0)
inn2["rrr"] = np.where(inn2["balls_remaining"] > 0,
                       inn2["runs_needed"] / inn2["balls_remaining"] * 6, 99.0)
inn2["rrr_diff"]        = inn2["crr"] - inn2["rrr"]
inn2["run_rate_ratio"]  = inn2["crr"] / inn2["rrr"].replace(0, 0.01)
inn2["balls_pct"]       = inn2["ball_num"] / 120
inn2["wickets_pct"]     = inn2["cum_wickets"] / 10

# First innings context
inn2["first_innings_run_rate"] = (inn2["target"] - 1) / 120 * 6

venue_avg = (
    deliveries[deliveries["innings"] == 1]
    .groupby(["file_id", "venue"])["runs_total"].sum().reset_index()
    .groupby("venue")["runs_total"].mean().rename("v_avg")
)
inn2 = inn2.merge(venue_avg, on="venue", how="left")
inn2["v_avg"] = inn2["v_avg"].fillna(160)
inn2["target_vs_venue_avg"] = inn2["target"] / inn2["v_avg"]

# ── Momentum features ────────────────────────────────────────────────────
print("Computing momentum features...")

def compute_partnership(group):
    cum_runs = group["cum_runs"].values
    is_wkt = group["is_wicket"].values
    ball_num = group["ball_num"].values
    pr, pb = [], []
    last_wkt_runs = 0
    last_wkt_ball = 0
    for i in range(len(group)):
        pr.append(cum_runs[i] - last_wkt_runs)
        pb.append(ball_num[i] - last_wkt_ball)
        if is_wkt[i]:
            last_wkt_runs = cum_runs[i]
            last_wkt_ball = ball_num[i]
    return pd.DataFrame({"partnership_runs": pr, "partnership_balls": pb}, index=group.index)

partnership = inn2.groupby("file_id", group_keys=False).apply(compute_partnership)
inn2["partnership_runs"] = partnership["partnership_runs"]
inn2["partnership_balls"] = partnership["partnership_balls"]

def rolling_18(group):
    runs = group["runs_total"].values
    wkts = group["is_wicket"].values
    r18, w18 = [], []
    for i in range(len(group)):
        start = max(0, i - 17)
        r18.append(runs[start:i+1].sum())
        w18.append(wkts[start:i+1].sum())
    return pd.DataFrame({"last_3ov_runs": r18, "last_3ov_wkts": w18}, index=group.index)

roll = inn2.groupby("file_id", group_keys=False).apply(rolling_18)
inn2["last_3ov_runs"] = roll["last_3ov_runs"]
inn2["last_3ov_wkts"] = roll["last_3ov_wkts"]

inn2["is_boundary"] = (inn2["runs_batter"] >= 4).astype(int)
inn2["is_dot"]      = (inn2["runs_batter"] == 0).astype(int)

def cumulative_pct(group, col):
    vals = group[col].values
    return pd.Series([vals[:i+1].mean() for i in range(len(group))], index=group.index)

inn2["boundary_pct"] = inn2.groupby("file_id", group_keys=False).apply(lambda g: cumulative_pct(g, "is_boundary"))
inn2["dot_ball_pct"] = inn2.groupby("file_id", group_keys=False).apply(lambda g: cumulative_pct(g, "is_dot"))

# ── Powerplay features ───────────────────────────────────────────────────
# Powerplay = overs 1-6 = ball_num 1-36
# pp_runs/pp_wickets = cumulative state at end of over 6 (ball_num == 36)
inn2_pp = (
    inn2[inn2["ball_num"] == 36]
    .groupby("file_id")[["cum_runs", "cum_wickets"]]
    .last()
    .rename(columns={"cum_runs": "pp_runs", "cum_wickets": "pp_wickets"})
    .reset_index()
)
inn2 = inn2.merge(inn2_pp, on="file_id", how="left")
# For balls inside the powerplay, pp_runs/pp_wickets don't exist yet — fill 0
inn2["pp_runs"]    = inn2["pp_runs"].fillna(0)
inn2["pp_wickets"] = inn2["pp_wickets"].fillna(0)

inn2["is_pp"]       = (inn2["ball_num"] <= 36).astype(int)
inn2["pp_run_rate"] = np.where(inn2["ball_num"] > 36, inn2["pp_runs"] / 36 * 6, 0.0)
inn2["pp_req_rate"] = np.where(
    inn2["ball_num"] > 36,
    (inn2["target"] - inn2["pp_runs"]) / 84 * 6,
    0.0
)
inn2["pp_rate_gap"] = inn2["pp_run_rate"] - inn2["pp_req_rate"]

# End-of-over snapshots
snapshots = inn2[inn2["ball_num"] % 6 == 0].copy()
snapshots = snapshots.dropna(subset=["chasing_won", "target"])

LIVE_FEATURES = [
    "ball_num", "balls_remaining", "balls_pct",
    "cum_runs", "runs_needed",
    "cum_wickets", "wickets_left", "wickets_pct",
    "crr", "rrr", "rrr_diff", "run_rate_ratio",
    "partnership_runs", "partnership_balls",
    "last_3ov_runs", "last_3ov_wkts",
    "boundary_pct", "dot_ball_pct",
    "first_innings_run_rate", "target_vs_venue_avg", "first_innings_wickets",
    # Powerplay phase features
    "is_pp", "pp_runs", "pp_wickets", "pp_run_rate", "pp_req_rate", "pp_rate_gap",
]

snapshots["season"] = snapshots["date"].dt.year.astype(str)
snap_train = snapshots[~snapshots["season"].isin(TEST_SEASONS)]
snap_test  = snapshots[snapshots["season"].isin(TEST_SEASONS)]

X_lt = snap_train[LIVE_FEATURES].fillna(0)
y_lt = snap_train["chasing_won"].astype(int)
X_lv = snap_test[LIVE_FEATURES].fillna(0)
y_lv = snap_test["chasing_won"].astype(int)

scaler = StandardScaler()
X_lt_s = scaler.fit_transform(X_lt)
X_lv_s = scaler.transform(X_lv)

print(f"Live train: {len(snap_train)} snapshots | Live test: {len(snap_test)} snapshots")

# LightGBM with GroupKFold calibration
print("Training live model (LightGBM + GroupKFold calibration)...")
base_live = lgb.LGBMClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=-1,
)

groups_train = snap_train["file_id"].values
gkf = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_lt_s, y_lt, groups=groups_train))

live_model = CalibratedClassifierCV(base_live, cv=cv_splits, method="isotonic")
live_model.fit(X_lt_s, y_lt)

probs_l = live_model.predict_proba(X_lv_s)[:, 1]
preds_l = (probs_l >= 0.5).astype(int)

print(f"\nLive model accuracy : {accuracy_score(y_lv, preds_l):.3f}")
print(f"Live model log-loss : {log_loss(y_lv, probs_l):.4f}")
print(f"Live model brier    : {brier_score_loss(y_lv, probs_l):.4f}")

live_bundle = {
    "model": live_model,
    "scaler": scaler,
    "features": LIVE_FEATURES,
}
with open("models/live_model.pkl", "wb") as f:
    pickle.dump(live_bundle, f)
print("Saved models/live_model.pkl")

# =====================================================================
# MODEL C — First innings live prediction (LightGBM)
# =====================================================================
print("\n" + "=" * 55)
print("MODEL C: First innings live (LightGBM)")
print("=" * 55)

from collections import defaultdict

# Venue averages (expanding, no leakage)
matches_sorted = pd.read_csv("data/matches.csv", parse_dates=["date"])
matches_sorted = matches_sorted.sort_values("date").reset_index(drop=True)
matches_sorted["file_id"] = matches_sorted["file_id"].astype(str)
matches_sorted["season"] = matches_sorted["season"].astype(str)

# Normalize venue
def norm_venue(v):
    if not isinstance(v, str): return v
    if "Chinnaswamy" in v: return "M Chinnaswamy Stadium, Bengaluru"
    if "Eden" in v: return "Eden Gardens, Kolkata"
    if "Wankhede" in v: return "Wankhede Stadium, Mumbai"
    if "Chepauk" in v or "Chidambaram" in v: return "MA Chidambaram Stadium, Chennai"
    if "Feroz" in v or "Arun Jaitley" in v or "Kotla" in v: return "Arun Jaitley Stadium, Delhi"
    if "Rajiv Gandhi" in v: return "Rajiv Gandhi Intl Stadium, Hyderabad"
    if "Sawai" in v: return "Sawai Mansingh Stadium, Jaipur"
    if "Mohali" in v or ("Punjab" in v and "Bindra" in v): return "PCA Stadium, Mohali"
    if "DY Patil" in v: return "DY Patil Stadium, Mumbai"
    if "Brabourne" in v: return "Brabourne Stadium, Mumbai"
    if "Narendra Modi" in v or "Motera" in v or "Sardar Patel" in v: return "Narendra Modi Stadium, Ahmedabad"
    if "Ekana" in v or "Atal Bihari" in v: return "Ekana Stadium, Lucknow"
    if "Maharashtra" in v and ("Pune" in v or "MCA" in v): return "MCA Stadium, Pune"
    if "Subrata" in v: return "Subrata Roy Sahara Stadium, Pune"
    if "Sharjah" in v: return "Sharjah Cricket Stadium"
    if "Dubai" in v: return "Dubai International Cricket Stadium"
    if "Sheikh Zayed" in v or "Abu Dhabi" in v: return "Sheikh Zayed Stadium, Abu Dhabi"
    return v

matches_sorted["venue"] = matches_sorted["venue"].apply(norm_venue)
deliveries["venue"] = deliveries["venue"].apply(norm_venue)

venue_inn1_expanding = defaultdict(list)
venue_avg_at_match = {}
for _, row in matches_sorted.iterrows():
    v = row["venue"]
    fid = row["file_id"]
    venue_avg_at_match[fid] = np.mean(venue_inn1_expanding[v]) if venue_inn1_expanding[v] else 165.0
    if pd.notna(row.get("inn1_runs")):
        venue_inn1_expanding[v].append(row["inn1_runs"])

# ELO at match time
def norm_team(t):
    if not isinstance(t, str): return t
    mapping = {"Delhi Daredevils":"Delhi Capitals","Deccan Chargers":"Sunrisers Hyderabad",
               "Punjab Kings":"Kings XI Punjab","Royal Challengers Bangalore":"Royal Challengers Bengaluru",
               "Rising Pune Supergiants":"Rising Pune Supergiant"}
    return mapping.get(t, t)

for c in ["team1","team2","winner","inn1_team","inn2_team"]:
    if c in matches_sorted.columns:
        matches_sorted[c] = matches_sorted[c].apply(norm_team)

team_elo_c = defaultdict(lambda: 1500)
team_form_c = defaultdict(list)
match_context = {}
for _, row in matches_sorted.iterrows():
    if pd.isna(row.get("winner")): continue
    t1, t2 = row["team1"], row["team2"]
    match_context[row["file_id"]] = {
        "team1": t1, "team2": t2,
        "elo1": team_elo_c[t1], "elo2": team_elo_c[t2],
        "form1": np.mean(team_form_c[t1][-5:]) if team_form_c[t1] else 0.5,
        "form2": np.mean(team_form_c[t2][-5:]) if team_form_c[t2] else 0.5,
        "inn1_team": row.get("inn1_team"),
    }
    won = 1 if row["winner"] == t1 else 0
    exp = 1 / (1 + 10 ** ((team_elo_c[t2] - team_elo_c[t1]) / 400))
    team_elo_c[t1] += 24 * (won - exp)
    team_elo_c[t2] += 24 * ((1 - won) - (1 - exp))
    team_form_c[t1].append(won)
    team_form_c[t2].append(1 - won)

# First innings deliveries
inn1 = deliveries[deliveries["innings"] == 1].copy()
inn1 = inn1.sort_values(["file_id", "over", "ball_in_over"])
inn1["file_id"] = inn1["file_id"].astype(str)
valid_fids = set(matches_sorted[matches_sorted["winner"].notna()]["file_id"])
inn1 = inn1[inn1["file_id"].isin(valid_fids)]
inn1["ball_num"] = inn1.groupby("file_id").cumcount() + 1

fid_to_winner = dict(zip(matches_sorted["file_id"], matches_sorted["winner"]))
fid_to_inn1_team = dict(zip(matches_sorted["file_id"], matches_sorted["inn1_team"]))

print("Building first innings snapshots...")
inn1_snaps = []
for fid, group in inn1.groupby("file_id"):
    if fid not in match_context: continue
    ctx = match_context[fid]
    venue_avg = venue_avg_at_match.get(fid, 165.0)
    bat_first_won = int(fid_to_inn1_team.get(fid) == fid_to_winner.get(fid))
    batting_team = fid_to_inn1_team.get(fid)
    season = matches_sorted[matches_sorted["file_id"]==fid]["season"].iloc[0] if fid in matches_sorted["file_id"].values else "2020"

    if batting_team == ctx["team1"]:
        bat_elo, bowl_elo = ctx["elo1"], ctx["elo2"]
        bat_form, bowl_form = ctx["form1"], ctx["form2"]
    else:
        bat_elo, bowl_elo = ctx["elo2"], ctx["elo1"]
        bat_form, bowl_form = ctx["form2"], ctx["form1"]

    group = group.sort_values(["over", "ball_in_over"])
    boundary_count, dot_count = 0, 0
    recent_runs, recent_wkts = [], []
    partnership_runs, partnership_balls = 0, 0
    pp_runs_at_6, pp_wkts_at_6 = 0, 0  # locked at end of over 6

    for _, ball in group.iterrows():
        recent_runs.append(ball["runs_total"])
        recent_wkts.append(int(ball["is_wicket"]))
        if len(recent_runs) > 18: recent_runs.pop(0); recent_wkts.pop(0)
        if ball["runs_batter"] in [4, 6]: boundary_count += 1
        if ball["runs_total"] == 0: dot_count += 1
        partnership_runs += ball["runs_total"]
        partnership_balls += 1
        if ball["is_wicket"]: partnership_runs = 0; partnership_balls = 0

        bn = ball["ball_num"]
        if bn % 6 == 0 and bn <= 120:
            over_num = bn // 6
            cr = ball["cum_runs"]
            cw = ball["cum_wickets"]
            crr = cr / bn * 6 if bn > 0 else 0
            expected_at = venue_avg * (bn / 120)
            projected = cr + (crr * (120 - bn) / 6) if bn < 120 else cr

            # Lock powerplay stats at end of over 6
            if over_num == 6:
                pp_runs_at_6, pp_wkts_at_6 = cr, cw

            pp_vs_venue = (pp_runs_at_6 / (venue_avg * 6 / 20)) if over_num > 6 and venue_avg > 0 else 0.0

            inn1_snaps.append({
                "file_id": fid, "season": season, "over": over_num,
                "cum_runs": cr, "cum_wickets": cw, "crr": crr,
                "balls_remaining": 120 - bn, "balls_pct": bn / 120,
                "wickets_pct": cw / 10,
                "projected_score": projected, "venue_avg": venue_avg,
                "score_vs_expected": cr - expected_at,
                "score_vs_expected_pct": cr / expected_at if expected_at > 0 else 1.0,
                "partnership_runs": partnership_runs,
                "partnership_balls": partnership_balls,
                "last_3ov_runs": sum(recent_runs), "last_3ov_wkts": sum(recent_wkts),
                "boundary_pct": boundary_count / bn if bn > 0 else 0,
                "dot_pct": dot_count / bn if bn > 0 else 0,
                "acceleration": (sum(recent_runs) / min(len(recent_runs), 18) * 6 - crr) if bn >= 36 else 0,
                "elo_diff": bat_elo - bowl_elo,
                "form_diff": bat_form - bowl_form,
                # Powerplay features
                "is_pp": int(over_num <= 6),
                "pp_runs": pp_runs_at_6 if over_num > 6 else 0,
                "pp_wickets": pp_wkts_at_6 if over_num > 6 else 0,
                "pp_vs_venue_avg": pp_vs_venue,
                "batting_first_won": bat_first_won,
            })

inn1_snap_df = pd.DataFrame(inn1_snaps)
print(f"First innings snapshots: {len(inn1_snap_df)} ({inn1_snap_df['file_id'].nunique()} matches)")

INN1_FEATURES = [
    "cum_runs", "cum_wickets", "crr", "balls_remaining", "balls_pct", "wickets_pct",
    "projected_score", "venue_avg", "score_vs_expected", "score_vs_expected_pct",
    "partnership_runs", "partnership_balls",
    "last_3ov_runs", "last_3ov_wkts", "boundary_pct", "dot_pct", "acceleration",
    "elo_diff", "form_diff",
    # Powerplay features
    "is_pp", "pp_runs", "pp_wickets", "pp_vs_venue_avg",
]

i1_train = inn1_snap_df[~inn1_snap_df["season"].isin(TEST_SEASONS)]
i1_test = inn1_snap_df[inn1_snap_df["season"].isin(TEST_SEASONS)]

X_i1_tr = i1_train[INN1_FEATURES].fillna(0)
y_i1_tr = i1_train["batting_first_won"].astype(int)
X_i1_te = i1_test[INN1_FEATURES].fillna(0)
y_i1_te = i1_test["batting_first_won"].astype(int)

scaler_inn1 = StandardScaler()
X_i1_tr_s = scaler_inn1.fit_transform(X_i1_tr)
X_i1_te_s = scaler_inn1.transform(X_i1_te)

print("Training first innings model (LightGBM)...")
inn1_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.03,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, random_state=42, verbosity=-1,
)
inn1_model.fit(X_i1_tr_s, y_i1_tr)

probs_i1 = inn1_model.predict_proba(X_i1_te_s)[:, 1]
preds_i1 = (probs_i1 >= 0.5).astype(int)

print(f"\n1st innings accuracy : {accuracy_score(y_i1_te, preds_i1):.3f}")
print(f"1st innings log-loss : {log_loss(y_i1_te, probs_i1):.4f}")
print(f"1st innings brier    : {brier_score_loss(y_i1_te, probs_i1):.4f}")

# Per-over accuracy
i1_test_copy = i1_test.copy()
i1_test_copy["prob"] = probs_i1
i1_test_copy["pred"] = preds_i1
i1_test_copy["correct"] = (preds_i1 == y_i1_te.values).astype(int)
print("\nPer-over accuracy (1st innings):")
for ov in [3, 6, 10, 15, 20]:
    sub = i1_test_copy[i1_test_copy["over"] == ov]
    if len(sub) > 0:
        print(f"  Over {ov:>2}: {sub['correct'].mean()*100:.1f}% ({len(sub)} matches)")

inn1_bundle = {
    "model": inn1_model,
    "scaler": scaler_inn1,
    "features": INN1_FEATURES,
}
with open("models/inn1_live_model.pkl", "wb") as f:
    pickle.dump(inn1_bundle, f)
print("Saved models/inn1_live_model.pkl")

print("\nAll models trained and saved.")
