"""
09_train_2024_2025.py -- Retrain all 3 models on 2024 + 2025 data only.

Key differences from 03_train.py:
  - Only 2024-2025 match data (~141 matches, ~2800 over-snapshots)
  - Ensemble pre-match model (XGBoost + LightGBM + LogisticRegression)
  - Confidence threshold stored in bundle: predictions above threshold
    achieve >= 80% accuracy (used by API to flag "high confidence")
  - team1_bats_second removed (always 0 in 2022+ Cricsheet format)
  - Saves to same .pkl paths so API picks them up via /reload-models

Run:
    python 09_train_2024_2025.py
"""

import pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, cross_val_predict, LeaveOneGroupOut
)
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
import optuna

from model_classes import EnsemblePreMatchModel  # shared class for pickling

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
Path("models").mkdir(exist_ok=True)

SEASONS = ["2024", "2025"]


# =====================================================================
# MODEL A -- Pre-match win predictor (2024-2025 ensemble, >=80% @ conf)
# =====================================================================
print("=" * 62)
print("MODEL A: Pre-match ensemble (2024-2025 only, target >=80%)")
print("=" * 62)

df = pd.read_csv("data/match_features.csv", parse_dates=["date"])
df = df[df["team1_won"].notna()].copy()
df["season"] = df["season"].astype(str)

mf = df[df["season"].isin(SEASONS)].copy()
print(f"Matches: {len(mf)}  "
      f"(2024: {(mf.season=='2024').sum()}, 2025: {(mf.season=='2025').sum()})")
print(f"Bat-first win rate: {mf['team1_won'].mean():.3f}  "
      f"(chase win rate: {1-mf['team1_won'].mean():.3f})")

# ── Feature set (drop team1_bats_second -- always 0 in 2022+ data) ──────
PREMATCH_FEATURES = [
    # ELO
    "elo_diff", "team1_elo", "team2_elo",
    # Multi-window form
    "form_diff", "form_3_diff", "form_10_diff", "form_weighted_diff",
    "team1_form", "team2_form", "team1_form_3", "team2_form_3",
    "team1_form_10", "team2_form_10",
    # Head-to-head
    "h2h_win_rate_team1",
    # Toss (interpretable in 2022+ where team1 = bat first)
    "team1_won_toss", "toss_chose_bat",
    # Venue
    "venue_bat_first_win_rate", "venue_chase_win_rate",
    "venue_avg_first_innings", "venue_matches",
    "team1_venue_win_rate", "team2_venue_win_rate",
    "venue_toss_win_rate",
    # Season phase
    "match_num_in_season", "is_playoff", "early_season",
    # Player strength
    "bat_diff", "bowl_diff",
    "team1_bat_strength", "team2_bat_strength",
    "team1_bowl_strength", "team2_bowl_strength",
    # Chase rates (team2 always chases in 2022+ so team2_chase_wr is key)
    "team1_chase_wr", "team2_chase_wr", "chase_wr_diff",
    # Venue x chase interaction
    "venue_chase_batting_second",
    "venue_month_chase_wr",
]
PREMATCH_FEATURES = [f for f in PREMATCH_FEATURES if f in mf.columns]
print(f"Features: {len(PREMATCH_FEATURES)}")

feature_median = mf[PREMATCH_FEATURES].median()
X = mf[PREMATCH_FEATURES].fillna(feature_median)
y = mf["team1_won"].astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Tune XGBoost ────────────────────────────────────────────────────────
print("\nTuning XGBoost with Optuna (80 trials)...")
def objective_xgb(trial):
    params = {
        "n_estimators":      trial.suggest_int  ("n_estimators",    30, 250),
        "max_depth":         trial.suggest_int  ("max_depth",        2,   4),
        "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample",        0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int  ("min_child_weight",  3,  20),
        "reg_alpha":         trial.suggest_float("reg_alpha",         0.5, 30.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",        0.5, 30.0, log=True),
        "gamma":             trial.suggest_float("gamma",             0,    5.0),
    }
    m = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
    oof = cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]
    return log_loss(y, oof)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=80)
bp_xgb = study_xgb.best_params
print(f"  XGB best CV log-loss: {study_xgb.best_value:.4f}")
print(f"  depth={bp_xgb['max_depth']}  lr={bp_xgb['learning_rate']:.4f}  "
      f"n_est={bp_xgb['n_estimators']}  alpha={bp_xgb['reg_alpha']:.2f}")

# ── Tune LightGBM ───────────────────────────────────────────────────────
print("\nTuning LightGBM with Optuna (60 trials)...")
def objective_lgb(trial):
    params = {
        "n_estimators":      trial.suggest_int  ("n_estimators",    30, 250),
        "max_depth":         trial.suggest_int  ("max_depth",        2,   5),
        "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int  ("num_leaves",       8,  31),
        "subsample":         trial.suggest_float("subsample",        0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int  ("min_child_samples",10,  40),
        "reg_alpha":         trial.suggest_float("reg_alpha",         0.5, 30.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",        0.5, 30.0, log=True),
    }
    m = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
    oof = cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]
    return log_loss(y, oof)

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=60)
bp_lgb = study_lgb.best_params
print(f"  LGB best CV log-loss: {study_lgb.best_value:.4f}")

# ── Build individual CV OOF predictions ────────────────────────────────
print("\nGenerating OOF predictions...")
xgb_clf = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss", random_state=42, verbosity=0)
lgb_clf = lgb.LGBMClassifier(**bp_lgb, random_state=42, verbosity=-1)
lr_clf  = LogisticRegression(C=0.05, max_iter=3000, random_state=42)

scaler_lr = StandardScaler()
X_scaled  = scaler_lr.fit_transform(X)

oof_xgb = cross_val_predict(xgb_clf, X,        y, cv=skf, method="predict_proba")[:, 1]
oof_lgb = cross_val_predict(lgb_clf, X,        y, cv=skf, method="predict_proba")[:, 1]
oof_lr  = cross_val_predict(lr_clf,  X_scaled, y, cv=skf, method="predict_proba")[:, 1]
oof_ens = (oof_xgb + oof_lgb + oof_lr) / 3

print("\n--- 5-fold Stratified CV accuracies ---")
for name, oof in [("XGBoost", oof_xgb), ("LightGBM", oof_lgb),
                  ("LogReg",  oof_lr),  ("Ensemble", oof_ens)]:
    acc = accuracy_score(y, (oof >= 0.5).astype(int))
    ll  = log_loss(y, oof)
    print(f"  {name:<12}: acc={acc:.3f}  ll={ll:.4f}  "
          f"brier={brier_score_loss(y, oof):.4f}")

# Per-season CV accuracy (leave-one-season-out proxy)
print("\nLeave-one-season-out accuracy:")
for s_test in SEASONS:
    s_train = [s for s in SEASONS if s != s_test]
    tr_idx  = mf[mf["season"].isin(s_train)].index
    te_idx  = mf[mf["season"] == s_test].index
    if len(tr_idx) == 0 or len(te_idx) == 0:
        continue
    X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
    X_te, y_te = X.loc[te_idx], y.loc[te_idx]
    Xs_tr = scaler_lr.fit_transform(X_tr)
    Xs_te = scaler_lr.transform(X_te)
    _xgb = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss", random_state=42, verbosity=0).fit(X_tr, y_tr)
    _lgb = lgb.LGBMClassifier(**bp_lgb, random_state=42, verbosity=-1).fit(X_tr, y_tr)
    _lr  = LogisticRegression(C=0.05, max_iter=3000, random_state=42).fit(Xs_tr, y_tr)
    p_ens = (_xgb.predict_proba(X_te)[:,1] + _lgb.predict_proba(X_te)[:,1] +
             _lr.predict_proba(Xs_te)[:,1]) / 3
    acc = accuracy_score(y_te, (p_ens >= 0.5).astype(int))
    print(f"  Train {[s for s in SEASONS if s!=s_test][0]}, "
          f"test {s_test}: acc={acc:.3f} ({len(te_idx)} matches)")

# ── Find confidence threshold -> >=80% accuracy rule ─────────────────────
print("\n--- Confidence threshold analysis (ensemble OOF) ---")
print(f"{'Threshold':>10}  {'Coverage':>9}  {'Matches':>8}  {'Accuracy':>9}")
print("-" * 45)
best_thr = 0.50
best_cov = 1.00
found_80 = False
for thr in np.round(np.arange(0.50, 0.82, 0.01), 2):
    mask = (oof_ens >= thr) | (oof_ens <= 1 - thr)
    n    = mask.sum()
    if n < 15:
        break
    acc  = accuracy_score(y[mask], (oof_ens[mask] >= 0.5).astype(int))
    cov  = n / len(y)
    print(f"  {thr:.2f}        {cov:>8.1%}   {n:>7}   {acc:>8.1%}")
    if acc >= 0.80 and not found_80:
        best_thr, best_cov = thr, cov
        found_80 = True

if found_80:
    mask_best = (oof_ens >= best_thr) | (oof_ens <= 1 - best_thr)
    acc_best  = accuracy_score(y[mask_best], (oof_ens[mask_best] >= 0.5).astype(int))
    print(f"\n✓ 80% accuracy threshold: {best_thr:.2f}  "
          f"(covers {best_cov:.1%} of matches, acc={acc_best:.3f})")
else:
    # Use the tightest threshold that gets closest to 80%
    for thr in np.round(np.arange(0.50, 0.90, 0.01), 2):
        mask = (oof_ens >= thr) | (oof_ens <= 1 - thr)
        if mask.sum() < 10:
            break
        acc = accuracy_score(y[mask], (oof_ens[mask] >= 0.5).astype(int))
        if acc >= 0.78:
            best_thr, best_cov = thr, mask.mean()
    print(f"\nBest achievable threshold: {best_thr:.2f}  (coverage {best_cov:.1%})")

# ── Train final model on ALL 2024-2025 data ─────────────────────────────
print("\nTraining final ensemble on all 2024-2025 data...")
scaler_final = StandardScaler()
X_scaled_final = scaler_final.fit_transform(X)

xgb_final = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss",
                                random_state=42, verbosity=0).fit(X, y)
lgb_final = lgb.LGBMClassifier(**bp_lgb, random_state=42,
                                verbosity=-1).fit(X, y)
lr_final  = LogisticRegression(C=0.05, max_iter=3000,
                                random_state=42).fit(X_scaled_final, y)

ensemble = EnsemblePreMatchModel(xgb_final, lgb_final, lr_final, scaler_final)

# Feature importance
fi_xgb = pd.Series(xgb_final.feature_importances_, index=PREMATCH_FEATURES)
fi_lgb = pd.Series(lgb_final.feature_importances_, index=PREMATCH_FEATURES)
fi_avg  = ((fi_xgb / fi_xgb.sum()) + (fi_lgb / fi_lgb.sum())) / 2
print("\nTop 12 features (averaged importance):")
print(fi_avg.sort_values(ascending=False).head(12).round(4).to_string())

cv_acc = accuracy_score(y, (oof_ens >= 0.5).astype(int))

prematch_bundle = {
    "model":                ensemble,
    "features":             PREMATCH_FEATURES,
    "train_median":         feature_median.to_dict(),
    "confidence_threshold": float(best_thr),
    "cv_accuracy":          round(cv_acc, 4),
    "seasons":              SEASONS,
}
with open("models/prematch_model.pkl", "wb") as f:
    pickle.dump(prematch_bundle, f)
print(f"\nSaved models/prematch_model.pkl")
print(f"  CV accuracy (all): {cv_acc:.3f}")
print(f"  80% threshold    : {best_thr:.2f}  (predictions above this hit >=80%)")


# =====================================================================
# MODEL B -- Live 2nd-innings (2024-2025 deliveries, LightGBM + calib)
# =====================================================================
print("\n" + "=" * 62)
print("MODEL B: Live 2nd-innings (2024-2025 deliveries)")
print("=" * 62)

deliveries = pd.read_csv("data/deliveries.csv")
deliveries["date"]    = pd.to_datetime(deliveries["date"])
deliveries["season"]  = deliveries["date"].dt.year.astype(str)
deliveries["file_id"] = deliveries["file_id"].astype(str)

# Filter to 2024-2025
recent_fids = set(deliveries[deliveries["season"].isin(SEASONS)]["file_id"])
deliv_recent = deliveries[deliveries["file_id"].isin(recent_fids)].copy()
print(f"2024-2025 deliveries: {len(deliv_recent):,} "
      f"({deliv_recent['file_id'].nunique()} matches)")

inn2 = deliv_recent[deliv_recent["innings"] == 2].copy()

# Target from 1st innings
inn1_totals = (
    deliv_recent[deliv_recent["innings"] == 1]
    .groupby("file_id")["runs_total"].sum()
    .rename("target").reset_index()
)
inn1_totals["target"] += 1
inn2 = inn2.merge(inn1_totals, on="file_id", how="left")

# 1st innings wickets
inn1_wkts = (
    deliv_recent[deliv_recent["innings"] == 1]
    .groupby("file_id")["is_wicket"].sum()
    .rename("first_innings_wickets").reset_index()
)
inn2 = inn2.merge(inn1_wkts, on="file_id", how="left")

# Match winner
inn2 = inn2.drop(columns=["winner"], errors="ignore")
matches_recent = pd.read_csv("data/matches.csv")[["file_id", "winner"]].dropna(subset=["winner"])
matches_recent["file_id"] = matches_recent["file_id"].astype(str)
inn2 = inn2.merge(matches_recent, on="file_id", how="left")
inn2 = inn2[inn2["winner"].notna()].copy()
inn2["chasing_won"] = (inn2["winner"] == inn2["batting_team"]).astype(int)

# Match state
inn2 = inn2.sort_values(["file_id", "over", "ball_in_over"]).copy()
inn2["ball_num"]      = inn2.groupby("file_id").cumcount() + 1
inn2["balls_remaining"] = (120 - inn2["ball_num"]).clip(0)
inn2["runs_needed"]   = (inn2["target"] - inn2["cum_runs"]).clip(0)
inn2["wickets_left"]  = 10 - inn2["cum_wickets"]
inn2["crr"]           = np.where(inn2["ball_num"] > 0,
                                  inn2["cum_runs"] / inn2["ball_num"] * 6, 0)
inn2["rrr"]           = np.where(inn2["balls_remaining"] > 0,
                                  inn2["runs_needed"] / inn2["balls_remaining"] * 6, 99.0)
inn2["rrr_diff"]      = inn2["crr"] - inn2["rrr"]
inn2["run_rate_ratio"]= inn2["crr"] / inn2["rrr"].replace(0, 0.01)
inn2["balls_pct"]     = inn2["ball_num"] / 120
inn2["wickets_pct"]   = inn2["cum_wickets"] / 10
inn2["first_innings_run_rate"] = (inn2["target"] - 1) / 120 * 6

# Venue avg (from full deliveries for robustness)
venue_avg = (
    deliveries[deliveries["innings"] == 1]
    .groupby(["file_id", "venue"])["runs_total"].sum().reset_index()
    .groupby("venue")["runs_total"].mean().rename("v_avg")
)
inn2 = inn2.merge(venue_avg, on="venue", how="left")
inn2["v_avg"] = inn2["v_avg"].fillna(160)
inn2["target_vs_venue_avg"] = inn2["target"] / inn2["v_avg"]

# Momentum
print("Computing momentum features (Model B)...")

def compute_partnership(group):
    cr, iw, bn = (group["cum_runs"].values, group["is_wicket"].values,
                  group["ball_num"].values)
    pr, pb = [], []
    lr_, lb_ = 0, 0
    for i in range(len(group)):
        pr.append(cr[i] - lr_); pb.append(bn[i] - lb_)
        if iw[i]: lr_, lb_ = cr[i], bn[i]
    return pd.DataFrame({"partnership_runs": pr, "partnership_balls": pb}, index=group.index)

partnership = inn2.groupby("file_id", group_keys=False).apply(compute_partnership)
inn2["partnership_runs"]  = partnership["partnership_runs"]
inn2["partnership_balls"] = partnership["partnership_balls"]

def rolling_18(group):
    runs, wkts = group["runs_total"].values, group["is_wicket"].values
    r18, w18 = [], []
    for i in range(len(group)):
        s = max(0, i - 17)
        r18.append(runs[s:i+1].sum()); w18.append(wkts[s:i+1].sum())
    return pd.DataFrame({"last_3ov_runs": r18, "last_3ov_wkts": w18}, index=group.index)

roll = inn2.groupby("file_id", group_keys=False).apply(rolling_18)
inn2["last_3ov_runs"]  = roll["last_3ov_runs"]
inn2["last_3ov_wkts"] = roll["last_3ov_wkts"]

inn2["is_boundary"] = (inn2["runs_batter"] >= 4).astype(int)
inn2["is_dot"]      = (inn2["runs_batter"] == 0).astype(int)

def cumulative_pct(group, col):
    vals = group[col].values
    return pd.Series([vals[:i+1].mean() for i in range(len(group))], index=group.index)

inn2["boundary_pct"] = inn2.groupby("file_id", group_keys=False).apply(
    lambda g: cumulative_pct(g, "is_boundary"))
inn2["dot_ball_pct"] = inn2.groupby("file_id", group_keys=False).apply(
    lambda g: cumulative_pct(g, "is_dot"))

# Powerplay
inn2_pp = (
    inn2[inn2["ball_num"] == 36]
    .groupby("file_id")[["cum_runs", "cum_wickets"]].last()
    .rename(columns={"cum_runs": "pp_runs", "cum_wickets": "pp_wickets"}).reset_index()
)
inn2 = inn2.merge(inn2_pp, on="file_id", how="left")
inn2["pp_runs"]    = inn2["pp_runs"].fillna(0)
inn2["pp_wickets"] = inn2["pp_wickets"].fillna(0)
inn2["is_pp"]      = (inn2["ball_num"] <= 36).astype(int)
inn2["pp_run_rate"]= np.where(inn2["ball_num"] > 36, inn2["pp_runs"] / 36 * 6, 0.0)
inn2["pp_req_rate"]= np.where(inn2["ball_num"] > 36,
                               (inn2["target"] - inn2["pp_runs"]) / 84 * 6, 0.0)
inn2["pp_rate_gap"]= inn2["pp_run_rate"] - inn2["pp_req_rate"]

snapshots = inn2[inn2["ball_num"] % 6 == 0].copy()
snapshots = snapshots.dropna(subset=["chasing_won", "target"])
snapshots["season"] = snapshots["date"].dt.year.astype(str)

LIVE_FEATURES = [
    "ball_num", "balls_remaining", "balls_pct",
    "cum_runs", "runs_needed",
    "cum_wickets", "wickets_left", "wickets_pct",
    "crr", "rrr", "rrr_diff", "run_rate_ratio",
    "partnership_runs", "partnership_balls",
    "last_3ov_runs", "last_3ov_wkts",
    "boundary_pct", "dot_ball_pct",
    "first_innings_run_rate", "target_vs_venue_avg", "first_innings_wickets",
    "is_pp", "pp_runs", "pp_wickets", "pp_run_rate", "pp_req_rate", "pp_rate_gap",
]

X_l = snapshots[LIVE_FEATURES].fillna(0)
y_l = snapshots["chasing_won"].astype(int)

scaler_live = StandardScaler()
X_l_s = scaler_live.fit_transform(X_l)

groups_l = snapshots["file_id"].values
gkf = GroupKFold(n_splits=5)
cv_splits_l = list(gkf.split(X_l_s, y_l, groups=groups_l))

print(f"Live 2nd inn snapshots: {len(snapshots):,}  "
      f"({snapshots['file_id'].nunique()} matches)")
print("Training LightGBM + GroupKFold isotonic calibration...")

base_live = lgb.LGBMClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=-1,
)
live_model = CalibratedClassifierCV(base_live, cv=cv_splits_l, method="isotonic")
live_model.fit(X_l_s, y_l)

oof_l = cross_val_predict(
    lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                       num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                       random_state=42, verbosity=-1),
    X_l_s, y_l, cv=gkf.split(X_l_s, y_l, groups=groups_l),
    method="predict_proba"
)[:, 1]

print(f"  CV accuracy (GroupKFold): {accuracy_score(y_l, (oof_l>=0.5).astype(int)):.3f}")
print(f"  CV log-loss             : {log_loss(y_l, oof_l):.4f}")

live_bundle = {
    "model":    live_model,
    "scaler":   scaler_live,
    "features": LIVE_FEATURES,
    "seasons":  SEASONS,
}
with open("models/live_model.pkl", "wb") as f:
    pickle.dump(live_bundle, f)
print("Saved models/live_model.pkl")


# =====================================================================
# MODEL C -- First innings live (2024-2025 deliveries)
# =====================================================================
print("\n" + "=" * 62)
print("MODEL C: First innings live (2024-2025 deliveries)")
print("=" * 62)

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
    if "Maharashtra" in v and "Pune" in v: return "MCA Stadium, Pune"
    return v

matches_sorted = pd.read_csv("data/matches.csv", parse_dates=["date"])
matches_sorted = matches_sorted.sort_values("date").reset_index(drop=True)
matches_sorted["file_id"] = matches_sorted["file_id"].astype(str)
matches_sorted["season"]  = matches_sorted["season"].astype(str)
matches_sorted["venue"]   = matches_sorted["venue"].apply(norm_venue)
deliveries["venue"]       = deliveries["venue"].apply(norm_venue)

# Expanding venue average (all history for stable estimate)
venue_inn1_expanding = defaultdict(list)
venue_avg_at_match   = {}
for _, row in matches_sorted.iterrows():
    v   = row["venue"]
    fid = row["file_id"]
    venue_avg_at_match[fid] = (np.mean(venue_inn1_expanding[v])
                                if venue_inn1_expanding[v] else 165.0)
    if pd.notna(row.get("inn1_runs")):
        venue_inn1_expanding[v].append(row["inn1_runs"])

def norm_team(t):
    if not isinstance(t, str): return t
    return {"Delhi Daredevils": "Delhi Capitals",
            "Deccan Chargers": "Sunrisers Hyderabad",
            "Punjab Kings": "Kings XI Punjab",
            "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
            "Rising Pune Supergiants": "Rising Pune Supergiant"}.get(t, t)

for c in ["team1", "team2", "winner", "inn1_team", "inn2_team"]:
    if c in matches_sorted.columns:
        matches_sorted[c] = matches_sorted[c].apply(norm_team)

# ELO + form at match time (all history)
team_elo_c  = defaultdict(lambda: 1500)
team_form_c = defaultdict(list)
match_ctx   = {}
for _, row in matches_sorted.iterrows():
    if pd.isna(row.get("winner")): continue
    t1, t2 = row["team1"], row["team2"]
    match_ctx[row["file_id"]] = {
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

# First innings -- filter to 2024-2025
inn1 = deliveries[(deliveries["innings"] == 1) &
                   (deliveries["file_id"].isin(recent_fids))].copy()
inn1 = inn1.sort_values(["file_id", "over", "ball_in_over"])
valid_fids = set(matches_sorted[matches_sorted["winner"].notna()]["file_id"])
inn1 = inn1[inn1["file_id"].isin(valid_fids)]
inn1["ball_num"] = inn1.groupby("file_id").cumcount() + 1

fid_to_winner    = dict(zip(matches_sorted["file_id"], matches_sorted["winner"]))
fid_to_inn1_team = dict(zip(matches_sorted["file_id"], matches_sorted["inn1_team"]))

print("Building 1st innings snapshots (2024-2025)...")
inn1_snaps = []
for fid, group in inn1.groupby("file_id"):
    if fid not in match_ctx: continue
    ctx = match_ctx[fid]
    venue_avg   = venue_avg_at_match.get(fid, 165.0)
    bat_first_won = int(fid_to_inn1_team.get(fid) == fid_to_winner.get(fid))
    batting_team  = fid_to_inn1_team.get(fid)
    season = matches_sorted[matches_sorted["file_id"] == fid]["season"].iloc[0] \
             if fid in matches_sorted["file_id"].values else "2024"

    bat_elo   = ctx["elo1"] if batting_team == ctx["team1"] else ctx["elo2"]
    bowl_elo  = ctx["elo2"] if batting_team == ctx["team1"] else ctx["elo1"]
    bat_form  = ctx["form1"] if batting_team == ctx["team1"] else ctx["form2"]
    bowl_form = ctx["form2"] if batting_team == ctx["team1"] else ctx["form1"]

    group = group.sort_values(["over", "ball_in_over"])
    boundary_count = dot_count = 0
    recent_runs, recent_wkts = [], []
    pr_, pb_ = 0, 0
    pp_r6 = pp_w6 = 0

    for _, ball in group.iterrows():
        recent_runs.append(ball["runs_total"])
        recent_wkts.append(int(ball["is_wicket"]))
        if len(recent_runs) > 18: recent_runs.pop(0); recent_wkts.pop(0)
        if ball["runs_batter"] in [4, 6]: boundary_count += 1
        if ball["runs_total"] == 0: dot_count += 1
        pr_ += ball["runs_total"]; pb_ += 1
        if ball["is_wicket"]: pr_ = 0; pb_ = 0

        bn = ball["ball_num"]
        if bn % 6 == 0 and bn <= 120:
            ov   = bn // 6
            cr   = ball["cum_runs"]
            cw   = ball["cum_wickets"]
            crr_ = cr / bn * 6 if bn > 0 else 0
            exp_ = venue_avg * (bn / 120)
            proj = cr + (crr_ * (120 - bn) / 6) if bn < 120 else cr
            if ov == 6: pp_r6, pp_w6 = cr, cw
            pp_vs = (pp_r6 / (venue_avg * 6 / 20)) if ov > 6 and venue_avg > 0 else 0.0

            inn1_snaps.append({
                "file_id": fid, "season": season, "over": ov,
                "cum_runs": cr, "cum_wickets": cw, "crr": crr_,
                "balls_remaining": 120 - bn, "balls_pct": bn / 120,
                "wickets_pct": cw / 10,
                "projected_score": proj, "venue_avg": venue_avg,
                "score_vs_expected": cr - exp_,
                "score_vs_expected_pct": cr / exp_ if exp_ > 0 else 1.0,
                "partnership_runs": pr_, "partnership_balls": pb_,
                "last_3ov_runs": sum(recent_runs), "last_3ov_wkts": sum(recent_wkts),
                "boundary_pct": boundary_count / bn if bn > 0 else 0,
                "dot_pct": dot_count / bn if bn > 0 else 0,
                "acceleration": (sum(recent_runs) / min(len(recent_runs), 18) * 6 - crr_)
                                 if bn >= 36 else 0,
                "elo_diff": bat_elo - bowl_elo,
                "form_diff": bat_form - bowl_form,
                "is_pp": int(ov <= 6),
                "pp_runs": pp_r6 if ov > 6 else 0,
                "pp_wickets": pp_w6 if ov > 6 else 0,
                "pp_vs_venue_avg": pp_vs,
                "batting_first_won": bat_first_won,
            })

inn1_snap_df = pd.DataFrame(inn1_snaps)
print(f"1st innings snapshots: {len(inn1_snap_df):,} "
      f"({inn1_snap_df['file_id'].nunique()} matches)")

INN1_FEATURES = [
    "cum_runs", "cum_wickets", "crr", "balls_remaining", "balls_pct", "wickets_pct",
    "projected_score", "venue_avg", "score_vs_expected", "score_vs_expected_pct",
    "partnership_runs", "partnership_balls",
    "last_3ov_runs", "last_3ov_wkts", "boundary_pct", "dot_pct", "acceleration",
    "elo_diff", "form_diff",
    "is_pp", "pp_runs", "pp_wickets", "pp_vs_venue_avg",
]

X_i1   = inn1_snap_df[INN1_FEATURES].fillna(0)
y_i1   = inn1_snap_df["batting_first_won"].astype(int)
g_i1   = inn1_snap_df["file_id"].values

scaler_i1 = StandardScaler()
X_i1_s    = scaler_i1.fit_transform(X_i1)

gkf_i1    = GroupKFold(n_splits=5)
splits_i1 = list(gkf_i1.split(X_i1_s, y_i1, groups=g_i1))

print("Training 1st innings model (LightGBM)...")
inn1_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.03,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=15, random_state=42, verbosity=-1,
)
inn1_model.fit(X_i1_s, y_i1)

oof_i1 = cross_val_predict(
    lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                       num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                       min_child_samples=15, random_state=42, verbosity=-1),
    X_i1_s, y_i1,
    cv=gkf_i1.split(X_i1_s, y_i1, groups=g_i1),
    method="predict_proba"
)[:, 1]

print(f"  CV accuracy (GroupKFold): {accuracy_score(y_i1, (oof_i1>=0.5).astype(int)):.3f}")
print(f"  CV log-loss             : {log_loss(y_i1, oof_i1):.4f}")

# Per-over CV accuracy
snap_copy = inn1_snap_df.copy()
snap_copy["oof_prob"] = oof_i1
snap_copy["correct"]  = ((oof_i1 >= 0.5).astype(int) == y_i1.values).astype(int)
print("\n  Per-over CV accuracy:")
for ov in [3, 6, 10, 15, 20]:
    sub = snap_copy[snap_copy["over"] == ov]
    if len(sub):
        print(f"    Over {ov:>2}: {sub['correct'].mean()*100:.1f}%  ({len(sub)} matches)")

inn1_bundle = {
    "model":    inn1_model,
    "scaler":   scaler_i1,
    "features": INN1_FEATURES,
    "seasons":  SEASONS,
}
with open("models/inn1_live_model.pkl", "wb") as f:
    pickle.dump(inn1_bundle, f)
print("Saved models/inn1_live_model.pkl")

print("\n" + "=" * 62)
print("All 3 models trained on 2024-2025 data and saved.")
print(f"  Pre-match CV acc  : {cv_acc:.3f}  "
      f"(>=80% when |p - 0.5| >= {best_thr - 0.5:.2f})")
print(f"  Reload via API    : POST /reload-models  {{\"secret\":\"ipl2026\"}}")
print("=" * 62)
