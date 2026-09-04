"""
backtest_posttoss.py — Post-toss model backtest for seasons 2021-2025
Rebuilds the full post-toss feature pipeline (same as 10_post_toss_model.py)
then uses the trained posttoss_model.pkl to predict and report per-season accuracy.

Note: 2020 IPL data is not in the dataset (UAE bubble season absent from source data).
"""

import pickle, warnings, os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

BACKTEST_SEASONS = ["2021", "2022", "2023", "2024", "2025"]
THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.65, 0.70]

print("=" * 65)
print("POST-TOSS MODEL BACKTEST — IPL 2021-2025")
print("=" * 65)

# ── Load model ────────────────────────────────────────────────────────────
with open("models/posttoss_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]
features = bundle["features"]
median_  = bundle["train_median"]
conf_thr = bundle.get("confidence_threshold", 0.58)
print(f"Model loaded. Features: {len(features)}, confidence threshold: {conf_thr}")

# ── Load data ─────────────────────────────────────────────────────────────
print("\nLoading data...")
del_df = pd.read_csv("data/deliveries.csv")
del_df["file_id"] = del_df["file_id"].astype(str)
del_df["date"]    = pd.to_datetime(del_df["date"])

TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils":            "Delhi Capitals",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
    "Punjab Kings":                "Kings XI Punjab",
}
for col in ["batting_team", "bowling_team", "winner"]:
    if col in del_df.columns:
        del_df[col] = del_df[col].replace(TEAM_NAME_MAP)

mf = pd.read_csv("data/match_features.csv")
mf["file_id"] = mf["file_id"].astype(str)
mf["date"]    = pd.to_datetime(mf["date"])
mf["season"]  = mf["season"].astype(str)
mf = mf[mf["team1_won"].notna()].copy()

valid_fids = set(del_df["file_id"].unique()) & set(mf["file_id"].unique())
mf = mf[mf["file_id"].isin(valid_fids)].copy()
print(f"  Matches with deliveries + features: {len(mf)}")

# Filter to backtest seasons (but keep ALL seasons for expanding-window features)
backtest_fids = set(mf[mf["season"].isin(BACKTEST_SEASONS)]["file_id"])
print(f"  Backtest matches (2021-2025): {len(backtest_fids)}")

# ── Bat-first team ────────────────────────────────────────────────────────
inn1_teams = (del_df[del_df["innings"] == 1]
              .groupby("file_id")["batting_team"].first()
              .reset_index().rename(columns={"batting_team": "bat_first_team"}))
inn2_teams = (del_df[del_df["innings"] == 2]
              .groupby("file_id")["batting_team"].first()
              .reset_index().rename(columns={"batting_team": "bat_second_team"}))

mf = mf.merge(inn1_teams, on="file_id", how="left")
mf = mf.merge(inn2_teams, on="file_id", how="left")

del_winners = (del_df.groupby("file_id")["winner"].first()
               .reset_index().rename(columns={"winner": "del_winner"}))
mf = mf.merge(del_winners, on="file_id", how="left")
mf["bat_first_won"] = (mf["bat_first_team"] == mf["del_winner"]).astype(int)
print(f"  Bat-first win rate: {mf['bat_first_won'].mean():.3f}")

# ── Extract XI ────────────────────────────────────────────────────────────
def extract_xi(fid):
    match = del_df[del_df["file_id"] == fid]
    inn1  = match[match["innings"] == 1]
    inn2  = match[match["innings"] == 2]
    bf_players = set(inn1["batter"].unique()) | set(inn1["non_striker"].unique())
    if len(inn2): bf_players |= set(inn2["bowler"].unique())
    bf_players |= set(inn1.loc[inn1["player_out"].notna(), "player_out"].unique())
    bs_players = set(inn1["bowler"].unique())
    if len(inn2):
        bs_players |= set(inn2["batter"].unique()) | set(inn2["non_striker"].unique())
        bs_players |= set(inn2.loc[inn2["player_out"].notna(), "player_out"].unique())
    bf_players = [p for p in bf_players if pd.notna(p) and str(p).strip()]
    bs_players = [p for p in bs_players if pd.notna(p) and str(p).strip()]
    return bf_players, bs_players

print("Extracting XI for all matches...")
xi_data = {fid: extract_xi(fid) for fid in mf["file_id"].unique()}

# ── Expanding-window player scores ────────────────────────────────────────
print("Computing player scores (expanding window)...")
del_sorted = del_df.sort_values(["date", "file_id", "innings", "over", "ball_in_over"]).copy()

bat_legal = del_sorted[del_sorted["is_wide"] == 0].copy()
bat_inns  = (bat_legal.groupby(["batter", "file_id", "date"])
             .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "count"))
             .reset_index().sort_values(["batter", "date", "file_id"]))
bat_inns["cum_runs"]  = bat_inns.groupby("batter")["runs"].cumsum()
bat_inns["cum_balls"] = bat_inns.groupby("batter")["balls"].cumsum()
bat_inns["cum_inns"]  = bat_inns.groupby("batter").cumcount() + 1
bat_inns["prev_runs"]  = bat_inns.groupby("batter")["cum_runs"].shift(1, fill_value=0)
bat_inns["prev_balls"] = bat_inns.groupby("batter")["cum_balls"].shift(1, fill_value=0)
bat_inns["prev_inns"]  = bat_inns.groupby("batter")["cum_inns"].shift(1, fill_value=0)
bat_inns["career_avg"] = np.where(bat_inns["prev_inns"] > 0, bat_inns["prev_runs"] / bat_inns["prev_inns"].clip(1), 0)
bat_inns["career_sr"]  = np.where(bat_inns["prev_balls"] > 0, bat_inns["prev_runs"] / bat_inns["prev_balls"].clip(1) * 100, 0)
bat_inns["form5_avg"]  = bat_inns.groupby("batter")["runs"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
bat_inns["form5_sr_n"] = bat_inns.groupby("batter")["runs"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bat_inns["form5_sr_d"] = bat_inns.groupby("batter")["balls"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bat_inns["form5_sr"]   = np.where(bat_inns["form5_sr_d"] > 0, bat_inns["form5_sr_n"] / bat_inns["form5_sr_d"] * 100, 0)

def compute_bat_score(row):
    if row["prev_inns"] < 1: return np.nan
    career = (row["career_avg"] / 40) * 0.5 + (row["career_sr"] / 150) * 0.5
    fa     = row["form5_avg"]  if pd.notna(row["form5_avg"])  else row["career_avg"]
    fs     = row["form5_sr"]   if pd.notna(row["form5_sr"]) and row["form5_sr"] > 0 else row["career_sr"]
    form   = (fa / 40) * 0.5 + (fs / 150) * 0.5
    return max(0, (0.5 * career + 0.5 * form) * 100)

bat_inns["bat_score"] = bat_inns.apply(compute_bat_score, axis=1)
bat_score_lookup = {(r.batter, r.file_id): r.bat_score
                    for _, r in bat_inns[["batter", "file_id", "bat_score"]].dropna(subset=["bat_score"]).iterrows()}

bowl_inns = (del_sorted.groupby(["bowler", "file_id", "date"])
             .agg(runs_conceded=("runs_total", "sum"),
                  balls_bowled=("is_wide", lambda x: (x == 0).sum()),
                  wickets=("is_wicket", "sum"))
             .reset_index().sort_values(["bowler", "date", "file_id"]))
bowl_inns["cum_runs_c"]    = bowl_inns.groupby("bowler")["runs_conceded"].cumsum()
bowl_inns["cum_balls_b"]   = bowl_inns.groupby("bowler")["balls_bowled"].cumsum()
bowl_inns["cum_wkts"]      = bowl_inns.groupby("bowler")["wickets"].cumsum()
bowl_inns["cum_bowl_inns"] = bowl_inns.groupby("bowler").cumcount() + 1
bowl_inns["prev_runs_c"]   = bowl_inns.groupby("bowler")["cum_runs_c"].shift(1, fill_value=0)
bowl_inns["prev_balls_b"]  = bowl_inns.groupby("bowler")["cum_balls_b"].shift(1, fill_value=0)
bowl_inns["prev_wkts"]     = bowl_inns.groupby("bowler")["cum_wkts"].shift(1, fill_value=0)
bowl_inns["prev_bowl_inns"]= bowl_inns.groupby("bowler")["cum_bowl_inns"].shift(1, fill_value=0)
bowl_inns["career_econ"]   = np.where(bowl_inns["prev_balls_b"] > 0,
                               bowl_inns["prev_runs_c"] / (bowl_inns["prev_balls_b"] / 6), 12.0)
bowl_inns["career_wkt_rate"] = np.where(bowl_inns["prev_bowl_inns"] > 0,
                                bowl_inns["prev_wkts"] / bowl_inns["prev_bowl_inns"], 0)
bowl_inns["form5_econ_runs"] = bowl_inns.groupby("bowler")["runs_conceded"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bowl_inns["form5_econ_balls"]= bowl_inns.groupby("bowler")["balls_bowled"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bowl_inns["form5_econ"]      = np.where(bowl_inns["form5_econ_balls"] > 0,
                                bowl_inns["form5_econ_runs"] / (bowl_inns["form5_econ_balls"] / 6),
                                bowl_inns["career_econ"])
bowl_inns["form5_wkts"]      = bowl_inns.groupby("bowler")["wickets"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

def compute_bowl_score(row):
    if row["prev_bowl_inns"] < 1: return np.nan
    c_econ = max(0, (10 - row["career_econ"]) / 6)
    c_wkt  = min(row["career_wkt_rate"] / 2, 1)
    f_econ = max(0, (10 - row["form5_econ"]) / 6) if pd.notna(row["form5_econ"]) else c_econ
    f_wkt  = min(row["form5_wkts"] / 2, 1) if pd.notna(row["form5_wkts"]) else c_wkt
    return max(0, (0.5 * (c_econ * 0.5 + c_wkt * 0.5) + 0.5 * (f_econ * 0.5 + f_wkt * 0.5)) * 100)

bowl_inns["bowl_score"] = bowl_inns.apply(compute_bowl_score, axis=1)
bowl_score_lookup = {(r.bowler, r.file_id): r.bowl_score
                     for _, r in bowl_inns[["bowler", "file_id", "bowl_score"]].dropna(subset=["bowl_score"]).iterrows()}

# Venue batting
bat_venue_info = del_sorted[["file_id", "venue"]].drop_duplicates()
biv = bat_inns.merge(bat_venue_info, on="file_id", how="left").sort_values(["batter", "venue", "date", "file_id"])
biv["v_cum_runs"]  = biv.groupby(["batter", "venue"])["runs"].cumsum()
biv["v_cum_balls"] = biv.groupby(["batter", "venue"])["balls"].cumsum()
biv["v_cum_inns"]  = biv.groupby(["batter", "venue"]).cumcount() + 1
biv["v_prev_runs"]  = biv.groupby(["batter", "venue"])["v_cum_runs"].shift(1, fill_value=0)
biv["v_prev_balls"] = biv.groupby(["batter", "venue"])["v_cum_balls"].shift(1, fill_value=0)
biv["v_prev_inns"]  = biv.groupby(["batter", "venue"])["v_cum_inns"].shift(1, fill_value=0)
biv["v_career_avg"] = np.where(biv["v_prev_inns"] >= 2, biv["v_prev_runs"] / biv["v_prev_inns"].clip(1), np.nan)
biv["v_career_sr"]  = np.where(biv["v_prev_balls"] >= 10, biv["v_prev_runs"] / biv["v_prev_balls"].clip(1) * 100, np.nan)
biv["venue_bat_score"] = np.where(
    biv["v_career_avg"].notna() & biv["v_career_sr"].notna(),
    ((biv["v_career_avg"] / 40) * 0.5 + (biv["v_career_sr"] / 150) * 0.5) * 100, np.nan)
venue_bat_lookup = {(r.batter, r.file_id): r.venue_bat_score
                    for _, r in biv[["batter", "file_id", "venue_bat_score"]].dropna(subset=["venue_bat_score"]).iterrows()}

DEFAULT_BAT  = float(bat_inns["bat_score"].dropna().median())  if bat_inns["bat_score"].notna().any()  else 50.0
DEFAULT_BOWL = float(bowl_inns["bowl_score"].dropna().median()) if bowl_inns["bowl_score"].notna().any() else 40.0
print(f"  Defaults: bat={DEFAULT_BAT:.1f}, bowl={DEFAULT_BOWL:.1f}")

def xi_features(fid, players):
    bat_scores, bowl_scores = [], []
    for p in players:
        cb = bat_score_lookup.get((p, fid), np.nan)
        vb = venue_bat_lookup.get((p, fid), np.nan)
        blended_bat = (0.6 * cb + 0.4 * vb) if not np.isnan(cb) and not np.isnan(vb) else (vb if not np.isnan(vb) else (cb if not np.isnan(cb) else DEFAULT_BAT))
        cb2 = bowl_score_lookup.get((p, fid), np.nan)
        bat_scores.append(blended_bat)
        bowl_scores.append(cb2 if not np.isnan(cb2) else DEFAULT_BOWL)
    if not bat_scores:
        return DEFAULT_BAT, DEFAULT_BOWL, 0, DEFAULT_BAT, 0
    bs = sorted(bat_scores, reverse=True); ws = sorted(bowl_scores, reverse=True)
    return np.mean(bs[:6]), np.mean(ws[:4]), (np.std(bs) if len(bs) > 1 else 0), bs[0], len(players)

print("Computing XI features for all matches...")
xi_feats = []
for _, row in mf.iterrows():
    fid = row["file_id"]
    if fid not in xi_data: continue
    bf_pl, bs_pl = xi_data[fid]
    bf_bat, bf_bowl, bf_dep, bf_max, bf_n = xi_features(fid, bf_pl)
    bs_bat, bs_bowl, bs_dep, bs_max, bs_n = xi_features(fid, bs_pl)
    xi_feats.append({"file_id": fid,
                     "bf_xi_bat": bf_bat, "bf_xi_bowl": bf_bowl, "bf_xi_depth": bf_dep, "bf_xi_max_bat": bf_max, "bf_xi_n": bf_n,
                     "bs_xi_bat": bs_bat, "bs_xi_bowl": bs_bowl, "bs_xi_depth": bs_dep, "bs_xi_max_bat": bs_max, "bs_xi_n": bs_n,
                     "xi_bat_diff": bf_bat - bs_bat, "xi_bowl_diff": bf_bowl - bs_bowl})

xi_df = pd.DataFrame(xi_feats)

# ── Weather (from cache) ──────────────────────────────────────────────────
print("Loading weather cache...")
WEATHER_CACHE = "data/weather_cache_v2.csv"
if os.path.exists(WEATHER_CACHE):
    weather_df = pd.read_csv(WEATHER_CACHE)
    weather_df["file_id"] = weather_df["file_id"].astype(str)
    weather_df = weather_df.drop_duplicates(subset="file_id", keep="first")
    weather_df["heat_factor"] = (weather_df["temperature"].fillna(30) >= 35).astype(int)
    print(f"  {len(weather_df)} cached records")
else:
    weather_df = pd.DataFrame(columns=["file_id", "temperature", "humidity", "cloud_cover", "wind_speed", "heat_factor"])
    print("  No cache found — weather features will be missing")

# ── Build feature matrix ──────────────────────────────────────────────────
print("Building feature matrix...")
df = mf.merge(xi_df, on="file_id", how="left")
wcols = [c for c in ["file_id", "temperature", "humidity", "cloud_cover", "wind_speed", "heat_factor"] if c in weather_df.columns]
if wcols:
    df = df.merge(weather_df[wcols], on="file_id", how="left")

# Date/time features
df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
date_counts = df.groupby("date_str")["file_id"].transform("count")
date_ranks  = df.groupby("date_str")["file_id"].transform("rank", method="first")
df["is_evening"]  = np.where((date_counts >= 2) & (date_ranks == 1), 0, 1)
df["match_hour"]  = np.where(df["is_evening"] == 1, 19, 15)
df["humidity"]    = df["humidity"].fillna(df["humidity"].median() if df["humidity"].notna().any() else 55)
df["dew_factor"]  = np.where((df["humidity"] >= 65) & (df["is_evening"] == 1),
                              np.clip((df["humidity"] - 65) / 35, 0, 1), 0)

# Bat-first orientation
df["bf_is_team1"] = (df["bat_first_team"] == df["team1"]).astype(int)
df["bf_elo"]      = np.where(df["bf_is_team1"] == 1, df["team1_elo"], df["team2_elo"])
df["bs_elo"]      = np.where(df["bf_is_team1"] == 1, df["team2_elo"], df["team1_elo"])
df["elo_diff_bf"] = df["bf_elo"] - df["bs_elo"]
df["bf_form"]     = np.where(df["bf_is_team1"] == 1, df["team1_form"], df["team2_form"])
df["bs_form"]     = np.where(df["bf_is_team1"] == 1, df["team2_form"], df["team1_form"])
df["form_diff_bf"]= df["bf_form"] - df["bs_form"]
for w in ["form_3", "form_10"]:
    df[f"bf_{w}"] = np.where(df["bf_is_team1"] == 1, df[f"team1_{w}"], df[f"team2_{w}"])
    df[f"bs_{w}"] = np.where(df["bf_is_team1"] == 1, df[f"team2_{w}"], df[f"team1_{w}"])
    df[f"{w}_diff_bf"] = df[f"bf_{w}"] - df[f"bs_{w}"]
df["h2h_bf"]      = np.where(df["bf_is_team1"] == 1, df["h2h_win_rate_team1"], 1 - df["h2h_win_rate_team1"])
df["bf_venue_wr"] = np.where(df["bf_is_team1"] == 1, df["team1_venue_win_rate"], df["team2_venue_win_rate"])
df["bs_venue_wr"] = np.where(df["bf_is_team1"] == 1, df["team2_venue_win_rate"], df["team1_venue_win_rate"])

df["toss_winner_bats_first"] = np.where(
    ((df["team1_won_toss"] == 1) & (df["toss_chose_bat"] == 1)) |
    ((df["team1_won_toss"] == 0) & (df["toss_chose_bat"] == 0)), 1, 0)
df["toss_chose_field"] = 1 - df["toss_chose_bat"]
df["toss_venue_aligned_bf"] = np.where(
    (df["toss_chose_field"] == 1) & (df["venue_chase_win_rate"] > 0.5), 1,
    np.where((df["toss_chose_bat"] == 1) & (df["venue_bat_first_win_rate"] > 0.5), 1, 0))

if "team1_chase_wr" in df.columns:
    df["bf_chase_wr"] = np.where(df["bf_is_team1"] == 1, df["team1_chase_wr"], df["team2_chase_wr"])
    df["bs_chase_wr"] = np.where(df["bf_is_team1"] == 1, df["team2_chase_wr"], df["team1_chase_wr"])

if "venue_bat_wr" in df.columns and "venue_field_wr" in df.columns:
    df["venue_decision_wr_bf"] = np.where(
        df["toss_winner_bats_first"] == 1, df["venue_bat_wr"], 1.0 - df["venue_field_wr"])
else:
    df["venue_decision_wr_bf"] = 0.5

df["dew_chase_advantage"] = df["dew_factor"] * df["venue_chase_win_rate"]
df["humidity_x_evening"]  = df["humidity"].fillna(60) * df["is_evening"].fillna(0) / 100

for col in ["temperature", "humidity", "cloud_cover", "wind_speed", "dew_factor", "is_evening", "heat_factor"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

feature_median = pd.Series(median_)
X_all = df[features].fillna(feature_median).values
y_all = df["bat_first_won"].values
seasons_all = df["season"].values

# ── Predict ───────────────────────────────────────────────────────────────
print("Predicting with posttoss_model.pkl...")
raw_probs = model.predict_proba(X_all)[:, 1]
# Apply isotonic calibration if present in bundle
calibrator = bundle.get("calibrator") or getattr(model, "calibrator", None)
if calibrator is not None:
    probs = calibrator.predict(raw_probs)
else:
    probs = raw_probs
preds = (probs >= 0.5).astype(int)
correct = (preds == y_all).astype(int)

df_results = df[["file_id", "season", "bat_first_team", "bat_second_team", "bat_first_won"]].copy()
df_results["prob_bat_first"]  = probs
df_results["predicted_winner"]= np.where(probs >= 0.5, df_results["bat_first_team"], df_results["bat_second_team"])
df_results["correct"]         = correct
df_results["confidence"]      = np.abs(probs - 0.5) + 0.5  # distance from 0.5, mapped to [0.5, 1.0]

# ── Print results ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RESULTS — ALL PREDICTIONS (threshold >= 0.50)")
print("=" * 65)
print(f"\n{'Season':>8}  {'Matches':>7}  {'Correct':>7}  {'Accuracy':>9}  {'Tag'}")
print("-" * 50)
for s in ["2021", "2022", "2023", "2024", "2025"]:
    mask = df_results["season"] == s
    n = mask.sum()
    if n == 0: continue
    c = df_results.loc[mask, "correct"].sum()
    tag = "IS" if s in ["2021", "2022"] else "OOS"
    print(f"  {s:>6}  {n:>7}  {c:>7}  {c/n*100:>8.1f}%  {tag}")
mask_all = df_results["season"].isin(BACKTEST_SEASONS)
n_all = mask_all.sum(); c_all = df_results.loc[mask_all, "correct"].sum()
print("-" * 50)
print(f"{'2021-25':>8}  {n_all:>7}  {c_all:>7}  {c_all/n_all*100:>8.1f}%")
mask_oos = df_results["season"].isin(["2023", "2024", "2025"])
n_oos = mask_oos.sum(); c_oos = df_results.loc[mask_oos, "correct"].sum()
print(f"{'OOS 23-25':>9}  {n_oos:>7}  {c_oos:>7}  {c_oos/n_oos*100:>8.1f}%")

print()
print("=" * 65)
print("ACCURACY BY CONFIDENCE THRESHOLD (2021-2025)")
print("  Higher threshold = more confident predictions only")
print("=" * 65)
print(f"\n{'Threshold':>10}  {'Coverage':>9}  {'Matches':>7}  {'Accuracy':>9}")
print("-" * 45)
for thr in THRESHOLDS:
    conf_mask = df_results["confidence"] >= thr
    bt_mask   = df_results["season"].isin(BACKTEST_SEASONS) & conf_mask
    n = bt_mask.sum()
    if n < 10: continue
    c = df_results.loc[bt_mask, "correct"].sum()
    cov = n / mask_all.sum()
    print(f"  {thr:>8.2f}  {cov*100:>8.1f}%  {n:>7}  {c/n*100:>8.1f}%")

print()
print("=" * 65)
print("OOS ACCURACY BY CONFIDENCE (2023-2024-2025 only)")
print("=" * 65)
print(f"\n{'Threshold':>10}  {'Coverage':>9}  {'Matches':>7}  {'Accuracy':>9}")
print("-" * 45)
for thr in THRESHOLDS:
    conf_mask = df_results["confidence"] >= thr
    oos_mask  = mask_oos & conf_mask
    n = oos_mask.sum()
    if n < 10: continue
    c = df_results.loc[oos_mask, "correct"].sum()
    cov = n / n_oos
    print(f"  {thr:>8.2f}  {cov*100:>8.1f}%  {n:>7}  {c/n*100:>8.1f}%")

print()
print("=" * 65)
print("PER-SEASON BREAKDOWN BY CONFIDENCE THRESHOLD")
print("=" * 65)
for thr in [0.50, 0.58]:
    print(f"\nThreshold = {thr}")
    print(f"  {'Season':>8}  {'Matches':>7}  {'Correct':>7}  {'Accuracy':>9}  {'Coverage':>9}")
    print("  " + "-" * 50)
    for s in ["2021", "2022", "2023", "2024", "2025"]:
        conf_mask = df_results["confidence"] >= thr
        smask = (df_results["season"] == s) & conf_mask
        n = smask.sum()
        if n == 0: continue
        c = df_results.loc[smask, "correct"].sum()
        tot = (df_results["season"] == s).sum()
        tag = "IS" if s in ["2021", "2022"] else "OOS"
        print(f"  {s:>8}  {n:>7}  {c:>7}  {c/n*100:>8.1f}%  {n/tot*100:>8.1f}%  {tag}")

print()
print("=" * 65)
print("CONFIDENCE GAP (correct vs wrong predictions, 2021-2025)")
print("=" * 65)
bt_res = df_results[df_results["season"].isin(BACKTEST_SEASONS)].copy()
corr_conf = bt_res.loc[bt_res["correct"] == 1, "confidence"].mean()
wrong_conf = bt_res.loc[bt_res["correct"] == 0, "confidence"].mean()
print(f"  Avg confidence when CORRECT : {corr_conf*100:.1f}%")
print(f"  Avg confidence when WRONG   : {wrong_conf*100:.1f}%")
print(f"  Gap                         : {(corr_conf - wrong_conf)*100:.1f}pp")

# Save
out_path = "data/backtest_posttoss.csv"
df_results.to_csv(out_path, index=False)
print(f"\nDetailed results saved to {out_path}")
print("=" * 65)
