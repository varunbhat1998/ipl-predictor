"""Evaluate all 4 models per season (2015-2025)."""
import pickle, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from model_classes import EnsemblePreMatchModel

# Load models
with open("models/prematch_model.pkl", "rb") as f:
    pre = pickle.load(f)
with open("models/posttoss_model.pkl", "rb") as f:
    post = pickle.load(f)
with open("models/live_model.pkl", "rb") as f:
    live = pickle.load(f)
with open("models/inn1_live_model.pkl", "rb") as f:
    inn1_m = pickle.load(f)

# Load data
mf = pd.read_csv("data/match_features.csv")
mf["file_id"] = mf["file_id"].astype(str)
mf["date"] = pd.to_datetime(mf["date"])
mf = mf[mf["team1_won"].notna()].copy()

dl = pd.read_csv("data/deliveries.csv")
dl["file_id"] = dl["file_id"].astype(str)
dl["date"] = pd.to_datetime(dl["date"])

TEAM_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Punjab Kings": "Kings XI Punjab",
}
for col in ["batting_team", "bowling_team", "winner"]:
    dl[col] = dl[col].replace(TEAM_MAP)

valid_fids = set(dl["file_id"].unique()) & set(mf["file_id"].unique())
mf = mf[mf["file_id"].isin(valid_fids)].copy()

inn1_teams_map = dl[dl["innings"] == 1].groupby("file_id")["batting_team"].first()
del_winners_map = dl.groupby("file_id")["winner"].first()

seasons = [s for s in range(2015, 2026)
           if str(s) in mf["season"].astype(str).unique()]

results = {s: {} for s in seasons}

# ============================================================
# 1. PRE-TOSS MODEL
# ============================================================
print("Evaluating pre-toss model...")
pre_feats = pre["features"]
pre_median = pre.get("train_median", {})

for s in seasons:
    sub = mf[mf["season"].astype(str) == str(s)].copy()
    X_vals = []
    for _, row in sub.iterrows():
        X_vals.append([float(row[f]) if f in row.index and pd.notna(row[f])
                       else float(pre_median.get(f, 0)) for f in pre_feats])
    X = np.array(X_vals)
    y = sub["team1_won"].astype(int).values
    preds = (pre["model"].predict_proba(X)[:, 1] >= 0.5).astype(int)
    results[s]["pre_toss"] = (preds == y).mean()
    results[s]["n"] = len(sub)

# ============================================================
# 2. POST-TOSS MODEL  (use weather cache for actual weather)
# ============================================================
print("Evaluating post-toss model...")
post_feats = post["features"]
post_median = post.get("train_median", {})

# Load weather cache
weather_cache = {}
import os
wc_path = "data/weather_cache_v2.csv"
if os.path.exists(wc_path):
    wc = pd.read_csv(wc_path)
    wc["file_id"] = wc["file_id"].astype(str)
    for _, r in wc.iterrows():
        weather_cache[r["file_id"]] = {
            "temperature": r.get("temperature", 30),
            "humidity": r.get("humidity", 55),
            "cloud_cover": r.get("cloud_cover", 30),
        }

# Load expanding-window XI features from training (they were saved in the model training)
# Re-compute XI features per match using the same logic as 10_post_toss_model.py
# For speed, we reuse the training script's bat_score_lookup approach

# Actually, let's load the post-toss model's training data approach:
# Use the SAME feature computation as 10_post_toss_model.py Phase 3-5

# Pre-compute batting/bowling per-match stats
print("  Building expanding-window player scores...")
del_sorted = dl.sort_values(["date", "file_id", "innings", "over", "ball_in_over"]).copy()

# Batting per innings
bat_legal = del_sorted[del_sorted["is_wide"] == 0]
bat_inns = (bat_legal.groupby(["batter", "file_id", "date"])
            .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "count"))
            .reset_index()
            .sort_values(["batter", "date", "file_id"]))
bat_inns["cum_runs"] = bat_inns.groupby("batter")["runs"].cumsum()
bat_inns["cum_balls"] = bat_inns.groupby("batter")["balls"].cumsum()
bat_inns["cum_inns"] = bat_inns.groupby("batter").cumcount() + 1
bat_inns["prev_runs"] = bat_inns.groupby("batter")["cum_runs"].shift(1, fill_value=0)
bat_inns["prev_balls"] = bat_inns.groupby("batter")["cum_balls"].shift(1, fill_value=0)
bat_inns["prev_inns"] = bat_inns.groupby("batter")["cum_inns"].shift(1, fill_value=0)
bat_inns["career_avg"] = np.where(bat_inns["prev_inns"] > 0,
    bat_inns["prev_runs"] / bat_inns["prev_inns"].clip(1), 0)
bat_inns["career_sr"] = np.where(bat_inns["prev_balls"] > 0,
    bat_inns["prev_runs"] / bat_inns["prev_balls"].clip(1) * 100, 0)
bat_inns["form5_avg"] = bat_inns.groupby("batter")["runs"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
bat_inns["form5_sr_num"] = bat_inns.groupby("batter")["runs"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bat_inns["form5_sr_den"] = bat_inns.groupby("batter")["balls"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bat_inns["form5_sr"] = np.where(bat_inns["form5_sr_den"] > 0,
    bat_inns["form5_sr_num"] / bat_inns["form5_sr_den"] * 100, 0)

def _bat_score(row):
    if row["prev_inns"] < 1:
        return np.nan
    career = (row["career_avg"] / 40) * 0.5 + (row["career_sr"] / 150) * 0.5
    f_avg = row["form5_avg"] if pd.notna(row["form5_avg"]) else row["career_avg"]
    f_sr = row["form5_sr"] if pd.notna(row["form5_sr"]) and row["form5_sr"] > 0 else row["career_sr"]
    form = (f_avg / 40) * 0.5 + (f_sr / 150) * 0.5
    return max(0, (0.5 * career + 0.5 * form) * 100)

bat_inns["bat_score"] = bat_inns.apply(_bat_score, axis=1)
bat_score_lookup = {}
for _, r in bat_inns[["batter", "file_id", "bat_score"]].dropna(subset=["bat_score"]).iterrows():
    bat_score_lookup[(r["batter"], r["file_id"])] = r["bat_score"]

# Bowling per innings
bowl_inns = (del_sorted.groupby(["bowler", "file_id", "date"])
             .agg(runs_c=("runs_total", "sum"),
                  balls_b=("is_wide", lambda x: (x == 0).sum()),
                  wickets=("is_wicket", "sum"))
             .reset_index()
             .sort_values(["bowler", "date", "file_id"]))
bowl_inns["cum_rc"] = bowl_inns.groupby("bowler")["runs_c"].cumsum()
bowl_inns["cum_bb"] = bowl_inns.groupby("bowler")["balls_b"].cumsum()
bowl_inns["cum_wk"] = bowl_inns.groupby("bowler")["wickets"].cumsum()
bowl_inns["cum_bi"] = bowl_inns.groupby("bowler").cumcount() + 1
bowl_inns["prev_rc"] = bowl_inns.groupby("bowler")["cum_rc"].shift(1, fill_value=0)
bowl_inns["prev_bb"] = bowl_inns.groupby("bowler")["cum_bb"].shift(1, fill_value=0)
bowl_inns["prev_wk"] = bowl_inns.groupby("bowler")["cum_wk"].shift(1, fill_value=0)
bowl_inns["prev_bi"] = bowl_inns.groupby("bowler")["cum_bi"].shift(1, fill_value=0)
bowl_inns["career_econ"] = np.where(bowl_inns["prev_bb"] > 0,
    bowl_inns["prev_rc"] / (bowl_inns["prev_bb"] / 6), 12.0)
bowl_inns["career_wkt_rate"] = np.where(bowl_inns["prev_bi"] > 0,
    bowl_inns["prev_wk"] / bowl_inns["prev_bi"], 0)
bowl_inns["f5_econ_r"] = bowl_inns.groupby("bowler")["runs_c"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bowl_inns["f5_econ_b"] = bowl_inns.groupby("bowler")["balls_b"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum())
bowl_inns["f5_econ"] = np.where(bowl_inns["f5_econ_b"] > 0,
    bowl_inns["f5_econ_r"] / (bowl_inns["f5_econ_b"] / 6), bowl_inns["career_econ"])
bowl_inns["f5_wkt"] = bowl_inns.groupby("bowler")["wickets"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())

def _bowl_score(row):
    if row["prev_bi"] < 1:
        return np.nan
    c_e = max(0, (10 - row["career_econ"]) / 6)
    c_w = min(row["career_wkt_rate"] / 2, 1)
    career = c_e * 0.5 + c_w * 0.5
    f_e = max(0, (10 - row["f5_econ"]) / 6) if pd.notna(row["f5_econ"]) else c_e
    f_w = min(row["f5_wkt"] / 2, 1) if pd.notna(row["f5_wkt"]) else c_w
    form = f_e * 0.5 + f_w * 0.5
    return max(0, (0.5 * career + 0.5 * form) * 100)

bowl_inns["bowl_score"] = bowl_inns.apply(_bowl_score, axis=1)
bowl_score_lookup = {}
for _, r in bowl_inns[["bowler", "file_id", "bowl_score"]].dropna(subset=["bowl_score"]).iterrows():
    bowl_score_lookup[(r["bowler"], r["file_id"])] = r["bowl_score"]

print(f"  Bat scores: {len(bat_score_lookup)}, Bowl scores: {len(bowl_score_lookup)}")

# Extract XI per match
def extract_xi(fid):
    match = dl[dl["file_id"] == fid]
    i1 = match[match["innings"] == 1]
    i2 = match[match["innings"] == 2]
    bf_p = set(i1["batter"].unique()) | set(i1["non_striker"].unique())
    if len(i2): bf_p |= set(i2["bowler"].unique())
    bf_p |= set(i1.loc[i1["player_out"].notna(), "player_out"].unique())
    bs_p = set(i1["bowler"].unique())
    if len(i2):
        bs_p |= set(i2["batter"].unique()) | set(i2["non_striker"].unique())
        bs_p |= set(i2.loc[i2["player_out"].notna(), "player_out"].unique())
    bf_p = [p for p in bf_p if pd.notna(p) and p.strip()]
    bs_p = [p for p in bs_p if pd.notna(p) and p.strip()]
    return bf_p, bs_p

def xi_strength(fid, players):
    bats, bowls = [], []
    for p in players:
        b = bat_score_lookup.get((p, fid), 45.0)
        w = bowl_score_lookup.get((p, fid), 35.0)
        bats.append(b)
        bowls.append(w)
    if not bats:
        return 45.0, 35.0, 15.0, 45.0
    bats_s = sorted(bats, reverse=True)
    bowls_s = sorted(bowls, reverse=True)
    return (np.mean(bats_s[:6]), np.mean(bowls_s[:4]),
            np.std(bats_s) if len(bats_s) > 1 else 0, bats_s[0])

print("  Computing post-toss features per match...")
for s in seasons:
    sub = mf[mf["season"].astype(str) == str(s)].copy()
    sub["bat_first_team"] = sub["file_id"].map(inn1_teams_map)
    sub["del_winner"] = sub["file_id"].map(del_winners_map)
    sub["bat_first_won"] = (sub["bat_first_team"] == sub["del_winner"]).astype(int)
    sub["bf_is_team1"] = (sub["bat_first_team"] == sub["team1"]).astype(int)
    date_counts = sub.groupby(sub["date"].dt.strftime("%Y-%m-%d"))["file_id"].transform("count")
    date_ranks = sub.groupby(sub["date"].dt.strftime("%Y-%m-%d"))["file_id"].transform("rank", method="first")
    sub["is_evening"] = np.where((date_counts >= 2) & (date_ranks == 1), 0, 1)

    X_vals = []
    for _, row in sub.iterrows():
        fid = row["file_id"]
        bf1 = row["bf_is_team1"]
        bf_p, bs_p = extract_xi(fid)
        bf_bat, bf_bowl, bf_depth, bf_max = xi_strength(fid, bf_p)
        bs_bat, bs_bowl, bs_depth, bs_max = xi_strength(fid, bs_p)

        wc = weather_cache.get(fid, {})
        temp = wc.get("temperature", 30)
        hum = wc.get("humidity", 55)
        cloud = wc.get("cloud_cover", 30)
        is_eve = row.get("is_evening", 1)
        dew = max(0, min(1, (hum - 65) / 35)) if hum >= 65 and is_eve else 0.0

        fd = {}
        fd["bf_elo"] = row["team1_elo"] if bf1 else row["team2_elo"]
        fd["bs_elo"] = row["team2_elo"] if bf1 else row["team1_elo"]
        fd["elo_diff_bf"] = fd["bf_elo"] - fd["bs_elo"]
        fd["bf_form"] = row["team1_form"] if bf1 else row["team2_form"]
        fd["bs_form"] = row["team2_form"] if bf1 else row["team1_form"]
        fd["form_diff_bf"] = fd["bf_form"] - fd["bs_form"]
        for w in ["form_3", "form_10"]:
            fd[f"bf_{w}"] = row.get(f"team1_{w}", 0.5) if bf1 else row.get(f"team2_{w}", 0.5)
            fd[f"bs_{w}"] = row.get(f"team2_{w}", 0.5) if bf1 else row.get(f"team1_{w}", 0.5)
            fd[f"{w}_diff_bf"] = fd[f"bf_{w}"] - fd[f"bs_{w}"]
        h2h = row.get("h2h_win_rate_team1", 0.5)
        fd["h2h_bf"] = h2h if bf1 else 1 - h2h
        fd["bf_xi_bat"] = bf_bat
        fd["bs_xi_bat"] = bs_bat
        fd["xi_bat_diff"] = bf_bat - bs_bat
        fd["bf_xi_bowl"] = bf_bowl
        fd["bs_xi_bowl"] = bs_bowl
        fd["xi_bowl_diff"] = bf_bowl - bs_bowl
        fd["bf_xi_depth"] = bf_depth
        fd["bs_xi_depth"] = bs_depth
        fd["bf_xi_max_bat"] = bf_max
        fd["bs_xi_max_bat"] = bs_max
        fd["toss_chose_field"] = 1 - row.get("toss_chose_bat", 0)
        tw1 = row.get("team1_won_toss", 0)
        tb = row.get("toss_chose_bat", 0)
        fd["toss_winner_bats_first"] = int((tw1 == 1 and tb == 1) or (tw1 == 0 and tb == 0))
        vcw = row.get("venue_chase_win_rate", 0.5)
        fd["toss_venue_aligned_bf"] = int(
            (fd["toss_chose_field"] == 1 and vcw > 0.5) or
            (fd["toss_chose_field"] == 0 and vcw <= 0.5))
        fd["venue_chase_win_rate"] = vcw
        fd["venue_bat_first_win_rate"] = row.get("venue_bat_first_win_rate", 0.5)
        fd["venue_avg_first_innings"] = row.get("venue_avg_first_innings", 160)
        fd["venue_matches"] = row.get("venue_matches", 10)
        fd["bf_venue_wr"] = row.get("team1_venue_win_rate", 0.5) if bf1 else row.get("team2_venue_win_rate", 0.5)
        fd["bs_venue_wr"] = row.get("team2_venue_win_rate", 0.5) if bf1 else row.get("team1_venue_win_rate", 0.5)
        fd["venue_toss_win_rate"] = row.get("venue_toss_win_rate", 0.5)
        fd["bs_chase_wr"] = row.get("team2_chase_wr", 0.5) if bf1 else row.get("team1_chase_wr", 0.5)
        fd["temperature"] = temp
        fd["humidity"] = hum
        fd["cloud_cover"] = cloud
        fd["dew_factor"] = dew
        fd["is_evening"] = is_eve
        fd["heat_factor"] = 1 if temp >= 35 else 0
        fd["dew_chase_advantage"] = dew * vcw
        fd["humidity_x_evening"] = hum * is_eve / 100
        fd["match_num_in_season"] = row.get("match_num_in_season", 30)
        fd["is_playoff"] = row.get("is_playoff", 0)

        X_vals.append([fd.get(f, post_median.get(f, 0)) for f in post_feats])

    X = np.array(X_vals)
    y = sub["bat_first_won"].values
    preds = (post["model"].predict_proba(X)[:, 1] >= 0.5).astype(int)
    results[s]["post_toss"] = (preds == y).mean()

# ============================================================
# SHARED: Global venue avg matching training
# ============================================================
_inn1_totals = dl[dl["innings"] == 1].groupby(["file_id", "venue"])["runs_total"].sum().reset_index()
_venue_avg_global = _inn1_totals.groupby("venue")["runs_total"].mean()

# ============================================================
# 3. INN1 LIVE (after over 20 = end of 1st innings)
# ============================================================
print("Evaluating inn1 live model (end of 1st innings)...")
inn1_feats = inn1_m["features"]
inn1_median = inn1_m.get("train_median", {})

for s in seasons:
    sub = mf[mf["season"].astype(str) == str(s)]
    correct, total = 0, 0
    for _, mrow in sub.iterrows():
        fid = mrow["file_id"]
        md = dl[dl["file_id"] == fid]
        i1 = md[md["innings"] == 1]
        if len(i1) == 0:
            continue
        bat_team = i1["batting_team"].iloc[0]
        winner = md["winner"].iloc[0]
        venue = md["venue"].iloc[0] if "venue" in md.columns else ""

        # Training uses ALL deliveries (including wides) for ball_num
        i1_sorted = i1.sort_values(["over", "ball_in_over"]).reset_index(drop=True)
        i1_sorted["_bn"] = range(1, len(i1_sorted) + 1)

        # Take the last snapshot at ball_num % 6 == 0 (matching training)
        max_bn = len(i1_sorted)
        snap_bn = (max_bn // 6) * 6
        if snap_bn == 0:
            continue
        snapshot = i1_sorted.iloc[:snap_bn]
        ball_num = snap_bn

        runs = int(snapshot["cum_runs"].iloc[-1])
        wkts = int(snapshot["cum_wickets"].iloc[-1])
        crr = runs / ball_num * 6 if ball_num > 0 else 0

        # Venue avg: training uses global venue avg from all data
        v_avg = _venue_avg_global.get(venue, 165.0) if venue in _venue_avg_global.index else 165.0
        expected_at = v_avg * (ball_num / 120)
        projected = runs + (crr * (120 - ball_num) / 6) if ball_num < 120 else runs

        # ELO/form: model expects bat_first_elo - bowl_first_elo
        team1 = mrow.get("team1", "")
        elo_d = mrow.get("elo_diff", 0)
        form_d = mrow.get("form_diff", 0)
        if bat_team != team1:
            elo_d = -elo_d
            form_d = -form_d

        # Partnership: training tracks cum_runs - last_wkt_cum_runs
        wkt_mask = snapshot["is_wicket"] == 1
        if wkt_mask.any():
            last_wkt_pos = wkt_mask.values.nonzero()[0][-1]
            last_wkt_cum = int(snapshot.iloc[last_wkt_pos]["cum_runs"])
            last_wkt_bn = last_wkt_pos + 1
            part_runs = runs - last_wkt_cum
            part_balls = ball_num - last_wkt_bn
        else:
            part_runs = runs
            part_balls = ball_num

        # Rolling 18: training accumulates runs_total in deque
        w = min(18, len(snapshot))
        last_w = snapshot.iloc[-w:]
        l3_runs = int(last_w["runs_total"].sum())
        l3_wkts = int(last_w["is_wicket"].sum())

        # Boundary/dot pct: training uses runs_batter in [4,6] and runs_total==0
        boundary_count = int((snapshot["runs_batter"].isin([4, 6])).sum())
        dot_count = int((snapshot["runs_total"] == 0).sum())
        bound_pct = boundary_count / ball_num if ball_num > 0 else 0
        dot_pct_val = dot_count / ball_num if ball_num > 0 else 0

        # Acceleration: training = (recent_3ov_rr - crr) if ball_num >= 36
        if ball_num >= 36:
            accel = (l3_runs / min(w, 18) * 6) - crr
        else:
            accel = 0

        # Powerplay: training locks pp stats at over 6 (ball_num after 36 deliveries)
        if ball_num > 36:
            pp_snap = i1_sorted.iloc[:36]
            pp_r = int(pp_snap["cum_runs"].iloc[-1])
            pp_w = int(pp_snap["cum_wickets"].iloc[-1])
        else:
            pp_r, pp_w = 0, 0

        over_num = ball_num // 6
        pp_vs_venue = (pp_r / (v_avg * 6 / 20)) if over_num > 6 and v_avg > 0 else 0.0

        fd = {}
        fd["cum_runs"] = runs
        fd["cum_wickets"] = wkts
        fd["crr"] = crr
        fd["balls_remaining"] = 120 - ball_num
        fd["balls_pct"] = ball_num / 120
        fd["wickets_pct"] = wkts / 10
        fd["projected_score"] = projected
        fd["venue_avg"] = v_avg
        fd["score_vs_expected"] = runs - expected_at
        fd["score_vs_expected_pct"] = runs / expected_at if expected_at > 0 else 1.0
        fd["partnership_runs"] = part_runs
        fd["partnership_balls"] = part_balls
        fd["last_3ov_runs"] = l3_runs
        fd["last_3ov_wkts"] = l3_wkts
        fd["boundary_pct"] = bound_pct
        fd["dot_pct"] = dot_pct_val
        fd["acceleration"] = accel
        fd["elo_diff"] = elo_d
        fd["form_diff"] = form_d
        fd["is_pp"] = int(over_num <= 6)
        fd["pp_runs"] = pp_r if over_num > 6 else 0
        fd["pp_wickets"] = pp_w if over_num > 6 else 0
        fd["pp_vs_venue_avg"] = pp_vs_venue

        X = np.array([[fd.get(f, 0) for f in inn1_feats]])
        X = inn1_m["scaler"].transform(X)
        prob = inn1_m["model"].predict_proba(X)[0, 1]
        if (prob >= 0.5) == (bat_team == winner):
            correct += 1
        total += 1

    if total > 0:
        results[s]["inn1_ov20"] = correct / total

# ============================================================
# 4. INN2 LIVE (after over 6)
# ============================================================
# Training uses ALL deliveries (including wides) for ball_num.
# ball_num = cumcount+1, snapshots at ball_num % 6 == 0.
# first_innings_run_rate hardcoded as (target-1)/120*6.
# We must match this exactly.
print("Evaluating inn2 live model (after over 6 of chase)...")
live_feats = live["features"]
live_median = live.get("train_median", {})

for s in seasons:
    sub = mf[mf["season"].astype(str) == str(s)]
    correct, total = 0, 0
    for _, mrow in sub.iterrows():
        fid = mrow["file_id"]
        md = dl[dl["file_id"] == fid]
        i1 = md[md["innings"] == 1]
        i2 = md[md["innings"] == 2]
        if len(i1) == 0 or len(i2) == 0:
            continue
        chase_team = i2["batting_team"].iloc[0]
        winner = md["winner"].iloc[0]
        venue = md["venue"].iloc[0] if "venue" in md.columns else ""
        v_avg = _venue_avg_global.get(venue, 160)

        # 1st innings stats — training uses target column from deliveries
        if "target" in i2.columns and i2["target"].notna().any():
            target = int(i2["target"].iloc[0])
        else:
            target = int(i1["cum_runs"].iloc[-1]) + 1
        inn1_wkts = int(i1["cum_wickets"].iloc[-1])
        # Training: first_innings_run_rate = (target - 1) / 120 * 6  (hardcoded 120)
        inn1_rr = (target - 1) / 120 * 6

        # 2nd innings: use ALL deliveries (including wides) to match training
        i2_sorted = i2.sort_values(["over", "ball_in_over"])
        i2_sorted = i2_sorted.reset_index(drop=True)
        # ball_num = 1-indexed running count of ALL deliveries
        i2_sorted["_ball_num"] = range(1, len(i2_sorted) + 1)

        if len(i2_sorted) < 36:
            if len(i2_sorted) == 0:
                continue
            snapshot = i2_sorted
            ball_num = len(snapshot)
        else:
            snapshot = i2_sorted.iloc[:36]
            ball_num = 36

        runs = int(snapshot["cum_runs"].iloc[-1])
        wkts = int(snapshot["cum_wickets"].iloc[-1])

        # CRR/RRR matching training: crr = cum_runs / ball_num * 6
        crr = runs / ball_num * 6 if ball_num > 0 else 0
        balls_remaining = max(120 - ball_num, 0)
        rrr = (target - runs) / balls_remaining * 6 if balls_remaining > 0 else 99.0

        # Partnership: training tracks cum_runs - last_wkt_cum_runs
        wkt_mask = snapshot["is_wicket"] == 1
        if wkt_mask.any():
            last_wkt_pos = wkt_mask.values.nonzero()[0][-1]
            last_wkt_cum = int(snapshot.iloc[last_wkt_pos]["cum_runs"])
            last_wkt_bn = last_wkt_pos + 1  # 1-indexed
            part_runs = runs - last_wkt_cum
            part_balls = ball_num - last_wkt_bn
        else:
            part_runs = runs
            part_balls = ball_num

        # Rolling 18-ball window: sum of runs_total
        w = min(18, len(snapshot))
        l3 = snapshot.iloc[-w:]
        l3_runs = int(l3["runs_total"].sum())
        l3_wkts = int(l3["is_wicket"].sum())

        # Cumulative boundary/dot pct (training uses runs_batter>=4 and runs_batter==0)
        is_bound = (snapshot["runs_batter"] >= 4).values
        is_dot = (snapshot["runs_batter"] == 0).values
        bound_pct = is_bound[:ball_num].mean() if ball_num > 0 else 0
        dot_pct = is_dot[:ball_num].mean() if ball_num > 0 else 0

        fd = {}
        fd["ball_num"] = ball_num
        fd["balls_remaining"] = balls_remaining
        fd["balls_pct"] = ball_num / 120
        fd["cum_runs"] = runs
        fd["runs_needed"] = max(target - runs, 0)
        fd["cum_wickets"] = wkts
        fd["wickets_left"] = 10 - wkts
        fd["wickets_pct"] = wkts / 10
        fd["crr"] = crr
        fd["rrr"] = rrr
        fd["rrr_diff"] = crr - rrr
        fd["run_rate_ratio"] = crr / max(rrr, 0.01)
        fd["partnership_runs"] = part_runs
        fd["partnership_balls"] = part_balls
        fd["last_3ov_runs"] = l3_runs
        fd["last_3ov_wkts"] = l3_wkts
        fd["boundary_pct"] = bound_pct
        fd["dot_ball_pct"] = dot_pct
        fd["first_innings_run_rate"] = inn1_rr
        fd["target_vs_venue_avg"] = target / max(v_avg, 1)
        fd["first_innings_wickets"] = inn1_wkts
        # Powerplay: ball_num <= 36 → is_pp=1, pp_rate features = 0
        fd["is_pp"] = int(ball_num <= 36)
        fd["pp_runs"] = runs if ball_num <= 36 else runs  # at ball 36 these are same
        fd["pp_wickets"] = wkts
        fd["pp_run_rate"] = 0.0  # training: 0 when ball_num <= 36
        fd["pp_req_rate"] = 0.0
        fd["pp_rate_gap"] = 0.0

        X = np.array([[fd.get(f, 0) for f in live_feats]])
        X = live["scaler"].transform(X)
        prob = live["model"].predict_proba(X)[0, 1]
        if (prob >= 0.5) == (chase_team == winner):
            correct += 1
        total += 1

    if total > 0:
        results[s]["inn2_ov6"] = correct / total

# ============================================================
# DISPLAY
# ============================================================
print()
print("=" * 78)
print("  MODEL ACCURACY BY YEAR (2015-2025)")
print("=" * 78)
hdr = f"  {'Year':>4} | {'Pre-Toss':>9} | {'Post-Toss':>10} | {'Inn1 Ov20':>10} | {'Inn2 Ov6':>9} | {'Matches':>7}"
print()
print(hdr)
print("  " + "-" * 70)

totals_pre, totals_post, totals_i1, totals_i2, totals_n = [], [], [], [], []

for s in seasons:
    r = results[s]
    n = r.get("n", 0)
    pa = r.get("pre_toss")
    pta = r.get("post_toss")
    i1a = r.get("inn1_ov20")
    i2a = r.get("inn2_ov6")

    ps = f"{pa*100:5.1f}%" if pa is not None else "   -- "
    pts = f"{pta*100:5.1f}%" if pta is not None else "    -- "
    i1s = f"{i1a*100:5.1f}%" if i1a is not None else "    -- "
    i2s = f"{i2a*100:5.1f}%" if i2a is not None else "   -- "
    print(f"  {s:>4} |   {ps} |   {pts} |   {i1s} |   {i2s} |   {n:>4}")

    if pa is not None: totals_pre.append((pa, n))
    if pta is not None: totals_post.append((pta, n))
    if i1a is not None: totals_i1.append((i1a, n))
    if i2a is not None: totals_i2.append((i2a, n))
    totals_n.append(n)

print("  " + "-" * 70)

def wavg(pairs):
    t = sum(n for _, n in pairs)
    return sum(a * n for a, n in pairs) / t if t > 0 else 0

wp = wavg(totals_pre)
wpt = wavg(totals_post)
wi1 = wavg(totals_i1)
wi2 = wavg(totals_i2)
tn = sum(totals_n)
print(f"  {'AVG':>4} |   {wp*100:5.1f}% |   {wpt*100:5.1f}% |   {wi1*100:5.1f}% |   {wi2*100:5.1f}% |   {tn:>4}")
print()

# ============================================================
# 5. INN2 ACCURACY AT MULTIPLE OVERS (diagnostic)
# ============================================================
print("=" * 78)
print("  INN2 LIVE MODEL — ACCURACY BY OVER (all years pooled)")
print("=" * 78)
print()

for eval_ball in [36, 60, 90, 108]:
    eval_ov = eval_ball // 6
    correct_all, total_all = 0, 0
    for s in seasons:
        sub = mf[mf["season"].astype(str) == str(s)]
        for _, mrow in sub.iterrows():
            fid = mrow["file_id"]
            md = dl[dl["file_id"] == fid]
            i1 = md[md["innings"] == 1]
            i2 = md[md["innings"] == 2]
            if len(i1) == 0 or len(i2) == 0:
                continue
            chase_team = i2["batting_team"].iloc[0]
            winner = md["winner"].iloc[0]
            venue = md["venue"].iloc[0] if "venue" in md.columns else ""
            v_avg = _venue_avg_global.get(venue, 160) if venue in _venue_avg_global.index else 160

            if "target" in i2.columns and i2["target"].notna().any():
                target = int(i2["target"].iloc[0])
            else:
                target = int(i1["cum_runs"].iloc[-1]) + 1
            inn1_wkts = int(i1["cum_wickets"].iloc[-1])
            inn1_rr = (target - 1) / 120 * 6

            i2s = i2.sort_values(["over", "ball_in_over"]).reset_index(drop=True)
            if len(i2s) < eval_ball:
                continue
            snap = i2s.iloc[:eval_ball]
            bn = eval_ball
            runs = int(snap["cum_runs"].iloc[-1])
            wkts = int(snap["cum_wickets"].iloc[-1])

            crr = runs / bn * 6 if bn > 0 else 0
            br = max(120 - bn, 0)
            rrr = (target - runs) / br * 6 if br > 0 else 99.0

            wm = snap["is_wicket"] == 1
            if wm.any():
                lp = wm.values.nonzero()[0][-1]
                pr = runs - int(snap.iloc[lp]["cum_runs"])
                pb = bn - (lp + 1)
            else:
                pr, pb = runs, bn

            w = min(18, len(snap))
            lw = snap.iloc[-w:]
            l3r = int(lw["runs_total"].sum())
            l3w = int(lw["is_wicket"].sum())

            bpct = (snap["runs_batter"] >= 4).mean()
            dpct = (snap["runs_batter"] == 0).mean()

            # Powerplay features
            if bn > 36:
                pp_s = i2s.iloc[:36]
                pp_r = int(pp_s["cum_runs"].iloc[-1])
                pp_w = int(pp_s["cum_wickets"].iloc[-1])
                pp_rr = pp_r / 36 * 6
                pp_reqr = (target - pp_r) / 84 * 6
                pp_gap = pp_rr - pp_reqr
            else:
                pp_r, pp_w = runs, wkts
                pp_rr, pp_reqr, pp_gap = 0.0, 0.0, 0.0

            fd = {
                "ball_num": bn, "balls_remaining": br, "balls_pct": bn / 120,
                "cum_runs": runs, "runs_needed": max(target - runs, 0),
                "cum_wickets": wkts, "wickets_left": 10 - wkts, "wickets_pct": wkts / 10,
                "crr": crr, "rrr": rrr, "rrr_diff": crr - rrr,
                "run_rate_ratio": crr / max(rrr, 0.01),
                "partnership_runs": pr, "partnership_balls": pb,
                "last_3ov_runs": l3r, "last_3ov_wkts": l3w,
                "boundary_pct": bpct, "dot_ball_pct": dpct,
                "first_innings_run_rate": inn1_rr,
                "target_vs_venue_avg": target / max(v_avg, 1),
                "first_innings_wickets": inn1_wkts,
                "is_pp": int(bn <= 36),
                "pp_runs": pp_r, "pp_wickets": pp_w,
                "pp_run_rate": pp_rr, "pp_req_rate": pp_reqr, "pp_rate_gap": pp_gap,
            }

            X = np.array([[fd.get(f, 0) for f in live_feats]])
            X = live["scaler"].transform(X)
            prob = live["model"].predict_proba(X)[0, 1]
            if (prob >= 0.5) == (chase_team == winner):
                correct_all += 1
            total_all += 1

    acc = correct_all / total_all * 100 if total_all > 0 else 0
    print(f"  Over {eval_ov:>2}: {acc:5.1f}%  ({correct_all}/{total_all})")

print()
