"""
Live model backtest: First innings + Second innings prediction accuracy
Walk-forward: train on all data before test season, test on that season
Tests 2015–2025. Includes powerplay features (pp_runs, pp_wickets, pp_rate_gap).
Reports accuracy per over, per season, and overall.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

TEST_SEASONS = [str(y) for y in range(2015, 2026)]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
matches    = pd.read_csv("data/matches.csv", parse_dates=["date"])
deliveries = pd.read_csv("data/deliveries.csv", parse_dates=["date"])

def norm_team(t):
    if not isinstance(t, str): return t
    m = {"Delhi Daredevils":"Delhi Capitals","Deccan Chargers":"Sunrisers Hyderabad",
         "Punjab Kings":"Kings XI Punjab","Royal Challengers Bangalore":"Royal Challengers Bengaluru",
         "Rising Pune Supergiants":"Rising Pune Supergiant"}
    return m.get(t, t)

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

for col in ["team1","team2","winner","toss_winner","inn1_team","inn2_team"]:
    if col in matches.columns:
        matches[col] = matches[col].apply(norm_team)
for col in ["batting_team","bowling_team","winner"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].apply(norm_team)

matches["venue"]    = matches["venue"].apply(norm_venue)
deliveries["venue"] = deliveries["venue"].apply(norm_venue)
matches["season"]   = matches["season"].astype(str)
deliveries["season"] = deliveries["season"].astype(str)
matches = matches[matches["winner"].notna()].sort_values("date").reset_index(drop=True)
valid_fids = set(matches["file_id"])

# ── Venue average (expanding, no leakage) ────────────────────────────────
venue_inn1_hist = defaultdict(list)
venue_avg_map   = {}
for _, row in matches.iterrows():
    v = row["venue"]
    venue_avg_map[row["file_id"]] = np.mean(venue_inn1_hist[v]) if venue_inn1_hist[v] else 165.0
    if pd.notna(row.get("inn1_runs")):
        venue_inn1_hist[v].append(row["inn1_runs"])

# ── ELO (expanding) ───────────────────────────────────────────────────────
team_elo  = defaultdict(lambda: 1500)
match_elo = {}
for _, row in matches.iterrows():
    t1, t2 = row["team1"], row["team2"]
    e1, e2 = team_elo[t1], team_elo[t2]
    match_elo[row["file_id"]] = {"team1":t1,"team2":t2,"elo1":e1,"elo2":e2}
    won = 1 if row["winner"] == t1 else 0
    exp = 1 / (1 + 10 ** ((e2 - e1) / 400))
    team_elo[t1] = e1 + 24 * (won - exp)
    team_elo[t2] = e2 + 24 * ((1 - won) - (1 - exp))

# ── Team form (expanding) ─────────────────────────────────────────────────
team_form_hist = defaultdict(list)
match_form = {}
for _, row in matches.iterrows():
    t1, t2 = row["team1"], row["team2"]
    f1 = np.mean(team_form_hist[t1][-5:]) if team_form_hist[t1] else 0.5
    f2 = np.mean(team_form_hist[t2][-5:]) if team_form_hist[t2] else 0.5
    match_form[row["file_id"]] = {"team1":t1,"team2":t2,"form1":f1,"form2":f2}
    won = 1 if row["winner"] == t1 else 0
    team_form_hist[t1].append(won)
    team_form_hist[t2].append(1 - won)

fid_to_winner    = dict(zip(matches["file_id"], matches["winner"]))
fid_to_inn1_team = dict(zip(matches["file_id"], matches["inn1_team"]))
fid_to_inn2_team = dict(zip(matches["file_id"], matches["inn2_team"]))
fid_to_inn1_runs = dict(zip(matches["file_id"], matches["inn1_runs"]))
fid_to_season    = dict(zip(matches["file_id"], matches["season"]))

# ══════════════════════════════════════════════════════════════════════════
# BUILD FIRST INNINGS SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════
print("Building 1st innings snapshots...")
inn1_raw = deliveries[deliveries["innings"] == 1].copy()
inn1_raw = inn1_raw[inn1_raw["file_id"].isin(valid_fids)]
inn1_raw = inn1_raw.sort_values(["file_id", "over", "ball_in_over"])
inn1_raw["ball_num"] = inn1_raw.groupby("file_id").cumcount() + 1

inn1_snaps = []
for fid, group in inn1_raw.groupby("file_id"):
    season     = fid_to_season.get(fid, "2010")
    venue_avg  = venue_avg_map.get(fid, 165.0)
    bat1_won   = int(fid_to_inn1_team.get(fid) == fid_to_winner.get(fid))
    elo_info   = match_elo.get(fid, {})
    form_info  = match_form.get(fid, {})
    bat_team   = fid_to_inn1_team.get(fid)

    if bat_team and elo_info:
        if bat_team == elo_info.get("team1"):
            bat_elo, bowl_elo = elo_info["elo1"], elo_info["elo2"]
            bat_form, bowl_form = form_info.get("form1", 0.5), form_info.get("form2", 0.5)
        else:
            bat_elo, bowl_elo = elo_info["elo2"], elo_info["elo1"]
            bat_form, bowl_form = form_info.get("form2", 0.5), form_info.get("form1", 0.5)
    else:
        bat_elo = bowl_elo = 1500
        bat_form = bowl_form = 0.5

    group = group.sort_values(["over", "ball_in_over"])
    boundary_cnt = dot_cnt = 0
    recent_r, recent_w = [], []
    p_runs = p_balls = 0
    pp_runs_at6 = pp_wkts_at6 = 0

    for _, ball in group.iterrows():
        recent_r.append(ball["runs_total"])
        recent_w.append(int(ball["is_wicket"]))
        if len(recent_r) > 18: recent_r.pop(0); recent_w.pop(0)
        if ball["runs_batter"] in [4, 6]: boundary_cnt += 1
        if ball["runs_total"] == 0: dot_cnt += 1
        p_runs += ball["runs_total"]
        p_balls += 1
        if ball["is_wicket"]: p_runs = p_balls = 0

        bn = ball["ball_num"]
        if bn % 6 == 0 and bn <= 120:
            ov = bn // 6
            cr, cw = ball["cum_runs"], ball["cum_wickets"]
            crr = cr / bn * 6 if bn > 0 else 0
            exp_at = venue_avg * (bn / 120)
            proj   = cr + crr * (120 - bn) / 6 if bn < 120 else cr

            if ov == 6:
                pp_runs_at6, pp_wkts_at6 = cr, cw

            accel = 0.0
            if bn >= 36:
                accel = (sum(recent_r) / min(len(recent_r), 18) * 6) - crr

            pp_vs_venue = (pp_runs_at6 / (venue_avg * 6 / 20)) if ov > 6 and venue_avg > 0 else 0.0

            inn1_snaps.append({
                "file_id": fid, "season": season, "over": ov,
                "cum_runs": cr, "cum_wickets": cw, "crr": crr,
                "balls_remaining": 120 - bn, "balls_pct": bn / 120,
                "wickets_pct": cw / 10,
                "projected_score": proj, "venue_avg": venue_avg,
                "score_vs_expected": cr - exp_at,
                "score_vs_expected_pct": cr / exp_at if exp_at > 0 else 1.0,
                "partnership_runs": p_runs, "partnership_balls": p_balls,
                "last_3ov_runs": sum(recent_r), "last_3ov_wkts": sum(recent_w),
                "boundary_pct": boundary_cnt / bn if bn > 0 else 0,
                "dot_pct": dot_cnt / bn if bn > 0 else 0,
                "acceleration": accel,
                "elo_diff": bat_elo - bowl_elo,
                "form_diff": bat_form - bowl_form,
                "is_pp": int(ov <= 6),
                "pp_runs": pp_runs_at6 if ov > 6 else 0,
                "pp_wickets": pp_wkts_at6 if ov > 6 else 0,
                "pp_vs_venue_avg": pp_vs_venue,
                "batting_first_won": bat1_won,
            })

inn1_df = pd.DataFrame(inn1_snaps)
print(f"  {len(inn1_df)} snapshots, {inn1_df['file_id'].nunique()} matches")

# ══════════════════════════════════════════════════════════════════════════
# BUILD SECOND INNINGS SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════════
print("Building 2nd innings snapshots...")
inn2_raw = deliveries[deliveries["innings"] == 2].copy()
inn2_raw = inn2_raw[inn2_raw["file_id"].isin(valid_fids)]
inn2_raw = inn2_raw.sort_values(["file_id", "over", "ball_in_over"])
inn2_raw["ball_num"] = inn2_raw.groupby("file_id").cumcount() + 1

# Pre-compute per-match first innings totals and powerplay stats
inn1_totals = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")["runs_total"].sum().rename("inn1_runs").reset_index()
)
inn1_totals["file_id"] = inn1_totals["file_id"].astype(str)

# 2nd innings powerplay state (at ball 36)
inn2_pp = (
    inn2_raw[inn2_raw["ball_num"] == 36]
    .groupby("file_id")[["cum_runs","cum_wickets"]].last()
    .rename(columns={"cum_runs":"pp_runs","cum_wickets":"pp_wickets"})
    .reset_index()
)
inn2_raw = inn2_raw.merge(inn2_pp, on="file_id", how="left")
inn2_raw["pp_runs"]    = inn2_raw["pp_runs"].fillna(0)
inn2_raw["pp_wickets"] = inn2_raw["pp_wickets"].fillna(0)

inn2_snaps = []
for fid, group in inn2_raw.groupby("file_id"):
    season    = fid_to_season.get(fid, "2010")
    venue_avg = venue_avg_map.get(fid, 165.0)
    target    = (fid_to_inn1_runs.get(fid) or 160) + 1
    chase_won = int(fid_to_inn2_team.get(fid) == fid_to_winner.get(fid))
    elo_info  = match_elo.get(fid, {})
    form_info = match_form.get(fid, {})
    chase_team = fid_to_inn2_team.get(fid)

    if chase_team and elo_info:
        if chase_team == elo_info.get("team1"):
            c_elo, d_elo = elo_info["elo1"], elo_info["elo2"]
            c_form, d_form = form_info.get("form1", 0.5), form_info.get("form2", 0.5)
        else:
            c_elo, d_elo = elo_info["elo2"], elo_info["elo1"]
            c_form, d_form = form_info.get("form2", 0.5), form_info.get("form1", 0.5)
    else:
        c_elo = d_elo = 1500
        c_form = d_form = 0.5

    # Powerplay stats from the pre-merged columns
    pp_r = group["pp_runs"].iloc[0] if len(group) else 0
    pp_w = group["pp_wickets"].iloc[0] if len(group) else 0

    group = group.sort_values(["over", "ball_in_over"])
    boundary_cnt = dot_cnt = 0
    recent_r, recent_w = [], []
    p_runs = p_balls = 0
    first_inn_rr = (target - 1) / 120 * 6

    for _, ball in group.iterrows():
        recent_r.append(ball["runs_total"])
        recent_w.append(int(ball["is_wicket"]))
        if len(recent_r) > 18: recent_r.pop(0); recent_w.pop(0)
        if ball["runs_batter"] in [4, 6]: boundary_cnt += 1
        if ball["runs_total"] == 0: dot_cnt += 1
        p_runs += ball["runs_total"]
        p_balls += 1
        if ball["is_wicket"]: p_runs = p_balls = 0

        bn = ball["ball_num"]
        if bn % 6 == 0 and bn <= 120:
            ov = bn // 6
            cr, cw = ball["cum_runs"], ball["cum_wickets"]
            needed = target - cr
            balls_left = 120 - bn
            crr = cr / bn * 6 if bn > 0 else 0
            rrr = needed / balls_left * 6 if balls_left > 0 else (0 if needed <= 0 else 99.0)

            is_pp       = int(bn <= 36)
            pp_run_rate = pp_r / 36 * 6 if bn > 36 else 0.0
            pp_req_rate = (target - pp_r) / 84 * 6 if bn > 36 else 0.0
            pp_rate_gap = pp_run_rate - pp_req_rate

            inn2_snaps.append({
                "file_id": fid, "season": season, "over": ov,
                "ball_num": bn,
                "cum_runs": cr, "cum_wickets": cw,
                "runs_needed": needed, "balls_remaining": balls_left,
                "crr": crr, "rrr": rrr,
                "rrr_diff": crr - rrr,
                "run_rate_ratio": crr / max(rrr, 0.01),
                "balls_pct": bn / 120, "wickets_pct": cw / 10,
                "partnership_runs": p_runs, "partnership_balls": p_balls,
                "last_3ov_runs": sum(recent_r), "last_3ov_wkts": sum(recent_w),
                "boundary_pct": boundary_cnt / bn if bn > 0 else 0,
                "dot_ball_pct": dot_cnt / bn if bn > 0 else 0,
                "first_innings_run_rate": first_inn_rr,
                "target_vs_venue_avg": target / venue_avg if venue_avg > 0 else 1.0,
                "elo_diff": c_elo - d_elo,
                "form_diff": c_form - d_form,
                "is_pp": is_pp,
                "pp_runs": pp_r if bn > 36 else 0,
                "pp_wickets": pp_w if bn > 36 else 0,
                "pp_run_rate": pp_run_rate,
                "pp_req_rate": pp_req_rate,
                "pp_rate_gap": pp_rate_gap,
                "chasing_won": chase_won,
            })

inn2_df = pd.DataFrame(inn2_snaps)
print(f"  {len(inn2_df)} snapshots, {inn2_df['file_id'].nunique()} matches")

# ══════════════════════════════════════════════════════════════════════════
# FEATURE LISTS
# ══════════════════════════════════════════════════════════════════════════
INN1_FEATURES = [
    "cum_runs", "cum_wickets", "crr", "balls_remaining", "balls_pct", "wickets_pct",
    "projected_score", "venue_avg", "score_vs_expected", "score_vs_expected_pct",
    "partnership_runs", "partnership_balls",
    "last_3ov_runs", "last_3ov_wkts", "boundary_pct", "dot_pct", "acceleration",
    "elo_diff", "form_diff",
    "is_pp", "pp_runs", "pp_wickets", "pp_vs_venue_avg",
]

INN2_FEATURES = [
    "ball_num", "balls_remaining", "balls_pct",
    "cum_runs", "runs_needed",
    "cum_wickets", "wickets_pct",
    "crr", "rrr", "rrr_diff", "run_rate_ratio",
    "partnership_runs", "partnership_balls",
    "last_3ov_runs", "last_3ov_wkts",
    "boundary_pct", "dot_ball_pct",
    "first_innings_run_rate", "target_vs_venue_avg",
    "elo_diff", "form_diff",
    "is_pp", "pp_runs", "pp_wickets", "pp_run_rate", "pp_req_rate", "pp_rate_gap",
]

# ══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST: 2015–2025
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("WALK-FORWARD BACKTEST: 2015–2025")
print("(each season: trained on ALL prior seasons)")
print("="*70)

# Accumulators
inn1_by_over = {ov: {"correct":0,"total":0,"probs":[],"actuals":[]} for ov in range(1,21)}
inn2_by_over = {ov: {"correct":0,"total":0,"probs":[],"actuals":[]} for ov in range(1,21)}
season_results = []

for test_season in TEST_SEASONS:
    yr = int(test_season)

    # ── INN1 ──
    tr1 = inn1_df[inn1_df["season"].astype(int) < yr]
    te1 = inn1_df[inn1_df["season"] == test_season]
    inn1_acc = inn1_matches = 0
    if len(tr1) >= 50 and len(te1) > 0:
        X_tr = tr1[INN1_FEATURES].fillna(0)
        y_tr = tr1["batting_first_won"].astype(int)
        X_te = te1[INN1_FEATURES].fillna(0)
        y_te = te1["batting_first_won"].astype(int)
        sc = StandardScaler()
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=20, random_state=42, verbosity=-1)
        m.fit(sc.fit_transform(X_tr), y_tr)
        p = m.predict_proba(sc.transform(X_te))[:, 1]
        pr = (p >= 0.5).astype(int)
        inn1_acc = accuracy_score(y_te, pr)
        inn1_matches = te1["file_id"].nunique()
        for i, (_, row) in enumerate(te1.iterrows()):
            ov = int(row["over"])
            inn1_by_over[ov]["correct"] += int(pr[i] == int(row["batting_first_won"]))
            inn1_by_over[ov]["total"]   += 1
            inn1_by_over[ov]["probs"].append(float(p[i]))
            inn1_by_over[ov]["actuals"].append(int(row["batting_first_won"]))

    # ── INN2 ──
    tr2 = inn2_df[inn2_df["season"].astype(int) < yr]
    te2 = inn2_df[inn2_df["season"] == test_season]
    inn2_acc = inn2_matches = 0
    if len(tr2) >= 50 and len(te2) > 0:
        X_tr = tr2[INN2_FEATURES].fillna(0)
        y_tr = tr2["chasing_won"].astype(int)
        X_te = te2[INN2_FEATURES].fillna(0)
        y_te = te2["chasing_won"].astype(int)
        sc = StandardScaler()
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03,
                                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=20, random_state=42, verbosity=-1)
        m.fit(sc.fit_transform(X_tr), y_tr)
        p = m.predict_proba(sc.transform(X_te))[:, 1]
        pr = (p >= 0.5).astype(int)
        inn2_acc = accuracy_score(y_te, pr)
        inn2_matches = te2["file_id"].nunique()
        for i, (_, row) in enumerate(te2.iterrows()):
            ov = int(row["over"])
            inn2_by_over[ov]["correct"] += int(pr[i] == int(row["chasing_won"]))
            inn2_by_over[ov]["total"]   += 1
            inn2_by_over[ov]["probs"].append(float(p[i]))
            inn2_by_over[ov]["actuals"].append(int(row["chasing_won"]))

    season_results.append({
        "season": test_season,
        "inn1_acc": inn1_acc, "inn1_matches": inn1_matches,
        "inn2_acc": inn2_acc, "inn2_matches": inn2_matches,
    })
    print(f"  {test_season}: 1st inn {inn1_acc*100:5.1f}% ({inn1_matches}m) | "
          f"2nd inn {inn2_acc*100:5.1f}% ({inn2_matches}m)")

# ══════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PER-SEASON SUMMARY")
print("="*70)
print(f"{'Season':>7} | {'Inn1 Acc':>9} | {'Inn1 M':>7} | {'Inn2 Acc':>9} | {'Inn2 M':>7}")
print("-" * 55)
i1_tots = i1_ms = i2_tots = i2_ms = 0
for r in season_results:
    if r["inn1_matches"]:
        print(f"  {r['season']:>5}  | {r['inn1_acc']*100:>7.1f}%  | {r['inn1_matches']:>5}    | "
              f"{r['inn2_acc']*100:>7.1f}%  | {r['inn2_matches']:>5}")
        i1_tots += r["inn1_acc"] * r["inn1_matches"]
        i1_ms   += r["inn1_matches"]
        i2_tots += r["inn2_acc"] * r["inn2_matches"]
        i2_ms   += r["inn2_matches"]
print("-" * 55)
print(f"  {'TOTAL':>5}  | {i1_tots/i1_ms*100:>7.1f}%  | {i1_ms:>5}    | "
      f"{i2_tots/i2_ms*100:>7.1f}%  | {i2_ms:>5}")

print("\n" + "="*70)
print("1ST INNINGS — ACCURACY BY OVER (2015–2025, walk-forward)")
print("="*70)
print(f"{'Over':>4} | {'Accuracy':>8} | {'LogLoss':>8} | {'Samples':>7}")
print("-" * 40)
for ov in range(1, 21):
    r = inn1_by_over[ov]
    if r["total"] < 10: continue
    acc = r["correct"] / r["total"]
    try:
        ll = log_loss(r["actuals"], r["probs"])
    except: ll = 0
    bar = "#" * int(acc * 35)
    print(f"  {ov:>2}  | {acc*100:>6.1f}%  | {ll:>7.4f}  | {r['total']:>5}   {bar}")

print("\n" + "="*70)
print("2ND INNINGS — ACCURACY BY OVER (2015–2025, walk-forward)")
print("Powerplay overs 1–6: pp features = 0 (no signal yet)")
print("Post-powerplay 7–20: pp_rate_gap active")
print("="*70)
print(f"{'Over':>4} | {'Accuracy':>8} | {'LogLoss':>8} | {'Samples':>7}")
print("-" * 40)
for ov in range(1, 21):
    r = inn2_by_over[ov]
    if r["total"] < 10: continue
    acc = r["correct"] / r["total"]
    try:
        ll = log_loss(r["actuals"], r["probs"])
    except: ll = 0
    marker = " ← pp active" if ov == 7 else ""
    bar = "#" * int(acc * 35)
    print(f"  {ov:>2}  | {acc*100:>6.1f}%  | {ll:>7.4f}  | {r['total']:>5}   {bar}{marker}")

# Combined over-by-over match timeline
print("\n" + "="*70)
print("FULL MATCH TIMELINE — ACCURACY PROGRESSION")
print("="*70)
all_i1 = sum(r["correct"] for r in inn1_by_over.values())
all_i1_t = sum(r["total"] for r in inn1_by_over.values())
all_i2 = sum(r["correct"] for r in inn2_by_over.values())
all_i2_t = sum(r["total"] for r in inn2_by_over.values())
print(f"  1st innings overall: {all_i1/all_i1_t*100:.1f}% ({all_i1_t} predictions)")
print(f"  2nd innings overall: {all_i2/all_i2_t*100:.1f}% ({all_i2_t} predictions)")

# pp_rate_gap impact: compare over 6 vs overs 7–10
r6  = inn2_by_over[6]
r7  = inn2_by_over[7]
r8  = inn2_by_over[8]
r10 = inn2_by_over[10]
if r6["total"] and r7["total"]:
    acc6  = r6["correct"]  / r6["total"]
    acc7  = r7["correct"]  / r7["total"]
    acc10 = r10["correct"] / r10["total"]
    print(f"\n  Accuracy jump when pp_rate_gap activates:")
    print(f"    Over 6  (pp just ended, no gap yet): {acc6*100:.1f}%")
    print(f"    Over 7  (first over with pp_rate_gap): {acc7*100:.1f}%")
    print(f"    Over 10 (pp context fully embedded):   {acc10*100:.1f}%")

print("\nDone.")
