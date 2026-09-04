"""
backtest_5yr.py — 5-year historical backtest (2021-2025)

Models tested:
  1. Pre-match  — prediction before toss (toss features zeroed out)
  2. Post-toss  — prediction after toss (actual toss features from history)
  3. Unified Live Inn1 @Over10 — unified model at over 10 of 1st innings
  4. Unified Live Inn2 @Over10 — unified model at over 10 of 2nd innings

Uses match_features.csv (historical ELO/form per match, no leakage) and
deliveries.csv (ball-by-ball) — does NOT call the live API.

Note:
  2021-2022 = in-sample (model trained on these)  [marked IS]
  2023-2025 = genuine out-of-sample test           [marked OOS]
"""

import pickle, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

SEASONS = ["2021", "2022", "2023", "2024", "2025"]
OOS     = {"2023", "2024", "2025"}   # out-of-sample seasons

# ── Load models ───────────────────────────────────────────────────────
print("Loading models...")
with open("models/prematch_model.pkl",    "rb") as f: pre_bundle  = pickle.load(f)
with open("models/unified_live_model.pkl","rb") as f: uni_bundle  = pickle.load(f)

pre_model    = pre_bundle["model"]
pre_features = pre_bundle["features"]
pre_median   = pre_bundle.get("train_median", {})

uni_model    = uni_bundle["model"]
uni_scaler   = uni_bundle["scaler"]
uni_features = uni_bundle["features"]

# ── Load data ─────────────────────────────────────────────────────────
print("Loading data...")
mf = pd.read_csv("data/match_features.csv")
mf["season"]  = mf["season"].astype(str)
mf["file_id"] = mf["file_id"].astype(str)
mf = mf[mf["season"].isin(SEASONS)].copy()

matches = pd.read_csv("data/matches.csv")
matches["season"]  = matches["season"].astype(str)
matches["file_id"] = matches["file_id"].astype(str)
matches = matches[matches["season"].isin(SEASONS)].copy()

deliveries = pd.read_csv("data/deliveries.csv")
deliveries["season"]  = deliveries["date"].str[:4]
deliveries["file_id"] = deliveries["file_id"].astype(str)
deliveries = deliveries[deliveries["season"].isin(SEASONS)].copy()
deliveries = deliveries.sort_values(["file_id", "innings", "over", "ball_in_over"])

print(f"  Match features: {len(mf)} rows")
print(f"  Matches: {len(matches)} rows")
print(f"  Deliveries: {len(deliveries)} rows")

# ── Pre-process deliveries: ball_num per innings per match ─────────────
deliveries["ball_num"] = deliveries.groupby(["file_id","innings"]).cumcount() + 1

# Powerplay snapshot (ball 36 = end of over 6)
pp_snap = (
    deliveries[deliveries["ball_num"] == 36]
    .groupby(["file_id","innings"])[["cum_runs","cum_wickets","batting_team","bowling_team"]]
    .last().reset_index()
    .rename(columns={"cum_runs":"pp_runs","cum_wickets":"pp_wkts"})
)

# Over-10 snapshot (ball 60 = end of over 10)
ov10_snap = (
    deliveries[deliveries["ball_num"] == 60]
    .groupby(["file_id","innings"])[["cum_runs","cum_wickets","batting_team","bowling_team"]]
    .last().reset_index()
    .rename(columns={"cum_runs":"ov10_runs","cum_wickets":"ov10_wkts"})
)

# Momentum at ball 60: last-18-ball runs/wkts and partnership
print("Computing momentum at over 10...")
mom_rows = []
for (fid, inn), grp in deliveries[deliveries["innings"].isin([1,2])].groupby(["file_id","innings"]):
    grp = grp.sort_values("ball_num")
    cr_r, cr_w, bd, dt = 0, 0, 0, 0
    pr_r, pr_b = 0, 0
    recent_r, recent_w = [], []
    last_pr_r, last_pr_b = 0, 0
    for _, row in grp.iterrows():
        bn = row["ball_num"]
        cr_r += row["runs_total"]
        cr_w += int(row["is_wicket"])
        if row["runs_batter"] >= 4: bd += 1
        if row["runs_total"] == 0:  dt += 1
        pr_r += row["runs_total"]
        pr_b += 1
        recent_r.append(row["runs_total"])
        recent_w.append(int(row["is_wicket"]))
        if len(recent_r) > 18: recent_r.pop(0); recent_w.pop(0)
        if row["is_wicket"]: pr_r = 0; pr_b = 0
        if bn == 60:
            mom_rows.append({
                "file_id": fid, "innings": inn,
                "partnership_runs": pr_r, "partnership_balls": pr_b,
                "last_3ov_runs": sum(recent_r), "last_3ov_wkts": sum(recent_w),
                "boundary_pct": bd / bn, "dot_pct": dt / bn,
            })
            break

mom_df = pd.DataFrame(mom_rows)

# ── Join all at-over-10 data together ────────────────────────────────
ov10 = ov10_snap.merge(pp_snap, on=["file_id","innings"], how="left",
                       suffixes=("","_pp"))
ov10 = ov10.merge(mom_df, on=["file_id","innings"], how="left")

# inn1 final totals (for target in Inn2)
inn1_final = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")
    .agg(inn1_final_runs=("cum_runs","max"),
         inn1_final_wkts=("cum_wickets","max"))
    .reset_index()
)

# ── Venue averages (at match time, already in mf) ─────────────────────
venue_avg_map = dict(zip(mf["file_id"], mf["venue_avg_first_innings"].fillna(160)))

# ── PP wicket win rates (per-match expanding window) ──────────────────
_pp_wr_by_match = pd.read_csv("data/pp_wicket_win_rates_by_match.csv")
_pp_wr_by_match["file_id"] = _pp_wr_by_match["file_id"].astype(str)
_pp_wr_lookup = {}
for _, r in _pp_wr_by_match.iterrows():
    key = (str(r["file_id"]), int(r["innings"]), r["phase"])
    _pp_wr_lookup[key] = r["prior_win_rate"] if pd.notna(r["prior_win_rate"]) else 0.0

def _bt_phase_wkt_wr(file_id, innings, over_num):
    """For backtest: get team_phase_wkt_wr at a given over."""
    if over_num <= 6:
        return 0.0
    elif over_num <= 15:
        return _pp_wr_lookup.get((str(file_id), int(innings), "pp"), 0.0)
    else:
        return _pp_wr_lookup.get((str(file_id), int(innings), "middle"), 0.0)

print(f"  PP wicket WR lookup: {len(_pp_wr_lookup)} entries")

# ── Features for pre-match model ─────────────────────────────────────
TOSS_ZERO = {
    "team1_won_toss": 0, "toss_chose_bat": 0,
    "team1_bats_second": 0, "toss_venue_aligned": 0,
    "team1_chase_advantage": 0, "team2_chase_advantage": 0,
    "chase_advantage_diff": 0, "early_chase_boost": 0,
    "venue_chase_batting_second": 0,
}

def prematch_prob(row, use_toss=False):
    """P(team1 wins) from pre-match model using match_features row."""
    feats = {f: row.get(f, pre_median.get(f, 0)) for f in pre_features}
    if not use_toss:
        feats.update(TOSS_ZERO)
    X = np.array([[feats.get(f, 0) for f in pre_features]])
    return float(pre_model.predict_proba(X)[0, 1])

# ── Unified live model helper ─────────────────────────────────────────
def unified_prob(innings, inn1_runs_final, inn1_wkts_final,
                 inn2_runs, inn2_wkts, balls, target, venue_avg,
                 elo_diff, form_diff, venue_bfwr, venue_cwr,
                 partnership_runs, partnership_balls,
                 last_3ov_runs, last_3ov_wkts,
                 boundary_pct, dot_pct,
                 pp_runs, pp_wkts,
                 team_phase_wkt_wr=0.0):
    """P(bat_first wins)."""
    crr = inn2_runs / balls * 6 if balls > 0 else 0.0
    pp_run_rate = pp_runs / 36 * 6 if balls > 36 and pp_runs else 0.0

    if innings == 1:
        expected_at = venue_avg * (balls / 120)
        projected   = inn1_runs_final + crr * (120 - balls) / 6
        feats = {
            "current_innings": 1, "innings_balls": balls,
            "innings_balls_rem": 120 - balls, "innings_balls_pct": balls / 120,
            "inn1_runs": inn1_runs_final, "inn1_wickets": inn1_wkts_final,
            "inn1_crr": crr, "inn1_projected": projected,
            "inn1_vs_avg": inn1_runs_final - expected_at,
            "inn1_vs_avg_pct": inn1_runs_final / expected_at if expected_at > 0 else 1.0,
            "inn1_balls_pct": balls / 120,
            "inn1_acceleration": (last_3ov_runs / 18 * 6 - crr) if balls >= 36 else 0.0,
            "inn2_runs": 0, "inn2_wickets": 0, "inn2_crr": 0.0,
            "inn2_rrr": 0.0, "inn2_rrr_diff": 0.0, "inn2_run_rate_ratio": 0.0,
            "inn2_runs_needed": 0, "inn2_balls_rem": 0, "inn2_balls_pct": 0.0,
            "first_innings_wickets": 0, "target": 0, "target_vs_venue_avg": 0.0,
            "pp_req_rate": 0.0, "pp_rate_gap": 0.0,
        }
    else:
        balls_rem   = max(0, 120 - balls)
        runs_needed = max(0, target - inn2_runs)
        rrr         = runs_needed / balls_rem * 6 if balls_rem > 0 else 99.0
        rrr_diff    = rrr - crr
        rr_ratio    = min(crr / rrr if rrr > 0 else 1.0, 3.0)
        pp_req_rate = (target - (pp_runs or 0)) / 84 * 6 if balls > 36 else 0.0
        pp_rate_gap = pp_run_rate - pp_req_rate if balls > 36 else 0.0
        feats = {
            "current_innings": 2, "innings_balls": balls,
            "innings_balls_rem": balls_rem, "innings_balls_pct": balls / 120,
            "inn1_runs": inn1_runs_final, "inn1_wickets": inn1_wkts_final,
            "inn1_crr": inn1_runs_final / 120 * 6,
            "inn1_projected": 0.0,
            "inn1_vs_avg": inn1_runs_final - venue_avg,
            "inn1_vs_avg_pct": inn1_runs_final / venue_avg if venue_avg > 0 else 1.0,
            "inn1_balls_pct": 1.0, "inn1_acceleration": 0.0,
            "inn2_runs": inn2_runs, "inn2_wickets": inn2_wkts,
            "inn2_crr": crr, "inn2_rrr": rrr, "inn2_rrr_diff": rrr_diff,
            "inn2_run_rate_ratio": rr_ratio,
            "inn2_runs_needed": runs_needed, "inn2_balls_rem": balls_rem,
            "inn2_balls_pct": balls / 120,
            "first_innings_wickets": int(inn1_wkts_final),
            "target": target,
            "target_vs_venue_avg": target / venue_avg if venue_avg > 0 else 1.0,
            "pp_req_rate": pp_req_rate, "pp_rate_gap": pp_rate_gap,
        }

    feats.update({
        "partnership_runs": partnership_runs or 0,
        "partnership_balls": max(partnership_balls or 1, 1),
        "last_3ov_runs": last_3ov_runs or 0,
        "last_3ov_wkts": last_3ov_wkts or 0,
        "boundary_pct": boundary_pct or 0.25,
        "dot_pct": dot_pct or 0.42,
        "is_pp": int(balls <= 36),
        "pp_runs": (pp_runs or 0) if balls > 36 else 0,
        "pp_wickets": (pp_wkts or 0) if balls > 36 else 0,
        "pp_run_rate": pp_run_rate,
        "elo_diff": elo_diff, "form_diff": form_diff,
        "venue_avg": venue_avg,
        "venue_bat_first_win_rate": venue_bfwr,
        "venue_chase_win_rate": venue_cwr,
        "team_phase_wkt_wr": team_phase_wkt_wr,
    })

    X   = np.array([[feats.get(f, 0) for f in uni_features]])
    X_s = uni_scaler.transform(X)
    return float(uni_model.predict_proba(X_s)[0, 1])


# ══════════════════════════════════════════════════════════════════════
# RUN BACKTEST
# ══════════════════════════════════════════════════════════════════════
print("\nRunning backtest...")

results = []
skipped = 0

# Join match_features with matches for inn1_team and toss info
mf_extra = mf.merge(
    matches[["file_id","inn1_team","inn2_team","toss_winner","toss_decision",
             "inn1_runs","inn1_wickets","inn2_runs","inn2_wickets"]],
    on="file_id", how="left"
)

# Join ov10 data
ov10_inn1 = ov10[ov10["innings"] == 1].rename(
    columns={"ov10_runs":"i1_ov10_r","ov10_wkts":"i1_ov10_w",
             "pp_runs":"i1_pp_r","pp_wkts":"i1_pp_w",
             "batting_team":"i1_bat","bowling_team":"i1_bowl",
             "partnership_runs":"i1_par_r","partnership_balls":"i1_par_b",
             "last_3ov_runs":"i1_l3r","last_3ov_wkts":"i1_l3w",
             "boundary_pct":"i1_bpct","dot_pct":"i1_dpct"})
ov10_inn2 = ov10[ov10["innings"] == 2].rename(
    columns={"ov10_runs":"i2_ov10_r","ov10_wkts":"i2_ov10_w",
             "pp_runs":"i2_pp_r","pp_wkts":"i2_pp_w",
             "batting_team":"i2_bat","bowling_team":"i2_bowl",
             "partnership_runs":"i2_par_r","partnership_balls":"i2_par_b",
             "last_3ov_runs":"i2_l3r","last_3ov_wkts":"i2_l3w",
             "boundary_pct":"i2_bpct","dot_pct":"i2_dpct"})

mf_extra = mf_extra.merge(ov10_inn1[["file_id","i1_ov10_r","i1_ov10_w",
    "i1_pp_r","i1_pp_w","i1_par_r","i1_par_b","i1_l3r","i1_l3w","i1_bpct","i1_dpct"]],
    on="file_id", how="left")
mf_extra = mf_extra.merge(ov10_inn2[["file_id","i2_ov10_r","i2_ov10_w",
    "i2_pp_r","i2_pp_w","i2_par_r","i2_par_b","i2_l3r","i2_l3w","i2_bpct","i2_dpct"]],
    on="file_id", how="left")
mf_extra = mf_extra.merge(inn1_final, on="file_id", how="left")

for _, row in mf_extra.iterrows():
    fid    = row["file_id"]
    season = str(row["season"])
    winner = row.get("winner")
    team1  = row["team1"]
    team2  = row["team2"]
    inn1_t = row.get("inn1_team")
    if pd.isna(winner) or pd.isna(inn1_t):
        skipped += 1
        continue

    bat_first_won = int(inn1_t == winner)
    team1_won     = int(team1 == winner)
    va            = float(row.get("venue_avg_first_innings") or 160.0)
    vbfwr         = float(row.get("venue_bat_first_win_rate") or 0.5)
    vcwr          = float(row.get("venue_chase_win_rate") or 0.5)
    elo_d         = float(row.get("elo_diff") or 0)   # team1 vs team2
    form_d        = float(row.get("form_diff") or 0)
    # From inn1_team perspective
    if inn1_t == team1:
        elo_d_bf  = elo_d
        form_d_bf = form_d
    else:
        elo_d_bf  = -elo_d
        form_d_bf = -form_d

    # ── Pre-match prediction ──────────────────────────────────────────
    try:
        p1_pre = prematch_prob(row, use_toss=False)
        pre_pred_t1_wins = int(p1_pre >= 0.5)
        pre_correct = int(pre_pred_t1_wins == team1_won)
        pre_prob = p1_pre if pre_pred_t1_wins else 1 - p1_pre
    except Exception:
        pre_correct = None; pre_prob = None

    # ── Post-toss prediction (pre-match model + toss features) ───────
    try:
        p1_pt = prematch_prob(row, use_toss=True)
        pt_pred_t1_wins = int(p1_pt >= 0.5)
        pt_correct = int(pt_pred_t1_wins == team1_won)
        pt_prob = p1_pt if pt_pred_t1_wins else 1 - p1_pt
    except Exception:
        pt_correct = None; pt_prob = None

    # ── Unified live Inn1 @Over10 ─────────────────────────────────────
    live1_correct = None; live1_prob = None
    i1r  = row.get("i1_ov10_r")
    i1w  = row.get("i1_ov10_w")
    if pd.notna(i1r) and pd.notna(i1w):
        try:
            p_bf = unified_prob(
                innings=1,
                inn1_runs_final=int(i1r), inn1_wkts_final=int(i1w),
                inn2_runs=0, inn2_wkts=0, balls=60,
                target=0, venue_avg=va,
                elo_diff=elo_d_bf, form_diff=form_d_bf,
                venue_bfwr=vbfwr, venue_cwr=vcwr,
                partnership_runs=row.get("i1_par_r"),
                partnership_balls=row.get("i1_par_b"),
                last_3ov_runs=row.get("i1_l3r"),
                last_3ov_wkts=row.get("i1_l3w"),
                boundary_pct=row.get("i1_bpct"),
                dot_pct=row.get("i1_dpct"),
                pp_runs=row.get("i1_pp_r"),
                pp_wkts=row.get("i1_pp_w"),
                team_phase_wkt_wr=_bt_phase_wkt_wr(fid, 1, 10),
            )
            live1_pred_bf_wins = int(p_bf >= 0.5)
            live1_correct = int(live1_pred_bf_wins == bat_first_won)
            live1_prob = p_bf if live1_pred_bf_wins else 1 - p_bf
        except Exception:
            pass

    # ── Unified live Inn2 @Over10 ─────────────────────────────────────
    live2_correct = None; live2_prob = None
    i2r  = row.get("i2_ov10_r")
    i2w  = row.get("i2_ov10_w")
    i1fr = row.get("inn1_final_runs")
    i1fw = row.get("inn1_final_wkts")
    if pd.notna(i2r) and pd.notna(i2w) and pd.notna(i1fr):
        try:
            target = int(i1fr) + 1
            p_bf = unified_prob(
                innings=2,
                inn1_runs_final=int(i1fr), inn1_wkts_final=int(i1fw or 0),
                inn2_runs=int(i2r), inn2_wkts=int(i2w), balls=60,
                target=target, venue_avg=va,
                elo_diff=elo_d_bf, form_diff=form_d_bf,
                venue_bfwr=vbfwr, venue_cwr=vcwr,
                partnership_runs=row.get("i2_par_r"),
                partnership_balls=row.get("i2_par_b"),
                last_3ov_runs=row.get("i2_l3r"),
                last_3ov_wkts=row.get("i2_l3w"),
                boundary_pct=row.get("i2_bpct"),
                dot_pct=row.get("i2_dpct"),
                pp_runs=row.get("i2_pp_r"),
                pp_wkts=row.get("i2_pp_w"),
                team_phase_wkt_wr=_bt_phase_wkt_wr(fid, 2, 10),
            )
            live2_pred_bf_wins = int(p_bf >= 0.5)
            live2_correct = int(live2_pred_bf_wins == bat_first_won)
            live2_prob = p_bf if live2_pred_bf_wins else 1 - p_bf
        except Exception:
            pass

    results.append({
        "file_id": fid, "season": season,
        "team1": team1, "team2": team2, "winner": winner,
        "oos": season in OOS,
        "pre_correct": pre_correct, "pre_prob": pre_prob,
        "pt_correct":  pt_correct,  "pt_prob":  pt_prob,
        "live1_correct": live1_correct, "live1_prob": live1_prob,
        "live2_correct": live2_correct, "live2_prob": live2_prob,
    })

df = pd.DataFrame(results)
print(f"  Processed: {len(df)} matches | Skipped: {skipped}")

# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════
W = 72
print("\n" + "=" * W)
print("5-YEAR BACKTEST RESULTS — IPL 2021-2025")
print("IS = in-sample (trained on) | OOS = out-of-sample (genuine test)")
print("=" * W)

def pct(s, col):
    v = s[col].dropna()
    if len(v) == 0: return "  N/A  "
    return f"{v.mean()*100:5.1f}% ({int(v.sum())}/{len(v)})"

# ── By season ─────────────────────────────────────────────────────────
print(f"\n{'Season':>7} {'Tag':>4}  {'Pre-Match':>14}  {'Post-Toss':>14}  {'Live Inn1@10':>14}  {'Live Inn2@10':>14}")
print("-" * W)
for season in SEASONS:
    s = df[df["season"] == season]
    tag = "OOS" if season in OOS else "IS"
    print(f"{season:>7}  {tag:>3}  "
          f"{pct(s,'pre_correct'):>14}  "
          f"{pct(s,'pt_correct'):>14}  "
          f"{pct(s,'live1_correct'):>14}  "
          f"{pct(s,'live2_correct'):>14}")

# ── OOS summary (2023-2025) ───────────────────────────────────────────
oos = df[df["oos"]]
print("-" * W)
print(f"{'OOS 23-25':>7}  {'***':>3}  "
      f"{pct(oos,'pre_correct'):>14}  "
      f"{pct(oos,'pt_correct'):>14}  "
      f"{pct(oos,'live1_correct'):>14}  "
      f"{pct(oos,'live2_correct'):>14}")

# ── Overall (all 5 years) ─────────────────────────────────────────────
print(f"{'Overall':>7}  {'   ':>3}  "
      f"{pct(df,'pre_correct'):>14}  "
      f"{pct(df,'pt_correct'):>14}  "
      f"{pct(df,'live1_correct'):>14}  "
      f"{pct(df,'live2_correct'):>14}")
print("=" * W)

# ── Average confidence (correct vs wrong) ─────────────────────────────
print("\nAverage confidence (predicted probability for chosen winner):")
print(f"{'':>12}  {'Correct':>10}  {'Wrong':>10}  {'Gap':>8}")
for col, label in [("pre","Pre-Match"),("pt","Post-Toss"),
                   ("live1","Live Inn1"),("live2","Live Inn2")]:
    prob_col = f"{col}_prob"
    corr_col = f"{col}_correct"
    sub = df[[corr_col, prob_col]].dropna()
    if len(sub) == 0: continue
    correct_conf = sub[sub[corr_col]==1][prob_col].mean() * 100
    wrong_conf   = sub[sub[corr_col]==0][prob_col].mean() * 100
    print(f"  {label:<12}  {correct_conf:>9.1f}%  {wrong_conf:>9.1f}%  {correct_conf-wrong_conf:>7.1f}pp")

# ── Save ──────────────────────────────────────────────────────────────
df.to_csv("data/backtest_5yr.csv", index=False)
print(f"\nDetailed results saved to data/backtest_5yr.csv")
