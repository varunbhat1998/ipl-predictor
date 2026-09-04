"""
backtest_2024_25.py — Comprehensive backtest of all 3 models on 2024-2025 data

Models tested:
  1. Pre-toss   — no toss info, estimated XI (simulates pre-match prediction)
  2. Post-toss  — actual toss + actual XI
  3. Live Unified — at multiple over checkpoints (Inn1: 3,6,10,15,20; Inn2: 3,6,10,15)

All 2024-2025 data is genuine out-of-sample (models trained on ≤2022).
"""

import pickle, numpy as np, pandas as pd, warnings, os, sys
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# -- Load models ------------------------------------------------------
print("Loading models...")
pre_b    = pickle.load(open("models/prematch_model.pkl","rb"))
post_b   = pickle.load(open("models/posttoss_model.pkl","rb"))
uni_b    = pickle.load(open("models/unified_live_model.pkl","rb"))

# -- Load data --------------------------------------------------------
print("Loading data...")
matches = pd.read_csv("data/matches.csv")
matches["season"] = matches["season"].astype(str)
matches["file_id"] = matches["file_id"].astype(str)

mf = pd.read_csv("data/match_features.csv")
mf["season"] = mf["season"].astype(str)
mf["file_id"] = mf["file_id"].astype(str)

# Load ALL deliveries — needed for expanding-window player score pipeline
TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils":            "Delhi Capitals",
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
    "Punjab Kings":                "Kings XI Punjab",
}
del_df = pd.read_csv("data/deliveries.csv")
del_df["file_id"] = del_df["file_id"].astype(str)
del_df["date"]    = pd.to_datetime(del_df["date"])
for col in ["batting_team", "bowling_team", "winner"]:
    if col in del_df.columns:
        del_df[col] = del_df[col].replace(TEAM_NAME_MAP)

# Filter to 2024-2025
SEASONS = ["2024", "2025"]
matches = matches[matches["season"].isin(SEASONS)].copy()
mf_all  = mf.copy()  # Keep full for lookups
mf      = mf[mf["season"].astype(str).isin(SEASONS)].copy()

# deliveries filtered to 2024-25 for snapshot computation
deliveries = del_df[del_df["date"].dt.year.astype(str).isin(SEASONS)].copy()
deliveries = deliveries.sort_values(["file_id","innings","over","ball_in_over"])
deliveries["ball_num"] = deliveries.groupby(["file_id","innings"]).cumcount() + 1

# PP wicket win rates
pp_wr_bm = pd.read_csv("data/pp_wicket_win_rates_by_match.csv")
pp_wr_bm["file_id"] = pp_wr_bm["file_id"].astype(str)
pp_wr_lookup = {}
for _, r in pp_wr_bm.iterrows():
    pp_wr_lookup[(str(r["file_id"]), int(r["innings"]), r["phase"])] = (
        r["prior_win_rate"] if pd.notna(r["prior_win_rate"]) else 0.0
    )

# Final cumulative PP WR (for pre-toss / cases not in per-match)
pp_wr_final = pd.read_csv("data/pp_wicket_win_rates.csv")

# H2H batter-bowler matchup lookup — latest expanding-window stats per (batter, bowler)
h2h_bvb_bt = {}  # (batter, bowler) -> matchup_adv_final
_h2h_path = "data/h2h_matchup_matrix.csv"
if os.path.exists(_h2h_path):
    _h2h_df = pd.read_csv(_h2h_path)
    # Use only rows where file_id is BEFORE 2024 (expanding window: no leakage into OOS period)
    _h2h_df["date"] = pd.to_datetime(_h2h_df["date"])
    _h2h_pre2024 = _h2h_df[_h2h_df["date"].dt.year < 2024]
    _h2h_latest = _h2h_pre2024.sort_values("date").groupby(["batter", "bowler"]).last().reset_index()
    for _, r in _h2h_latest.iterrows():
        if pd.notna(r["matchup_adv_final"]):
            h2h_bvb_bt[(r["batter"], r["bowler"])] = float(r["matchup_adv_final"])
    print(f"  H2H pairs (pre-2024, for OOS backtest): {len(h2h_bvb_bt)}")

def _match_adv(bf_players, bs_players, top_bat=6, top_bowl=4):
    advs = [h2h_bvb_bt.get((b, w), 0.5)
            for b in (bf_players or [])[:top_bat]
            for w in (bs_players or [])[:top_bowl]]
    return float(np.mean(advs)) if advs else 0.5
pp_wr_final_map = {}
for _, r in pp_wr_final.iterrows():
    pp_wr_final_map[(r["team"], r["phase"], int(r["wicket_bucket"]), r["role"])] = (
        r["win_rate"] if pd.notna(r["win_rate"]) else 0.0
    )

print(f"  Matches: {len(matches)} | Deliveries (2024-25): {len(deliveries)} | Total del: {len(del_df)}")

# ── Expanding-window player score pipeline (same as backtest_posttoss.py) ─────
# Uses ALL historical deliveries so 2024-25 scores are genuinely out-of-sample.
print("Computing expanding-window player scores from all deliveries...")
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

# Current-season expanding window (within-season, before this match)
bat_inns_s = bat_inns.copy()
bat_inns_s["season"] = bat_inns_s["date"].dt.year
bat_inns_s = bat_inns_s.sort_values(["batter", "season", "date", "file_id"])
bat_inns_s["s_cum_r"] = bat_inns_s.groupby(["batter","season"])["runs"].cumsum()
bat_inns_s["s_cum_b"] = bat_inns_s.groupby(["batter","season"])["balls"].cumsum()
bat_inns_s["s_cum_i"] = bat_inns_s.groupby(["batter","season"]).cumcount() + 1
bat_inns_s["s_prev_r"] = bat_inns_s.groupby(["batter","season"])["s_cum_r"].shift(1, fill_value=0)
bat_inns_s["s_prev_b"] = bat_inns_s.groupby(["batter","season"])["s_cum_b"].shift(1, fill_value=0)
bat_inns_s["s_prev_i"] = bat_inns_s.groupby(["batter","season"])["s_cum_i"].shift(1, fill_value=0)
bat_inns_s["season_avg"] = np.where(bat_inns_s["s_prev_i"] >= 2, bat_inns_s["s_prev_r"] / bat_inns_s["s_prev_i"].clip(1), np.nan)
bat_inns_s["season_sr"]  = np.where(bat_inns_s["s_prev_b"] >= 10, bat_inns_s["s_prev_r"] / bat_inns_s["s_prev_b"].clip(1) * 100, np.nan)
bat_inns = bat_inns.merge(bat_inns_s[["batter","file_id","season_avg","season_sr"]], on=["batter","file_id"], how="left")

def compute_bat_score(row):
    if row["prev_inns"] < 1: return np.nan
    career = (row["career_avg"] / 40) * 0.5 + (row["career_sr"] / 150) * 0.5
    fa = row["form5_avg"] if pd.notna(row["form5_avg"]) else row["career_avg"]
    fs = row["form5_sr"] if pd.notna(row["form5_sr"]) and row["form5_sr"] > 0 else row["career_sr"]
    form = (fa / 40) * 0.5 + (fs / 150) * 0.5
    if pd.notna(row.get("season_avg")) and pd.notna(row.get("season_sr")):
        season = (row["season_avg"] / 40) * 0.5 + (row["season_sr"] / 150) * 0.5
        return max(0, (0.35 * career + 0.30 * form + 0.35 * season) * 100)
    return max(0, (0.50 * career + 0.50 * form) * 100)

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

# Current-season bowling stats
bowl_inns_s = bowl_inns.copy()
bowl_inns_s["season"] = bowl_inns_s["date"].dt.year
bowl_inns_s = bowl_inns_s.sort_values(["bowler","season","date","file_id"])
bowl_inns_s["s_cum_rc"] = bowl_inns_s.groupby(["bowler","season"])["runs_conceded"].cumsum()
bowl_inns_s["s_cum_bb"] = bowl_inns_s.groupby(["bowler","season"])["balls_bowled"].cumsum()
bowl_inns_s["s_cum_wk"] = bowl_inns_s.groupby(["bowler","season"])["wickets"].cumsum()
bowl_inns_s["s_cum_in"] = bowl_inns_s.groupby(["bowler","season"]).cumcount() + 1
bowl_inns_s["s_prev_rc"] = bowl_inns_s.groupby(["bowler","season"])["s_cum_rc"].shift(1, fill_value=0)
bowl_inns_s["s_prev_bb"] = bowl_inns_s.groupby(["bowler","season"])["s_cum_bb"].shift(1, fill_value=0)
bowl_inns_s["s_prev_wk"] = bowl_inns_s.groupby(["bowler","season"])["s_cum_wk"].shift(1, fill_value=0)
bowl_inns_s["s_prev_in"] = bowl_inns_s.groupby(["bowler","season"])["s_cum_in"].shift(1, fill_value=0)
bowl_inns_s["season_econ"]     = np.where(bowl_inns_s["s_prev_bb"] >= 6, bowl_inns_s["s_prev_rc"] / (bowl_inns_s["s_prev_bb"] / 6), np.nan)
bowl_inns_s["season_wkt_rate"] = np.where(bowl_inns_s["s_prev_in"] >= 2, bowl_inns_s["s_prev_wk"] / bowl_inns_s["s_prev_in"].clip(1), np.nan)
bowl_inns = bowl_inns.merge(bowl_inns_s[["bowler","file_id","season_econ","season_wkt_rate"]], on=["bowler","file_id"], how="left")

def compute_bowl_score(row):
    # Non-bowlers: fewer than 60 career balls → NaN (excluded from lookup)
    if row["prev_balls_b"] < 60: return np.nan
    # /4 denominator: gives realistic spread for T20 (6–10 RPO range)
    c_econ = min(1.0, max(0, (10 - row["career_econ"]) / 4))
    c_wkt  = min(row["career_wkt_rate"] / 2, 1)
    career = c_econ * 0.5 + c_wkt * 0.5
    f_econ = min(1.0, max(0, (10 - row["form5_econ"]) / 4)) if pd.notna(row["form5_econ"]) else c_econ
    f_wkt  = min(row["form5_wkts"] / 2, 1) if pd.notna(row["form5_wkts"]) else c_wkt
    form = f_econ * 0.5 + f_wkt * 0.5
    if pd.notna(row.get("season_econ")) and pd.notna(row.get("season_wkt_rate")):
        s_econ = min(1.0, max(0, (10 - row["season_econ"]) / 4))
        s_wkt  = min(row["season_wkt_rate"] / 2, 1)
        season = s_econ * 0.5 + s_wkt * 0.5
        return max(0, (0.35 * career + 0.30 * form + 0.35 * season) * 100)
    return max(0, (0.50 * career + 0.50 * form) * 100)

bowl_inns["bowl_score"] = bowl_inns.apply(compute_bowl_score, axis=1)
bowl_score_lookup = {(r.bowler, r.file_id): r.bowl_score
                     for _, r in bowl_inns[["bowler", "file_id", "bowl_score"]].dropna(subset=["bowl_score"]).iterrows()}

# Venue batting scores
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
print(f"  Player score defaults: bat={DEFAULT_BAT:.1f}, bowl={DEFAULT_BOWL:.1f}")

# Extract XI from deliveries for all 2024-25 matches
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

print("Extracting XI for 2024-25 matches...")
_bt_fids = set(matches["file_id"].unique()) & set(del_df["file_id"].unique())
xi_data  = {fid: extract_xi(fid) for fid in _bt_fids}

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
        return DEFAULT_BAT, DEFAULT_BOWL, 0.0, DEFAULT_BAT, 0
    bs = sorted(bat_scores, reverse=True); ws = sorted(bowl_scores, reverse=True)
    return np.mean(bs[:6]), np.mean(ws[:4]), (np.std(bs) if len(bs) > 1 else 0), bs[0], len(players)

# -- Build lookup maps ------------------------------------------------

# ELO/form from match_features (expanding window — use row BEFORE each match)
# For backtest: use the row's own features (already expanding-window in match_features)
fid_to_winner = dict(zip(matches["file_id"], matches["winner"]))
fid_to_inn1   = dict(zip(matches["file_id"], matches["inn1_team"]))
fid_to_inn2   = dict(zip(matches["file_id"], matches["inn2_team"]))

# mf keyed by file_id (drop duplicates)
mf_dedup = mf.drop_duplicates("file_id", keep="last")
mfd = {r["file_id"]: dict(r) for _, r in mf_dedup.iterrows()}

# Venue stats from full history (not just 2024-25)
venue_stats_map = {}
for v, grp in mf_all.groupby("venue"):
    venue_stats_map[v] = {
        "venue_avg_first_innings": grp["venue_avg_first_innings"].iloc[-1] if "venue_avg_first_innings" in grp else 160.0,
        "venue_bat_first_win_rate": grp["venue_bat_first_win_rate"].iloc[-1] if "venue_bat_first_win_rate" in grp else 0.5,
        "venue_chase_win_rate": grp["venue_chase_win_rate"].iloc[-1] if "venue_chase_win_rate" in grp else 0.5,
        "venue_toss_win_rate": grp.get("venue_toss_win_rate", pd.Series([0.5])).iloc[-1],
        "venue_matches": len(grp),
    }

# player_score_map removed — using expanding-window xi_features() from deliveries.csv instead

# Impact player lookup: file_id -> {team -> player_name}
matches_raw = pd.read_csv("data/matches.csv")
matches_raw["file_id"] = matches_raw["file_id"].astype(str)
impact_nominee_map = {}  # file_id -> {team: impact_player_name}
for _, mr in matches_raw.iterrows():
    fid_r = str(mr["file_id"])
    t1_pl = str(mr.get("team1_players","")).split("|")
    t2_pl = str(mr.get("team2_players","")).split("|")
    imp = {}
    if len(t1_pl) >= 12 and t1_pl[11].strip():
        imp[mr["team1"]] = t1_pl[11].strip()
    if len(t2_pl) >= 12 and t2_pl[11].strip():
        imp[mr["team2"]] = t2_pl[11].strip()
    impact_nominee_map[fid_r] = imp

def _get_impact_bowl_score_bt(player, fid):
    """Expanding-window bowl score for impact player at match time."""
    if not player or pd.isna(player):
        return DEFAULT_BOWL
    score = bowl_score_lookup.get((player, fid), np.nan)
    return score if not np.isnan(score) else DEFAULT_BOWL

# H2H from match_features
h2h_map = {}
if "h2h_win_rate_team1" in mf_all.columns:
    for _, r in mf_all.iterrows():
        key = tuple(sorted([r["team1"], r["team2"]]))
        h2h_map[key] = r["h2h_win_rate_team1"]

# Chase WR
chase_wr_map = {}
for _, r in mf_all.iterrows():
    if "team1_chase_wr" in r and pd.notna(r.get("team1_chase_wr")):
        chase_wr_map[r["team1"]] = r["team1_chase_wr"]
    if "team2_chase_wr" in r and pd.notna(r.get("team2_chase_wr")):
        chase_wr_map[r["team2"]] = r["team2_chase_wr"]

# ELO/form maps (latest values for each team from match_features)
elo_map, form_map, form3_map, form10_map = {}, {}, {}, {}
for _, r in mf_all.sort_values("date").iterrows():
    elo_map[r["team1"]] = r.get("team1_elo", 1500)
    elo_map[r["team2"]] = r.get("team2_elo", 1500)
    form_map[r["team1"]] = r.get("team1_form", 0.5)
    form_map[r["team2"]] = r.get("team2_form", 0.5)
    if "team1_form_3" in r:
        form3_map[r["team1"]] = r.get("team1_form_3", 0.5)
        form3_map[r["team2"]] = r.get("team2_form_3", 0.5)
    if "team1_form_10" in r:
        form10_map[r["team1"]] = r.get("team1_form_10", 0.5)
        form10_map[r["team2"]] = r.get("team2_form_10", 0.5)

# Venue-team WR
venue_team_map = {}
if "team1_venue_win_rate" in mf_all.columns:
    for _, r in mf_all.iterrows():
        v = r.get("venue", "")
        venue_team_map[(v, r["team1"])] = r.get("team1_venue_win_rate", 0.5)
        venue_team_map[(v, r["team2"])] = r.get("team2_venue_win_rate", 0.5)

# Inn1 final totals
inn1_final = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")
    .agg(inn1_fr=("cum_runs", "max"), inn1_fw=("cum_wickets", "max"))
    .to_dict("index")
)


# ══════════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════

def norm_venue(v):
    if not isinstance(v, str): return v
    if "Chinnaswamy" in v: return "M Chinnaswamy Stadium, Bengaluru"
    if "Eden" in v: return "Eden Gardens, Kolkata"
    if "Wankhede" in v: return "Wankhede Stadium, Mumbai"
    if "Chepauk" in v or "Chidambaram" in v: return "MA Chidambaram Stadium, Chennai"
    if "Feroz" in v or "Arun Jaitley" in v or "Kotla" in v: return "Arun Jaitley Stadium, Delhi"
    if "Sawai" in v or "Jaipur" in v: return "Sawai Mansingh Stadium, Jaipur"
    if "Rajiv Gandhi" in v and "Hyderabad" in v: return "Rajiv Gandhi Intl Stadium, Hyderabad"
    if "Ekana" in v: return "Ekana Stadium, Lucknow"
    if "Narendra Modi" in v or "Motera" in v: return "Narendra Modi Stadium, Ahmedabad"
    if "Holkar" in v or "Indore" in v: return "Holkar Cricket Stadium, Indore"
    if "Punjab" in v or "Mohali" in v or "PCA" in v: return "Punjab Cricket Association Stadium, Mohali"
    return v

# get_xi_strengths removed — using xi_features() with expanding-window pipeline instead


# -- 1. Pre-match (pre-toss) prediction ------------------------------

def predict_prematch(mf_row):
    """P(team1 wins) — pre-match model with toss features zeroed out."""
    TOSS_ZERO = {
        "team1_won_toss": 0, "toss_chose_bat": 0,
        "team1_bats_second": 0, "toss_venue_aligned": 0,
        "team1_chase_advantage": 0, "team2_chase_advantage": 0,
        "chase_advantage_diff": 0, "early_chase_boost": 0,
        "venue_chase_batting_second": 0,
    }
    tm = pre_b.get("train_median", {})
    feats = {f: mf_row.get(f, tm.get(f, 0)) for f in pre_b["features"]}
    feats.update(TOSS_ZERO)
    X = np.array([[feats.get(f, 0) for f in pre_b["features"]]])
    return float(pre_b["model"].predict_proba(X)[0, 1])


# -- 2. Post-toss prediction -----------------------------------------

def predict_posttoss(match_row, mf_row, fid):
    """P(bat_first wins) — post-toss model with actual XI (expanding-window) and toss."""
    bat_first  = match_row["inn1_team"]
    bat_second = match_row["inn2_team"]
    venue = norm_venue(match_row["venue"])

    # XI strengths via expanding-window pipeline
    bf_pl, bs_pl = xi_data.get(fid, ([], []))
    if bf_pl or bs_pl:
        bf_bat, bf_bowl, bf_depth, bf_max, _bfn = xi_features(fid, bf_pl)
        bs_bat, bs_bowl, bs_depth, bs_max, _bsn = xi_features(fid, bs_pl)
    else:
        bf_bat = bs_bat = DEFAULT_BAT
        bf_bowl = bs_bowl = DEFAULT_BOWL
        bf_depth = bs_depth = 0.0
        bf_max = bs_max = DEFAULT_BAT

    # H2H batter-bowler matchup advantage (Phase 4 feature)
    matchup_adv_bf   = _match_adv(bf_pl, bs_pl)
    matchup_adv_diff = matchup_adv_bf - 0.5

    # Impact player bowling strength (Phase 6)
    nominees = impact_nominee_map.get(fid, {})
    bf_imp_player = nominees.get(bat_first)
    bs_imp_player = nominees.get(bat_second)
    bf_impact_bowl = _get_impact_bowl_score_bt(bf_imp_player, fid)
    bs_impact_bowl = _get_impact_bowl_score_bt(bs_imp_player, fid)
    impact_bowl_diff = bf_impact_bowl - bs_impact_bowl

    # ELO/form (bat_first perspective)
    bf_elo = elo_map.get(bat_first, 1500)
    bs_elo = elo_map.get(bat_second, 1500)
    bf_form = form_map.get(bat_first, 0.5)
    bs_form = form_map.get(bat_second, 0.5)
    bf_form3 = form3_map.get(bat_first, 0.5)
    bs_form3 = form3_map.get(bat_second, 0.5)
    bf_form10 = form10_map.get(bat_first, 0.5)
    bs_form10 = form10_map.get(bat_second, 0.5)

    # H2H
    key = tuple(sorted([bat_first, bat_second]))
    raw_h2h = h2h_map.get(key, 0.5)
    h2h_bf = raw_h2h if bat_first <= bat_second else 1 - raw_h2h

    # Venue
    vs = venue_stats_map.get(venue, {})
    bf_venue_wr = venue_team_map.get((match_row["venue"], bat_first), 0.5)
    bs_venue_wr = venue_team_map.get((match_row["venue"], bat_second), 0.5)
    venue_chase_wr = vs.get("venue_chase_win_rate", 0.5)

    # Toss features
    toss_decision = str(match_row["toss_decision"]).lower()
    toss_chose_field = 1 if toss_decision in ("field", "bowl") else 0
    toss_winner = match_row["toss_winner"]
    tw_bats_first = int(
        (toss_winner == bat_first and toss_decision == "bat") or
        (toss_winner == bat_second and toss_decision in ("field", "bowl"))
    )
    toss_venue_aligned = int(
        (toss_chose_field == 1 and venue_chase_wr > 0.5) or
        (toss_chose_field == 0 and venue_chase_wr <= 0.5)
    )
    venue_decision_wr = mf_row.get("venue_decision_wr_bf", 0.5)

    bs_chase_wr = chase_wr_map.get(bat_second, 0.5)

    # Weather defaults
    temp, hum, cloud, is_eve = 30.0, 55.0, 30.0, 1
    dew = max(0, min(1, (hum - 65) / 35)) if hum >= 65 and is_eve else 0.0
    heat = 1 if temp >= 35 else 0

    match_num = int(mf_row.get("match_num_in_season", 30))

    feats = {
        "elo_diff_bf": bf_elo - bs_elo,
        "bf_elo": bf_elo, "bs_elo": bs_elo,
        "form_diff_bf": bf_form - bs_form,
        "bf_form": bf_form, "bs_form": bs_form,
        "form_3_diff_bf": bf_form3 - bs_form3,
        "form_10_diff_bf": bf_form10 - bs_form10,
        "h2h_bf": h2h_bf,
        "bf_xi_bat": bf_bat, "bs_xi_bat": bs_bat, "xi_bat_diff": bf_bat - bs_bat,
        "bf_xi_bowl": bf_bowl, "bs_xi_bowl": bs_bowl, "xi_bowl_diff": bf_bowl - bs_bowl,
        "bf_xi_depth": bf_depth, "bs_xi_depth": bs_depth,
        "bf_xi_max_bat": bf_max, "bs_xi_max_bat": bs_max,
        "matchup_advantage_bf":   matchup_adv_bf,
        "matchup_advantage_diff": matchup_adv_diff,
        "bf_impact_bowl":   bf_impact_bowl,
        "bs_impact_bowl":   bs_impact_bowl,
        "impact_bowl_diff": impact_bowl_diff,
        "toss_chose_field": toss_chose_field,
        "toss_winner_bats_first": tw_bats_first,
        "toss_venue_aligned_bf": toss_venue_aligned,
        "venue_decision_wr_bf": venue_decision_wr,
        "venue_chase_win_rate": venue_chase_wr,
        "venue_bat_first_win_rate": vs.get("venue_bat_first_win_rate", 0.5),
        "venue_avg_first_innings": vs.get("venue_avg_first_innings", 160),
        "venue_matches": vs.get("venue_matches", 10),
        "bf_venue_wr": bf_venue_wr, "bs_venue_wr": bs_venue_wr,
        "venue_toss_win_rate": vs.get("venue_toss_win_rate", 0.5),
        "bs_chase_wr": bs_chase_wr,
        "temperature": temp, "humidity": hum, "cloud_cover": cloud,
        "dew_factor": dew, "is_evening": is_eve, "heat_factor": heat,
        "dew_chase_advantage": dew * venue_chase_wr,
        "humidity_x_evening": hum * is_eve / 100,
        "match_num_in_season": match_num,
        "is_playoff": int(match_num > 56),
    }

    tm = post_b.get("train_median", {})
    X = np.array([[feats.get(f, tm.get(f, 0)) for f in post_b["features"]]])
    p_bf = float(post_b["model"].predict_proba(X)[0, 1])
    return max(0.05, min(0.95, p_bf))


# -- 3. Live Unified prediction --------------------------------------

def predict_unified(innings, runs, wkts, balls, target, venue_avg,
                    elo_diff, form_diff, venue_bfwr, venue_cwr,
                    partnership_runs, partnership_balls,
                    last_3ov_runs, last_3ov_wkts,
                    boundary_pct, dot_pct,
                    pp_runs, pp_wkts,
                    inn1_final_runs=0, inn1_final_wkts=0,
                    team_phase_wkt_wr=0.0):
    """P(bat_first wins) — unified live model."""
    crr = runs / balls * 6 if balls > 0 else 0.0
    pp_run_rate = (pp_runs or 0) / 36 * 6 if balls > 36 and pp_runs else 0.0

    if innings == 1:
        expected_at = venue_avg * (balls / 120)
        # Phase-aware projection
        venue_rr = venue_avg / 20
        if balls <= 36:
            blend = balls / 120
        elif balls <= 90:
            blend = 0.3 + 0.7 * (balls - 36) / 54
        else:
            blend = 1.0
        proj_rr = blend * crr + (1 - blend) * venue_rr
        balls_rem = max(0, 120 - balls)
        projected = round(runs + proj_rr * balls_rem / 6, 1) if balls_rem > 0 else float(runs)

        feats = {
            "current_innings": 1, "innings_balls": balls,
            "innings_balls_rem": 120 - balls, "innings_balls_pct": balls / 120,
            "inn1_runs": runs, "inn1_wickets": wkts, "inn1_crr": crr,
            "inn1_projected": projected,
            "inn1_vs_avg": runs - expected_at,
            "inn1_vs_avg_pct": runs / expected_at if expected_at > 0 else 1.0,
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
        runs_needed = max(0, target - runs)
        rrr         = runs_needed / balls_rem * 6 if balls_rem > 0 else 99.0
        rrr_diff    = rrr - crr
        rr_ratio    = min(crr / rrr if rrr > 0 else 1.0, 3.0)
        pp_req_rate = (target - (pp_runs or 0)) / 84 * 6 if balls > 36 else 0.0
        pp_rate_gap = pp_run_rate - pp_req_rate if balls > 36 else 0.0
        feats = {
            "current_innings": 2, "innings_balls": balls,
            "innings_balls_rem": balls_rem, "innings_balls_pct": balls / 120,
            "inn1_runs": inn1_final_runs, "inn1_wickets": inn1_final_wkts,
            "inn1_crr": inn1_final_runs / 120 * 6,
            "inn1_projected": 0.0,
            "inn1_vs_avg": inn1_final_runs - venue_avg,
            "inn1_vs_avg_pct": inn1_final_runs / venue_avg if venue_avg > 0 else 1.0,
            "inn1_balls_pct": 1.0, "inn1_acceleration": 0.0,
            "inn2_runs": runs, "inn2_wickets": wkts, "inn2_crr": crr,
            "inn2_rrr": rrr, "inn2_rrr_diff": rrr_diff,
            "inn2_run_rate_ratio": rr_ratio,
            "inn2_runs_needed": runs_needed, "inn2_balls_rem": balls_rem,
            "inn2_balls_pct": balls / 120,
            "first_innings_wickets": int(inn1_final_wkts),
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

    X   = np.array([[feats.get(f, 0) for f in uni_b["features"]]])
    X_s = uni_b["scaler"].transform(X)
    return float(uni_b["model"].predict_proba(X_s)[0, 1])


# ══════════════════════════════════════════════════════════════════════
# BUILD OVER-BY-OVER SNAPSHOTS
# ══════════════════════════════════════════════════════════════════════

print("Computing ball-by-ball momentum snapshots...")

CHECKPOINTS = [3, 6, 10, 15, 20]  # overs to evaluate

# For each match+innings, compute running stats at each checkpoint
snap_data = {}  # (file_id, innings, over) -> dict of stats

for (fid, inn), grp in deliveries.groupby(["file_id", "innings"]):
    grp = grp.sort_values("ball_num").reset_index(drop=True)
    cr, cw, bd, dt = 0, 0, 0, 0
    pr_r, pr_b = 0, 0
    recent_r, recent_w = [], []
    pp_r, pp_w = 0, 0

    for _, row in grp.iterrows():
        bn = row["ball_num"]
        cr = int(row["cum_runs"])
        cw = int(row["cum_wickets"])
        bd += int(row["runs_batter"] >= 4)
        dt += int(row["runs_total"] == 0)
        pr_r += int(row["runs_total"])
        pr_b += 1
        recent_r.append(int(row["runs_total"]))
        recent_w.append(int(row["is_wicket"]))
        if len(recent_r) > 18:
            recent_r.pop(0); recent_w.pop(0)
        if row["is_wicket"]:
            pr_r = 0; pr_b = 0

        # At end of each over checkpoint
        if bn % 6 == 0:
            over_num = bn // 6
            if over_num == 6:
                pp_r = cr
                pp_w = cw
            if over_num in CHECKPOINTS:
                snap_data[(fid, int(inn), over_num)] = {
                    "runs": cr, "wkts": cw,
                    "partnership_runs": pr_r, "partnership_balls": pr_b,
                    "last_3ov_runs": sum(recent_r), "last_3ov_wkts": sum(recent_w),
                    "boundary_pct": bd / bn if bn > 0 else 0.25,
                    "dot_pct": dt / bn if bn > 0 else 0.42,
                    "pp_runs": pp_r if over_num > 6 else None,
                    "pp_wkts": pp_w if over_num > 6 else None,
                }

print(f"  Snapshots computed: {len(snap_data)}")


# ══════════════════════════════════════════════════════════════════════
# RUN BACKTEST
# ══════════════════════════════════════════════════════════════════════

print("\nRunning comprehensive backtest...\n")

results = []
skipped = 0

for _, match in matches.iterrows():
    fid    = match["file_id"]
    season = str(match["season"])
    winner = match.get("winner")
    team1  = match["team1"]
    team2  = match["team2"]
    inn1_t = match.get("inn1_team")
    inn2_t = match.get("inn2_team")

    if pd.isna(winner) or pd.isna(inn1_t):
        skipped += 1
        continue

    mf_row = mfd.get(fid, {})
    if not mf_row:
        skipped += 1
        continue

    bat_first_won = int(inn1_t == winner)
    team1_won     = int(team1 == winner)

    venue      = match["venue"]
    venue_norm = norm_venue(venue)
    vs         = venue_stats_map.get(venue_norm, venue_stats_map.get(venue, {}))
    va         = vs.get("venue_avg_first_innings", 160.0)
    vbfwr      = vs.get("venue_bat_first_win_rate", 0.5)
    vcwr       = vs.get("venue_chase_win_rate", 0.5)

    # ELO/form from bat_first perspective
    if inn1_t == team1:
        elo_d_bf  = float(mf_row.get("elo_diff", 0))
        form_d_bf = float(mf_row.get("form_diff", 0))
    else:
        elo_d_bf  = -float(mf_row.get("elo_diff", 0))
        form_d_bf = -float(mf_row.get("form_diff", 0))

    row_result = {
        "file_id": fid, "season": season, "date": match["date"],
        "team1": team1, "team2": team2,
        "bat_first": inn1_t, "bat_second": inn2_t,
        "winner": winner, "bat_first_won": bat_first_won,
    }

    # -- 1. Pre-match prediction --------------------------------------
    try:
        p_t1 = predict_prematch(mf_row)
        pre_pred = int(p_t1 >= 0.5)
        row_result["pre_prob"] = p_t1
        row_result["pre_correct"] = int(pre_pred == team1_won)
        row_result["pre_confidence"] = p_t1 if pre_pred else 1 - p_t1
    except Exception as e:
        row_result["pre_correct"] = None

    # -- 2. Post-toss prediction --------------------------------------
    try:
        p_bf = predict_posttoss(match, mf_row, fid)
        pt_pred = int(p_bf >= 0.5)
        row_result["pt_prob"] = p_bf
        row_result["pt_correct"] = int(pt_pred == bat_first_won)
        row_result["pt_confidence"] = p_bf if pt_pred else 1 - p_bf
    except Exception as e:
        row_result["pt_correct"] = None

    # -- 3. Live Unified — Inn1 checkpoints ---------------------------
    i1_data = inn1_final.get(fid, {})
    for ov in CHECKPOINTS:
        snap = snap_data.get((fid, 1, ov))
        if snap is None:
            row_result[f"inn1_ov{ov}_correct"] = None
            continue
        try:
            phase_wr = pp_wr_lookup.get((fid, 1, "pp"), 0.0) if ov > 6 else 0.0
            p_bf = predict_unified(
                innings=1, runs=snap["runs"], wkts=snap["wkts"],
                balls=ov*6, target=0, venue_avg=va,
                elo_diff=elo_d_bf, form_diff=form_d_bf,
                venue_bfwr=vbfwr, venue_cwr=vcwr,
                partnership_runs=snap["partnership_runs"],
                partnership_balls=snap["partnership_balls"],
                last_3ov_runs=snap["last_3ov_runs"],
                last_3ov_wkts=snap["last_3ov_wkts"],
                boundary_pct=snap["boundary_pct"],
                dot_pct=snap["dot_pct"],
                pp_runs=snap["pp_runs"], pp_wkts=snap["pp_wkts"],
                team_phase_wkt_wr=phase_wr,
            )
            pred = int(p_bf >= 0.5)
            row_result[f"inn1_ov{ov}_correct"] = int(pred == bat_first_won)
            row_result[f"inn1_ov{ov}_prob"] = p_bf
            row_result[f"inn1_ov{ov}_conf"] = p_bf if pred else 1 - p_bf
        except Exception:
            row_result[f"inn1_ov{ov}_correct"] = None

    # -- 4. Live Unified — Inn2 checkpoints ---------------------------
    i1fr = i1_data.get("inn1_fr", None)
    i1fw = i1_data.get("inn1_fw", 0)
    if i1fr is not None:
        target = int(i1fr) + 1
        for ov in CHECKPOINTS[:-1]:  # No over 20 for inn2 (match likely over)
            snap = snap_data.get((fid, 2, ov))
            if snap is None:
                row_result[f"inn2_ov{ov}_correct"] = None
                continue
            try:
                phase_wr = pp_wr_lookup.get((fid, 2, "pp"), 0.0) if ov > 6 else 0.0
                p_bf = predict_unified(
                    innings=2, runs=snap["runs"], wkts=snap["wkts"],
                    balls=ov*6, target=target, venue_avg=va,
                    elo_diff=elo_d_bf, form_diff=form_d_bf,
                    venue_bfwr=vbfwr, venue_cwr=vcwr,
                    partnership_runs=snap["partnership_runs"],
                    partnership_balls=snap["partnership_balls"],
                    last_3ov_runs=snap["last_3ov_runs"],
                    last_3ov_wkts=snap["last_3ov_wkts"],
                    boundary_pct=snap["boundary_pct"],
                    dot_pct=snap["dot_pct"],
                    pp_runs=snap["pp_runs"], pp_wkts=snap["pp_wkts"],
                    inn1_final_runs=int(i1fr), inn1_final_wkts=int(i1fw),
                    team_phase_wkt_wr=phase_wr,
                )
                pred = int(p_bf >= 0.5)
                row_result[f"inn2_ov{ov}_correct"] = int(pred == bat_first_won)
                row_result[f"inn2_ov{ov}_prob"] = p_bf
                row_result[f"inn2_ov{ov}_conf"] = p_bf if pred else 1 - p_bf
            except Exception:
                row_result[f"inn2_ov{ov}_correct"] = None

    results.append(row_result)

df = pd.DataFrame(results)
print(f"Processed: {len(df)} matches | Skipped: {skipped}")


# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════

W = 90

def pct(s, col):
    v = s[col].dropna()
    if len(v) == 0: return "  N/A    "
    return f"{v.mean()*100:5.1f}% ({int(v.sum()):>3}/{len(v):<3})"

def conf_stats(s, corr_col, prob_col):
    sub = s[[corr_col, prob_col]].dropna()
    if len(sub) == 0: return None, None
    correct_conf = sub[sub[corr_col]==1][prob_col].mean()
    wrong_conf   = sub[sub[corr_col]==0][prob_col].mean()
    return correct_conf, wrong_conf


print("\n" + "=" * W)
print("COMPREHENSIVE MODEL BACKTEST — IPL 2024-2025 (Out-of-Sample)")
print("=" * W)

# -- Section 1: Pre-Match & Post-Toss --------------------------------
print(f"\n{'-'*W}")
print("SECTION 1: PRE-MATCH & POST-TOSS ACCURACY")
print(f"{'-'*W}")
print(f"  {'Model':<20} {'2024':>16} {'2025':>16} {'Combined':>16}")
print(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*16}")

for label, col in [("Pre-Match (no toss)", "pre_correct"),
                    ("Post-Toss (XI+toss)", "pt_correct")]:
    s24 = df[df["season"]=="2024"]
    s25 = df[df["season"]=="2025"]
    print(f"  {label:<20} {pct(s24, col):>16} {pct(s25, col):>16} {pct(df, col):>16}")

# Confidence breakdown
print(f"\n  {'Model':<20} {'Correct Conf':>14} {'Wrong Conf':>14} {'Gap':>8}")
print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*8}")
for label, cc, pc in [("Pre-Match", "pre_correct", "pre_confidence"),
                       ("Post-Toss", "pt_correct", "pt_confidence")]:
    c, w = conf_stats(df, cc, pc)
    if c is not None:
        print(f"  {label:<20} {c*100:>13.1f}% {w*100:>13.1f}% {(c-w)*100:>7.1f}pp")


# -- Section 2: Live Unified — Inn1 ----------------------------------
print(f"\n{'-'*W}")
print("SECTION 2: LIVE UNIFIED MODEL — 1st INNINGS")
print(f"{'-'*W}")
header = f"  {'Over':>4}"
for s in ["2024", "2025", "Combined"]:
    header += f"  {s:>16}"
print(header)
print(f"  {'-'*4}  {'-'*16}  {'-'*16}  {'-'*16}")

for ov in CHECKPOINTS:
    col = f"inn1_ov{ov}_correct"
    s24 = df[df["season"]=="2024"]
    s25 = df[df["season"]=="2025"]
    print(f"  {ov:>4}  {pct(s24, col):>16}  {pct(s25, col):>16}  {pct(df, col):>16}")

# Confidence
print(f"\n  {'Over':>4}  {'Correct Conf':>14} {'Wrong Conf':>14} {'Gap':>8}")
print(f"  {'-'*4}  {'-'*14} {'-'*14} {'-'*8}")
for ov in CHECKPOINTS:
    c, w = conf_stats(df, f"inn1_ov{ov}_correct", f"inn1_ov{ov}_conf")
    if c is not None:
        print(f"  {ov:>4}  {c*100:>13.1f}% {w*100:>13.1f}% {(c-w)*100:>7.1f}pp")


# -- Section 3: Live Unified — Inn2 ----------------------------------
print(f"\n{'-'*W}")
print("SECTION 3: LIVE UNIFIED MODEL — 2nd INNINGS (Chase)")
print(f"{'-'*W}")
header = f"  {'Over':>4}"
for s in ["2024", "2025", "Combined"]:
    header += f"  {s:>16}"
print(header)
print(f"  {'-'*4}  {'-'*16}  {'-'*16}  {'-'*16}")

for ov in CHECKPOINTS[:-1]:
    col = f"inn2_ov{ov}_correct"
    s24 = df[df["season"]=="2024"]
    s25 = df[df["season"]=="2025"]
    print(f"  {ov:>4}  {pct(s24, col):>16}  {pct(s25, col):>16}  {pct(df, col):>16}")

# Confidence
print(f"\n  {'Over':>4}  {'Correct Conf':>14} {'Wrong Conf':>14} {'Gap':>8}")
print(f"  {'-'*4}  {'-'*14} {'-'*14} {'-'*8}")
for ov in CHECKPOINTS[:-1]:
    c, w = conf_stats(df, f"inn2_ov{ov}_correct", f"inn2_ov{ov}_conf")
    if c is not None:
        print(f"  {ov:>4}  {c*100:>13.1f}% {w*100:>13.1f}% {(c-w)*100:>7.1f}pp")


# -- Section 4: Full Pipeline Accuracy Timeline ----------------------
print(f"\n{'-'*W}")
print("SECTION 4: FULL PREDICTION PIPELINE — ACCURACY TIMELINE")
print(f"{'-'*W}")
print(f"  {'Stage':<30} {'Accuracy':>16} {'Avg Confidence':>16}")
print(f"  {'-'*30} {'-'*16} {'-'*16}")

stages = [
    ("Pre-Match (no toss)", "pre_correct", "pre_confidence"),
    ("Post-Toss (with XI)", "pt_correct", "pt_confidence"),
]
for ov in CHECKPOINTS:
    stages.append((f"Inn1 Over {ov}", f"inn1_ov{ov}_correct", f"inn1_ov{ov}_conf"))
for ov in CHECKPOINTS[:-1]:
    stages.append((f"Inn2 Over {ov}", f"inn2_ov{ov}_correct", f"inn2_ov{ov}_conf"))

for label, cc, pc in stages:
    v = df[cc].dropna()
    if len(v) == 0:
        continue
    acc = f"{v.mean()*100:.1f}% ({int(v.sum())}/{len(v)})"
    p = df[pc].dropna()
    avg_conf = f"{p.mean()*100:.1f}%" if len(p) > 0 else "N/A"
    print(f"  {label:<30} {acc:>16} {avg_conf:>16}")


# -- Section 5: Per-season summary ------------------------------------
print(f"\n{'-'*W}")
print("SECTION 5: SEASON SUMMARY")
print(f"{'-'*W}")
for season in ["2024", "2025"]:
    ss = df[df["season"] == season]
    n = len(ss)
    print(f"\n  IPL {season} ({n} matches):")
    for label, cc in [("Pre-Match", "pre_correct"),
                       ("Post-Toss", "pt_correct"),
                       ("Inn1@Ov6", "inn1_ov6_correct"),
                       ("Inn1@Ov10", "inn1_ov10_correct"),
                       ("Inn1@Ov15", "inn1_ov15_correct"),
                       ("Inn1@Ov20", "inn1_ov20_correct"),
                       ("Inn2@Ov6", "inn2_ov6_correct"),
                       ("Inn2@Ov10", "inn2_ov10_correct"),
                       ("Inn2@Ov15", "inn2_ov15_correct")]:
        v = ss[cc].dropna()
        if len(v) > 0:
            print(f"    {label:<14} {v.mean()*100:5.1f}% ({int(v.sum()):>2}/{len(v):<2})")


# -- Save detailed results -------------------------------------------
df.to_csv("data/backtest_2024_25.csv", index=False)
print(f"\n{'='*W}")
print(f"Detailed results saved to data/backtest_2024_25.csv")
print("Done.")
