"""
02_features.py  —  Build match_features.csv
Fixes data leakage bugs, adds multi-window form, venue-team rates,
current-season player stats blend, chase success rate, phase-of-tournament.
Adds venue normalization, toss-venue alignment, team chase ability,
batting order context features for post-toss prediction.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

Path("data").mkdir(exist_ok=True)

# ── Team name normalisation ────────────────────────────────────────────────
TEAM_NORM = {
    "Delhi Daredevils":               "Delhi Capitals",
    "Deccan Chargers":                "Sunrisers Hyderabad",
    "Rising Pune Supergiants":        "Rising Pune Supergiant",
    "Punjab Kings":                   "Kings XI Punjab",
    "Kings XI Punjab":                "Kings XI Punjab",
    "Royal Challengers Bangalore":    "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru":    "Royal Challengers Bengaluru",
    "Kochi Tuskers Kerala":           "Kochi Tuskers Kerala",
    "Pune Warriors":                  "Pune Warriors",
}

def norm(name):
    if not isinstance(name, str):
        return name
    return TEAM_NORM.get(name, name)

# ── Venue name normalisation ──────────────────────────────────────────────
def norm_venue(v):
    """Normalize venue names — same stadium has 2-3 different names in data."""
    if not isinstance(v, str):
        return v
    if "Chinnaswamy" in v: return "M Chinnaswamy Stadium, Bengaluru"
    if "Eden" in v: return "Eden Gardens, Kolkata"
    if "Wankhede" in v: return "Wankhede Stadium, Mumbai"
    if "Chepauk" in v or "Chidambaram" in v: return "MA Chidambaram Stadium, Chennai"
    if "Feroz" in v or "Arun Jaitley" in v or "Kotla" in v: return "Arun Jaitley Stadium, Delhi"
    if "Rajiv Gandhi" in v and ("Uppal" in v or "Hyderabad" in v): return "Rajiv Gandhi Intl Stadium, Hyderabad"
    if "Rajiv Gandhi" in v: return "Rajiv Gandhi Intl Stadium, Hyderabad"
    if "Sawai" in v: return "Sawai Mansingh Stadium, Jaipur"
    if "Mohali" in v or ("Punjab" in v and "Bindra" in v): return "PCA Stadium, Mohali"
    if "DY Patil" in v: return "DY Patil Stadium, Mumbai"
    if "Brabourne" in v: return "Brabourne Stadium, Mumbai"
    if "Narendra Modi" in v or "Motera" in v or "Sardar Patel" in v: return "Narendra Modi Stadium, Ahmedabad"
    if "Ekana" in v or "Atal Bihari" in v: return "Ekana Stadium, Lucknow"
    if "Maharashtra" in v and ("Pune" in v or "MCA" in v): return "MCA Stadium, Pune"
    if "Subrata" in v: return "Subrata Roy Sahara Stadium, Pune"
    if "Himachal" in v or "Dharamsala" in v or "Dharamshala" in v: return "HPCA Stadium, Dharamsala"
    if "Holkar" in v: return "Holkar Stadium, Indore"
    if "Barabati" in v: return "Barabati Stadium, Cuttack"
    if "Greenfield" in v or "Thiruvananthapuram" in v: return "Greenfield Stadium, Thiruvananthapuram"
    if "Sharjah" in v: return "Sharjah Cricket Stadium"
    if "Dubai" in v: return "Dubai International Cricket Stadium"
    if "Sheikh Zayed" in v or "Abu Dhabi" in v: return "Sheikh Zayed Stadium, Abu Dhabi"
    return v

print("Loading data...")
matches    = pd.read_csv("data/matches.csv", parse_dates=["date"])
deliveries = pd.read_csv("data/deliveries.csv", parse_dates=["date"])

# Normalise team names everywhere
for col in ["team1","team2","winner","toss_winner","inn1_team","inn2_team"]:
    if col in matches.columns:
        matches[col] = matches[col].map(norm)

for col in ["batting_team","bowling_team","winner"]:
    if col in deliveries.columns:
        deliveries[col] = deliveries[col].map(norm)

# ── Recover tied matches: assign Super Over winners ──────────────────────
# These 15 matches tied in regulation; the Super Over winner is known from IPL history.
# Team names are already normalised at this point.
_SUPER_OVER_WINNERS = {
    ("2009-04-23", "Kolkata Knight Riders", "Rajasthan Royals"):       "Rajasthan Royals",
    ("2010-03-21", "Chennai Super Kings",   "Kings XI Punjab"):        "Kings XI Punjab",
    ("2013-04-07", "Sunrisers Hyderabad",   "Royal Challengers Bengaluru"): "Sunrisers Hyderabad",
    ("2013-04-16", "Royal Challengers Bengaluru", "Delhi Capitals"):   "Royal Challengers Bengaluru",
    ("2014-04-29", "Kolkata Knight Riders", "Rajasthan Royals"):       "Rajasthan Royals",
    ("2015-04-21", "Rajasthan Royals",      "Kings XI Punjab"):        "Kings XI Punjab",
    ("2017-04-29", "Gujarat Lions",         "Mumbai Indians"):         "Mumbai Indians",
    ("2019-03-30", "Kolkata Knight Riders", "Delhi Capitals"):         "Delhi Capitals",
    ("2019-05-02", "Mumbai Indians",        "Sunrisers Hyderabad"):    "Mumbai Indians",
    ("2020-09-20", "Delhi Capitals",        "Kings XI Punjab"):        "Delhi Capitals",
    ("2020-09-28", "Royal Challengers Bengaluru", "Mumbai Indians"):   "Royal Challengers Bengaluru",
    ("2020-10-18", "Kolkata Knight Riders", "Sunrisers Hyderabad"):    "Kolkata Knight Riders",
    ("2020-10-18", "Mumbai Indians",        "Kings XI Punjab"):        "Kings XI Punjab",
    ("2021-04-25", "Delhi Capitals",        "Sunrisers Hyderabad"):    "Delhi Capitals",
    ("2025-04-16", "Delhi Capitals",        "Rajasthan Royals"):       "Delhi Capitals",
}
_so_recovered = 0
for idx, row in matches.iterrows():
    if pd.notna(row.get("winner")):
        continue
    date_str = str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10]
    key = (date_str, row["team1"], row["team2"])
    if key in _SUPER_OVER_WINNERS:
        matches.at[idx, "winner"] = _SUPER_OVER_WINNERS[key]
        _so_recovered += 1
if _so_recovered:
    print(f"  Recovered {_so_recovered} Super Over winners from tied matches")

matches = matches[matches["winner"].notna()].copy()
matches = matches[matches["team1"].notna() & matches["team2"].notna()].copy()
matches = matches.sort_values("date").reset_index(drop=True)
matches["season"]    = matches["season"].astype(str)
matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

# Normalize venue names
matches["venue"] = matches["venue"].apply(norm_venue)
if "venue" in deliveries.columns:
    deliveries["venue"] = deliveries["venue"].apply(norm_venue)

# Ensure deliveries has season as str
deliveries["season"] = deliveries["season"].astype(str)

print(f"  {len(matches)} valid matches, seasons: {sorted(matches['season'].unique())}")
print(f"  Teams: {sorted(set(matches['team1'].tolist() + matches['team2'].tolist()))}")

# ── 1. Team ELO ────────────────────────────────────────────────────────────
K = 24
INIT = 1500

def expected_score(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

team_elo = defaultdict(lambda: INIT)
pre_elo1, pre_elo2 = [], []

for _, row in matches.iterrows():
    t1, t2 = row["team1"], row["team2"]
    e1, e2 = team_elo[t1], team_elo[t2]
    pre_elo1.append(e1)
    pre_elo2.append(e2)
    exp = expected_score(e1, e2)
    res = row["team1_won"]
    team_elo[t1] = e1 + K * (res - exp)
    team_elo[t2] = e2 + K * ((1 - res) - (1 - exp))

matches["team1_elo"] = pre_elo1
matches["team2_elo"] = pre_elo2
matches["elo_diff"]  = matches["team1_elo"] - matches["team2_elo"]

# ── 2. Multi-window team form ──────────────────────────────────────────────
print("Computing multi-window form...")
team_hist = defaultdict(list)
form1_3, form2_3 = [], []
form1_5, form2_5 = [], []
form1_10, form2_10 = [], []
form1_w, form2_w = [], []

EXP_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.12, 0.08])

def weighted_form(hist, weights=EXP_WEIGHTS):
    if not hist:
        return 0.5
    h = hist[-len(weights):]
    w = weights[-len(h):]
    w = w / w.sum()
    return float(np.dot(h, w))

for _, row in matches.iterrows():
    t1, t2 = row["team1"], row["team2"]
    h1, h2 = team_hist[t1], team_hist[t2]

    form1_3.append(np.mean(h1[-3:]) if h1 else 0.5)
    form2_3.append(np.mean(h2[-3:]) if h2 else 0.5)
    form1_5.append(np.mean(h1[-5:]) if h1 else 0.5)
    form2_5.append(np.mean(h2[-5:]) if h2 else 0.5)
    form1_10.append(np.mean(h1[-10:]) if h1 else 0.5)
    form2_10.append(np.mean(h2[-10:]) if h2 else 0.5)
    form1_w.append(weighted_form(h1))
    form2_w.append(weighted_form(h2))

    team_hist[t1].append(row["team1_won"])
    team_hist[t2].append(1 - row["team1_won"])

matches["team1_form"]    = form1_5
matches["team2_form"]    = form2_5
matches["form_diff"]     = matches["team1_form"] - matches["team2_form"]
matches["team1_form_3"]  = form1_3
matches["team2_form_3"]  = form2_3
matches["form_3_diff"]   = matches["team1_form_3"] - matches["team2_form_3"]
matches["team1_form_10"] = form1_10
matches["team2_form_10"] = form2_10
matches["form_10_diff"]  = matches["team1_form_10"] - matches["team2_form_10"]
matches["form_weighted_diff"] = np.array(form1_w) - np.array(form2_w)

# ── 3. Head-to-head (last 10) ─────────────────────────────────────────────
h2h = defaultdict(list)
h2h_rate = []

for _, row in matches.iterrows():
    t1, t2 = row["team1"], row["team2"]
    key = tuple(sorted([t1, t2]))
    hist = h2h[key][-10:]
    if hist:
        wins = sum(1 for (a, w) in hist if (a == t1 and w == t1) or (a != t1 and w == t1))
        h2h_rate.append(wins / len(hist))
    else:
        h2h_rate.append(0.5)
    h2h[key].append((t1, row["winner"]))

matches["h2h_win_rate_team1"] = h2h_rate

# ── 4. Toss features ──────────────────────────────────────────────────────
matches["team1_won_toss"]    = (matches["toss_winner"] == matches["team1"]).astype(int)
matches["toss_chose_bat"]    = (matches["toss_decision"] == "bat").astype(int)
matches["batting_first_won"] = (matches["inn1_team"] == matches["winner"]).astype(int)
matches["chasing_won"]       = (matches["inn2_team"] == matches["winner"]).astype(int)

# ── 5. Venue stats — EXPANDING MEANS (no data leakage) ────────────────────
print("Computing venue stats (expanding means, no leakage)...")

# Pre-compute venue-level expanding stats using only past data
venue_toss_wr, venue_bat1_wr, venue_avg_inn1, venue_chase_wr, venue_match_count = [], [], [], [], []

venue_toss_hist   = defaultdict(list)   # toss_winner_won per venue
venue_bat1_hist   = defaultdict(list)   # batting_first_won per venue
venue_inn1_hist   = defaultdict(list)   # first innings totals per venue
venue_chase_hist  = defaultdict(list)   # chasing_won per venue
_all_inn1_scores  = []                  # running list for global fallback average

for _, row in matches.iterrows():
    v = row["venue"]

    # Dynamic global average: used as fallback for new/unknown venues
    _global_avg = np.mean(_all_inn1_scores) if _all_inn1_scores else 160.0

    # Use ONLY data before this match (what's accumulated so far)
    venue_toss_wr.append(np.mean(venue_toss_hist[v]) if venue_toss_hist[v] else 0.5)
    venue_bat1_wr.append(np.mean(venue_bat1_hist[v]) if venue_bat1_hist[v] else 0.5)
    venue_avg_inn1.append(np.mean(venue_inn1_hist[v]) if venue_inn1_hist[v] else _global_avg)
    venue_chase_wr.append(np.mean(venue_chase_hist[v]) if venue_chase_hist[v] else 0.5)
    venue_match_count.append(len(venue_toss_hist[v]))

    # NOW append this match's data (for future matches to use)
    if pd.notna(row.get("toss_winner_won")):
        venue_toss_hist[v].append(row["toss_winner_won"])
    venue_bat1_hist[v].append(row["batting_first_won"])
    if pd.notna(row.get("inn1_runs")):
        venue_inn1_hist[v].append(row["inn1_runs"])
        _all_inn1_scores.append(row["inn1_runs"])
    venue_chase_hist[v].append(row["chasing_won"])

matches["venue_toss_win_rate"]      = venue_toss_wr
matches["venue_bat_first_win_rate"] = venue_bat1_wr
matches["venue_avg_first_innings"]  = venue_avg_inn1
matches["venue_chase_win_rate"]     = venue_chase_wr
matches["venue_matches"]            = venue_match_count

# ── 5b. Venue × toss-decision win rates (expanding, no leakage) ─────────────
# venue_bat_wr:   P(toss_winner wins | chose to BAT at this venue)
# venue_field_wr: P(toss_winner wins | chose to FIELD at this venue)
# Falls back to lag-1 venue toss win rate when fewer than 10 observations in a cell.
print("Computing venue×decision win rates (expanding, no leakage)...")
MIN_CELL = 10
venue_decision_hist = defaultdict(list)   # (venue, "bat"/"field") -> [1, 0, ...]
venue_bat_wr_vals   = []
venue_field_wr_vals = []

for i, (_, row) in enumerate(matches.iterrows()):
    v        = row["venue"]
    decision = row.get("toss_decision", "field")  # "bat" or "field"

    bat_hist   = venue_decision_hist[(v, "bat")]
    field_hist = venue_decision_hist[(v, "field")]
    fallback   = venue_toss_wr[i]  # lag-1 safe toss win rate at this venue

    venue_bat_wr_vals.append(float(np.mean(bat_hist)) if len(bat_hist) >= MIN_CELL else fallback)
    venue_field_wr_vals.append(float(np.mean(field_hist)) if len(field_hist) >= MIN_CELL else fallback)

    # Update AFTER computing (no leakage)
    if pd.notna(row.get("toss_winner_won")):
        venue_decision_hist[(v, decision)].append(int(row["toss_winner_won"]))

matches["venue_bat_wr"]   = venue_bat_wr_vals
matches["venue_field_wr"] = venue_field_wr_vals

# ── 6. Venue-team win rate (expanding) ─────────────────────────────────────
print("Computing venue-team win rates...")
venue_team_hist = defaultdict(list)  # (venue, team) -> [1, 0, 1, ...]
vt1_wr, vt2_wr = [], []

for _, row in matches.iterrows():
    v, t1, t2 = row["venue"], row["team1"], row["team2"]

    h1 = venue_team_hist[(v, t1)]
    h2 = venue_team_hist[(v, t2)]
    vt1_wr.append(np.mean(h1) if h1 else 0.5)
    vt2_wr.append(np.mean(h2) if h2 else 0.5)

    venue_team_hist[(v, t1)].append(row["team1_won"])
    venue_team_hist[(v, t2)].append(1 - row["team1_won"])

matches["team1_venue_win_rate"] = vt1_wr
matches["team2_venue_win_rate"] = vt2_wr

# ── 7. Phase-of-tournament ─────────────────────────────────────────────────
season_match_counter = defaultdict(int)
match_nums = []
for _, row in matches.iterrows():
    s = row["season"]
    season_match_counter[s] += 1
    match_nums.append(season_match_counter[s])

matches["match_num_in_season"] = match_nums
matches["is_playoff"] = (matches["match_num_in_season"] > 56).astype(int)

# ── 7b. Batting order features (post-toss) ────────────────────────────────
print("Computing post-toss batting order features...")

# Determine who bats first/second based on toss
matches["batting_second"] = np.where(
    matches["toss_decision"] == "field",
    matches["toss_winner"],  # toss winner chose to field = they bat second
    np.where(
        matches["toss_winner"] == matches["team1"],
        matches["team2"],  # toss winner (team1) chose to bat = team2 bats second
        matches["team1"]   # toss winner (team2) chose to bat = team1 bats second
    )
)
matches["team1_bats_second"] = (matches["batting_second"] == matches["team1"]).astype(int)

# ── 7c. Toss-venue alignment (THE BIG SIGNAL: 58% vs 42%) ────────────────
print("Computing toss-venue alignment...")
toss_align_vals = []
team1_chase_wr_vals = []
team2_chase_wr_vals = []
team_chase_hist = defaultdict(list)  # team -> [1, 0, 1, ...] when batting second
venue_chase_expanding = defaultdict(list)

# Also compute team-specific chase ability (expanding, no leakage)
for i, (_, row) in enumerate(matches.iterrows()):
    v = row["venue"]
    t1, t2 = row["team1"], row["team2"]

    # --- Toss-venue alignment ---
    hist_chase_wr = np.mean(venue_chase_expanding[v]) if venue_chase_expanding[v] else 0.5
    chose_field = row["toss_decision"] == "field"
    venue_favors_chase = hist_chase_wr > 0.5

    # Did toss winner make the right call for this venue?
    aligned = (chose_field and venue_favors_chase) or (not chose_field and not venue_favors_chase)
    # From team1's perspective: did team1 benefit from the alignment?
    team1_won_toss = row["toss_winner"] == t1
    if team1_won_toss:
        toss_align_vals.append(1 if aligned else 0)
    else:
        toss_align_vals.append(1 if not aligned else 0)  # if team2 made wrong call, benefits team1

    # --- Team chase ability ---
    h1 = team_chase_hist[t1]
    h2 = team_chase_hist[t2]
    team1_chase_wr_vals.append(np.mean(h1) if len(h1) >= 3 else 0.5)
    team2_chase_wr_vals.append(np.mean(h2) if len(h2) >= 3 else 0.5)

    # Update histories AFTER computing features (no leakage)
    chasing_won = row["batting_second"] == row["winner"]
    venue_chase_expanding[v].append(int(chasing_won))

    # Track team chase ability only when they actually batted second
    if row["batting_second"] == t1:
        team_chase_hist[t1].append(int(row["winner"] == t1))
    elif row["batting_second"] == t2:
        team_chase_hist[t2].append(int(row["winner"] == t2))

matches["toss_venue_aligned"] = toss_align_vals
matches["team1_chase_wr"] = team1_chase_wr_vals
matches["team2_chase_wr"] = team2_chase_wr_vals

# Chase ability diff: positive = team1 is better at chasing
matches["chase_wr_diff"] = matches["team1_chase_wr"] - matches["team2_chase_wr"]

# Interaction: team1 bats second AND is a good chaser
matches["team1_chase_advantage"] = matches["team1_bats_second"] * matches["team1_chase_wr"]
matches["team2_chase_advantage"] = (1 - matches["team1_bats_second"]) * matches["team2_chase_wr"]
matches["chase_advantage_diff"] = matches["team1_chase_advantage"] - matches["team2_chase_advantage"]

# ── 7d. Season phase interaction with chase ───────────────────────────────
# Early season = more dew = more chase advantage
matches["early_season"] = (matches["match_num_in_season"] <= 25).astype(int)
matches["early_chase_boost"] = matches["early_season"] * matches["team1_bats_second"]

# ── 7e. Venue-specific chase advantage for batting-second team ────────────
# If team1 bats second at a high-chase venue, that's a big advantage
matches["venue_chase_batting_second"] = np.where(
    matches["team1_bats_second"] == 1,
    matches["venue_chase_win_rate"],
    1 - matches["venue_chase_win_rate"]
)

# ── 7f. Venue × Month chase win rate (expanding, no leakage) ──────────────
# Captures e.g. Ahmedabad April 73% vs May 33%, Hyderabad chase only 35%
print("Computing venue-month chase win rates...")
matches["month"] = matches["date"].dt.month
matches["is_march"] = (matches["month"] == 3).astype(int)

venue_month_chase_hist = defaultdict(list)
venue_month_wr_vals = []

for _, row in matches.iterrows():
    v = row["venue"]
    m = row["month"]
    key = (v, m)
    hist = venue_month_chase_hist[key]
    venue_month_wr_vals.append(np.mean(hist) if hist else 0.5)
    venue_month_chase_hist[key].append(row["chasing_won"])

matches["venue_month_chase_wr"] = venue_month_wr_vals

# Flip to batting-second team's perspective
matches["venue_month_chase_batting_second"] = np.where(
    matches["team1_bats_second"] == 1,
    matches["venue_month_chase_wr"],
    1 - matches["venue_month_chase_wr"]
)

# ── 8. Player batting & bowling scores — DATE-FILTERED (no leakage) ───────
print("Computing player stats (date-filtered, no leakage)...")

# Pre-compute cumulative player stats by building an index
# Group deliveries by (season, batter/bowler, date) for efficient filtering
deliveries["file_id_str"] = deliveries["file_id"].astype(str)

# Build per-match player batting aggregates
bat_by_match = (
    deliveries[deliveries["innings"].isin([1, 2])]
    .groupby(["season", "date", "batter"])
    .agg(bat_balls=("runs_batter", "count"), bat_runs=("runs_batter", "sum"))
    .reset_index()
)

# Build per-match player bowling aggregates
bowl_by_match = (
    deliveries[deliveries["innings"].isin([1, 2])]
    .groupby(["season", "date", "bowler"])
    .agg(bowl_balls=("runs_total", "count"), bowl_runs=("runs_total", "sum"),
         bowl_wkts=("is_wicket", "sum"))
    .reset_index()
)

def compute_bat_score(balls, runs):
    """Normalized batting score 0-100. SR=150 → 100, SR=0 → 0. Matches post-toss model scale."""
    if balls < 20:
        return np.nan
    sr = runs / balls * 100
    return max(0.0, min(100.0, (sr / 150.0) * 100.0))

def compute_bowl_score(balls, runs, wkts):
    """Normalized bowling score 0-100. Matches 10_post_toss_model.py formula."""
    if balls < 12:
        return np.nan
    economy = runs / balls * 6
    # Proxy wickets per innings: assume ~24 balls per bowling spell on average
    wkts_per_inns = wkts / max(1, balls / 24)
    econ_score = max(0.0, (10.0 - economy) / 6.0)
    wkt_score  = min(wkts_per_inns / 2.0, 1.0)
    return max(0.0, (econ_score * 0.5 + wkt_score * 0.5) * 100.0)

def get_player_bat_scores(players, match_date, match_season, all_seasons_list):
    """Get blended bat scores: 0.6*current + 0.4*prev when enough data."""
    if not isinstance(players, str):
        return 0.0
    player_list = players.split("|")

    # Current season stats (before this match date)
    curr = bat_by_match[
        (bat_by_match["season"] == match_season) &
        (bat_by_match["date"] < match_date) &
        (bat_by_match["batter"].isin(player_list))
    ].groupby("batter").agg(bat_balls=("bat_balls", "sum"), bat_runs=("bat_runs", "sum"))

    # Previous season stats (full season)
    idx = all_seasons_list.index(match_season) if match_season in all_seasons_list else 0
    prev_season = all_seasons_list[idx - 1] if idx > 0 else match_season
    prev = bat_by_match[
        (bat_by_match["season"] == prev_season) &
        (bat_by_match["batter"].isin(player_list))
    ].groupby("batter").agg(bat_balls=("bat_balls", "sum"), bat_runs=("bat_runs", "sum"))

    scores = []
    for p in player_list:
        curr_score = np.nan
        prev_score = np.nan
        if p in curr.index:
            r = curr.loc[p]
            curr_score = compute_bat_score(r["bat_balls"], r["bat_runs"])
        if p in prev.index:
            r = prev.loc[p]
            prev_score = compute_bat_score(r["bat_balls"], r["bat_runs"])

        # Blend: prefer current season if enough data
        if not np.isnan(curr_score) and not np.isnan(prev_score):
            scores.append(0.6 * curr_score + 0.4 * prev_score)
        elif not np.isnan(curr_score):
            scores.append(curr_score)
        elif not np.isnan(prev_score):
            scores.append(prev_score)

    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return float(np.mean(scores[:5]))  # top 5

def get_player_bowl_scores(players, match_date, match_season, all_seasons_list):
    """Get blended bowl scores: 0.6*current + 0.4*prev when enough data."""
    if not isinstance(players, str):
        return 0.0
    player_list = players.split("|")

    curr = bowl_by_match[
        (bowl_by_match["season"] == match_season) &
        (bowl_by_match["date"] < match_date) &
        (bowl_by_match["bowler"].isin(player_list))
    ].groupby("bowler").agg(
        bowl_balls=("bowl_balls", "sum"), bowl_runs=("bowl_runs", "sum"),
        bowl_wkts=("bowl_wkts", "sum")
    )

    idx = all_seasons_list.index(match_season) if match_season in all_seasons_list else 0
    prev_season = all_seasons_list[idx - 1] if idx > 0 else match_season
    prev = bowl_by_match[
        (bowl_by_match["season"] == prev_season) &
        (bowl_by_match["bowler"].isin(player_list))
    ].groupby("bowler").agg(
        bowl_balls=("bowl_balls", "sum"), bowl_runs=("bowl_runs", "sum"),
        bowl_wkts=("bowl_wkts", "sum")
    )

    scores = []
    for p in player_list:
        curr_score = np.nan
        prev_score = np.nan
        if p in curr.index:
            r = curr.loc[p]
            curr_score = compute_bowl_score(r["bowl_balls"], r["bowl_runs"], r["bowl_wkts"])
        if p in prev.index:
            r = prev.loc[p]
            prev_score = compute_bowl_score(r["bowl_balls"], r["bowl_runs"], r["bowl_wkts"])

        if not np.isnan(curr_score) and not np.isnan(prev_score):
            scores.append(0.6 * curr_score + 0.4 * prev_score)
        elif not np.isnan(curr_score):
            scores.append(curr_score)
        elif not np.isnan(prev_score):
            scores.append(prev_score)

    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return float(np.mean(scores[:3]))  # top 3

all_seasons = sorted(matches["season"].unique())

t1b, t2b, t1w, t2w = [], [], [], []
for i, (_, row) in enumerate(matches.iterrows()):
    if i % 100 == 0:
        print(f"  Processing match {i+1}/{len(matches)}...")
    md = row["date"]
    ms = row["season"]
    t1b.append(get_player_bat_scores(row["team1_players"], md, ms, all_seasons))
    t2b.append(get_player_bat_scores(row["team2_players"], md, ms, all_seasons))
    t1w.append(get_player_bowl_scores(row["team1_players"], md, ms, all_seasons))
    t2w.append(get_player_bowl_scores(row["team2_players"], md, ms, all_seasons))

matches["team1_bat_strength"]  = t1b
matches["team2_bat_strength"]  = t2b
matches["team1_bowl_strength"] = t1w
matches["team2_bowl_strength"] = t2w
matches["bat_diff"]            = matches["team1_bat_strength"] - matches["team2_bat_strength"]
matches["bowl_diff"]           = matches["team1_bowl_strength"] - matches["team2_bowl_strength"]

# ── 9. Save ───────────────────────────────────────────────────────────────
KEEP = [
    "file_id", "match_number", "season", "date", "venue", "city",
    "team1", "team2", "winner", "team1_won",
    "team1_elo", "team2_elo", "elo_diff",
    "team1_form", "team2_form", "form_diff",
    "team1_form_3", "team2_form_3", "form_3_diff",
    "team1_form_10", "team2_form_10", "form_10_diff",
    "form_weighted_diff",
    "h2h_win_rate_team1",
    "team1_won_toss", "toss_chose_bat", "toss_decision",
    "venue_toss_win_rate", "venue_bat_first_win_rate",
    "venue_avg_first_innings", "venue_chase_win_rate", "venue_matches",
    "team1_venue_win_rate", "team2_venue_win_rate",
    "match_num_in_season", "is_playoff",
    "team1_bat_strength", "team2_bat_strength", "bat_diff",
    "team1_bowl_strength", "team2_bowl_strength", "bowl_diff",
    # New post-toss features
    "team1_bats_second", "toss_venue_aligned",
    "team1_chase_wr", "team2_chase_wr", "chase_wr_diff",
    "team1_chase_advantage", "team2_chase_advantage", "chase_advantage_diff",
    "early_season", "early_chase_boost",
    "venue_chase_batting_second",
    "month", "is_march",
    "venue_month_chase_wr", "venue_month_chase_batting_second",
    "venue_bat_wr", "venue_field_wr",
]
KEEP = [c for c in KEEP if c in matches.columns]
out = matches[KEEP].copy()
out.to_csv("data/match_features.csv", index=False)
print(f"\nSaved data/match_features.csv  ({len(out)} rows x {len(out.columns)} cols)")

# Show ELO spread — sanity check
elo_spread = abs(matches["elo_diff"])
print(f"ELO diff — mean: {elo_spread.mean():.1f}, max: {elo_spread.max():.1f}  (should be >50 if working)")
print(out.tail(3)[["team1", "team2", "team1_elo", "team2_elo", "elo_diff", "winner"]].to_string())

# Also save player stats CSVs for the API (full-season aggregates for latest season lookup)
bat_stats_full = (
    deliveries[deliveries["innings"].isin([1, 2])]
    .groupby(["season", "batter"])
    .agg(bat_balls=("runs_batter", "count"), bat_runs=("runs_batter", "sum"))
    .reset_index()
)
bat_stats_full = bat_stats_full[bat_stats_full["bat_balls"] >= 20].copy()
bat_stats_full["strike_rate"] = bat_stats_full["bat_runs"] / bat_stats_full["bat_balls"] * 100
bat_stats_full["bat_score"] = (bat_stats_full["strike_rate"] / 150.0 * 100.0).clip(0, 100)

bowl_stats_full = (
    deliveries[deliveries["innings"].isin([1, 2])]
    .groupby(["season", "bowler"])
    .agg(bowl_balls=("runs_total", "count"), bowl_runs=("runs_total", "sum"),
         bowl_wkts=("is_wicket", "sum"))
    .reset_index()
)
bowl_stats_full = bowl_stats_full[bowl_stats_full["bowl_balls"] >= 12].copy()
bowl_stats_full["economy"] = bowl_stats_full["bowl_runs"] / bowl_stats_full["bowl_balls"] * 6
bowl_stats_full["wkts_per_6"] = bowl_stats_full["bowl_wkts"] / bowl_stats_full["bowl_balls"] * 6
bowl_stats_full["econ_score"] = ((10.0 - bowl_stats_full["economy"]) / 6.0).clip(0, 1)
bowl_stats_full["wkt_score"]  = (bowl_stats_full["wkts_per_6"] / 2.0).clip(0, 1)
bowl_stats_full["bowl_score"] = (bowl_stats_full["econ_score"] * 0.5 + bowl_stats_full["wkt_score"] * 0.5) * 100.0

bat_stats_full.to_csv("data/player_bat_stats.csv", index=False)
bowl_stats_full.to_csv("data/player_bowl_stats.csv", index=False)
print("\nSaved player stats CSVs.")
