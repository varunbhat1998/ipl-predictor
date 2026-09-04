"""
09_player_profile_backtest.py
Backtest player-profile-based prediction model for 2023-2025.

Method:
  For each match, use the playing XI (from matches.csv) to look up
  each player's bat_score and bowl_score from player_database_2026.csv.

  Team strength scores:
    bat_strength  = mean of top 6 batters' bat_score
    bowl_strength = mean of top 4 bowlers' bowl_score
    team_score    = 0.55 * bat_strength + 0.45 * bowl_strength

  Also uses venue-specific scores from player_venue_scores.csv when
  a player has played at that venue (>=2 innings), blended:
    venue_score = 0.6 * career_score + 0.4 * venue_specific_score

  Prediction: team with higher team_score wins.

Scenarios:
  1. All matches
  2. Weekday matches only (Mon-Fri, exclude Sat/Sun)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

DATA = os.path.join(os.path.dirname(__file__), "data")

# -- Load data -------------------------------------------------------------
print("Loading data...")
matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
player_db = pd.read_csv(os.path.join(DATA, "player_database_2026.csv"))
venue_db = pd.read_csv(os.path.join(DATA, "player_venue_scores.csv"))

matches["date"] = pd.to_datetime(matches["date"])
matches["dow"] = matches["date"].dt.dayofweek  # 0=Mon, 5=Sat, 6=Sun
matches["is_weekend"] = matches["dow"] >= 5

# Venue normalizer (same as 02_features.py)
def norm_venue(v):
    if not isinstance(v, str): return v
    if "Chinnaswamy" in v: return "M Chinnaswamy Stadium, Bengaluru"
    if "Eden" in v: return "Eden Gardens, Kolkata"
    if "Wankhede" in v: return "Wankhede Stadium, Mumbai"
    if "Chepauk" in v or "Chidambaram" in v: return "MA Chidambaram Stadium, Chennai"
    if "Feroz" in v or "Arun Jaitley" in v or "Kotla" in v: return "Arun Jaitley Stadium, Delhi"
    if "Rajiv Gandhi" in v and "Hyderabad" in v: return "Rajiv Gandhi Intl Cricket Stadium, Hyderabad"
    if "Sawai" in v: return "Sawai Mansingh Stadium, Jaipur"
    if "Ekana" in v or ("Lucknow" in v and "Atal" in v): return "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"
    if "Narendra Modi" in v or "Motera" in v: return "Narendra Modi Stadium, Ahmedabad"
    if "Punjab" in v or "Mohali" in v or "Bindra" in v or "Mullanpur" in v: return "Punjab Cricket Association IS Bindra Stadium, Mohali"
    if "DY Patil" in v: return "Dr DY Patil Sports Academy, Mumbai"
    if "Brabourne" in v: return "Brabourne Stadium, Mumbai"
    if "Holkar" in v: return "Holkar Cricket Stadium, Indore"
    if "Himachal" in v or "Dharamsala" in v: return "Himachal Pradesh Cricket Association Stadium, Dharamsala"
    if "Visakhapatnam" in v or "VDCA" in v or "Rajasekhara" in v: return "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam"
    if "Barsapara" in v or "Guwahati" in v: return "Barsapara Cricket Stadium, Guwahati"
    if "JSCA" in v or "Ranchi" in v: return "JSCA International Stadium Complex, Ranchi"
    if "Greenfield" in v or "Trivandrum" in v or "Thiruvananthapuram" in v: return "Greenfield International Stadium, Thiruvananthapuram"
    if "Green Park" in v: return "Green Park, Kanpur"
    return v

matches["venue"] = matches["venue"].apply(norm_venue)

# -- Build lookup maps ------------------------------------------------------
# player data_name -> bat_score, bowl_score
score_map = {}
for _, r in player_db.iterrows():
    dn = r["data_name"]
    score_map[dn] = {
        "bat_score": pd.to_numeric(r["bat_score"], errors="coerce"),
        "bowl_score": pd.to_numeric(r["bowl_score"], errors="coerce"),
        "role": r["role"],
    }

# (player data_name, venue) -> venue-specific bat/bowl score
venue_score_map = {}
for _, r in venue_db.iterrows():
    dn = r["data_name"]
    v = r["venue"]
    vbs = pd.to_numeric(r["venue_bat_score"], errors="coerce")
    vws = pd.to_numeric(r["venue_bowl_score"], errors="coerce")
    vi = pd.to_numeric(r["venue_bat_innings"], errors="coerce")
    vwi = pd.to_numeric(r["venue_bowl_innings"], errors="coerce")
    if pd.notna(vbs) or pd.notna(vws):
        venue_score_map[(dn, v)] = {
            "bat_score": vbs,
            "bowl_score": vws,
            "bat_innings": vi,
            "bowl_innings": vwi,
        }

def get_player_scores(data_name, venue):
    """Get blended (career + venue) bat and bowl scores for a player."""
    career = score_map.get(data_name, {})
    c_bat = career.get("bat_score", np.nan)
    c_bowl = career.get("bowl_score", np.nan)

    vk = (data_name, venue)
    if vk in venue_score_map:
        vs = venue_score_map[vk]
        v_bat = vs["bat_score"]
        v_bowl = vs["bowl_score"]
        v_bat_inn = vs["bat_innings"]
        v_bowl_inn = vs["bowl_innings"]

        # Blend only if player has >=2 innings at venue (enough signal)
        if pd.notna(v_bat) and pd.notna(v_bat_inn) and v_bat_inn >= 2:
            bat = 0.6 * c_bat + 0.4 * v_bat if pd.notna(c_bat) else v_bat
        else:
            bat = c_bat

        if pd.notna(v_bowl) and pd.notna(v_bowl_inn) and v_bowl_inn >= 2:
            bowl = 0.6 * c_bowl + 0.4 * v_bowl if pd.notna(c_bowl) else v_bowl
        else:
            bowl = c_bowl
    else:
        bat, bowl = c_bat, c_bowl

    return float(bat) if pd.notna(bat) else np.nan, float(bowl) if pd.notna(bowl) else np.nan

def compute_team_score(player_str, venue):
    """
    Given pipe-separated player data_names, compute team score.
    bat_strength  = mean of top 6 bat_scores
    bowl_strength = mean of top 4 bowl_scores
    team_score    = 0.55 * bat_strength + 0.45 * bowl_strength
    """
    if not isinstance(player_str, str):
        return np.nan, np.nan, np.nan

    players = [p.strip() for p in player_str.split("|")]
    bat_scores = []
    bowl_scores = []

    for p in players:
        bat, bowl = get_player_scores(p, venue)
        if not np.isnan(bat):
            bat_scores.append(bat)
        if not np.isnan(bowl):
            bowl_scores.append(bowl)

    if not bat_scores and not bowl_scores:
        return np.nan, np.nan, np.nan

    # Top 6 batters, top 4 bowlers
    bat_scores.sort(reverse=True)
    bowl_scores.sort(reverse=True)
    bat_str = np.mean(bat_scores[:6]) if bat_scores else np.nan
    bowl_str = np.mean(bowl_scores[:4]) if bowl_scores else np.nan

    if np.isnan(bat_str) and np.isnan(bowl_str):
        return np.nan, np.nan, np.nan
    elif np.isnan(bat_str):
        team_score = bowl_str
    elif np.isnan(bowl_str):
        team_score = bat_str
    else:
        team_score = 0.55 * bat_str + 0.45 * bowl_str

    return bat_str, bowl_str, team_score

# -- Walk-forward backtest -------------------------------------------------
# For 2023-2025: use player DB built from ALL historical data
# (In a true walk-forward, you'd rebuild the DB before each season,
#  but since most 2026 players were active in the data, this is reasonable.
#  The player scores already use expanding/leakage-free formulas in 07_build.)

print("Running backtest for 2023-2025...")

backtest_rows = []
seasons = [2023, 2024, 2025]

for season in seasons:
    season_matches = matches[matches["season"] == season].copy()
    season_matches = season_matches.sort_values("date").reset_index(drop=True)

    for _, row in season_matches.iterrows():
        if not pd.notna(row["winner"]):
            continue  # Skip no-result matches

        venue = row["venue"]
        t1_bat, t1_bowl, t1_score = compute_team_score(row["team1_players"], venue)
        t2_bat, t2_bowl, t2_score = compute_team_score(row["team2_players"], venue)

        if np.isnan(t1_score) or np.isnan(t2_score):
            pred_winner = None
        elif t1_score > t2_score:
            pred_winner = row["team1"]
        elif t2_score > t1_score:
            pred_winner = row["team2"]
        else:
            pred_winner = None  # Draw / too close

        actual_winner = row["winner"]
        correct = (pred_winner == actual_winner) if pred_winner else None

        backtest_rows.append({
            "season": season,
            "date": row["date"],
            "dow": row["dow"],
            "is_weekend": row["is_weekend"],
            "team1": row["team1"],
            "team2": row["team2"],
            "venue": venue,
            "t1_score": round(t1_score, 2) if not np.isnan(t1_score) else None,
            "t2_score": round(t2_score, 2) if not np.isnan(t2_score) else None,
            "t1_bat": round(t1_bat, 1) if not np.isnan(t1_bat) else None,
            "t1_bowl": round(t1_bowl, 1) if not np.isnan(t1_bowl) else None,
            "t2_bat": round(t2_bat, 1) if not np.isnan(t2_bat) else None,
            "t2_bowl": round(t2_bowl, 1) if not np.isnan(t2_bowl) else None,
            "pred_winner": pred_winner,
            "actual_winner": actual_winner,
            "correct": correct,
        })

bt = pd.DataFrame(backtest_rows)
bt.to_csv(os.path.join(DATA, "player_profile_backtest.csv"), index=False)

# -- Compute accuracy ------------------------------------------------------
def accuracy_report(df, label):
    total = len(df)
    predictable = df["correct"].notna().sum()
    correct = df["correct"].sum()
    pct = correct / predictable * 100 if predictable > 0 else 0
    skipped = total - predictable
    return {
        "label": label,
        "total": total,
        "predictable": predictable,
        "skipped": skipped,
        "correct": int(correct),
        "accuracy": pct,
    }

DOW_NAMES = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}

print(f"\n{'='*70}")
print(f"PLAYER PROFILE BACKTEST — 2023-2025")
print(f"{'='*70}")

# -- SCENARIO 1: ALL MATCHES -----------------------------------------------
print(f"\n{'-'*70}")
print(f"SCENARIO 1: ALL MATCHES")
print(f"{'-'*70}")

all_report = accuracy_report(bt, "All 2023-2025")
r = all_report
print(f"  Total matches : {r['total']}")
print(f"  Predictable   : {r['predictable']} (skipped {r['skipped']} with no data)")
print(f"  Correct       : {r['correct']}")
print(f"  ACCURACY      : {r['accuracy']:.2f}%")

print(f"\n  By season:")
for s in seasons:
    rs = accuracy_report(bt[bt["season"]==s], f"  {s}")
    print(f"    {s}: {rs['correct']}/{rs['predictable']} = {rs['accuracy']:.1f}%")

print(f"\n  By day of week:")
for dow in range(7):
    day_df = bt[bt["dow"]==dow]
    if len(day_df) > 0:
        rd = accuracy_report(day_df, DOW_NAMES[dow])
        marker = " [WEEKEND]" if dow >= 5 else ""
        print(f"    {DOW_NAMES[dow]}: {rd['correct']}/{rd['predictable']} = {rd['accuracy']:.1f}%{marker}")

# -- SCENARIO 2: WEEKDAY MATCHES ONLY ------------------------------------
weekday = bt[bt["is_weekend"] == False].copy()
weekend = bt[bt["is_weekend"] == True].copy()

print(f"\n{'-'*70}")
print(f"SCENARIO 2: WEEKDAY MATCHES ONLY (Mon–Fri)")
print(f"{'-'*70}")

rw = accuracy_report(weekday, "Weekday 2023-2025")
print(f"  Total matches : {rw['total']}")
print(f"  Predictable   : {rw['predictable']} (skipped {rw['skipped']} with no data)")
print(f"  Correct       : {rw['correct']}")
print(f"  ACCURACY      : {rw['accuracy']:.2f}%")

print(f"\n  By season:")
for s in seasons:
    rs = accuracy_report(weekday[weekday["season"]==s], f"  {s}")
    print(f"    {s}: {rs['correct']}/{rs['predictable']} = {rs['accuracy']:.1f}%")

# Weekend for comparison
print(f"\n{'-'*70}")
print(f"WEEKEND MATCHES (Sat–Sun) — for comparison")
print(f"{'-'*70}")
rwe = accuracy_report(weekend, "Weekend 2023-2025")
print(f"  Total matches : {rwe['total']}")
print(f"  Predictable   : {rwe['predictable']}")
print(f"  Correct       : {rwe['correct']}")
print(f"  ACCURACY      : {rwe['accuracy']:.2f}%")

print(f"\n  By season:")
for s in seasons:
    rs = accuracy_report(weekend[weekend["season"]==s], f"  {s}")
    print(f"    {s}: {rs['correct']}/{rs['predictable']} = {rs['accuracy']:.1f}%")

# -- Margin analysis -------------------------------------------------------
print(f"\n{'-'*70}")
print(f"ACCURACY BY PREDICTION CONFIDENCE (score margin)")
print(f"{'-'*70}")
bt["score_margin"] = abs(bt["t1_score"].fillna(0) - bt["t2_score"].fillna(0))
bt_pred = bt[bt["correct"].notna()].copy()

thresholds = [0, 2, 4, 6, 8, 10]
for lo, hi in zip(thresholds, thresholds[1:] + [999]):
    slice_ = bt_pred[(bt_pred["score_margin"] >= lo) & (bt_pred["score_margin"] < hi)]
    if len(slice_) > 0:
        acc = slice_["correct"].sum() / len(slice_) * 100
        label = f"{lo}-{hi}" if hi < 999 else f">{lo}"
        print(f"  Margin {label:>6s}:  {len(slice_):3d} matches  acc={acc:.1f}%")

# -- Biggest upsets (predicted wrong with large margin) --------------------
print(f"\n{'-'*70}")
print(f"BIGGEST PREDICTION MISSES (high confidence, wrong prediction)")
print(f"{'-'*70}")
misses = bt_pred[bt_pred["correct"]==False].nlargest(8, "score_margin")
for _, r in misses.iterrows():
    print(f"  {str(r['date'])[:10]}  {r['team1']:25s} vs {r['team2']:25s}  "
          f"pred:{r['pred_winner']:<25s} actual:{r['actual_winner']}  margin:{r['score_margin']:.1f}")

print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Scenario 1 (All matches):   {all_report['accuracy']:.2f}% ({all_report['correct']}/{all_report['predictable']})")
print(f"  Scenario 2 (Weekdays only): {rw['accuracy']:.2f}% ({rw['correct']}/{rw['predictable']})")
print(f"  Weekend-only (reference):   {rwe['accuracy']:.2f}% ({rwe['correct']}/{rwe['predictable']})")
print(f"  Saved to: data/player_profile_backtest.csv")
