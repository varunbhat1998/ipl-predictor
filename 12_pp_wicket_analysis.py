"""
12_pp_wicket_analysis.py  —  Powerplay / Phase Wicket Win Rate Analysis

Builds an expanding-window lookup: for each (team, phase, wicket_bucket, role),
what is the historical win rate?

Phases:
  pp     = overs 0-5  (powerplay)
  middle = overs 6-14
  death  = overs 15-19

Wicket buckets: 0, 1, 2, 3+

Role: bat_first, bat_second (chasing)

Output: data/pp_wicket_win_rates.csv
  Columns: team, phase, wicket_bucket, role, matches, wins, win_rate

Expanding window: for each match date, only uses prior matches (no leakage).
The final output contains the LATEST cumulative values (usable at prediction time).
Also outputs data/pp_wicket_win_rates_by_match.csv for training (per-match lookups).
"""

import pandas as pd
import numpy as np
from collections import defaultdict

print("Loading data...")
del_df = pd.read_csv("data/deliveries.csv")
mat_df = pd.read_csv("data/matches.csv")

# Align file_id dtype (deliveries has int, matches has str for some rows)
del_df["file_id"] = del_df["file_id"].astype(str)
mat_df["file_id"] = mat_df["file_id"].astype(str)

# Merge winner info into deliveries via file_id
match_info = mat_df[["file_id", "date", "winner", "inn1_team"]].copy()
match_info["date"] = pd.to_datetime(match_info["date"])
match_info = match_info.sort_values("date")

# Get per-innings PP stats from deliveries
print("Computing per-innings phase stats...")

def compute_innings_phase_stats(del_df):
    """For each (file_id, innings), compute wickets at end of each phase."""
    rows = []
    for (fid, inn), grp in del_df.groupby(["file_id", "innings"]):
        team = grp["batting_team"].iloc[0]

        pp = grp[grp["over"] < 6]
        mid = grp[(grp["over"] >= 6) & (grp["over"] < 15)]
        death = grp[grp["over"] >= 15]

        pp_wkts = int(pp["is_wicket"].sum()) if len(pp) > 0 else 0
        mid_wkts = int(mid["is_wicket"].sum()) if len(mid) > 0 else 0
        death_wkts = int(death["is_wicket"].sum()) if len(death) > 0 else 0

        # Cumulative wickets at each phase boundary
        pp_cum_wkts = pp_wkts
        mid_cum_wkts = pp_wkts + mid_wkts

        rows.append({
            "file_id": fid,
            "innings": inn,
            "batting_team": team,
            "pp_wickets": pp_wkts,
            "mid_wickets": mid_wkts,
            "death_wickets": death_wkts,
            "pp_cum_wkts": pp_cum_wkts,
            "mid_cum_wkts": mid_cum_wkts,
        })
    return pd.DataFrame(rows)

phase_stats = compute_innings_phase_stats(del_df)

# Merge with match info
phase_stats = phase_stats.merge(match_info, on="file_id", how="inner")

# Determine role (bat_first / bat_second)
phase_stats["role"] = phase_stats.apply(
    lambda r: "bat_first" if r["innings"] == 1 else "bat_second", axis=1
)

# Determine if this batting team won
phase_stats["won"] = (phase_stats["batting_team"] == phase_stats["winner"]).astype(int)

# Wicket bucket function
def wkt_bucket(w):
    if w == 0: return 0
    if w == 1: return 1
    if w == 2: return 2
    return 3  # 3+

phase_stats["pp_bucket"] = phase_stats["pp_wickets"].apply(wkt_bucket)
phase_stats["mid_bucket"] = phase_stats["mid_wickets"].apply(wkt_bucket)
phase_stats["death_bucket"] = phase_stats["death_wickets"].apply(wkt_bucket)

# Sort by date for expanding window
phase_stats = phase_stats.sort_values("date").reset_index(drop=True)

# ── Build expanding-window lookup (per-match, for training) ────────────────

print("Building expanding-window lookup (per-match)...")

# Track cumulative stats per (team, phase, bucket, role)
# Key: (team, phase, bucket, role) -> {"matches": N, "wins": W}
cumulative = defaultdict(lambda: {"matches": 0, "wins": 0})

per_match_rows = []

for idx, row in phase_stats.iterrows():
    team = row["batting_team"]
    role = row["role"]
    won = row["won"]
    fid = row["file_id"]
    inn = row["innings"]

    # For each phase, look up the CURRENT cumulative win rate BEFORE this match
    for phase, bucket_col in [("pp", "pp_bucket"), ("middle", "mid_bucket"), ("death", "death_bucket")]:
        bucket = row[bucket_col]
        key = (team, phase, bucket, role)

        prior = cumulative[key]
        wr = prior["wins"] / prior["matches"] if prior["matches"] >= 3 else np.nan

        per_match_rows.append({
            "file_id": fid,
            "innings": inn,
            "batting_team": team,
            "phase": phase,
            "wicket_bucket": bucket,
            "role": role,
            "prior_matches": prior["matches"],
            "prior_wins": prior["wins"],
            "prior_win_rate": wr,
        })

    # Now UPDATE cumulative for all three phases (after recording)
    for phase, bucket_col in [("pp", "pp_bucket"), ("middle", "mid_bucket"), ("death", "death_bucket")]:
        bucket = row[bucket_col]
        key = (team, phase, bucket, role)
        cumulative[key]["matches"] += 1
        cumulative[key]["wins"] += won

per_match_df = pd.DataFrame(per_match_rows)

# Save per-match lookup (for training — each row has the prior win rate at that point)
per_match_df.to_csv("data/pp_wicket_win_rates_by_match.csv", index=False)
print(f"  Saved per-match lookup: {len(per_match_df)} rows")

# ── Build final cumulative lookup (for live prediction) ───────────────────

print("Building final cumulative lookup...")

final_rows = []
for (team, phase, bucket, role), stats in cumulative.items():
    wr = stats["wins"] / stats["matches"] if stats["matches"] >= 3 else np.nan
    final_rows.append({
        "team": team,
        "phase": phase,
        "wicket_bucket": bucket,
        "role": role,
        "matches": stats["matches"],
        "wins": stats["wins"],
        "win_rate": wr,
    })

final_df = pd.DataFrame(final_rows)
final_df = final_df.sort_values(["team", "phase", "role", "wicket_bucket"])
final_df.to_csv("data/pp_wicket_win_rates.csv", index=False)
print(f"  Saved final lookup: {len(final_df)} rows")

# ── Print summary ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("POWERPLAY WICKET WIN RATE SUMMARY (bat_first)")
print("="*70)

# Show PP analysis for current teams
current_teams = [
    "Royal Challengers Bengaluru", "Mumbai Indians", "Chennai Super Kings",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
    "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants",
]

pp_bf = final_df[(final_df["phase"] == "pp") & (final_df["role"] == "bat_first")]
for team in current_teams:
    team_data = pp_bf[pp_bf["team"] == team].sort_values("wicket_bucket")
    if len(team_data) == 0:
        # Try older name variants
        continue
    print(f"\n{team}:")
    for _, r in team_data.iterrows():
        b = f"{int(r['wicket_bucket'])}+" if r["wicket_bucket"] == 3 else str(int(r["wicket_bucket"]))
        wr = f"{r['win_rate']:.0%}" if pd.notna(r["win_rate"]) else "n/a"
        print(f"  {b} wkts: {wr} ({int(r['matches'])} matches)")

print("\n" + "="*70)
print("KEY INSIGHT: Team win rates with 3+ PP wickets (bat first)")
print("="*70)
three_plus = pp_bf[pp_bf["wicket_bucket"] == 3].copy()
three_plus = three_plus[three_plus["matches"] >= 3].sort_values("win_rate")
for _, r in three_plus.iterrows():
    print(f"  {r['team']:<40} {r['win_rate']:.0%} ({int(r['matches'])} matches)")

print("\nDone.")
