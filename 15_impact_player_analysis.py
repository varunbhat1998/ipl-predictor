"""
15_impact_player_analysis.py — Impact Player Analysis (IPL 2023-2025)

The Impact Player rule was introduced in IPL 2023.
Each team nominates a 12th player; they can substitute him in during the match.
The nominated player is always listed last (index 11) in team1/team2_players.

This script:
  1. Extracts all impact player nominations from matches.csv (2023-2025)
  2. Identifies who actually played (appeared in deliveries)
  3. Computes their role (bat/bowl) and performance stats
  4. Outputs data/impact_player_log.csv for use in post-toss model

Output columns:
  file_id, date, season, venue, team, player, role,
  balls_bowled, runs_conceded, wickets, economy,
  balls_faced, runs_scored, was_used
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("Impact Player Analysis — IPL 2023-2025")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
m  = pd.read_csv("data/matches.csv")
d  = pd.read_csv("data/deliveries.csv")
m["date"] = pd.to_datetime(m["date"])
d["date"] = pd.to_datetime(d["date"])
m["file_id"] = m["file_id"].astype(str)
d["file_id"] = d["file_id"].astype(str)

# Impact player rule started IPL 2023
m_impact = m[m["season"] >= 2023].copy()
d_impact = d[d["file_id"].isin(m_impact["file_id"].astype(str))].copy()

print(f"Matches (2023-25): {len(m_impact)}")
print(f"Deliveries:        {len(d_impact)}")

# ── Extract impact player nominations ─────────────────────────────────
records = []
for _, row in m_impact.iterrows():
    fid  = str(row["file_id"])
    date = row["date"]
    season = row["season"]
    venue  = row["venue"]

    t1_players = str(row["team1_players"]).split("|")
    t2_players = str(row["team2_players"]).split("|")

    match_del = d_impact[d_impact["file_id"] == fid]
    all_batters = set(match_del["batter"].dropna()) | set(match_del["non_striker"].dropna())
    all_bowlers = set(match_del["bowler"].dropna())

    for team, players in [(row["team1"], t1_players), (row["team2"], t2_players)]:
        if len(players) < 12:
            continue  # no impact player nominated (pre-2023 style or missing data)

        impact = players[11].strip()
        if not impact:
            continue

        regular_xi = set(p.strip() for p in players[:11])

        # Did they actually play?
        batted  = impact in all_batters
        bowled  = impact in all_bowlers
        was_used = batted or bowled

        # Stats
        bowl_del = match_del[match_del["bowler"] == impact]
        bat_del  = match_del[match_del["batter"] == impact]

        balls_bowled   = int((bowl_del["is_wide"] == 0).sum())
        runs_conceded  = int(bowl_del["runs_total"].sum()) if len(bowl_del) else 0
        wickets        = int(bowl_del["is_wicket"].sum())  if len(bowl_del) else 0
        economy        = round(runs_conceded / balls_bowled * 6, 2) if balls_bowled >= 6 else np.nan

        balls_faced    = int((bat_del["is_wide"] == 0).sum()) if len(bat_del) else 0
        runs_scored    = int(bat_del["runs_batter"].sum())    if len(bat_del) else 0

        role = []
        if bowled: role.append("bowl")
        if batted: role.append("bat")
        role_str = "+".join(role) if role else "none"

        records.append({
            "file_id":       fid,
            "date":          date,
            "season":        int(season),
            "venue":         venue,
            "team":          team,
            "player":        impact,
            "role":          role_str,
            "was_used":      was_used,
            "balls_bowled":  balls_bowled,
            "runs_conceded": runs_conceded,
            "wickets":       wickets,
            "economy":       economy,
            "balls_faced":   balls_faced,
            "runs_scored":   runs_scored,
        })

df = pd.DataFrame(records)
print(f"\nTotal nominations: {len(df)}")
print(f"Actually used:     {df['was_used'].sum()} ({df['was_used'].mean()*100:.1f}%)")

# ── Summary stats ──────────────────────────────────────────────────────
print("\nRole breakdown (used players):")
print(df[df["was_used"]]["role"].value_counts())

print("\nSeason breakdown:")
print(df.groupby("season").agg(
    nominations=("player", "count"),
    used=("was_used", "sum"),
    use_rate=("was_used", "mean"),
    bowl_pct=("role", lambda x: (x == "bowl").mean()),
).round(3))

used = df[df["was_used"] & (df["balls_bowled"] >= 6)]
print(f"\nBowling impact players (>=6 balls): {len(used)}")
print(f"  Avg overs bowled: {used['balls_bowled'].mean()/6:.1f}")
print(f"  Avg economy:      {used['economy'].mean():.2f}")
print(f"  Avg wickets:      {used['wickets'].mean():.2f}")

# ── Who are the most-used impact players? ────────────────────────────
print("\nTop impact player nominees (by usage count):")
top = (df[df["was_used"]]
       .groupby("player")
       .agg(appearances=("file_id","count"),
            total_wkts=("wickets","sum"),
            avg_eco=("economy","mean"))
       .sort_values("appearances", ascending=False)
       .head(15))
print(top.round(2))

# ── Save ──────────────────────────────────────────────────────────────
out = Path("data/impact_player_log.csv")
df.to_csv(out, index=False)
print(f"\nSaved {out} ({len(df)} rows)")
print("Done.")
