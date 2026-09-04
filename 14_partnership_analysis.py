"""
14_partnership_analysis.py — Partnership Impact Analysis

Computes per-partnership statistics from ball-by-ball deliveries.
Correlates 50+/100+ partnerships with team win rates.
Output: data/partnership_stats.csv (per-match innings partnership breakdown)

Findings feed into 11_unified_live_model.py (max_partnership, partnership_quality).
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("Partnership Impact Analysis")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
deliveries = pd.read_csv("data/deliveries.csv")
deliveries["date"]    = pd.to_datetime(deliveries["date"])
deliveries["file_id"] = deliveries["file_id"].astype(str)

matches = pd.read_csv("data/matches.csv", parse_dates=["date"])
matches["file_id"] = matches["file_id"].astype(str)

fid_to_winner   = dict(zip(matches["file_id"], matches["winner"]))
fid_to_inn1team = dict(zip(matches["file_id"], matches["team1"]))  # bat_first
# Actually use toss info to determine bat_first
toss_winner_bats = matches["toss_decision"] == "bat"
bat_first_map = {}
for _, r in matches.iterrows():
    if r["toss_decision"] == "bat":
        bat_first_map[r["file_id"]] = r["toss_winner"]
    else:
        other = r["team2"] if r["toss_winner"] == r["team1"] else r["team1"]
        bat_first_map[r["file_id"]] = other

print(f"Matches: {len(matches)} | Deliveries: {len(deliveries)}")

# ── Compute partnerships ball-by-ball ─────────────────────────────────
print("\nComputing partnerships...")

all_partnerships = []

for (fid, innings), grp in deliveries[deliveries["innings"].isin([1, 2])].sort_values(
        ["file_id", "innings", "over", "ball_in_over"]
).groupby(["file_id", "innings"]):
    winner      = fid_to_winner.get(fid)
    bat_first   = bat_first_map.get(fid)
    if not winner or not bat_first:
        continue

    batting_team   = bat_first if innings == 1 else (
        [t for t in [grp["batting_team"].iloc[0]] if t][0]
        if "batting_team" in grp.columns else "unknown"
    )
    bat_first_won  = int(bat_first == winner)

    part_runs  = 0
    part_balls = 0
    part_num   = 1
    max_part   = 0

    for _, row in grp.iterrows():
        part_runs  += row["runs_total"]
        part_balls += 1

        if row["is_wicket"]:
            all_partnerships.append({
                "file_id":      fid,
                "innings":      innings,
                "partnership":  part_num,
                "runs":         part_runs,
                "balls":        part_balls,
                "run_rate":     part_runs / part_balls * 6 if part_balls > 0 else 0.0,
                "bat_first_won": bat_first_won,
            })
            if part_runs > max_part:
                max_part = part_runs
            part_runs  = 0
            part_balls = 0
            part_num  += 1

    # Final (unbroken) partnership
    if part_balls > 0:
        all_partnerships.append({
            "file_id":      fid,
            "innings":      innings,
            "partnership":  part_num,
            "runs":         part_runs,
            "balls":        part_balls,
            "run_rate":     part_runs / part_balls * 6 if part_balls > 0 else 0.0,
            "bat_first_won": bat_first_won,
        })

pship_df = pd.DataFrame(all_partnerships)
print(f"Total partnerships: {len(pship_df)}")
print(f"Avg runs per partnership: {pship_df['runs'].mean():.1f}")
print(f"Median: {pship_df['runs'].median():.1f}")

# ── Compute max_partnership per innings ────────────────────────────────
print("\nMax partnership per match-innings:")
max_part = pship_df.groupby(["file_id", "innings"])["runs"].max().reset_index()
max_part.rename(columns={"runs": "max_partnership"}, inplace=True)

print(f"  Mean max partnership: {max_part['max_partnership'].mean():.1f}")
print(f"  Median max partnership: {max_part['max_partnership'].median():.1f}")
print(f"  75th percentile: {max_part['max_partnership'].quantile(0.75):.1f}")
print(f"  90th percentile: {max_part['max_partnership'].quantile(0.90):.1f}")
print(f"  Max ever: {max_part['max_partnership'].max():.0f}")

# ── Win rate by max partnership bracket ──────────────────────────────
print("\nBat-first win rate by max partnership (1st innings):")
inn1_max = max_part[max_part["innings"] == 1].merge(
    matches[["file_id", "winner"]].merge(
        pd.DataFrame([{"file_id": k, "bat_first": v} for k, v in bat_first_map.items()]),
        on="file_id"
    ),
    on="file_id"
)
inn1_max["bat_first_won"] = (inn1_max["winner"] == inn1_max["bat_first"]).astype(int)

brackets = [0, 30, 50, 75, 100, 150, 300]
labels   = ["0-29", "30-49", "50-74", "75-99", "100-149", "150+"]
inn1_max["bracket"] = pd.cut(inn1_max["max_partnership"], bins=brackets, labels=labels, right=False)

summary = inn1_max.groupby("bracket", observed=True).agg(
    matches=("file_id", "count"),
    win_rate=("bat_first_won", "mean")
).reset_index()
print(f"  {'Bracket':>10}  {'Matches':>7}  {'BatFirst Win%':>13}")
for _, r in summary.iterrows():
    print(f"  {r['bracket']:>10}  {r['matches']:>7}  {r['win_rate']*100:>12.1f}%")

# ── Partnership threshold win rates (chasing team, Inn2) ─────────────
print("\nChase win rate by max partnership in Inn2:")
inn2_max = max_part[max_part["innings"] == 2].merge(
    matches[["file_id", "winner"]].merge(
        pd.DataFrame([{"file_id": k, "bat_first": v} for k, v in bat_first_map.items()]),
        on="file_id"
    ),
    on="file_id"
)
inn2_max["chaser_won"] = (inn2_max["winner"] != inn2_max["bat_first"]).astype(int)
inn2_max["bracket"] = pd.cut(inn2_max["max_partnership"], bins=brackets, labels=labels, right=False)

summary2 = inn2_max.groupby("bracket", observed=True).agg(
    matches=("file_id", "count"),
    chase_win_rate=("chaser_won", "mean")
).reset_index()
print(f"  {'Bracket':>10}  {'Matches':>7}  {'Chase Win%':>10}")
for _, r in summary2.iterrows():
    print(f"  {r['bracket']:>10}  {r['matches']:>7}  {r['chase_win_rate']*100:>9.1f}%")

# ── Correlation: max_partnership vs actual outcome ────────────────────
from scipy import stats
inn1_valid = inn1_max.dropna(subset=["max_partnership", "bat_first_won"])
corr, pval = stats.pointbiserialr(inn1_valid["bat_first_won"], inn1_valid["max_partnership"])
print(f"\nCorrelation (Inn1 max_partnership vs bat_first_won): r={corr:.3f}, p={pval:.4f}")

inn2_valid = inn2_max.dropna(subset=["max_partnership", "chaser_won"])
corr2, pval2 = stats.pointbiserialr(inn2_valid["chaser_won"], inn2_valid["max_partnership"])
print(f"Correlation (Inn2 max_partnership vs chaser_won):   r={corr2:.3f}, p={pval2:.4f}")

# ── Save partnership stats per match-innings ──────────────────────────
max_part_full = pship_df.groupby(["file_id", "innings"]).agg(
    max_partnership=("runs", "max"),
    total_partnerships=("partnership", "max"),
    avg_partnership=("runs", "mean"),
    partnerships_50plus=("runs", lambda x: (x >= 50).sum()),
    partnerships_100plus=("runs", lambda x: (x >= 100).sum()),
).reset_index()

out_path = Path("data/partnership_stats.csv")
max_part_full.to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(max_part_full)} rows)")

# ── Top partnerships ever ─────────────────────────────────────────────
print("\nTop 10 highest individual partnerships:")
top10 = pship_df.nlargest(10, "runs")[["file_id", "innings", "partnership", "runs", "balls", "run_rate"]]
print(top10.to_string(index=False))

print("\nDone.")
