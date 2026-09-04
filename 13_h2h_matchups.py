"""
13_h2h_matchups.py — Expanding-window batter vs bowler H2H matchup matrix

For each match M in the dataset, computes for every (batter, bowler) pair
the head-to-head advantage USING ONLY deliveries from matches BEFORE M
(no data leakage).

Matchup advantage (0-1 scale, 0.5 = neutral):
    If ≥12 balls faced:
        adv = 0.5 × (SR/150) + 0.5 × (1 - dismissal_rate × 5), clipped [0,1]
    If <12 balls:
        fallback = bowler's average matchup_advantage vs all batters (0.5 if no data)

Team-level feature per match:
    matchup_advantage_bf = mean advantage of bat-first top-6 batters
                           vs bowl-second top-4 bowlers
    matchup_advantage_diff = matchup_advantage_bf - 0.5  (positive = bat-first has edge)

Outputs:
    data/h2h_matchup_matrix.csv        — per-(batter, bowler, file_id) lookup
    data/h2h_team_matchup.csv          — per-match matchup_advantage_bf / diff
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
Path("data").mkdir(exist_ok=True)

print("=" * 62)
print("H2H MATCHUP MATRIX — Expanding Window")
print("=" * 62)

# ── Load data ─────────────────────────────────────────────────────────
print("\nLoading deliveries and match features...")
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

print(f"  Deliveries: {len(del_df):,}  |  Matches: {len(mf):,}")

# ── Sort for expanding window ──────────────────────────────────────────
# Sort by date then file_id so ties on same day are stable
del_sorted = del_df.sort_values(["date", "file_id", "innings", "over", "ball_in_over"]).copy()

# Only legal (non-wide) balls for batting stats
bat_legal = del_sorted[del_sorted["is_wide"] == 0].copy()

# ── Step 1: Per-(batter, bowler, file_id) raw stats ───────────────────
print("\nAggregating batter-bowler stats per match...")

bb = (bat_legal.groupby(["batter", "bowler", "file_id", "date"])
      .agg(
          runs=("runs_batter", "sum"),
          balls=("runs_batter", "count"),
          dismissals=("is_wicket", "sum"),
      )
      .reset_index()
      .sort_values(["batter", "bowler", "date", "file_id"]))

print(f"  Unique (batter, bowler, match) entries: {len(bb):,}")

# ── Step 2: Expanding cumulative sums for each (batter, bowler) pair ──
print("Computing expanding-window cumulative stats...")

bb["cum_runs"]       = bb.groupby(["batter", "bowler"])["runs"].cumsum()
bb["cum_balls"]      = bb.groupby(["batter", "bowler"])["balls"].cumsum()
bb["cum_dismissals"] = bb.groupby(["batter", "bowler"])["dismissals"].cumsum()

# Shift by 1 — prior stats = cumulative BEFORE current match
bb["prior_runs"]       = bb.groupby(["batter", "bowler"])["cum_runs"].shift(1, fill_value=0)
bb["prior_balls"]      = bb.groupby(["batter", "bowler"])["cum_balls"].shift(1, fill_value=0)
bb["prior_dismissals"] = bb.groupby(["batter", "bowler"])["cum_dismissals"].shift(1, fill_value=0)
bb["prior_matches"]    = bb.groupby(["batter", "bowler"]).cumcount()  # 0 for first encounter

# ── Step 3: Compute matchup_advantage per (batter, bowler, file_id) ───
print("Computing matchup advantages...")

MIN_BALLS = 12  # minimum balls for reliable H2H stats

def matchup_adv(balls, runs, dismissals):
    """Advantage score from batter's perspective: 0=bowler dominates, 1=batter dominates."""
    if balls < MIN_BALLS:
        return np.nan  # will fall back to bowler average
    sr = (runs / balls) * 100
    dis_rate = dismissals / (balls / 6)  # dismissals per over → normalise by 5
    adv = 0.5 * (sr / 150) + 0.5 * (1 - dis_rate * 5)
    return float(np.clip(adv, 0.0, 1.0))

bb["matchup_adv"] = bb.apply(
    lambda r: matchup_adv(r["prior_balls"], r["prior_runs"], r["prior_dismissals"]), axis=1
)

print(f"  H2H entries with >={MIN_BALLS} prior balls: "
      f"{bb['matchup_adv'].notna().sum():,} / {len(bb):,}")

# ── Step 4: Bowler-level fallback average ─────────────────────────────
# For pairs with < MIN_BALLS, fall back to this bowler's mean advantage vs all batters
print("Computing bowler-level fallback averages...")

bowler_avg_adv = (bb[bb["matchup_adv"].notna()]
                  .groupby(["bowler", "file_id"])["matchup_adv"]
                  .mean()
                  .reset_index()
                  .rename(columns={"matchup_adv": "bowler_avg_adv"}))

# Also compute a global bowler average (for bowlers not seen before in this match)
bowler_global_avg = (bb[bb["matchup_adv"].notna()]
                     .groupby("bowler")["matchup_adv"]
                     .mean()
                     .to_dict())

bb = bb.merge(bowler_avg_adv, on=["bowler", "file_id"], how="left")

# Fill NaN matchup_adv with bowler-level average, then 0.5 (neutral)
bb["matchup_adv_final"] = np.where(
    bb["matchup_adv"].notna(),
    bb["matchup_adv"],
    np.where(
        bb["bowler_avg_adv"].notna(),
        bb["bowler_avg_adv"],
        0.5,
    )
)

# ── Step 5: Build per-match lookup dict ───────────────────────────────
# (batter, bowler, file_id) → matchup_adv_final
print("Building per-match H2H lookup...")

h2h_lookup = {
    (r.batter, r.bowler, r.file_id): r.matchup_adv_final
    for _, r in bb[["batter", "bowler", "file_id", "matchup_adv_final"]].iterrows()
}

# Save per-(batter, bowler, match) matrix for reference
matrix_df = bb[["batter", "bowler", "file_id", "date",
                 "prior_balls", "prior_runs", "prior_dismissals",
                 "matchup_adv", "matchup_adv_final"]].copy()
matrix_df.to_csv("data/h2h_matchup_matrix.csv", index=False)
print(f"  Saved data/h2h_matchup_matrix.csv ({len(matrix_df):,} rows)")

# ── Step 6: Extract XI per match ──────────────────────────────────────
print("\nExtracting bat-first / bat-second XIs from deliveries...")

def extract_xi_for_match(fid):
    match = del_df[del_df["file_id"] == fid]
    inn1  = match[match["innings"] == 1]
    inn2  = match[match["innings"] == 2]

    bf_batters = list(set(inn1["batter"].dropna()) | set(inn1["non_striker"].dropna()))
    bf_batters += list(inn1.loc[inn1["player_out"].notna(), "player_out"].dropna())
    if len(inn2):
        bf_batters += list(inn2["bowler"].dropna())

    bs_bowlers = list(inn1["bowler"].dropna())
    if len(inn2):
        bs_bowlers += list(set(inn2["batter"].dropna()) | set(inn2["non_striker"].dropna()))

    bf_batters = list({p for p in bf_batters if p and str(p).strip()})
    bs_bowlers = list({p for p in bs_bowlers if p and str(p).strip()})
    return bf_batters, bs_bowlers

valid_fids = set(del_df["file_id"].unique()) & set(mf["file_id"].unique())
print(f"  Building XI for {len(valid_fids):,} matches...")
xi_cache = {fid: extract_xi_for_match(fid) for fid in valid_fids}

# ── Step 7: Compute per-match team matchup advantage ──────────────────
# bat-first top-6 batters vs bat-second top-4 bowlers
print("Computing per-match team matchup advantage...")

def team_matchup_advantage(fid, bf_batters, bs_bowlers, top_n_bat=6, top_n_bowl=4):
    """
    Mean H2H advantage of bat-first top batters vs bat-second top bowlers.
    > 0.5 means bat-first has H2H edge; < 0.5 means bat-second (bowling side) has edge.
    """
    advs = []
    # Use top-N batters/bowlers by frequency in the XI (already de-duped)
    for batter in bf_batters[:top_n_bat]:
        for bowler in bs_bowlers[:top_n_bowl]:
            key = (batter, bowler, fid)
            adv = h2h_lookup.get(key)
            if adv is None:
                # Fallback: global bowler average, then 0.5
                adv = bowler_global_avg.get(bowler, 0.5)
            advs.append(adv)
    return float(np.mean(advs)) if advs else 0.5


team_h2h = []
for fid, (bf_bat, bs_bowl) in xi_cache.items():
    adv_bf  = team_matchup_advantage(fid, bf_bat, bs_bowl)
    # Reverse: bat-second batters vs bat-first bowlers
    # (bat-first's bowlers are bat-second's innings bowlers)
    adv_bs  = team_matchup_advantage(fid, bs_bowl[:6], bf_bat[:4])  # reuse bs_bowl as batters
    diff    = adv_bf - 0.5

    # Get match date for joining
    match_dates = del_df[del_df["file_id"] == fid]["date"].iloc[0] if fid in set(del_df["file_id"]) else pd.NaT
    team_h2h.append({
        "file_id": fid,
        "matchup_advantage_bf":   round(adv_bf, 4),
        "matchup_advantage_diff": round(diff, 4),
    })

h2h_team_df = pd.DataFrame(team_h2h)
h2h_team_df.to_csv("data/h2h_team_matchup.csv", index=False)
print(f"  Saved data/h2h_team_matchup.csv ({len(h2h_team_df):,} rows)")

# ── Diagnostics ───────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("DIAGNOSTICS")
print("=" * 62)

print(f"\n  matchup_advantage_bf distribution:")
adv_series = h2h_team_df["matchup_advantage_bf"]
print(f"    mean={adv_series.mean():.3f}  std={adv_series.std():.3f}  "
      f"min={adv_series.min():.3f}  max={adv_series.max():.3f}")

# Show a famous matchup example if available
FAMOUS = [
    ("V Kohli", "Sandeep Sharma"),
    ("RG Sharma", "JJ Bumrah"),
    ("KL Rahul", "Y Chahal"),
]
print("\n  Notable H2H pairs (prior career stats):")
print(f"  {'Batter':<20} {'Bowler':<20} {'Balls':>6} {'SR':>6} {'Dis/ov':>7} {'Adv':>6}")
print(f"  {'-'*20} {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")
for batter, bowler in FAMOUS:
    sub = bb[(bb["batter"] == batter) & (bb["bowler"] == bowler)].sort_values("date")
    if len(sub) == 0:
        print(f"  {batter:<20} {bowler:<20}    N/A")
        continue
    last = sub.iloc[-1]
    pb = int(last["prior_balls"])
    pr = int(last["prior_runs"])
    pd_ = int(last["prior_dismissals"])
    sr = (pr / pb * 100) if pb > 0 else 0
    dis_rate = pd_ / (pb / 6) if pb > 0 else 0
    adv = last["matchup_adv_final"]
    print(f"  {batter:<20} {bowler:<20} {pb:>6} {sr:>5.1f}% {dis_rate:>6.2f}/ov {adv:>5.3f}")

print(f"\n  Matches with H2H data: {h2h_team_df['matchup_advantage_bf'].notna().sum():,} / {len(h2h_team_df):,}")
print("\nDone. Run 10_post_toss_model.py next to retrain with H2H features.")
