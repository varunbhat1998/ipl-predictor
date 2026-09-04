"""
11_unified_live_model.py — Unified Live Match Predictor

Trains a SINGLE LightGBM model on both 1st and 2nd innings snapshots.
Label: bat_first_wins (consistent across both innings)
- Inn1 snapshot: did the batting-first team win?
- Inn2 snapshot: did the chasing team LOSE? (= bat_first won)

Key advantage over two separate models:
- No model switch at innings break → no probability jump
- Model sees full context in Inn2: both "what inn1 scored" AND "how the chase is going"
- Learns cross-inning patterns (e.g. high inn1 score + early chase wickets)

Output: models/unified_live_model.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from collections import defaultdict

Path("models").mkdir(exist_ok=True)


def _phase_aware_projection(runs, balls, crr, venue_avg, total_balls=120):
    """Phase-aware projected score. Must match 04_api.py exactly."""
    balls_rem = max(0, total_balls - balls)
    if balls_rem == 0:
        return float(runs)
    venue_rr = venue_avg / 20
    if balls <= 36:
        blend = balls / 120
        proj_rr = blend * crr + (1 - blend) * venue_rr
    elif balls <= 90:
        blend = 0.3 + 0.7 * (balls - 36) / 54
        proj_rr = blend * crr + (1 - blend) * venue_rr
    else:
        proj_rr = crr
    return round(runs + proj_rr * balls_rem / 6, 1)


print("=" * 60)
print("Unified Live Model (Inn1 + Inn2 combined)")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────
deliveries = pd.read_csv("data/deliveries.csv")
deliveries["date"]    = pd.to_datetime(deliveries["date"])
deliveries["season"]  = deliveries["date"].dt.year.astype(str)
deliveries["file_id"] = deliveries["file_id"].astype(str)

matches = pd.read_csv("data/matches.csv", parse_dates=["date"])
matches["file_id"] = matches["file_id"].astype(str)
matches["season"]  = matches["season"].astype(str)
matches = matches.sort_values("date").reset_index(drop=True)

TEST_SEASONS = ["2023", "2024", "2025"]

# ── Venue name normaliser (mirrors 03_train.py) ───────────────────────
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

deliveries["venue"] = deliveries["venue"].apply(norm_venue)
matches["venue"]    = matches["venue"].apply(norm_venue)

# ── Expanding venue averages (no leakage) ────────────────────────────
inn1_totals_all = (
    deliveries[deliveries["innings"] == 1]
    .groupby("file_id")["runs_total"].sum()
    .rename("inn1_score")
    .reset_index()
)
inn1_totals_all["file_id"] = inn1_totals_all["file_id"].astype(str)
matches_with_score = matches.merge(inn1_totals_all, on="file_id", how="left")
matches_with_score = matches_with_score.merge(
    deliveries[deliveries["innings"] == 1][["file_id", "venue"]].drop_duplicates(),
    on="file_id", how="left"
)

venue_avg_at_match = {}  # file_id -> venue avg up to (not including) this match
_venue_runs, _venue_counts = defaultdict(list), defaultdict(int)
for _, row in matches_with_score.iterrows():
    fid = row["file_id"]
    venue = row.get("venue_y") or row.get("venue_x") or row.get("venue", "")
    score = row.get("inn1_score")
    if venue:
        existing = _venue_runs[venue]
        venue_avg_at_match[fid] = np.mean(existing) if existing else 160.0
        if pd.notna(score):
            _venue_runs[venue].append(score)

# ── Load per-match PP/phase wicket win rates (expanding-window) ──────
_pp_wkt_wr_df = pd.read_csv("data/pp_wicket_win_rates_by_match.csv")
_pp_wkt_wr_df["file_id"] = _pp_wkt_wr_df["file_id"].astype(str)
# Lookup: (file_id, innings, phase) → prior_win_rate
_pp_wkt_wr_lookup = {}
for _, r in _pp_wkt_wr_df.iterrows():
    key = (str(r["file_id"]), int(r["innings"]), r["phase"])
    _pp_wkt_wr_lookup[key] = r["prior_win_rate"] if pd.notna(r["prior_win_rate"]) else 0.0
print(f"Loaded PP wicket win rate lookup: {len(_pp_wkt_wr_lookup)} entries")

def _get_phase_wkt_wr(file_id, innings, over_num):
    """Get team's historical phase-wicket win rate for completed phases.
    Over 1-6:  no completed phase → 0
    Over 7-15: PP complete → PP wicket win rate
    Over 16+:  middle complete → middle wicket win rate
    """
    if over_num <= 6:
        return 0.0  # PP not yet complete
    elif over_num <= 15:
        return _pp_wkt_wr_lookup.get((str(file_id), int(innings), "pp"), 0.0)
    else:
        return _pp_wkt_wr_lookup.get((str(file_id), int(innings), "middle"), 0.0)

# ── Match context: ELO, form, winner, inn1 team ───────────────────────
fid_to_winner     = dict(zip(matches["file_id"], matches["winner"]))
fid_to_inn1_team  = dict(zip(matches["file_id"], matches["inn1_team"]))
fid_to_season     = dict(zip(matches["file_id"], matches["season"]))

# Build ELO/form maps from match_features.csv
mf = pd.read_csv("data/match_features.csv")
mf["file_id"] = mf["file_id"].astype(str)

match_context = {}  # file_id -> {elo_diff, form_diff, venue_bat_first_win_rate, venue_chase_win_rate}
for _, r in mf.iterrows():
    fid = r["file_id"]
    inn1_team = fid_to_inn1_team.get(fid)
    if not inn1_team: continue
    if inn1_team == r["team1"]:
        elo_diff  = r.get("elo_diff", 0)
        form_diff = r.get("form_diff", 0)
    else:
        elo_diff  = -r.get("elo_diff", 0)
        form_diff = -r.get("form_diff", 0)
    match_context[fid] = {
        "elo_diff":  elo_diff,
        "form_diff": form_diff,
        "venue_bat_first_win_rate": r.get("venue_bat_first_win_rate", 0.5),
        "venue_chase_win_rate":     r.get("venue_chase_win_rate", 0.5),
    }

# ── Helper: momentum features over a ball sequence ───────────────────
def compute_momentum(balls_df):
    """Given sorted ball-level rows, compute running momentum state per ball."""
    runs   = balls_df["runs_total"].values
    wkts   = balls_df["is_wicket"].values
    bounds = (balls_df["runs_batter"].values >= 4).astype(int)
    dots   = (balls_df["runs_total"].values == 0).astype(int)

    partnership_runs, partnership_balls = 0, 0
    max_partnership = 0   # does NOT reset on wicket — tracks largest partnership this innings
    recent_runs, recent_wkts = [], []
    total_bounds, total_dots = 0, 0

    rows = []
    for i in range(len(balls_df)):
        partnership_runs += runs[i]
        partnership_balls += 1
        recent_runs.append(runs[i])
        recent_wkts.append(int(wkts[i]))
        if len(recent_runs) > 18: recent_runs.pop(0); recent_wkts.pop(0)
        total_bounds += bounds[i]
        total_dots   += dots[i]
        # Update max before reset so it captures the just-ended partnership
        max_partnership = max(max_partnership, partnership_runs)
        if wkts[i]: partnership_runs = 0; partnership_balls = 0
        rows.append({
            "partnership_runs": partnership_runs,
            "partnership_balls": partnership_balls,
            "max_partnership":  max_partnership,
            "last_3ov_runs":  sum(recent_runs),
            "last_3ov_wkts":  sum(recent_wkts),
            "boundary_pct":   total_bounds / (i + 1),
            "dot_pct":        total_dots   / (i + 1),
        })
    return rows

# ═══════════════════════════════════════════════════════════════════════
# BUILD TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding unified snapshots from both innings...")

all_snaps = []
valid_fids = set(matches[matches["winner"].notna()]["file_id"])

# ── Group deliveries by match ─────────────────────────────────────────
inn1_del = deliveries[deliveries["innings"] == 1].copy()
inn2_del = deliveries[deliveries["innings"] == 2].copy()

# Inn2: compute target and first-innings wickets per match
inn1_totals = (
    inn1_del.groupby("file_id")["runs_total"].sum().rename("inn1_final_runs") + 1
).reset_index()
inn1_totals.columns = ["file_id", "target"]

inn1_wkts = (
    inn1_del.groupby("file_id")["is_wicket"].sum().rename("first_innings_wickets")
).reset_index()

inn2_del = inn2_del.merge(inn1_totals, on="file_id", how="left")
inn2_del = inn2_del.merge(inn1_wkts, on="file_id", how="left")

# ── Process 1st innings ───────────────────────────────────────────────
print("  Processing 1st innings...")
inn1_snaps_count = 0

for fid, group in inn1_del.sort_values(["file_id","over","ball_in_over"]).groupby("file_id"):
    if fid not in valid_fids: continue
    winner   = fid_to_winner.get(fid)
    inn1team = fid_to_inn1_team.get(fid)
    if not winner or not inn1team: continue

    bat_first_won = int(inn1team == winner)
    venue_avg     = venue_avg_at_match.get(fid, 160.0)
    ctx           = match_context.get(fid, {})
    season        = fid_to_season.get(fid, "2020")

    group = group.sort_values(["over", "ball_in_over"]).reset_index(drop=True)
    group["ball_num"] = range(1, len(group) + 1)

    momentum = compute_momentum(group)
    pp_runs_locked, pp_wkts_locked = 0, 0

    for i, row in group.iterrows():
        bn = row["ball_num"]
        if bn % 6 != 0 or bn > 120: continue
        over_num = bn // 6

        cr  = row["cum_runs"]
        cw  = row["cum_wickets"]
        crr = cr / bn * 6 if bn > 0 else 0.0
        expected_at = venue_avg * (bn / 120)
        projected   = _phase_aware_projection(cr, bn, crr, venue_avg)

        # Lock powerplay at end of over 6
        if over_num == 6:
            pp_runs_locked = cr
            pp_wkts_locked = cw

        mom = momentum[i]

        all_snaps.append({
            "file_id": fid, "season": season,
            # Context
            "current_innings":     1,
            "innings_balls":       bn,
            "innings_balls_rem":   120 - bn,
            "innings_balls_pct":   bn / 120,
            # Inn1 state (running)
            "inn1_runs":           cr,
            "inn1_wickets":        cw,
            "inn1_crr":            crr,
            "inn1_projected":      projected,
            "inn1_vs_avg":         cr - expected_at,
            "inn1_vs_avg_pct":     cr / expected_at if expected_at > 0 else 1.0,
            "inn1_balls_pct":      bn / 120,
            "inn1_acceleration":   (mom["last_3ov_runs"] / 18 * 6 - crr) if bn >= 36 else 0.0,
            # Inn2 state (not happened yet → 0)
            "inn2_runs":           0,
            "inn2_wickets":        0,
            "inn2_crr":            0.0,
            "inn2_rrr":            0.0,
            "inn2_rrr_diff":       0.0,
            "inn2_run_rate_ratio": 0.0,
            "inn2_runs_needed":    0,
            "inn2_balls_rem":      0,
            "inn2_balls_pct":      0.0,
            "first_innings_wickets": 0,
            "target":              0,
            "target_vs_venue_avg": 0.0,
            # Momentum (bat_first's)
            "partnership_runs":    mom["partnership_runs"],
            "partnership_balls":   mom["partnership_balls"],
            "partnership_quality": mom["partnership_runs"] / max(venue_avg / 20, 1.0),
            "max_partnership":     mom["max_partnership"],
            "last_3ov_runs":       mom["last_3ov_runs"],
            "last_3ov_wkts":       mom["last_3ov_wkts"],
            "boundary_pct":        mom["boundary_pct"],
            "dot_pct":             mom["dot_pct"],
            # Powerplay (current innings)
            "is_pp":               int(over_num <= 6),
            "pp_runs":             pp_runs_locked if over_num > 6 else 0,
            "pp_wickets":          pp_wkts_locked if over_num > 6 else 0,
            "pp_run_rate":         pp_runs_locked / 36 * 6 if over_num > 6 else 0.0,
            "pp_req_rate":         0.0,   # no target in Inn1
            "pp_rate_gap":         0.0,
            # Pre-match (bat_first perspective)
            "elo_diff":            ctx.get("elo_diff", 0),
            "form_diff":           ctx.get("form_diff", 0),
            "venue_avg":           venue_avg,
            "venue_bat_first_win_rate": ctx.get("venue_bat_first_win_rate", 0.5),
            "venue_chase_win_rate":     ctx.get("venue_chase_win_rate", 0.5),
            # Phase wicket win rate (team-specific)
            "team_phase_wkt_wr":   _get_phase_wkt_wr(fid, 1, over_num),
            # Label
            "bat_first_wins":      bat_first_won,
        })
        inn1_snaps_count += 1

print(f"  Inn1 snapshots: {inn1_snaps_count}")

# ── Process 2nd innings ───────────────────────────────────────────────
print("  Processing 2nd innings...")
inn2_snaps_count = 0

for fid, group in inn2_del.sort_values(["file_id","over","ball_in_over"]).groupby("file_id"):
    if fid not in valid_fids: continue
    winner   = fid_to_winner.get(fid)
    inn1team = fid_to_inn1_team.get(fid)
    if not winner or not inn1team: continue

    bat_first_won = int(inn1team == winner)
    target        = group["target"].iloc[0] if "target" in group.columns else None
    inn1_wkt_cnt  = group["first_innings_wickets"].iloc[0] if "first_innings_wickets" in group.columns else 0
    if pd.isna(target): continue
    target = int(target)
    inn1_final_runs = target - 1
    venue_avg       = venue_avg_at_match.get(fid, 160.0)
    ctx             = match_context.get(fid, {})
    season          = fid_to_season.get(fid, "2020")

    # Inn1 derived
    inn1_final_crr  = inn1_final_runs / 120 * 6
    inn1_vs_avg     = inn1_final_runs - venue_avg

    group = group.sort_values(["over", "ball_in_over"]).reset_index(drop=True)
    group["ball_num"] = range(1, len(group) + 1)

    momentum = compute_momentum(group)
    pp_runs_locked, pp_wkts_locked = 0, 0

    for i, row in group.iterrows():
        bn = row["ball_num"]
        if bn % 6 != 0 or bn > 120: continue
        over_num = bn // 6

        cr           = row["cum_runs"]
        cw           = row["cum_wickets"]
        crr          = cr / bn * 6 if bn > 0 else 0.0
        balls_rem    = 120 - bn
        runs_needed  = max(0, target - cr)
        rrr          = runs_needed / balls_rem * 6 if balls_rem > 0 else 99.0
        rrr_diff     = rrr - crr
        rr_ratio     = crr / rrr if rrr > 0 else 1.0

        # Lock powerplay at end of over 6
        if over_num == 6:
            pp_runs_locked = cr
            pp_wkts_locked = cw

        pp_run_rate = pp_runs_locked / 36 * 6 if over_num > 6 else 0.0
        pp_req_rate = (target - pp_runs_locked) / 84 * 6 if over_num > 6 else 0.0
        pp_rate_gap = pp_run_rate - pp_req_rate if over_num > 6 else 0.0

        mom = momentum[i]

        all_snaps.append({
            "file_id": fid, "season": season,
            # Context
            "current_innings":     2,
            "innings_balls":       bn,
            "innings_balls_rem":   balls_rem,
            "innings_balls_pct":   bn / 120,
            # Inn1 final state (known)
            "inn1_runs":           inn1_final_runs,
            "inn1_wickets":        inn1_wkt_cnt,
            "inn1_crr":            inn1_final_crr,
            "inn1_projected":      0.0,           # already happened
            "inn1_vs_avg":         inn1_vs_avg,
            "inn1_vs_avg_pct":     inn1_final_runs / venue_avg if venue_avg > 0 else 1.0,
            "inn1_balls_pct":      1.0,           # complete
            "inn1_acceleration":   0.0,           # not applicable
            # Inn2 current state
            "inn2_runs":           cr,
            "inn2_wickets":        cw,
            "inn2_crr":            crr,
            "inn2_rrr":            rrr,
            "inn2_rrr_diff":       rrr_diff,
            "inn2_run_rate_ratio": min(rr_ratio, 3.0),
            "inn2_runs_needed":    runs_needed,
            "inn2_balls_rem":      balls_rem,
            "inn2_balls_pct":      bn / 120,
            "first_innings_wickets": int(inn1_wkt_cnt) if pd.notna(inn1_wkt_cnt) else 0,
            "target":              target,
            "target_vs_venue_avg": target / venue_avg if venue_avg > 0 else 1.0,
            # Momentum (bat_second's, the chasing team)
            "partnership_runs":    mom["partnership_runs"],
            "partnership_balls":   mom["partnership_balls"],
            "partnership_quality": mom["partnership_runs"] / max(venue_avg / 20, 1.0),
            "max_partnership":     mom["max_partnership"],
            "last_3ov_runs":       mom["last_3ov_runs"],
            "last_3ov_wkts":       mom["last_3ov_wkts"],
            "boundary_pct":        mom["boundary_pct"],
            "dot_pct":             mom["dot_pct"],
            # Powerplay (chasing team's)
            "is_pp":               int(over_num <= 6),
            "pp_runs":             pp_runs_locked if over_num > 6 else 0,
            "pp_wickets":          pp_wkts_locked if over_num > 6 else 0,
            "pp_run_rate":         pp_run_rate,
            "pp_req_rate":         pp_req_rate,
            "pp_rate_gap":         pp_rate_gap,
            # Pre-match (bat_first perspective — consistent with Inn1)
            "elo_diff":            ctx.get("elo_diff", 0),
            "form_diff":           ctx.get("form_diff", 0),
            "venue_avg":           venue_avg,
            "venue_bat_first_win_rate": ctx.get("venue_bat_first_win_rate", 0.5),
            "venue_chase_win_rate":     ctx.get("venue_chase_win_rate", 0.5),
            # Phase wicket win rate (team-specific)
            "team_phase_wkt_wr":   _get_phase_wkt_wr(fid, 2, over_num),
            # Label: bat_first_wins = chaser LOST
            "bat_first_wins":      bat_first_won,
        })
        inn2_snaps_count += 1

print(f"  Inn2 snapshots: {inn2_snaps_count}")
print(f"  Total snapshots: {len(all_snaps)}")

# ── Build DataFrame ───────────────────────────────────────────────────
snap_df = pd.DataFrame(all_snaps)
snap_df["season"] = snap_df["season"].astype(str)

UNIFIED_FEATURES = [
    # Innings context
    "current_innings", "innings_balls", "innings_balls_rem", "innings_balls_pct",
    # Inn1 state
    "inn1_runs", "inn1_wickets", "inn1_crr",
    "inn1_projected", "inn1_vs_avg", "inn1_vs_avg_pct",
    "inn1_balls_pct", "inn1_acceleration",
    # Inn2 state (0 in Inn1)
    "inn2_runs", "inn2_wickets", "inn2_crr",
    "inn2_rrr", "inn2_rrr_diff", "inn2_run_rate_ratio",
    "inn2_runs_needed", "inn2_balls_rem", "inn2_balls_pct",
    "first_innings_wickets", "target", "target_vs_venue_avg",
    # Momentum
    "partnership_runs", "partnership_balls",
    "partnership_quality", "max_partnership",
    "last_3ov_runs", "last_3ov_wkts",
    "boundary_pct", "dot_pct",
    # Powerplay
    "is_pp", "pp_runs", "pp_wickets",
    "pp_run_rate", "pp_req_rate", "pp_rate_gap",
    # Pre-match
    "elo_diff", "form_diff", "venue_avg",
    "venue_bat_first_win_rate", "venue_chase_win_rate",
    # Team-specific phase wicket win rate
    "team_phase_wkt_wr",
]

snap_train = snap_df[~snap_df["season"].isin(TEST_SEASONS)].copy()
snap_test  = snap_df[ snap_df["season"].isin(TEST_SEASONS)].copy()

X_tr = snap_train[UNIFIED_FEATURES].fillna(0)
y_tr = snap_train["bat_first_wins"].astype(int)
X_te = snap_test[UNIFIED_FEATURES].fillna(0)
y_te = snap_test["bat_first_wins"].astype(int)

print(f"\nTrain: {len(snap_train)} snapshots | Test: {len(snap_test)} snapshots")

# ── Train unified LightGBM ────────────────────────────────────────────
print("Training unified LightGBM (GroupKFold calibration)...")

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

base = lgb.LGBMClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.03,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, random_state=42, verbosity=-1,
)

groups = snap_train["file_id"].values
gkf    = GroupKFold(n_splits=5)
cv_splits = list(gkf.split(X_tr_s, y_tr, groups=groups))

model = CalibratedClassifierCV(base, cv=cv_splits, method="isotonic")
model.fit(X_tr_s, y_tr)

probs = model.predict_proba(X_te_s)[:, 1]
preds = (probs >= 0.5).astype(int)

print(f"\nOverall accuracy : {accuracy_score(y_te, preds):.3f}")
print(f"Log-loss         : {log_loss(y_te, probs):.4f}")
print(f"Brier score      : {brier_score_loss(y_te, probs):.4f}")

# ── Per-over accuracy breakdown ───────────────────────────────────────
snap_test_cp = snap_test.copy()
snap_test_cp["prob"]  = probs
snap_test_cp["pred"]  = preds
snap_test_cp["over"]  = snap_test_cp["innings_balls"] // 6
snap_test_cp["correct"] = (preds == y_te.values).astype(int)

print("\nPer-over accuracy breakdown:")
print(f"  {'Over':>4}  {'Inn':>3}  {'Acc':>6}  {'N':>5}")
for innings in [1, 2]:
    sub_inn = snap_test_cp[snap_test_cp["current_innings"] == innings]
    for ov in [1, 3, 6, 10, 15, 20]:
        sub = sub_inn[sub_inn["over"] == ov]
        if len(sub) > 0:
            print(f"  {ov:>4}  Inn{innings}  {sub['correct'].mean()*100:>5.1f}%  {len(sub):>5}")

# ── Compare with incumbent models on test set ─────────────────────────
print("\nComparison vs separate models on same test set...")
try:
    with open("models/live_model.pkl", "rb") as f:
        old_inn2 = pickle.load(f)
    X_inn2 = snap_test_cp[snap_test_cp["current_innings"] == 2]
    old_feats = old_inn2["features"]
    # Inn2 test set: label from chaser's perspective = 1-bat_first_wins
    y_inn2_chase = 1 - X_inn2["bat_first_wins"]
    X_inn2_feats = X_inn2.reindex(columns=old_feats, fill_value=0).fillna(0)
    X_inn2_s = old_inn2["scaler"].transform(X_inn2_feats)
    old_probs = old_inn2["model"].predict_proba(X_inn2_s)[:, 1]
    old_acc = accuracy_score(y_inn2_chase, (old_probs >= 0.5).astype(int))
    new_inn2_probs = snap_test_cp[snap_test_cp["current_innings"] == 2]["prob"]
    new_inn2_y    = snap_test_cp[snap_test_cp["current_innings"] == 2]["bat_first_wins"]
    new_acc = accuracy_score(new_inn2_y, (new_inn2_probs >= 0.5).astype(int))
    print(f"  Inn2: old model {old_acc:.3f}  |  unified {new_acc:.3f}")
except Exception as e:
    print(f"  (comparison skipped: {e})")

try:
    with open("models/inn1_live_model.pkl", "rb") as f:
        old_inn1 = pickle.load(f)
    X_inn1 = snap_test_cp[snap_test_cp["current_innings"] == 1]
    old_feats1 = old_inn1["features"]
    X_inn1_feats = X_inn1.reindex(columns=old_feats1, fill_value=0).fillna(0)
    X_inn1_s = old_inn1["scaler"].transform(X_inn1_feats)
    old_probs1 = old_inn1["model"].predict_proba(X_inn1_s)[:, 1]
    old_acc1 = accuracy_score(X_inn1["bat_first_wins"], (old_probs1 >= 0.5).astype(int))
    new_inn1_probs = snap_test_cp[snap_test_cp["current_innings"] == 1]["prob"]
    new_inn1_y    = snap_test_cp[snap_test_cp["current_innings"] == 1]["bat_first_wins"]
    new_acc1 = accuracy_score(new_inn1_y, (new_inn1_probs >= 0.5).astype(int))
    print(f"  Inn1: old model {old_acc1:.3f}  |  unified {new_acc1:.3f}")
except Exception as e:
    print(f"  (comparison skipped: {e})")

# ── Save ──────────────────────────────────────────────────────────────
bundle = {
    "model":    model,
    "scaler":   scaler,
    "features": UNIFIED_FEATURES,
}
with open("models/unified_live_model.pkl", "wb") as f:
    pickle.dump(bundle, f)
print("\nSaved models/unified_live_model.pkl")
print("Done.")
