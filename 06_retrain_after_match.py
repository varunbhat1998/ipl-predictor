"""
06_retrain_after_match.py
Run this after each match result is confirmed.
Appends the new match to matches.csv, reruns features + training, reloads the API.

Usage:
  python 06_retrain_after_match.py \
    --match_id 1234567 \
    --team1 "Royal Challengers Bengaluru" \
    --team2 "Sunrisers Hyderabad" \
    --winner "Royal Challengers Bengaluru" \
    --toss_winner "Sunrisers Hyderabad" \
    --toss_decision field \
    --venue "M Chinnaswamy Stadium, Bengaluru" \
    --date 2026-03-28 \
    --inn1_score 186 --inn1_wkts 5 \
    --inn2_score 174 --inn2_wkts 8

This script is called automatically by n8n Workflow 1 after result is confirmed.
"""

import argparse, subprocess, requests, sys, os
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--match_id",      required=True)
parser.add_argument("--team1",         required=True)
parser.add_argument("--team2",         required=True)
parser.add_argument("--winner",        default="", help="Winner team name (empty for tied/no-result matches)")
parser.add_argument("--toss_winner",   required=True)
parser.add_argument("--toss_decision", required=True)
parser.add_argument("--venue",         required=True)
parser.add_argument("--date",          required=True)
parser.add_argument("--inn1_score",    type=int, default=0)
parser.add_argument("--inn1_wkts",     type=int, default=10)
parser.add_argument("--inn2_score",    type=int, default=0)
parser.add_argument("--inn2_wkts",     type=int, default=10)
parser.add_argument("--team1_players", default="", help="Pipe-delimited player list for team1")
parser.add_argument("--team2_players", default="", help="Pipe-delimited player list for team2")
parser.add_argument("--player_of_match", default=None)
parser.add_argument("--season",        default="2026")
parser.add_argument("--api_url",       default="http://localhost:8000")
args = parser.parse_args()

print(f"\n{'='*50}")
print(f"Post-match retrain: {args.team1} vs {args.team2}")
print(f"Winner: {args.winner or '(tied/no result)'}")
print(f"{'='*50}")

# ── 1. Append result to matches.csv ───────────────────────────────────────
matches_path = Path("data/matches.csv")
df = pd.read_csv(matches_path)

# Check not already added
if str(args.match_id) in df["file_id"].astype(str).values:
    print(f"Match {args.match_id} already in dataset. Skipping append.")
else:
    inn1_team = args.team1   # batting first = team1 by convention here
    inn2_team = args.team2
    new_row = {
        "file_id":          args.match_id,
        "match_number":     None,
        "season":           args.season,
        "date":             args.date,
        "venue":            args.venue,
        "city":             None,
        "team1":            args.team1,
        "team2":            args.team2,
        "toss_winner":      args.toss_winner,
        "toss_decision":    args.toss_decision,
        "toss_winner_won":  int(args.toss_winner == args.winner) if args.winner else None,
        "winner":           args.winner if args.winner else None,
        "win_by_runs":      None,
        "win_by_wickets":   None,
        "player_of_match":  args.player_of_match,
        "team1_players":    args.team1_players,
        "team2_players":    args.team2_players,
        "inn1_team":        inn1_team,
        "inn1_runs":        args.inn1_score,
        "inn1_wickets":     args.inn1_wkts,
        "inn2_team":        inn2_team,
        "inn2_runs":        args.inn2_score,
        "inn2_wickets":     args.inn2_wkts,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(matches_path, index=False)
    print(f"[OK] Appended match to matches.csv ({len(df)} total rows)")

# ── 2. Rebuild player database (picks up new players from deliveries) ─────
print("\nRebuilding player database (07_build_player_db.py)...")
result = subprocess.run([sys.executable, "07_build_player_db.py"], capture_output=True, text=True)
if result.returncode != 0:
    print("WARNING in 07_build_player_db.py:", result.stderr[-500:])
else:
    # Print key summary lines
    for line in result.stdout.split("\n"):
        if any(kw in line for kw in ["Auto-discovered", "Total", "Saved", "players"]):
            print(" ", line.strip())
    print("[OK] Player database rebuilt")

# ── 3. Rerun feature engineering ──────────────────────────────────────────
print("\nRunning 02_features.py...")
result = subprocess.run([sys.executable, "02_features.py"], capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR in 02_features.py:", result.stderr)
    sys.exit(1)
print("[OK] Features rebuilt")

# ── 4. Retrain models ─────────────────────────────────────────────────────
print("\nRunning 03_train.py (prematch + live + inn1)...")
result = subprocess.run([sys.executable, "03_train.py"], capture_output=True, text=True)
if result.returncode != 0:
    print("ERROR in 03_train.py:", result.stderr[-500:])
    # Try live model standalone
    result2 = subprocess.run([sys.executable, "03_live_model.py"], capture_output=True, text=True)
    if result2.returncode != 0:
        print("ERROR in 03_live_model.py too:", result2.stderr[-300:])
        sys.exit(1)

# Extract accuracy from output
for line in result.stdout.split("\n"):
    if "accuracy" in line.lower() or "log-loss" in line.lower():
        print(" ", line.strip())
print("[OK] 03_train.py done")

# ── 4b. Retrain post-toss model ───────────────────────────────────────────
print("\nRunning 10_post_toss_model.py...")
result_pt = subprocess.run([sys.executable, "10_post_toss_model.py"], capture_output=True, text=True)
if result_pt.returncode != 0:
    print("ERROR in 10_post_toss_model.py:", result_pt.stderr[-500:])
else:
    for line in result_pt.stdout.split("\n"):
        if "accuracy" in line.lower() or "cv accuracy" in line.lower() or "features" in line.lower():
            print(" ", line.strip())
    print("[OK] Post-toss model retrained")

# ── 4c. Retrain unified live model ────────────────────────────────────────
print("\nRunning 11_unified_live_model.py...")
result_ul = subprocess.run([sys.executable, "11_unified_live_model.py"], capture_output=True, text=True)
if result_ul.returncode != 0:
    print("ERROR in 11_unified_live_model.py:", result_ul.stderr[-500:])
else:
    for line in result_ul.stdout.split("\n"):
        if "accuracy" in line.lower() or "saved" in line.lower() or "inn" in line.lower():
            print(" ", line.strip())
    print("[OK] Unified live model retrained")

print("[OK] All models retrained")

# ── 5. Hot-reload API ─────────────────────────────────────────────────────
print("\nReloading API models...")
try:
    resp = requests.post(f"{args.api_url}/reload-models",
                         json={"secret": "ipl2026"}, timeout=10)
    if resp.status_code == 200:
        print("[OK] API models reloaded:", resp.json())
    else:
        print("Warning: API reload returned", resp.status_code)
except Exception as e:
    print(f"Warning: Could not reach API for reload: {e}")
    print("Restart the API server manually to pick up new models.")

print(f"\n[OK] Retrain complete. Ready for next match.")
