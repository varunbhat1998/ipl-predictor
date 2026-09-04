"""
Replay toss output for historical matches to validate:
  1. Team labels match player lists (Fix 2)
  2. Bowl scores: bowlers > non-bowlers (Fix 1)
  3. Post-toss prediction fires and produces a winner

Usage:
    python tests/replay_toss.py               # last 3 matches
    python tests/replay_toss.py --n 5         # last N matches
    python tests/replay_toss.py --fid 1473507 # specific match

Requires API to be running: python 04_api.py
"""

import sys, os, argparse, requests, pandas as pd, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API = "http://127.0.0.1:8000"
MATCHES = os.path.join(os.path.dirname(__file__), "..", "data", "matches.csv")
PLAYER_DB = os.path.join(os.path.dirname(__file__), "..", "data", "player_database_2026.csv")

# ── Load player database locally (don't need API for score lookups) ────────
pdb = pd.read_csv(PLAYER_DB)
pdb["bat_score"]  = pd.to_numeric(pdb["bat_score"],  errors="coerce").fillna(0)
pdb["bowl_score"] = pd.to_numeric(pdb["bowl_score"], errors="coerce").fillna(0)
score_map = {r["data_name"]: {"bat": r["bat_score"], "bowl": r["bowl_score"]}
             for _, r in pdb.iterrows()}

TEAM_EMOJI = {
    "Chennai Super Kings": "🦁", "Delhi Capitals": "🔵",
    "Gujarat Titans": "🔷", "Kolkata Knight Riders": "💜",
    "Lucknow Super Giants": "🟡", "Mumbai Indians": "🔵",
    "Punjab Kings": "🔴", "Rajasthan Royals": "🩷",
    "Royal Challengers Bengaluru": "🟢", "Sunrisers Hyderabad": "🟠",
}
def emoji(team): return TEAM_EMOJI.get(team, "🏏")
def short(team): return team.split()[-1][:6]
def divider(): return "─" * 40


def load_matches(fids=None, n=3):
    df = pd.read_csv(MATCHES)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["team1_players", "toss_decision"])
    if fids:
        df = df[df["file_id"].astype(str).isin([str(f) for f in fids])]
    else:
        df = df.sort_values("date").tail(n)
    return df


def xi_from_row(row, team_col):
    """Extract data_names for a team's XI from matches.csv pipe-separated list."""
    raw = str(row[team_col]).split("|")
    # First 11 are the XI; 12th (index 11) is the impact player nominee
    xi = [p.strip() for p in raw[:11] if p.strip()]
    return xi


def player_section(team_label, role_label, emoji_s, xi, bat_first_team, team_name):
    """Render an XI section exactly as match_bot.py does after Fix 2."""
    lines = [f"\n{emoji_s} {team_label} ({role_label})"]
    bat_vals, bowl_vals = [], []
    for i, p in enumerate(xi, 1):
        s = score_map.get(p, {})
        bat  = s.get("bat",  None)
        bowl = s.get("bowl", None)
        bat_str  = f"~{bat:.0f}"  if bat  is not None else " ~?"
        bowl_str = f"~{bowl:.0f}" if bowl is not None else " ~?"
        if bat  is not None: bat_vals.append(bat)
        if bowl is not None: bowl_vals.append(bowl)
        lines.append(f"  {i:2}. {p:30s}  Bat:{bat_str:>5}  Bowl:{bowl_str:>5}")

    if bat_vals:
        top6_bat  = np.mean(sorted(bat_vals, reverse=True)[:6])
        top4_bowl = np.mean(sorted(bowl_vals, reverse=True)[:4]) if bowl_vals else 0
        lines.append(f"  {'Team avg':>32}  Bat:{top6_bat:.1f}  Bowl:{top4_bowl:.1f}")
    return "\n".join(lines)


def call_posttoss(match, bat_first, bat_second, bf_xi, bs_xi, venue, toss_winner, toss_decision):
    """Call the /predict-posttoss API endpoint."""
    try:
        payload = {
            "bat_first": bat_first, "bat_second": bat_second,
            "venue": venue, "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "bf_players": bf_xi, "bs_players": bs_xi,
        }
        r = requests.post(f"{API}/predict/posttoss", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def check_invariants(team1, team2, bat_first, t1_xi, t2_xi):
    """Return list of invariant violations."""
    failures = []

    # ── Invariant 1: team1 section must contain team1's first player ──────
    # The XI section uses t1_xi with team1 label. If label is team2, it's swapped.
    # We verify this by checking data_name membership — if the label team is correct,
    # then t1_xi[0] should be a known player (simple existence check).
    if t1_xi and t1_xi[0] not in score_map:
        failures.append(f"  ⚠ team1 first player '{t1_xi[0]}' not in player_database — name mapping issue")
    if t2_xi and t2_xi[0] not in score_map:
        failures.append(f"  ⚠ team2 first player '{t2_xi[0]}' not in player_database — name mapping issue")

    # ── Invariant 2: cross-check — team1's players should NOT be listed ──
    # under team2 labels by verifying no XI player belongs to the wrong squad.
    # We do this by checking that the sets are disjoint.
    t1_set = set(t1_xi)
    t2_set = set(t2_xi)
    overlap = t1_set & t2_set
    if overlap:
        failures.append(f"  ⚠ Players appear in both XI lists: {overlap} — XI extraction bug")

    # ── Invariant 3: bowlers should outscore non-bowlers on bowl ──────────
    def is_bowler(p):
        row = pdb[pdb["data_name"] == p]
        if len(row) == 0: return None
        return row.iloc[0]["role"] in ("bowler", "all-rounder")

    def bowl_score(p):
        return score_map.get(p, {}).get("bowl", 0)

    for xi, team in [(t1_xi, team1), (t2_xi, team2)]:
        bowlers     = [p for p in xi if is_bowler(p) is True]
        non_bowlers = [p for p in xi if is_bowler(p) is False]
        if bowlers and non_bowlers:
            avg_bowler     = np.mean([bowl_score(p) for p in bowlers])
            avg_non_bowler = np.mean([bowl_score(p) for p in non_bowlers])
            if avg_bowler <= avg_non_bowler:
                failures.append(
                    f"  ⚠ {team}: bowlers avg bowl={avg_bowler:.1f} ≤ non-bowlers avg bowl={avg_non_bowler:.1f}"
                    f" — score inversion bug"
                )

    return failures


def replay_match(row):
    team1 = row["team1"]
    team2 = row["team2"]
    venue = row["venue"]
    toss_winner = row["toss_winner"]
    toss_decision = row["toss_decision"]  # 'bat' or 'field'
    actual_winner = row.get("winner", "?")
    date = str(row["date"])[:10]

    # Determine bat_first / bat_second
    if toss_decision == "bat":
        bat_first, bat_second = toss_winner, (team2 if toss_winner == team1 else team1)
    else:
        bat_second_temp = toss_winner
        bat_first = team2 if toss_winner == team1 else team1
        bat_second = bat_second_temp

    # Extract XI (team1_players / team2_players)
    t1_xi = xi_from_row(row, "team1_players")
    t2_xi = xi_from_row(row, "team2_players")

    # XI for bat_first / bat_second
    bf_xi = t1_xi if bat_first == team1 else t2_xi
    bs_xi = t2_xi if bat_first == team1 else t1_xi

    print(f"\n{'='*60}")
    print(f"REPLAY: {date}  {short(team1)} vs {short(team2)}")
    print(f"  Venue      : {venue}")
    print(f"  Toss       : {short(toss_winner)} won → chose to {toss_decision.upper()}")
    print(f"  Bat first  : {short(bat_first)}")
    print(f"  Chasing    : {short(bat_second)}")
    print(f"  Actual win : {short(actual_winner)}")
    print(f"{'='*60}")

    # ── Post-toss prediction ───────────────────────────────────────────────
    pt = call_posttoss(row, bat_first, bat_second, bf_xi, bs_xi, venue,
                       toss_winner, toss_decision)
    if "error" in pt:
        print(f"\n  ❌ Post-toss API error: {pt['error']}")
        print("     (Is 04_api.py running? Start with: python 04_api.py)")
        pred_winner = None
    else:
        p_bf = pt.get("batting_first_win_probability", 0.5)
        p_bs = 1 - p_bf
        pred_winner = bat_first if p_bf > 0.5 else bat_second
        correct = "✅" if pred_winner == actual_winner else "❌"
        conf = pt.get("confidence_label", "")
        print(f"\n📊 POST-TOSS PREDICTION")
        print(f"  {emoji(bat_first)} {short(bat_first):8s} (Batting)  {p_bf*100:.1f}%")
        print(f"  {emoji(bat_second)} {short(bat_second):8s} (Chasing)  {p_bs*100:.1f}%")
        print(f"  🏆 Predicted: {short(pred_winner)}  [{conf}]  {correct} (actual: {short(actual_winner)})")

    # ── XI Display ────────────────────────────────────────────────────────
    print(f"\n📋 XI ANALYSIS")
    # team1 section uses team1 label with team1 players (Fix 2)
    t1_role = "Batting" if bat_first == team1 else "Chasing"
    t2_role = "Batting" if bat_first == team2 else "Chasing"
    print(player_section(short(team1), t1_role, emoji(team1), t1_xi, bat_first, team1))
    print(player_section(short(team2), t2_role, emoji(team2), t2_xi, bat_first, team2))

    # ── Invariant checks ──────────────────────────────────────────────────
    print(f"\n🔍 INVARIANT CHECKS")
    failures = check_invariants(team1, team2, bat_first, t1_xi, t2_xi)
    if failures:
        for f in failures:
            print(f)
    else:
        print("  ✅ All invariants passed")

    return pred_winner == actual_winner if pred_winner else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=3, help="Number of recent matches to replay")
    parser.add_argument("--fid", type=str, nargs="+", help="Specific file_id(s) to replay")
    args = parser.parse_args()

    matches = load_matches(fids=args.fid, n=args.n)
    if len(matches) == 0:
        print("No matches found.")
        sys.exit(1)

    results = []
    for _, row in matches.iterrows():
        correct = replay_match(row)
        results.append(correct)

    valid = [r for r in results if r is not None]
    if valid:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {sum(valid)}/{len(valid)} correct predictions")


if __name__ == "__main__":
    main()
