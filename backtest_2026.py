"""
backtest_2026.py — Backtest all 3 models on 2026 IPL season matches.

Pre-match: prediction before toss
Post-toss: prediction after toss (with 60/40 anchor blend as used in bot)
Live Inn1 @ Over 10: 1st innings prediction at over 10 (estimated score)
Live Inn2 @ Over 10: 2nd innings prediction at over 10 (estimated score)

Requires the API to be running on localhost:8000.
"""

import requests, pandas as pd, numpy as np, sys

API = "http://localhost:8000"

# ── Load 2026 matches ────────────────────────────────────────────────────
df = pd.read_csv("data/matches.csv")
df = df[df["season"].astype(str) == "2026"].copy()
df = df[df["winner"].notna()].copy()  # skip abandoned matches

# Skip the CSK vs KXIP incomplete match (inn2_runs=0, inn1_runs=24)
df = df[df["inn1_runs"] > 50].copy()

print(f"Backtesting {len(df)} matches from IPL 2026\n")
print("=" * 100)

results = []

for _, row in df.iterrows():
    t1, t2 = row["team1"], row["team2"]
    venue = row["venue"]
    winner = row["winner"]
    toss_w = row["toss_winner"]
    toss_d = row["toss_decision"]
    inn1_team = row["inn1_team"]
    inn2_team = row["inn2_team"]
    inn1_r = int(row["inn1_runs"])
    inn1_w = int(row["inn1_wickets"])
    inn2_r = int(row["inn2_runs"]) if pd.notna(row["inn2_runs"]) else 0
    inn2_w = int(row["inn2_wickets"]) if pd.notna(row["inn2_wickets"]) else 0
    target = inn1_r + 1

    match_label = f"{row['date'][:10]}  {t1} vs {t2}"
    print(f"\n{match_label}")
    print(f"  Venue: {venue}")
    print(f"  Toss: {toss_w} chose to {toss_d}")
    print(f"  Result: {winner} won ({inn1_team} {inn1_r}/{inn1_w} vs {inn2_team} {inn2_r}/{inn2_w})")

    entry = {
        "date": str(row["date"])[:10], "team1": t1, "team2": t2,
        "winner": winner, "venue": venue,
        "inn1_team": inn1_team, "inn2_team": inn2_team,
        "inn1_score": f"{inn1_r}/{inn1_w}", "inn2_score": f"{inn2_r}/{inn2_w}",
    }

    # ── 1. PRE-MATCH ─────────────────────────────────────────────────────
    try:
        r = requests.post(f"{API}/predict/prematch", json={
            "team1": t1, "team2": t2, "venue": venue
        }, timeout=10)
        pre = r.json()
        p1_pre = pre["team1_win_probability"]
        pred_pre = pre["predicted_winner"]
        correct_pre = int(pred_pre == winner)
        conf_pre = abs(p1_pre - 0.5)
        entry["pre_pred"] = pred_pre
        entry["pre_p1"] = round(p1_pre * 100, 1)
        entry["pre_correct"] = correct_pre
        print(f"  Pre-match:  {pred_pre:>35s}  {max(p1_pre,1-p1_pre)*100:5.1f}%  {'OK' if correct_pre else 'WRONG'}")
    except Exception as e:
        print(f"  Pre-match:  ERROR - {e}")
        entry["pre_pred"] = None
        entry["pre_correct"] = None

    # ── 2. POST-TOSS (with 60/40 anchor blend) ──────────────────────────
    try:
        bat_first = inn1_team
        bat_second = inn2_team
        r = requests.post(f"{API}/predict/posttoss", json={
            "bat_first": bat_first, "bat_second": bat_second,
            "venue": venue, "toss_winner": toss_w, "toss_decision": toss_d,
        }, timeout=10)
        pt = r.json()
        p_bf = pt["batting_first_win_probability"]
        p_bs = pt["batting_second_win_probability"]

        # Map to team1/team2
        if bat_first == t1:
            p1_pt = p_bf * 100
        else:
            p1_pt = p_bs * 100

        # Apply 60/40 anchor blend (same as bot)
        p1_blend = 0.60 * p1_pt + 0.40 * (p1_pre * 100)
        p2_blend = 100 - p1_blend
        pred_pt = t1 if p1_blend >= 50 else t2
        correct_pt = int(pred_pt == winner)
        entry["pt_pred"] = pred_pt
        entry["pt_p1"] = round(p1_blend, 1)
        entry["pt_p1_raw"] = round(p1_pt, 1)
        entry["pt_correct"] = correct_pt
        print(f"  Post-toss:  {pred_pt:>35s}  {max(p1_blend,100-p1_blend):5.1f}%  {'OK' if correct_pt else 'WRONG'}"
              f"  (raw={max(p1_pt,100-p1_pt):.1f}%, blended)")
    except Exception as e:
        print(f"  Post-toss:  ERROR - {e}")
        entry["pt_pred"] = None
        entry["pt_correct"] = None

    # ── 3. LIVE INN1 @ OVER 10 (estimated score) ────────────────────────
    # Estimate over-10 score: ~55% of total runs, ~25% of wickets by over 10
    inn1_r_10 = int(inn1_r * 0.55)
    inn1_w_10 = max(0, min(inn1_w, int(inn1_w * 0.25 + 0.5)))  # rough estimate
    inn1_balls_10 = 60  # 10 overs

    try:
        r = requests.post(f"{API}/predict/live_inn1", json={
            "batting_team": inn1_team, "bowling_team": inn2_team,
            "runs_scored": inn1_r_10, "wickets_fallen": inn1_w_10,
            "balls_bowled": inn1_balls_10, "venue": venue,
        }, timeout=10)
        inn1_ml = r.json()
        p_bat_inn1 = inn1_ml["batting_team_win_probability"]
        # Map: batting_team = inn1_team = bat_first
        pred_inn1 = inn1_team if p_bat_inn1 >= 0.5 else inn2_team
        correct_inn1 = int(pred_inn1 == winner)
        entry["inn1_10_pred"] = pred_inn1
        entry["inn1_10_prob"] = round(max(p_bat_inn1, 1-p_bat_inn1) * 100, 1)
        entry["inn1_10_score"] = f"{inn1_r_10}/{inn1_w_10}"
        entry["inn1_10_correct"] = correct_inn1
        print(f"  Inn1 Ov10:  {pred_inn1:>35s}  {max(p_bat_inn1,1-p_bat_inn1)*100:5.1f}%  {'OK' if correct_inn1 else 'WRONG'}"
              f"  (est {inn1_r_10}/{inn1_w_10} in 10 ov)")
    except Exception as e:
        print(f"  Inn1 Ov10:  ERROR - {e}")
        entry["inn1_10_correct"] = None

    # ── 4. LIVE INN2 @ OVER 10 (estimated score) ────────────────────────
    # Estimate over-10 chase score: ~55% of final inn2 runs, ~25% of wickets
    inn2_r_10 = int(inn2_r * 0.55)
    inn2_w_10 = max(0, min(inn2_w, int(inn2_w * 0.25 + 0.5)))
    inn2_balls_10 = 60

    try:
        r = requests.post(f"{API}/predict/live", json={
            "batting_team": inn2_team, "bowling_team": inn1_team,
            "runs_scored": inn2_r_10, "wickets_fallen": inn2_w_10,
            "balls_bowled": inn2_balls_10, "target": target,
            "first_innings_wickets": inn1_w, "venue": venue,
        }, timeout=10)
        inn2_ml = r.json()
        p_chase = inn2_ml["batting_team_win_probability"]
        # batting_team = inn2_team = chasing team
        pred_inn2 = inn2_team if p_chase >= 0.5 else inn1_team
        correct_inn2 = int(pred_inn2 == winner)
        entry["inn2_10_pred"] = pred_inn2
        entry["inn2_10_prob"] = round(max(p_chase, 1-p_chase) * 100, 1)
        entry["inn2_10_score"] = f"{inn2_r_10}/{inn2_w_10}"
        entry["inn2_10_correct"] = correct_inn2
        print(f"  Inn2 Ov10:  {pred_inn2:>35s}  {max(p_chase,1-p_chase)*100:5.1f}%  {'OK' if correct_inn2 else 'WRONG'}"
              f"  (est {inn2_r_10}/{inn2_w_10} chasing {target} in 10 ov)")
    except Exception as e:
        print(f"  Inn2 Ov10:  ERROR - {e}")
        entry["inn2_10_correct"] = None

    results.append(entry)

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("BACKTEST SUMMARY — IPL 2026")
print("=" * 100)

rdf = pd.DataFrame(results)
n = len(rdf)

for model, col in [("Pre-match", "pre_correct"), ("Post-toss (60/40 blend)", "pt_correct"),
                    ("Live Inn1 @Ov10", "inn1_10_correct"), ("Live Inn2 @Ov10", "inn2_10_correct")]:
    valid = rdf[col].dropna()
    correct = int(valid.sum())
    total = len(valid)
    pct = correct / total * 100 if total > 0 else 0
    print(f"  {model:30s}: {correct}/{total} correct ({pct:.1f}%)")

print(f"\nNote: Live model scores at over 10 are ESTIMATED from final scores")
print(f"      (55% of final runs, 25% of final wickets). Actual over-10 scores")
print(f"      would require ball-by-ball data which is not available for 2026.")

# Save results
rdf.to_csv("data/backtest_2026.csv", index=False)
print(f"\nResults saved to data/backtest_2026.csv")

# ── Detailed table ───────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("DETAILED RESULTS")
print("=" * 100)
print(f"\n{'Date':<12} {'Match':<25} {'Winner':<15} {'PreMatch':<12} {'PostToss':<12} {'Inn1@10':<12} {'Inn2@10':<12}")
print("-" * 100)
for _, r in rdf.iterrows():
    def fmt(pred, correct):
        if pred is None: return "N/A"
        mark = "OK" if correct else "X"
        short = pred.split()[-1][:8]
        return f"{short} {mark}"

    match = f"{r['team1'].split()[-1][:6]} v {r['team2'].split()[-1][:6]}"
    w = r['winner'].split()[-1][:8]
    print(f"{r['date']:<12} {match:<25} {w:<15} "
          f"{fmt(r.get('pre_pred'), r.get('pre_correct')):<12} "
          f"{fmt(r.get('pt_pred'), r.get('pt_correct')):<12} "
          f"{fmt(r.get('inn1_10_pred'), r.get('inn1_10_correct')):<12} "
          f"{fmt(r.get('inn2_10_pred'), r.get('inn2_10_correct')):<12}")
