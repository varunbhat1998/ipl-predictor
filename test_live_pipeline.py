"""
Test full live pipeline against AUS vs WI Women match currently live.
Simulates exactly what match_bot.py does on IPL match day.
"""
import requests, os, json, time
from dotenv import load_dotenv
load_dotenv()

KEY      = os.environ.get("CRICAPI_KEY") or os.environ.get("CRICAPI_KEY_EVENING")
TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

def send_telegram(msg):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    ok = r.status_code == 200
    print(f"  Telegram: {'OK' if ok else r.text[:80]}")
    return ok

def divider():   return "─" * 22
def prob_bar(p, width=10):
    import math
    filled = max(0, min(width, math.floor(p * width)))
    return "█" * filled + "░" * (width - filled)

# ── Step 1: Find the live match ─────────────────────────────────────────
print("Step 1: Finding live match on CricAPI...")
r = requests.get("https://api.cricapi.com/v1/matches",
                 params={"apikey": KEY, "offset": 0}, timeout=15)
matches = r.json().get("data", [])
live = [m for m in matches if m.get("matchStarted") and not m.get("matchEnded")]

if not live:
    print("  No live matches. Checking all matches for AUS/WI...")
    all_m = matches
    live = [m for m in all_m if
            any(t in m.get("name","") for t in ["Australia","West Indies","Women"])]

print(f"  Candidates: {len(live)}")
for m in live:
    print(f"  [{m['id']}] {m.get('name','?')} | {m.get('matchType','?')}")

if not live:
    print("No match found. Exiting.")
    exit()

# Pick the AUS vs WI match
match = next((m for m in live if
              "Australia" in m.get("name","") or "West Indies" in m.get("name","")), live[0])
match_id = match["id"]
print(f"\nUsing: {match.get('name','?')}")
print(f"ID: {match_id}")

# ── Step 2: Get full match info ──────────────────────────────────────────
print("\nStep 2: Fetching match info...")
r = requests.get("https://api.cricapi.com/v1/matchInfo",
                 params={"apikey": KEY, "id": match_id}, timeout=15)
info = r.json().get("data", {})
print(f"  Status: {info.get('status','?')}")
print(f"  Teams: {info.get('teams', info.get('teamInfo','?'))}")
toss_winner = info.get("tossWinner","")
toss_choice = info.get("tossChoice","")
print(f"  Toss: {toss_winner} chose to {toss_choice}")

# ── Step 3: Get live score ───────────────────────────────────────────────
print("\nStep 3: Fetching live score...")
r = requests.get("https://api.cricapi.com/v1/matchScore",
                 params={"apikey": KEY, "id": match_id}, timeout=15)
score_data = r.json().get("data", {})
scores = score_data.get("score", [])
print(f"  Match ended: {score_data.get('matchEnded', False)}")
for s in scores:
    print(f"  {s.get('inning','?')}: {s.get('r','?')}/{s.get('w','?')} ({s.get('o','?')} ov)")

# ── Step 4: Test live prediction endpoint ───────────────────────────────
print("\nStep 4: Testing live prediction API...")

# Determine batting/bowling teams from scores
teams = [s for s in scores]
if len(teams) >= 2:
    # 2nd innings in progress
    bat_inn = teams[-1]
    bowl_inn = teams[0]
    batting_team = bat_inn.get("inning","").replace(" Inning 2","").replace(" Inning 1","").strip()
    bowling_team = bowl_inn.get("inning","").replace(" Inning 2","").replace(" Inning 1","").strip()
    runs   = int(bat_inn.get("r", 0))
    wkts   = int(bat_inn.get("w", 0))
    overs  = float(bat_inn.get("o", 0))
    balls  = int(overs) * 6 + round((overs % 1) * 10)
    target = int(bowl_inn.get("r", 0)) + 1

    print(f"  2nd innings: {batting_team} chasing {target}")
    print(f"  Score: {runs}/{wkts} ({overs} ov)")

    r = requests.post(f"{API_BASE}/predict/live", json={
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "runs_scored": runs,
        "wickets_fallen": wkts,
        "balls_bowled": balls,
        "target": target,
    }, timeout=10)

    if r.status_code == 200:
        pred = r.json()
        p_chase = pred["batting_team_win_probability"]
        p_bowl  = 1 - p_chase
        ms      = pred["match_state"]
        rrr     = ms["required_run_rate"]
        crr     = ms["current_run_rate"]
        needed  = ms["runs_needed"]
        balls_left = ms["balls_remaining"]
        rrr_gap = round(float(rrr) - float(crr), 1)
        gap_str = f"+{rrr_gap}" if rrr_gap >= 0 else str(rrr_gap)

        msg = (
            f"🏏 <b>LIVE TEST — {batting_team.split()[-1]} vs {bowling_team.split()[-1]}</b>\n\n"
            f"🎯 Target: {target}  ·  Need: <b>{needed} off {balls_left} balls</b>\n"
            f"⚡ CRR: {crr:.1f}  ·  RRR: <b>{rrr:.1f}</b>  ·  Gap: {gap_str}\n\n"
            f"{divider()}\n"
            f"📊 <b>Win Probability</b>\n"
            f"{divider()}\n"
            f"🏏 <b>{batting_team.split()[-1]}</b>  {p_chase*100:.1f}%  {prob_bar(p_chase)}\n"
            f"🎳 <b>{bowling_team.split()[-1]}</b>  {p_bowl*100:.1f}%  {prob_bar(p_bowl)}\n\n"
            f"<i>Live pipeline test — IPL bot is working</i>"
        )
        print(f"\n  Chase win prob: {p_chase*100:.1f}%")
        print(f"  RRR: {rrr} | CRR: {crr}")
        send_telegram(msg)
    else:
        print(f"  API error {r.status_code}: {r.text[:200]}")

elif len(teams) == 1:
    # 1st innings
    bat_inn = teams[0]
    batting_team = bat_inn.get("inning","").replace(" Inning 1","").strip()
    bowling_team = "opposition"
    runs   = int(bat_inn.get("r", 0))
    wkts   = int(bat_inn.get("w", 0))
    overs  = float(bat_inn.get("o", 0))
    balls  = int(overs) * 6 + round((overs % 1) * 10)

    print(f"  1st innings: {batting_team} batting")
    print(f"  Score: {runs}/{wkts} ({overs} ov)")

    r = requests.post(f"{API_BASE}/predict/live_inn1", json={
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "runs_scored": runs,
        "wickets_fallen": wkts,
        "balls_bowled": balls,
        "venue": info.get("venue",""),
    }, timeout=10)

    if r.status_code == 200:
        pred = r.json()
        p_bat = pred["batting_team_win_probability"]
        ms    = pred["match_state"]
        proj  = ms["projected_score"]
        crr   = ms["current_run_rate"]
        proj_lo, proj_hi = int(proj) - 7, int(proj) + 7

        msg = (
            f"🏏 <b>LIVE TEST — 1st Innings</b>\n\n"
            f"<b>{batting_team.split()[-1]}</b>: {runs}/{wkts} ({overs} ov)\n"
            f"📈 Projected: <b>{proj_lo}–{proj_hi}</b>\n"
            f"⚡ CRR: {crr:.1f}\n\n"
            f"{divider()}\n"
            f"📊 <b>Win Probability</b>\n"
            f"{divider()}\n"
            f"🏏 Batting  {p_bat*100:.1f}%  {prob_bar(p_bat)}\n"
            f"🎳 Bowling  {(1-p_bat)*100:.1f}%  {prob_bar(1-p_bat)}\n\n"
            f"<i>Live pipeline test — IPL bot is working</i>"
        )
        print(f"\n  Projected score: {proj}")
        print(f"  Win prob (batting): {p_bat*100:.1f}%")
        send_telegram(msg)
    else:
        print(f"  API error {r.status_code}: {r.text[:200]}")

print("\nDone. Check Telegram for the live prediction message.")
