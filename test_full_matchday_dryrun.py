"""
Full match-day dry run — RCB vs SRH, March 28.
Simulates every stage: pre-toss → toss → inn1 overs → break → inn2 overs → result.
Sends real Telegram messages at each stage.
Tests every API endpoint in sequence.
"""
import requests, os, time, math
from dotenv import load_dotenv
load_dotenv()

TOKEN    = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

TEAM1  = "Royal Challengers Bengaluru"
TEAM2  = "Sunrisers Hyderabad"
VENUE  = "M Chinnaswamy Stadium, Bengaluru"
T1S    = "RCB"
T2S    = "SRH"

RCB_XI = ["V Kohli","RM Patidar","PD Salt","TH David","JG Bethell",
          "KH Pandya","R Shepherd","VR Iyer","JR Hazlewood","Yash Dayal","Vicky Ostwal"]
SRH_XI = ["TM Head","Abhishek Sharma","Ishan Kishan","H Klaasen",
          "Nitish Kumar Reddy","BKG Mendis","PJ Cummins","LS Livingstone",
          "HV Patel","B Carse","Harsh Dubey"]

def send(msg):
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    print(f"  Telegram: {'✓ sent' if r.status_code==200 else f'ERR {r.status_code}'}")

def divider(): return "─" * 22

def prob_bar(p, width=10):
    filled = max(0, min(width, math.floor(p * width)))
    return "█" * filled + "░" * (width - filled)

def conf_label(c):
    return {"high":"🟢 HIGH","medium":"🟡 MEDIUM","low":"🔴 LOW"}.get(c.lower(), c.upper())

def pause(label):
    print(f"\n  → {label}")
    time.sleep(1)

errors = []

# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  IPL 2026 MATCH DAY DRY RUN")
print(f"  {TEAM1} vs {TEAM2}")
print(f"  {VENUE}")
print("=" * 60)

# ── STAGE 1: Health check ─────────────────────────────────────────
pause("Stage 1: API health check")
r = requests.get(f"{API_BASE}/health", timeout=5)
assert r.status_code == 200, "API not running!"
print(f"  API: {r.json()}")

# ── STAGE 2: Pre-toss prediction ─────────────────────────────────
pause("Stage 2: Pre-toss prediction")
r = requests.post(f"{API_BASE}/predict/prematch",
                  json={"team1": TEAM1, "team2": TEAM2, "venue": VENUE}, timeout=10)
assert r.status_code == 200, f"prematch failed: {r.text}"
d = r.json()
p1 = d["team1_win_probability"] * 100
p2 = d["team2_win_probability"] * 100
winner = d["predicted_winner"]
conf = d["confidence"]
factors = d.get("key_factors", [])

print(f"  {T1S}: {p1:.1f}%  |  {T2S}: {p2:.1f}%")
print(f"  Winner: {winner}  |  Conf: {conf}")

msg = (
    f"🏏 <b>IPL 2026 — MATCH DAY [DRY RUN]</b>\n\n"
    f"<b>{T1S} vs {T2S}</b>\n"
    f"📍 {VENUE}\n"
    f"🗓 Saturday, 28 Mar · 7:30 PM IST\n\n"
    f"{divider()}\n"
    f"📊 <b>PRE-TOSS PREDICTION</b>\n"
    f"{divider()}\n"
    f"🔴 <b>{T1S}</b>  {p1:.1f}%  {prob_bar(p1/100)}\n"
    f"🟠 <b>{T2S}</b>  {p2:.1f}%  {prob_bar(p2/100)}\n\n"
    f"🏆 <b>Predicted: {winner.split()[-1]}</b>\n"
    f"⚡ Confidence: {conf_label(conf)}\n\n"
    f"{divider()}\n"
    f"🔍 <b>Key Factors</b>\n"
    + "\n".join(f"• {f}" for f in factors) +
    f"\n\n⏳ <i>Toss update to follow at 7:00 PM...</i>"
)
send(msg)

# ── STAGE 3: Post-toss (RCB won toss, chose to field) ─────────────
pause("Stage 3: Post-toss — RCB won toss, chose to field")
r = requests.post(f"{API_BASE}/predict/prematch", json={
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "toss_winner": TEAM1, "toss_decision": "field",
    "team1_players": RCB_XI, "team2_players": SRH_XI,
}, timeout=10)
assert r.status_code == 200, f"post-toss failed: {r.text}"
d2 = r.json()
p1t = d2["team1_win_probability"] * 100
p2t = d2["team2_win_probability"] * 100
winner2 = d2["predicted_winner"]
conf2 = d2["confidence"]
factors2 = d2.get("key_factors", [])
ps = d2.get("player_strengths", {})
xi_used = d2.get("xi_data_used", False)

print(f"  {T1S}: {p1t:.1f}%  |  {T2S}: {p2t:.1f}%")
print(f"  XI used: {xi_used}  |  Source: {ps.get('source')}")

msg2 = (
    f"🎯 <b>TOSS UPDATE — {T1S} vs {T2S}</b>\n\n"
    f"🪙 <b>RCB won toss · chose to FIELD</b>\n"
    f"🟠 Batting 1st: <b>SRH</b>\n"
    f"🔴 Chasing: <b>RCB</b>\n\n"
    f"{divider()}\n"
    f"📊 <b>UPDATED PREDICTION</b>\n"
    f"{divider()}\n"
    f"🔴 <b>{T1S}</b>  {p1t:.1f}%  {prob_bar(p1t/100)}\n"
    f"🟠 <b>{T2S}</b>  {p2t:.1f}%  {prob_bar(p2t/100)}\n\n"
    f"🏆 <b>Predicted: {winner2.split()[-1]}</b>\n"
    f"⚡ Confidence: {conf_label(conf2)}\n\n"
)
if xi_used and ps:
    msg2 += (
        f"{divider()}\n"
        f"💪 <b>XI Strength (venue-adjusted)</b>\n"
        f"{divider()}\n"
        f"        Bat    Bowl\n"
        f"🔴 RCB  {ps.get('team1_bat',0):.1f}   {ps.get('team1_bowl',0):.1f}\n"
        f"🟠 SRH  {ps.get('team2_bat',0):.1f}   {ps.get('team2_bowl',0):.1f}\n\n"
    )
msg2 += f"{divider()}\n🔍 <b>Key Factors</b>\n" + "\n".join(f"• {f}" for f in factors2)
send(msg2)

# ── STAGE 4: Live 1st innings — SRH batting ───────────────────────
inn1_snapshots = [
    (6,  52,  1),   # over 6  — end of powerplay
    (10, 89,  2),   # over 10
    (15, 131, 3),   # over 15
    (20, 174, 6),   # over 20 — innings end
]

print(f"\nStage 4: Live 1st innings (SRH batting)")
for over, runs, wkts in inn1_snapshots:
    balls = over * 6
    r = requests.post(f"{API_BASE}/predict/live_inn1", json={
        "batting_team": TEAM2, "bowling_team": TEAM1,
        "runs_scored": runs, "wickets_fallen": wkts,
        "balls_bowled": balls, "venue": VENUE,
    }, timeout=10)

    if r.status_code != 200:
        print(f"  Over {over}: live_inn1 error {r.status_code}: {r.text[:100]}")
        errors.append(f"live_inn1 over {over}")
        continue

    pred = r.json()
    p_bat = pred["batting_team_win_probability"]
    ms    = pred["match_state"]
    proj  = ms["projected_score"]
    crr   = ms["current_run_rate"]
    proj_lo, proj_hi = int(proj) - 7, int(proj) + 7

    print(f"  Over {over}: SRH {runs}/{wkts} | proj {proj_lo}-{proj_hi} | SRH win {p_bat*100:.1f}%")

    if over in (10, 20):   # only send over 10 and end-of-innings to Telegram
        msg = (
            f"🟠 <b>LIVE · Over {over} · SRH {runs}/{wkts}</b>\n\n"
            f"📈 Projected: <b>{proj_lo}–{proj_hi}</b>\n"
            f"⚡ CRR: {crr:.1f}\n\n"
            f"{divider()}\n"
            f"📊 <b>Win Probability</b>\n"
            f"{divider()}\n"
            f"🟠 <b>SRH</b>  {p_bat*100:.1f}%  {prob_bar(p_bat)}\n"
            f"🔴 <b>RCB</b>  {(1-p_bat)*100:.1f}%  {prob_bar(1-p_bat)}"
            + (f"\n\n<i>Over {over} update</i>" if over < 20 else "")
        )
        send(msg)

# ── STAGE 5: Innings break ────────────────────────────────────────
pause("Stage 5: Innings break")
target = 175
msg3 = (
    f"⏸ <b>INNINGS BREAK</b>\n\n"
    f"🟠 <b>SRH:</b> 174/6  (20.0 ov)\n\n"
    f"{divider()}\n"
    f"🎯 <b>Target for RCB: {target}</b>\n"
    f"{divider()}\n\n"
    f"⏳ <i>2nd innings begins shortly...</i>"
)
send(msg3)

# ── STAGE 6: Live 2nd innings — RCB chasing ───────────────────────
inn2_snapshots = [
    (6,  52,  1),   # over 6
    (10, 83,  2),   # over 10
    (15, 118, 3),   # over 15
    (18, 148, 5),   # over 18 — pressure
    (19, 162, 7),   # over 19 — very tight
]

print(f"\nStage 6: Live 2nd innings (RCB chasing {target})")
for over, runs, wkts in inn2_snapshots:
    balls = over * 6
    r = requests.post(f"{API_BASE}/predict/live", json={
        "batting_team": TEAM1, "bowling_team": TEAM2,
        "runs_scored": runs, "wickets_fallen": wkts,
        "balls_bowled": balls, "target": target,
    }, timeout=10)

    if r.status_code != 200:
        print(f"  Over {over}: live error {r.status_code}: {r.text[:100]}")
        errors.append(f"live_inn2 over {over}")
        continue

    pred   = r.json()
    p_rcb  = pred["batting_team_win_probability"]
    ms     = pred["match_state"]
    rrr    = ms["required_run_rate"]
    crr    = ms["current_run_rate"]
    needed = ms["runs_needed"]
    left   = ms["balls_remaining"]
    gap    = round(float(rrr) - float(crr), 1)
    gap_s  = f"+{gap}" if gap >= 0 else str(gap)

    print(f"  Over {over}: RCB {runs}/{wkts} | need {needed} off {left} | RCB {p_rcb*100:.1f}%")

    msg = (
        f"🔴 <b>LIVE · Over {over} · RCB {runs}/{wkts}</b>\n\n"
        f"🎯 Target: {target}  ·  Need: <b>{needed} off {left} balls</b>\n"
        f"⚡ CRR: {crr:.1f}  ·  RRR: <b>{rrr:.1f}</b>  ·  Gap: {gap_s}\n\n"
        f"{divider()}\n"
        f"📊 <b>Win Probability</b>\n"
        f"{divider()}\n"
        f"🔴 <b>RCB</b>  {p_rcb*100:.1f}%  {prob_bar(p_rcb)}\n"
        f"🟠 <b>SRH</b>  {(1-p_rcb)*100:.1f}%  {prob_bar(1-p_rcb)}"
    )
    send(msg)

# ── STAGE 7: Match result ─────────────────────────────────────────
pause("Stage 7: Match result")
msg4 = (
    f"🏁 <b>MATCH RESULT</b>\n\n"
    f"🏆 <b>SRH won by 13 runs</b>\n\n"
    f"{divider()}\n"
    f"🟠 SRH: <b>174/6</b>  (20.0 ov)\n"
    f"🔴 RCB: <b>161/9</b>  (20.0 ov)\n"
    f"{divider()}\n\n"
    f"🤖 <i>Models retraining with this result...</i>"
)
send(msg4)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DRY RUN COMPLETE")
print(f"  Errors: {len(errors)}")
if errors:
    for e in errors: print(f"    ✗ {e}")
else:
    print("  All stages passed ✓")
print()
print("  Messages sent to Telegram:")
print("  1. Pre-toss prediction")
print("  2. Post-toss update (with XI strengths)")
print("  3. Live — Over 10 (1st innings)")
print("  4. Live — Over 20 / innings end")
print("  5. Innings break")
print("  6. Live — Over 6  (2nd innings)")
print("  7. Live — Over 10 (2nd innings)")
print("  8. Live — Over 15 (2nd innings)")
print("  9. Live — Over 18 (pressure)")
print(" 10. Live — Over 19 (very tight)")
print(" 11. Match result")
print("=" * 60)
