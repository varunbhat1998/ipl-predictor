"""
Preview all 5 Telegram message templates with mock RCB vs SRH data.
"""
import os, requests
from dotenv import load_dotenv
load_dotenv()

TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def send(msg):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
        timeout=10
    )
    print("OK" if r.status_code == 200 else f"ERR {r.status_code}: {r.text[:100]}")

def divider():   return "─" * 22
def prob_bar(p, width=10):
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)
def conf_label(c):
    return {"high":"🟢 HIGH","medium":"🟡 MEDIUM","low":"🔴 LOW"}.get(c.lower(), c.upper())

# ── 1. PRE-TOSS ────────────────────────────────────────────────────────────
msg1 = (
    f"🏏 <b>IPL 2026 — MATCH DAY</b>\n\n"
    f"<b>RCB vs SRH</b>\n"
    f"📍 M Chinnaswamy Stadium, Bengaluru\n"
    f"🗓 Saturday, 28 Mar · 7:30 PM IST\n\n"
    f"{divider()}\n"
    f"📊 <b>PRE-TOSS PREDICTION</b>\n"
    f"{divider()}\n"
    f"🔴 <b>RCB</b>  48.4%  {prob_bar(0.484)}\n"
    f"🟠 <b>SRH</b>  51.6%  {prob_bar(0.516)}\n\n"
    f"🏆 <b>Predicted: SRH</b>\n"
    f"⚡ Confidence: {conf_label('low')}\n\n"
    f"{divider()}\n"
    f"🔍 <b>Key Factors</b>\n"
    f"• RCB has higher ELO rating\n"
    f"• SRH leads H2H (last 10 games)\n"
    f"• Chinnaswamy historically favours chasing\n\n"
    f"⏳ <i>Toss update to follow at 7:00 PM...</i>"
)
print("Sending template 1: Pre-toss"); send(msg1)

# ── 2. POST-TOSS ───────────────────────────────────────────────────────────
msg2 = (
    f"🎯 <b>TOSS UPDATE — RCB vs SRH</b>\n\n"
    f"🪙 <b>RCB won toss · chose to FIELD</b>\n"
    f"🟠 Batting 1st: <b>SRH</b>\n"
    f"🔴 Chasing: <b>RCB</b>\n\n"
    f"{divider()}\n"
    f"📊 <b>UPDATED PREDICTION</b>\n"
    f"{divider()}\n"
    f"🔴 <b>RCB</b>  52.1%  {prob_bar(0.521)}\n"
    f"🟠 <b>SRH</b>  47.9%  {prob_bar(0.479)}\n\n"
    f"🏆 <b>Predicted: RCB</b>\n"
    f"⚡ Confidence: {conf_label('low')}\n\n"
    f"{divider()}\n"
    f"💪 <b>XI Strength (venue-adjusted)</b>\n"
    f"{divider()}\n"
    f"{'':8}{'Bat':>6}  {'Bowl':>6}\n"
    f"🔴 {'RCB':<6}  {'93.5':>5}   {'39.2':>5}\n"
    f"🟠 {'SRH':<6}  {'93.5':>5}   {'37.4':>5}\n\n"
    f"{divider()}\n"
    f"🔍 <b>Key Factors</b>\n"
    f"• RCB has higher ELO rating\n"
    f"• Toss well-aligned with Chinnaswamy — venue favours chase\n"
    f"• RCB chasing at home: 58% win rate\n\n"
    f"<i>Chinnaswamy's dew factor in the second innings suits RCB's power hitters. "
    f"Kohli and Bethell are in strong form heading into the opener.</i>"
)
print("Sending template 2: Post-toss"); send(msg2)

# ── 3. LIVE 1ST INNINGS ────────────────────────────────────────────────────
msg3 = (
    f"🟠 <b>LIVE · Over 10 · SRH 82/1</b>\n\n"
    f"📈 Projected: <b>168–182</b>\n"
    f"⚡ CRR: 8.2\n\n"
    f"{divider()}\n"
    f"📊 <b>Win Probability</b>\n"
    f"{divider()}\n"
    f"🟠 <b>SRH</b>  54.0%  {prob_bar(0.54)}\n"
    f"🔴 <b>RCB</b>  46.0%  {prob_bar(0.46)}"
)
print("Sending template 3: Live 1st innings"); send(msg3)

# ── 4. INNINGS BREAK ───────────────────────────────────────────────────────
msg4 = (
    f"⏸ <b>INNINGS BREAK</b>\n\n"
    f"🟠 <b>SRH:</b> 174/6  (20.0 ov)\n\n"
    f"{divider()}\n"
    f"🎯 <b>Target for RCB: 175</b>\n"
    f"{divider()}\n\n"
    f"⏳ <i>2nd innings begins shortly...</i>"
)
print("Sending template 4: Innings break"); send(msg4)

# ── 5. LIVE 2ND INNINGS ────────────────────────────────────────────────────
msg5 = (
    f"🔴 <b>LIVE · Over 15 · RCB 118/3</b>\n\n"
    f"🎯 Target: 175  ·  Need: <b>57 off 30 balls</b>\n"
    f"⚡ CRR: 7.9  ·  RRR: <b>11.4</b>  ·  Gap: +3.5\n\n"
    f"{divider()}\n"
    f"📊 <b>Win Probability</b>\n"
    f"{divider()}\n"
    f"🟠 <b>SRH</b>  68.0%  {prob_bar(0.68)}\n"
    f"🔴 <b>RCB</b>  32.0%  {prob_bar(0.32)}"
)
print("Sending template 5: Live 2nd innings"); send(msg5)

# ── 6. MATCH RESULT ────────────────────────────────────────────────────────
msg6 = (
    f"🏁 <b>MATCH RESULT</b>\n\n"
    f"🏆 <b>SRH won by 28 runs</b>\n\n"
    f"{divider()}\n"
    f"🟠 SRH: <b>174/6</b>  (20.0 ov)\n"
    f"🔴 RCB: <b>146/9</b>  (20.0 ov)\n"
    f"{divider()}\n\n"
    f"🤖 <i>Models retraining with this result...</i>"
)
print("Sending template 6: Match result"); send(msg6)

print("\nAll 6 templates sent. Check Telegram.")
