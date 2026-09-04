"""
match_bot.py - Fully automated IPL match prediction bot

Smart polling strategy (under 100 CricAPI calls per match):
  - Pre-match: Sleep until 30 min before toss time (0 calls)
  - Toss window: Poll every 3 min from toss-15min to toss+15min (~10 calls)
  - 1st innings: Poll every 4 min for ~90 min (~22 calls)
  - Innings break: Sleep 20 min (0 calls)
  - 2nd innings: Poll every 3 min for ~90 min (~30 calls)
  - Total: ~62 calls per match, well under 100

Supports dual API keys for double-headers.
Auto-retrains after match ends.
Claude explains pre-toss and post-toss only.

Usage:
  python match_bot.py                      # Auto-detect today's match
  python match_bot.py --slot evening       # Evening game only (7:30 PM)
  python match_bot.py --slot afternoon     # Afternoon game only (3:30 PM)
  python match_bot.py --match-id abc123    # Specific CricAPI match ID

Requires:
  - API server running: python 04_api.py
  - Environment variables for tokens (see below)
"""

import requests
import time
import json
import re
import sys
import os
import subprocess
import argparse
from datetime import datetime, timedelta, timezone
import traceback
import threading
from name_map import xi_to_data_names, DATA_TO_FULL
from match_logger import mlog

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # dotenv not installed — use system env vars directly

# ======================================================================
# CONFIGURATION
# ======================================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_IDS = [
    cid.strip()
    for cid in os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID").split(",")
    if cid.strip()
]
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Dual CricAPI keys — one per match slot
CRICAPI_KEY_1 = os.environ.get("CRICAPI_KEY_1", "")
CRICAPI_KEY_2 = os.environ.get("CRICAPI_KEY_2", "")
CRICAPI_KEY_3 = os.environ.get("CRICAPI_KEY_3", "")
# Legacy fallbacks (backward compat with old .env files)
_CRICAPI_KEY_AFTERNOON = os.environ.get("CRICAPI_KEY_AFTERNOON", "")
_CRICAPI_KEY_EVENING = os.environ.get("CRICAPI_KEY_EVENING", "")
_CRICAPI_KEY_DEFAULT = os.environ.get("CRICAPI_KEY", "")

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# ── IPL Match Timings (IST) ──────────────────────────────────────────
# Source: Official IPL playing conditions
#   Toss: 15-30 min before match start
#   Innings break: 20 min (can be reduced to 10)
#   Match duration: ~3 hours
MATCH_SCHEDULE = {
    "afternoon": {
        "match_start": (15, 30),   # 3:30 PM IST
        "toss_time":   (15, 0),    # 3:00 PM IST
        "inn1_start":  (15, 30),   # 3:30 PM
        "inn1_end":    (17, 10),   # ~5:10 PM (100 min)
        "inn2_start":  (17, 30),   # 5:30 PM (20 min break)
        "inn2_end":    (19, 10),   # ~7:10 PM
    },
    "evening": {
        "match_start": (19, 30),   # 7:30 PM IST
        "toss_time":   (19, 0),    # 7:00 PM IST
        "inn1_start":  (19, 30),   # 7:30 PM
        "inn1_end":    (21, 10),   # ~9:10 PM
        "inn2_start":  (21, 30),   # 9:30 PM
        "inn2_end":    (23, 10),   # ~11:10 PM
    },
}

# Polling intervals per phase (seconds)
POLL_TOSS = 180          # Every 3 min during toss window
POLL_INN1 = 60           # Every 60s during 1st innings (catches end-of-over window)
POLL_INN2 = 60           # Every 60s during 2nd innings
POLL_FIND_MATCH = 900    # Every 15 min when waiting for match to appear

MAX_API_CALLS = 280   # 3 keys × ~100/day each; leave headroom for pre-match setup calls

# Team short names
TEAM_SHORT = {
    "Chennai Super Kings": "CSK", "Mumbai Indians": "MI",
    "Royal Challengers Bengaluru": "RCB", "Kolkata Knight Riders": "KKR",
    "Delhi Capitals": "DC", "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH", "Kings XI Punjab": "PBKS",
    "Gujarat Titans": "GT", "Lucknow Super Giants": "LSG",
}


# ======================================================================
# UTILITIES
# ======================================================================
def now_ist():
    return datetime.now(IST)


def ist_today_at(hour, minute):
    n = now_ist()
    return n.replace(hour=hour, minute=minute, second=0, microsecond=0)


def sleep_until(target_dt, label=""):
    """Sleep until a target IST datetime. Shows countdown."""
    while True:
        remaining = (target_dt - now_ist()).total_seconds()
        if remaining <= 0:
            break
        if remaining > 300:
            mins = int(remaining / 60)
            print(f"  Sleeping {mins} min until {label} ({target_dt.strftime('%H:%M IST')})...")
            time.sleep(min(remaining, 300))
        else:
            time.sleep(min(remaining, 30))
    print(f"  Reached {label} time ({now_ist().strftime('%H:%M IST')})")


def short(team):
    return TEAM_SHORT.get(team, team[:3].upper())


def overs_to_balls(overs_float):
    full = int(overs_float)
    partial = round((overs_float - full) * 10)
    return full * 6 + partial


def _parse_dl_target(status_str, fallback_target, current_runs=0):
    """Extract D/L target and max overs from status strings like:
    'MI need 61 runs off 30 balls' / 'target revised to 61 off 5 overs' / 'Target: 61'.
    Returns (target, max_balls) or (fallback_target, 120) if nothing found.
    current_runs: current score of chasing team (needed to compute target from 'need X off Y').
    """
    if not status_str:
        return fallback_target, 120
    s = status_str.lower()
    # "target revised to X" / "target: X" / "target of X" — most explicit, check first
    m2 = re.search(r'target\s+(?:revised\s+to|:?|of)\s+(\d+)', s)
    if m2:
        t = int(m2.group(1))
        mo = re.search(r'off\s+(\d+)\s+overs?', s)
        max_balls = int(mo.group(1)) * 6 if mo else 120
        return t, max_balls
    # "need X runs off Y balls" — compute target = current_runs + runs_needed
    m = re.search(r'need\s+(\d+)\s+(?:runs?\s+)?(?:off|from|in)\s+(\d+)\s+balls?', s)
    if m:
        runs_needed = int(m.group(1))
        max_balls = int(m.group(2))
        return current_runs + runs_needed, max_balls
    # "X to win off/from/in Y overs/balls"
    m3 = re.search(r'(\d+)\s+to\s+win\s+(?:off|from|in)\s+(\d+)\s+(overs?|balls?)', s)
    if m3:
        runs_to_win = int(m3.group(1))
        val = int(m3.group(2))
        max_balls = val * 6 if 'over' in m3.group(3) else val
        return current_runs + runs_to_win, max_balls
    # "need X more runs" (no ball info — keep default 120 balls)
    m4 = re.search(r'need\s+(\d+)\s+(?:more\s+)?runs?', s)
    if m4:
        return current_runs + int(m4.group(1)), 120
    return fallback_target, 120


def emoji_for_prob(prob):
    if prob > 0.60: return "[+]"
    if prob > 0.40: return "[=]"
    return "[-]"


def normalize_team_name(name):
    if not name: return name
    m = {"Punjab Kings": "Kings XI Punjab",
         "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
         "Delhi Daredevils": "Delhi Capitals"}
    return m.get(name, name)


def match_team_to_ours(cricapi_name, our_teams):
    """Match a CricAPI team name to our model's team name."""
    norm = normalize_team_name(cricapi_name)
    if norm in our_teams:
        return norm
    # Fuzzy: check word overlap
    for ot in our_teams:
        c_words = set(norm.lower().split())
        o_words = set(ot.lower().split())
        if len(c_words & o_words) >= 2:
            return ot
    return norm


# ======================================================================
# TELEGRAM FORMATTING HELPERS
# ======================================================================

TEAM_EMOJI = {
    "Royal Challengers Bengaluru": "🔴",
    "Sunrisers Hyderabad":         "🟠",
    "Mumbai Indians":              "💙",
    "Chennai Super Kings":         "💛",
    "Kolkata Knight Riders":       "💜",
    "Delhi Capitals":              "🔵",
    "Rajasthan Royals":            "🩷",
    "Lucknow Super Giants":        "🟢",
    "Gujarat Titans":              "🔷",
    "Punjab Kings":                "❤️",
    "Kings XI Punjab":             "❤️",
}

# GPS coordinates for weather lookups (Open-Meteo, no API key needed)
VENUE_COORDS = {
    "Wankhede Stadium":                                             (18.9388,  72.8258),
    "M.Chinnaswamy Stadium":                                        (12.9790,  77.5995),
    "MA Chidambaram Stadium":                                       (13.0629,  80.2792),
    "Eden Gardens":                                                 (22.5646,  88.3433),
    "Narendra Modi Stadium":                                        (23.0900,  72.0830),
    "Rajiv Gandhi International Stadium":                           (17.4046,  78.5481),
    "Punjab Cricket Association Stadium":                           (30.6943,  76.8601),
    "Sawai Mansingh Stadium":                                       (26.8869,  75.8063),
    "BRSABV Ekana Cricket Stadium":                                 (26.9034,  80.9450),
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": (26.9034,  80.9450),
    "Dr DY Patil Sports Academy":                                   (19.0443,  73.0168),
    "Arun Jaitley Stadium":                                         (28.6376,  77.2209),
    "Holkar Cricket Stadium":                                       (22.7196,  75.8577),
    "Himachal Pradesh Cricket Association Stadium":                 (31.8350,  76.9430),
}

def t_emoji(team):
    return TEAM_EMOJI.get(team, "🏏")

def prob_bar(p, width=10):
    import math
    filled = math.floor(p * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)

def conf_label(confidence):
    return {"high": "🟢 HIGH", "medium": "🟡 MEDIUM", "low": "🔴 LOW"}.get(
        confidence.lower(), confidence.upper()
    )

def divider():
    return "─" * 22

# ======================================================================
# STARTUP: Load last known XI from matches.csv (always available as fallback)
# ======================================================================
_last_xi_map: dict = {}   # team -> list[data_name str]
try:
    import pandas as _pd_xi
    _mcsv = _pd_xi.read_csv("data/matches.csv")
    _mcsv = _mcsv.sort_values("date")
    for _, _mr in _mcsv.iterrows():
        for _col, _team in [("team1_players", _mr.get("team1")), ("team2_players", _mr.get("team2"))]:
            _raw = str(_mr.get(_col, ""))
            if _raw not in ("nan", "") and _team:
                _players = [p.strip() for p in _raw.split("|") if p.strip()][:11]
                if _players:
                    _last_xi_map[_team] = _players
    print(f"[Startup] Last-known XI loaded for {len(_last_xi_map)} teams")
    del _pd_xi, _mcsv, _mr, _col, _team, _raw, _players
except Exception as _e:
    print(f"[Startup] Warning: could not load last-known XI: {_e}")

# ======================================================================
# TELEGRAM COMMAND LISTENER (runs in background thread)
# ======================================================================
# Shared match state for the /predict command
_match_state = {
    "team1": None, "team2": None, "venue": None,
    "bat_first": None, "bat_second": None,
    "toss_winner": None, "toss_decision": None,
    "t1_xi": [], "t2_xi": [],
    "phase": "idle",           # idle, pre_toss, post_toss, inn1, break, inn2, ended
    "inn1_runs": 0, "inn1_wkts": 0, "inn1_overs": 0,
    "inn2_runs": 0, "inn2_wkts": 0, "inn2_overs": 0,
    "inn1_final_wkts": 0,
    "target": 0,
    "last_prob": None,
    "match_id": None,
    "api_key": None,
    "cb_match_id": None,
    "cb_slug": None,
    # Rain tracking
    "rain_active": False,
    "rain_started_at": None,
    "rain_status": "",         # latest status text from API
    "last_score_change_at": None,
    "stall_notified": False,
    # Ball-by-ball mode (toggled via /predictASAP command)
    "ball_by_ball": False,
    # Impact player nominees (set via /impact or auto-detected from live squad)
    "bf_impact_player": None,   # bat_first team's impact nominee (data_name)
    "bs_impact_player": None,   # bat_second team's impact nominee (data_name)
    "impact_auto_detected": False,  # True once auto-detect has fired
}

# Keywords that indicate rain/interruption in API status strings
RAIN_KEYWORDS = [
    "rain", "wet outfield", "drizzle", "play stopped", "play suspended",
    "play has been suspended", "interruption", "delay", "bad light",
    "strategic time-out",  # not rain, but a pause
]
RAIN_RESUME_KEYWORDS = [
    "play resumed", "play resumes", "back in action", "play is on",
]

_tg_update_offset = 0

def _poll_telegram_commands():
    """Background thread: poll Telegram for /predict command every 5 seconds."""
    global _tg_update_offset
    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={"offset": _tg_update_offset, "timeout": 10}, timeout=15
            )
            if r.status_code != 200:
                time.sleep(5)
                continue
            updates = r.json().get("result", [])
            for u in updates:
                _tg_update_offset = u["update_id"] + 1
                msg = u.get("message", {})
                text = (msg.get("text") or "").strip().lower()
                chat_id = str(msg.get("chat", {}).get("id", ""))

                if text in ("predict", "/predict", "/p"):
                    _handle_predict_command(chat_id)
                elif text in ("/predictasap", "predictasap", "/asap"):
                    _handle_predictasap_command(chat_id)
                elif text in ("status", "/status"):
                    _handle_status_command(chat_id)
                elif text in ("rain", "/rain", "/weather"):
                    _handle_rain_command(chat_id)
                elif (msg.get("text") or "").strip().lower().startswith("/xi"):
                    _handle_xi_command(chat_id, (msg.get("text") or "").strip())
                elif (msg.get("text") or "").strip().lower().startswith("/impact"):
                    _handle_impact_command(chat_id, (msg.get("text") or "").strip())
                elif text in ("/retrain", "retrain"):
                    _handle_retrain_command(chat_id)
        except Exception:
            pass
        time.sleep(3)


def _handle_predict_command(chat_id):
    """Run current prediction and send to requesting chat."""
    s = _match_state
    if s["phase"] == "idle" or not s["team1"]:
        send_telegram("No active match. Waiting for today's IPL game.", chat_id)
        return

    t1, t2 = s["team1"], s["team2"]
    t1s, t2s = short(t1), short(t2)
    e1, e2 = t_emoji(t1), t_emoji(t2)

    if s["phase"] in ("pre_toss",):
        # Pre-toss prediction (uses post-toss model with estimated XI)
        ml = ml_pretoss(t1, t2, s["venue"])
        if ml:
            p1 = ml["team1_win_probability"] * 100
            p2 = ml["team2_win_probability"] * 100
            scenarios = ml.get("scenarios")
            msg = (
                f"📊 <b>ON-DEMAND PREDICTION</b>\n\n"
                f"<b>{t1s} vs {t2s}</b> · {s['venue'].split(',')[0]}\n"
                f"Phase: Pre-toss\n\n"
                f"{e1} <b>{t1s}</b>  {p1:.1f}%  {prob_bar(p1/100)}\n"
                f"{e2} <b>{t2s}</b>  {p2:.1f}%  {prob_bar(p2/100)}\n\n"
                f"🏆 <b>Predicted: {ml['predicted_winner'].split()[-1]}</b>"
            )
            if scenarios:
                _s_a = scenarios["team1_bats_first"]
                _s_b = scenarios["team2_bats_first"]
                msg += (
                    f"\n\n🔄 <b>Scenarios</b>\n"
                    f"• {t1s} bats: {_s_a['team1_win_prob']*100:.0f}%-{_s_a['team2_win_prob']*100:.0f}%\n"
                    f"• {t2s} bats: {_s_b['team1_win_prob']*100:.0f}%-{_s_b['team2_win_prob']*100:.0f}%"
                )
            send_telegram(msg, chat_id)

    elif s["phase"] in ("post_toss", "inn1"):
        # Fetch fresh score — Cricbuzz first, CricAPI as fallback
        inn_info = ""
        if s["phase"] == "inn1":
            fresh = None
            cb_id   = s.get("cb_match_id")
            cb_slug = s.get("cb_slug")
            if cb_id and cb_slug:
                fresh = get_cricbuzz_score(cb_id, cb_slug, s.get("bat_first"), s.get("bat_second"))
            if not fresh and s["api_key"]:
                fresh = (get_match_score(s["match_id"], s["api_key"]) if s["match_id"] else None) or \
                        get_score_from_cricscore(s["team1"], s["team2"], s["api_key"])
            if fresh and fresh.get("score"):
                for sc in fresh.get("score", []):
                    inning = sc.get("inning", "")
                    if s["bat_first"] and any(w in inning for w in s["bat_first"].split()):
                        fr, fw, fo = sc.get("r", 0), sc.get("w", 0), sc.get("o", 0)
                        s.update({"inn1_runs": fr, "inn1_wkts": fw, "inn1_overs": fo})
                        fballs = overs_to_balls(fo)
                        ml_l = ml_live_inn1(s["bat_first"], s["bat_second"], fr, fw, fballs, s["venue"])
                        if ml_l:
                            p_bat = ml_l["batting_team_win_probability"]
                            ms = ml_l["match_state"]
                            crr = ms["current_run_rate"]
                            proj = ms["projected_score"]
                            proj_lo, proj_hi = max(0, int(proj) - 7), int(proj) + 7
                            eb1, eb2 = t_emoji(s["bat_first"]), t_emoji(s["bat_second"])
                            msg = (
                                f"📊 <b>ON-DEMAND PREDICTION</b>\n\n"
                                f"<b>{t1s} vs {t2s}</b> · {s['venue'].split(',')[0]}\n"
                                f"Phase: 1st Innings (live score)\n\n"
                                f"🏏 {short(s['bat_first'])}: <b>{fr}/{fw}</b> ({fo} ov)\n"
                                f"📈 Projected: <b>{proj_lo}–{proj_hi}</b>\n"
                                f"⚡ CRR: {crr:.1f}\n\n"
                                f"{divider()}\n"
                                f"📊 <b>Win Probability</b>\n"
                                f"{divider()}\n"
                                f"{eb1} <b>{short(s['bat_first'])}</b>  {p_bat*100:.1f}%  {prob_bar(p_bat)}\n"
                                f"{eb2} <b>{short(s['bat_second'])}</b>  {(1-p_bat)*100:.1f}%  {prob_bar(1-p_bat)}"
                            )
                            send_telegram(msg, chat_id)
                            return
            inn_info = f"\n🏏 {short(s['bat_first'])}: {s['inn1_runs']}/{s['inn1_wkts']} ({s['inn1_overs']} ov)\n"

        # Post-toss or 1st innings fallback — use posttoss model
        if s.get("bat_first") and s.get("toss_winner"):
            _bf = s["bat_first"]
            _bs = s["bat_second"]
            _bf_xi = s["t1_xi"] if _bf == t1 else s["t2_xi"]
            _bs_xi = s["t2_xi"] if _bf == t1 else s["t1_xi"]
            _pt = ml_posttoss(_bf, _bs, s["venue"], s["toss_winner"], s["toss_decision"],
                              bf_players=_bf_xi, bs_players=_bs_xi,
                              weather=s.get("weather"),
                              bf_impact_player=s.get("bf_impact_player"),
                              bs_impact_player=s.get("bs_impact_player"))
            if _pt and "error" not in _pt:
                _p_bf = _pt["batting_first_win_probability"]
                p1 = (_p_bf if _bf == t1 else 1 - _p_bf) * 100
                p2 = 100 - p1
                ml = {"team1_win_probability": p1/100, "team2_win_probability": p2/100,
                      "predicted_winner": t1 if p1 >= p2 else t2}
            else:
                ml = ml_prematch(t1, t2, s["venue"], s["toss_winner"], s["toss_decision"],
                                 s["t1_xi"], s["t2_xi"])
        else:
            ml = ml_prematch(t1, t2, s["venue"], s["toss_winner"], s["toss_decision"],
                             s["t1_xi"], s["t2_xi"])
        if ml:
            p1 = ml["team1_win_probability"] * 100
            p2 = ml["team2_win_probability"] * 100
            if not inn_info and s["phase"] == "inn1" and s["inn1_overs"] > 0:
                inn_info = f"\n🏏 {short(s['bat_first'])}: {s['inn1_runs']}/{s['inn1_wkts']} ({s['inn1_overs']} ov)\n"
            msg = (
                f"📊 <b>ON-DEMAND PREDICTION</b>\n\n"
                f"<b>{t1s} vs {t2s}</b> · {s['venue'].split(',')[0]}\n"
                f"Phase: {'Post-toss' if s['phase'] == 'post_toss' else '1st Innings'}\n"
                f"{inn_info}\n"
                f"{e1} <b>{t1s}</b>  {p1:.1f}%  {prob_bar(p1/100)}\n"
                f"{e2} <b>{t2s}</b>  {p2:.1f}%  {prob_bar(p2/100)}\n\n"
                f"🏆 <b>Predicted: {ml['predicted_winner'].split()[-1]}</b>"
            )
            send_telegram(msg, chat_id)

    elif s["phase"] in ("break", "inn2"):
        # Fetch fresh score — Cricbuzz first, CricAPI as fallback
        inn2_r, inn2_w, inn2_o = s.get("inn2_runs", 0), s.get("inn2_wkts", 0), s.get("inn2_overs", 0)
        cb_id   = s.get("cb_match_id")
        cb_slug = s.get("cb_slug")
        fresh = None
        if cb_id and cb_slug:
            fresh = get_cricbuzz_score(cb_id, cb_slug, s.get("bat_first"), s.get("bat_second"))
        if not fresh and s["api_key"]:
            fresh = (get_match_score(s["match_id"], s["api_key"]) if s["match_id"] else None) or \
                    get_score_from_cricscore(s["team1"], s["team2"], s["api_key"])
        if fresh:
            for sc in fresh.get("score", []):
                inning = sc.get("inning", "")
                if s["bat_second"] and any(w in inning for w in s["bat_second"].split()):
                    inn2_r = sc.get("r", 0)
                    inn2_w = sc.get("w", 0)
                    inn2_o = sc.get("o", 0)
                    s.update({"inn2_runs": inn2_r, "inn2_wkts": inn2_w, "inn2_overs": inn2_o})

        # 2nd innings — run live model with fresh score
        if s["target"] > 0 and inn2_o > 0:
            balls = int(inn2_o) * 6 + round((inn2_o % 1) * 10)
            ml = ml_live_inn2(s["bat_second"], s["bat_first"],
                              inn2_r, inn2_w, balls, s["target"],
                              venue=s["venue"],
                              first_innings_wickets=s.get("inn1_final_wkts"))
            if ml:
                p_chase = ml["batting_team_win_probability"]
                ms = ml["match_state"]
                crr = ms["current_run_rate"]
                rrr = ms["required_run_rate"]
                needed = ms["runs_needed"]
                balls_left = ms["balls_remaining"]
                rrr_gap = round(float(rrr) - float(crr), 1)
                gap_str = f"+{rrr_gap}" if rrr_gap >= 0 else str(rrr_gap)
                eb1, eb2 = t_emoji(s["bat_first"]), t_emoji(s["bat_second"])
                msg = (
                    f"📊 <b>ON-DEMAND PREDICTION</b>\n\n"
                    f"<b>{t1s} vs {t2s}</b> · {s['venue'].split(',')[0]}\n"
                    f"Phase: 2nd Innings (live score)\n\n"
                    f"🏏 {short(s['bat_second'])}: <b>{inn2_r}/{inn2_w}</b> ({inn2_o} ov)\n"
                    f"🎯 Target: {s['target']}  ·  Need: <b>{needed} off {balls_left} balls</b>\n"
                    f"⚡ CRR: {crr:.1f}  ·  RRR: <b>{rrr:.1f}</b>  ·  Gap: {gap_str}\n\n"
                    f"{divider()}\n"
                    f"📊 <b>Win Probability</b>\n"
                    f"{divider()}\n"
                    f"{eb1} <b>{short(s['bat_first'])}</b>  {(1-p_chase)*100:.1f}%  {prob_bar(1-p_chase)}\n"
                    f"{eb2} <b>{short(s['bat_second'])}</b>  {p_chase*100:.1f}%  {prob_bar(p_chase)}\n\n"
                    f"🏆 <b>Predicted: {ml['predicted_winner'].split()[-1]}</b>"
                )
                send_telegram(msg, chat_id)
        else:
            send_telegram(f"⏸ Innings break. Target for {short(s['bat_second'])}: {s['target']}", chat_id)

    elif s["phase"] == "ended":
        send_telegram("Match has ended. Models are retraining.", chat_id)


def _handle_predictasap_command(chat_id):
    """Toggle ball-by-ball prediction mode."""
    current = _match_state.get("ball_by_ball", False)
    _match_state["ball_by_ball"] = not current
    if _match_state["ball_by_ball"]:
        send_telegram(
            "⚡ <b>Ball-by-ball mode ON</b>\n\n"
            "You'll receive a prediction after every ball.\n"
            "Send /predictASAP again to turn it off.",
            chat_id
        )
        print("  [CMD] Ball-by-ball mode enabled via /predictASAP")
    else:
        send_telegram(
            "📊 <b>Ball-by-ball mode OFF</b>\n\n"
            "Back to per-over predictions.",
            chat_id
        )
        print("  [CMD] Ball-by-ball mode disabled")


def _handle_status_command(chat_id):
    """Send current match status."""
    s = _match_state
    if s["phase"] == "idle" or not s["team1"]:
        send_telegram("🤖 Bot is running. No active match.", chat_id)
        return
    msg = (
        f"🤖 <b>Bot Status</b>\n\n"
        f"Match: {short(s['team1'])} vs {short(s['team2'])}\n"
        f"Phase: {s['phase']}\n"
        f"Venue: {s['venue']}"
    )
    send_telegram(msg, chat_id)


def _handle_rain_command(chat_id):
    """Send current rain/interruption status."""
    s = _match_state
    if s["phase"] == "idle" or not s["team1"]:
        send_telegram("No active match.", chat_id)
        return
    t1s, t2s = short(s["team1"]), short(s["team2"])
    if s["rain_active"]:
        started = s.get("rain_started_at", "")
        status_txt = s.get("rain_status", "Play stopped")
        elapsed = ""
        if started:
            try:
                from datetime import datetime as _dt
                st = _dt.strptime(started, "%H:%M IST")
                now_m = now_ist().hour * 60 + now_ist().minute
                st_m = st.hour * 60 + st.minute
                diff = now_m - st_m
                if diff > 0:
                    elapsed = f" ({diff} min ago)"
            except Exception:
                pass
        msg = (
            f"🌧 <b>RAIN DELAY — {t1s} vs {t2s}</b>\n\n"
            f"Status: {status_txt}\n"
            f"Since: {started}{elapsed}\n"
            f"Phase: {s['phase']}\n"
        )
        if s["phase"] == "inn1":
            msg += f"Score: {t1s if s.get('bat_first') == s['team1'] else t2s} {s['inn1_runs']}/{s['inn1_wkts']} ({s['inn1_overs']} ov)"
        elif s["phase"] == "inn2":
            msg += f"Score: {t2s if s.get('bat_second') == s['team2'] else t1s} {s['inn2_runs']}/{s['inn2_wkts']} ({s['inn2_overs']} ov)"
    else:
        msg = (
            f"☀️ <b>No rain delay — {t1s} vs {t2s}</b>\n\n"
            f"Phase: {s['phase']}\n"
            f"Play is on."
        )
    send_telegram(msg, chat_id)


def _check_rain_status(score):
    """Check score response for rain/interruption indicators.
    Returns (is_rain, status_text) tuple.

    Sources of rain info:
      - CricAPI status field: "Rain - Loss of play", "Play suspended due to rain"
      - Cricbuzz meta description: status text may mention rain
      - Score stall: no over change for 5+ consecutive polls (300+ seconds)
    """
    if not score:
        return False, ""

    status = (score.get("status") or "").lower()

    # Check for explicit rain/delay keywords in status
    for kw in RAIN_KEYWORDS:
        if kw in status:
            return True, score.get("status", "")

    # Check for resume keywords
    for kw in RAIN_RESUME_KEYWORDS:
        if kw in status:
            return False, score.get("status", "")

    return False, ""


def _handle_rain_change(is_rain, status_text, phase_label):
    """Detect transitions: play -> rain and rain -> play. Send Telegram alerts."""
    s = _match_state
    was_raining = s["rain_active"]

    if is_rain and not was_raining:
        # Rain just started
        s["rain_active"] = True
        s["rain_started_at"] = now_ist().strftime("%H:%M IST")
        s["rain_status"] = status_text
        s["stall_notified"] = False

        t1s = short(s.get("bat_first") or s["team1"])
        t2s = short(s.get("bat_second") or s["team2"])
        score_line = ""
        if s["phase"] == "inn1":
            score_line = f"\n📊 Score: {t1s} {s['inn1_runs']}/{s['inn1_wkts']} ({s['inn1_overs']} ov)"
        elif s["phase"] == "inn2":
            score_line = f"\n📊 Score: {t2s} {s['inn2_runs']}/{s['inn2_wkts']} ({s['inn2_overs']} ov)"
            score_line += f"\n🎯 Target: {s['target']}"
        msg = (
            f"🌧 <b>RAIN DELAY — PLAY STOPPED</b>\n\n"
            f"📍 {status_text}{score_line}\n\n"
            f"⏳ <i>Will notify when play resumes...</i>"
        )
        send_telegram(msg)
        mlog.error("rain", f"Play stopped: {status_text}")
        print(f"  [Rain] Play stopped: {status_text}")

    elif not is_rain and was_raining:
        # Rain stopped, play resumed
        started = s.get("rain_started_at", "?")
        s["rain_active"] = False
        s["rain_status"] = ""

        msg = (
            f"☀️ <b>PLAY RESUMED</b>\n\n"
            f"Rain delay ended (started {started}).\n"
            f"📍 {status_text or 'Play is back on'}\n\n"
            f"⏳ <i>Live updates resuming...</i>"
        )
        send_telegram(msg)
        print(f"  [Rain] Play resumed after delay since {started}")

    elif is_rain and was_raining:
        # Still raining — update status text silently
        s["rain_status"] = status_text


def _handle_xi_command(chat_id, raw_text):
    """
    Accept playing XI via Telegram.
    Format: /xi team1:P1,P2,...,P11 | team2:P1,P2,...,P11
    OR two lines:
      /xi
      MI: Rohit Sharma, Suryakumar Yadav, ...
      KKR: Shreyas Iyer, ...

    Updates _match_state["t1_xi"] and ["t2_xi"] in-place.
    """
    s = _match_state
    if not s["team1"]:
        send_telegram("No active match to apply XI to.", chat_id)
        return

    # Parse "team: p1, p2, ..." segments separated by newline or "|"
    text = raw_text[len("/xi"):].strip()
    if not text:
        send_telegram(
            "Usage: <code>/xi MI: Rohit Sharma, Hardik Pandya, ... | KKR: Shreyas Iyer, ...</code>\n"
            "Separate the two teams with a newline or |",
            chat_id
        )
        return

    segments = [seg.strip() for seg in text.replace("|", "\n").split("\n") if seg.strip()]
    updated = []
    for seg in segments:
        if ":" not in seg:
            continue
        team_raw, players_raw = seg.split(":", 1)
        team_raw = team_raw.strip()
        players = [p.strip() for p in players_raw.split(",") if p.strip()]
        if not players:
            continue
        mapped = xi_to_data_names(players)
        # Match to team1 or team2 by short name / keyword
        t1, t2 = s["team1"], s["team2"]
        if any(w in t1.lower() for w in team_raw.lower().split() if len(w) > 2):
            s["t1_xi"] = mapped
            updated.append(f"{short(t1)}: {len(mapped)} players mapped")
        elif any(w in t2.lower() for w in team_raw.lower().split() if len(w) > 2):
            s["t2_xi"] = mapped
            updated.append(f"{short(t2)}: {len(mapped)} players mapped")

    if updated:
        print(f"  [XI cmd] {'; '.join(updated)}")
        send_telegram("XI updated:\n" + "\n".join(f"- {u}" for u in updated), chat_id)
    else:
        send_telegram("Could not parse XI. Format: <code>/xi MI: Player1, Player2, ...</code>", chat_id)


def _handle_impact_command(chat_id, raw_text):
    """
    Option A: Manually set the impact player for each team.
    Format: /impact MI: Jasprit Bumrah | KKR: CV Varun
    OR single team: /impact MI: Jasprit Bumrah

    Updates _match_state["bf_impact_player"] and ["bs_impact_player"].
    Triggers an immediate re-prediction with the new impact player scores.
    """
    s = _match_state
    if not s.get("team1") or s.get("phase") == "idle":
        send_telegram("No active match to apply impact player to.", chat_id)
        return

    text = raw_text[len("/impact"):].strip()
    if not text:
        # Show current impact players
        bf_imp = s.get("bf_impact_player") or "not set"
        bs_imp = s.get("bs_impact_player") or "not set"
        bat_first  = s.get("bat_first", s.get("team1", "Team1"))
        bat_second = s.get("bat_second", s.get("team2", "Team2"))
        send_telegram(
            f"🔄 <b>Impact Players</b>\n"
            f"{short(bat_first)}: <b>{bf_imp}</b>\n"
            f"{short(bat_second)}: <b>{bs_imp}</b>\n\n"
            f"Usage: <code>/impact MI: Jasprit Bumrah | KKR: CV Varun</code>",
            chat_id
        )
        return

    # Parse segments: "Team: Player Name" separated by | or newline
    segments = [seg.strip() for seg in text.replace("|", "\n").split("\n") if seg.strip()]
    updated = []
    t1, t2 = s.get("team1"), s.get("team2")
    bat_first  = s.get("bat_first", t1)
    bat_second = s.get("bat_second", t2)

    for seg in segments:
        if ":" not in seg:
            continue
        team_raw, player_raw = seg.split(":", 1)
        team_raw = team_raw.strip()
        player = player_raw.strip()
        if not player:
            continue
        # Map player name to data_name format
        mapped = xi_to_data_names([player])
        data_name = mapped[0] if mapped else player

        # Match to bat_first or bat_second
        for team_ref in [bat_first, t1]:
            if team_ref and any(w in team_ref.lower()
                                for w in team_raw.lower().split() if len(w) > 2):
                s["bf_impact_player"] = data_name
                updated.append(f"{short(bat_first)} impact → {data_name}")
                break
        else:
            for team_ref in [bat_second, t2]:
                if team_ref and any(w in team_ref.lower()
                                    for w in team_raw.lower().split() if len(w) > 2):
                    s["bs_impact_player"] = data_name
                    updated.append(f"{short(bat_second)} impact → {data_name}")
                    break

    if updated:
        print(f"  [Impact cmd] {'; '.join(updated)}")
        send_telegram(
            "✅ <b>Impact player set:</b>\n" +
            "\n".join(f"• {u}" for u in updated) +
            "\n\nPost-toss model will use these scores in the next prediction.",
            chat_id
        )
    else:
        send_telegram(
            "Could not parse impact player.\n"
            "Format: <code>/impact MI: Jasprit Bumrah | KKR: CV Varun</code>",
            chat_id
        )


def _handle_retrain_command(chat_id):
    """Manually trigger model retraining via /retrain command."""
    s = _match_state
    if s.get("phase") not in ("idle", "ended", None):
        send_telegram(
            "⚠️ Match appears live — running retrain anyway.\n"
            "If a match is still in progress, retrain will also run automatically when it ends.",
            chat_id
        )

    send_telegram("🔄 <b>Manual retrain started...</b>\nThis takes 3-5 minutes.", chat_id)
    print("\n[CMD] Manual retrain triggered via Telegram")

    def _do_retrain():
        try:
            import subprocess as _sp
            errors = []
            # Step 1: features
            r1 = _sp.run([sys.executable, "02_features.py"], capture_output=True, text=True, timeout=120)
            if r1.returncode != 0:
                errors.append(f"02_features: {r1.stderr[-200:]}")

            # Step 2: main models
            r2 = _sp.run([sys.executable, "03_train.py"], capture_output=True, text=True, timeout=300)
            if r2.returncode != 0:
                errors.append(f"03_train: {r2.stderr[-200:]}")

            # Step 3: post-toss model
            r3 = _sp.run([sys.executable, "10_post_toss_model.py"], capture_output=True, text=True, timeout=300)
            if r3.returncode != 0:
                errors.append(f"10_post_toss: {r3.stderr[-200:]}")

            # Step 4: unified live model
            r4 = _sp.run([sys.executable, "11_unified_live_model.py"], capture_output=True, text=True, timeout=300)
            if r4.returncode != 0:
                errors.append(f"11_unified: {r4.stderr[-200:]}")

            # Step 5: reload API
            try:
                resp = requests.post(f"{API_BASE}/reload-models", json={"secret": "ipl2026"}, timeout=10)
                reload_ok = resp.status_code == 200
            except Exception:
                reload_ok = False

            if errors:
                send_telegram(
                    f"⚠️ <b>Retrain completed with errors:</b>\n" +
                    "\n".join(f"• {e}" for e in errors),
                    chat_id
                )
            else:
                send_telegram(
                    f"✅ <b>Retrain complete!</b>\n"
                    f"All 4 models updated.\n"
                    f"{'API reloaded ✓' if reload_ok else '⚠️ API reload failed — restart 04_api.py'}",
                    chat_id
                )
            print(f"[CMD] Retrain done. Errors: {errors}")
        except Exception as e:
            send_telegram(f"❌ Retrain failed: {e}", chat_id)
            print(f"[CMD] Retrain error: {e}")

    threading.Thread(target=_do_retrain, daemon=True).start()


def start_command_listener():
    """Start the Telegram command listener in a background thread."""
    global _tg_update_offset
    # Clear pending updates before starting
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            params={"offset": -1}, timeout=10
        )
        updates = r.json().get("result", [])
        if updates:
            _tg_update_offset = updates[-1]["update_id"] + 1
    except Exception:
        pass

    t = threading.Thread(target=_poll_telegram_commands, daemon=True)
    t.start()
    print("  Telegram command listener started (send 'predict' or 'status')")


# ======================================================================
# TELEGRAM
# ======================================================================
def _test_telegram_connectivity():
    """Send a startup test message to every configured chat ID and log the result."""
    print(f"Telegram chat IDs: {TELEGRAM_CHAT_IDS}")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for cid in TELEGRAM_CHAT_IDS:
        try:
            r = requests.post(url, json={
                "chat_id": cid,
                "text": "🤖 <b>IPL Bot started</b> — connected successfully.",
                "parse_mode": "HTML",
            }, timeout=10)
            if r.status_code == 200:
                print(f"  [TG] ✓ Chat {cid}: OK")
            else:
                err = r.json().get("description", r.text[:120])
                print(f"  [TG] ✗ Chat {cid}: ERROR {r.status_code} — {err}")
        except Exception as e:
            print(f"  [TG] ✗ Chat {cid}: EXCEPTION — {e}")


def send_telegram(message, chat_id=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    targets = [chat_id] if chat_id else TELEGRAM_CHAT_IDS
    for cid in targets:
        try:
            r = requests.post(url, json={
                "chat_id": cid,
                "text": message,
                "parse_mode": "HTML",
            }, timeout=10)
            if r.status_code == 200:
                print(f"  [TG] Sent to {cid} ({len(message)} chars)")
            else:
                print(f"  [TG] Error {r.status_code} for {cid}: {r.text[:200]}")
        except Exception as e:
            print(f"  [TG] Exception for {cid}: {e}")


# ======================================================================
# CRICAPI (with call counter + per-key failure tracking)
# ======================================================================
api_call_count = 0

# Per-key consecutive failure streaks and exhaustion state.
# Keys are the actual key strings; values are fail counts.
_key_fail_streaks: dict = {}
_key_exhausted: set = set()   # keys that have crossed the exhaustion threshold
_KEY_EXHAUST_THRESHOLD = 5    # consecutive failures before marking key dead

def _get_all_cricapi_keys(primary_key=None):
    """Build ordered list of unique CricAPI keys to try, primary first."""
    all_keys = []
    if primary_key:
        all_keys.append(primary_key)
    # Add numbered keys in order
    for k in [CRICAPI_KEY_1, CRICAPI_KEY_2, CRICAPI_KEY_3]:
        if k and k not in all_keys:
            all_keys.append(k)
    # Legacy fallback keys (from old .env format)
    for k in [_CRICAPI_KEY_AFTERNOON, _CRICAPI_KEY_EVENING, _CRICAPI_KEY_DEFAULT]:
        if k and k not in all_keys:
            all_keys.append(k)
    return all_keys


# Build key label lookup once at startup
_KEY_LABELS = {}
for _i, _k in enumerate([CRICAPI_KEY_1, CRICAPI_KEY_2, CRICAPI_KEY_3], 1):
    if _k:
        _KEY_LABELS[_k] = f"Key {_i}"
# Legacy labels
if _CRICAPI_KEY_AFTERNOON:
    _KEY_LABELS.setdefault(_CRICAPI_KEY_AFTERNOON, "Afternoon key")
if _CRICAPI_KEY_EVENING:
    _KEY_LABELS.setdefault(_CRICAPI_KEY_EVENING, "Evening key")
if _CRICAPI_KEY_DEFAULT:
    _KEY_LABELS.setdefault(_CRICAPI_KEY_DEFAULT, "Default key")


def _key_label(key):
    """Human-readable key name for logs/Telegram."""
    return _KEY_LABELS.get(key, f"Key ...{key[-6:]}" if key else "Unknown")


def cricapi_get(endpoint, params=None, api_key=None):
    global api_call_count, _key_fail_streaks, _key_exhausted
    if api_call_count >= MAX_API_CALLS:
        print(f"  [CRICAPI] Daily limit reached ({api_call_count})")
        return None
    keys_to_try = _get_all_cricapi_keys(api_key)
    url = f"https://api.cricapi.com/v1/{endpoint}"
    newly_exhausted = []   # keys that crossed threshold on this call
    last_err = None
    for i, key in enumerate(keys_to_try):
        if api_call_count >= MAX_API_CALLS:
            print(f"  [CRICAPI] Daily limit reached ({api_call_count})")
            return None
        p = {"apikey": key}
        if params: p.update(params)
        try:
            r = requests.get(url, params=p, timeout=15)
            api_call_count += 1
            if r.status_code == 200:
                if i > 0:
                    print(f"  [CRICAPI] Key #{i+1} succeeded for {endpoint}")
                # Reset failure streak on success
                _key_fail_streaks[key] = 0
                if key in _key_exhausted:
                    _key_exhausted.discard(key)
                    print(f"  [CRICAPI] {_key_label(key)} recovered")
                return r.json()
            last_err = f"status {r.status_code}"
            print(f"  [CRICAPI] Key #{i+1} failed for {endpoint}: {last_err}")
        except Exception as e:
            api_call_count += 1
            last_err = str(e)
            print(f"  [CRICAPI] Key #{i+1} failed for {endpoint}: {last_err}")
        # Track failure streak
        _key_fail_streaks[key] = _key_fail_streaks.get(key, 0) + 1
        if (_key_fail_streaks[key] >= _KEY_EXHAUST_THRESHOLD
                and key not in _key_exhausted):
            _key_exhausted.add(key)
            newly_exhausted.append(key)
    print(f"  [CRICAPI] All {len(keys_to_try)} keys exhausted for {endpoint}")
    # Attach newly exhausted info so callers can notify
    if newly_exhausted:
        # Store on thread-local or a module-level queue for _get_score to pick up
        _cricapi_newly_exhausted.extend(newly_exhausted)
    return None

# Queue for _get_score to pick up newly exhausted keys and send Telegram
_cricapi_newly_exhausted: list = []


def find_todays_ipl_match(api_key=None):
    data = cricapi_get("matches", {"offset": 0}, api_key)
    if not data or not data.get("data"): return None
    today = now_ist().strftime("%Y-%m-%d")
    matches = []
    for m in data["data"]:
        name   = (m.get("name")   or "").lower()
        series = (m.get("series") or "").lower()
        is_ipl = ("indian premier league" in name or "ipl" in name or
                  "indian premier league" in series or "ipl" in series)
        if is_ipl and (m.get("date") or "")[:10] == today:
            matches.append(m)
    return matches if matches else None


def find_live_ipl_match(team1, team2, api_key=None):
    """Search currentMatches (then cricScore fallback) for today's IPL match.
    currentMatches returns IDs compatible with matchInfo/matchScore endpoints.
    cricScore IDs are a different format that only works with cricScore itself."""
    today = now_ist().strftime("%Y-%m-%d")

    # ── Try currentMatches first (IDs work with matchInfo/matchScore) ──
    data = cricapi_get("currentMatches", {"offset": 0}, api_key)
    if data and data.get("data"):
        for m in data["data"]:
            name   = (m.get("name")   or "").lower()
            series = (m.get("series") or "").lower()
            is_ipl = ("indian premier league" in name or "ipl" in name or
                      "indian premier league" in series or "ipl" in series)
            if not is_ipl:
                continue
            # Date check (dateTimeGMT is UTC — allow None so upcoming shows up)
            date = (m.get("dateTimeGMT") or "")[:10]
            if date and date != today:
                continue
            # Match team names
            teams_raw = m.get("teams", [])
            t_str = " ".join(t.lower() for t in teams_raw) if teams_raw else name
            t1_match = any(w in t_str for w in team1.lower().split() if len(w) > 3)
            t2_match = any(w in t_str for w in team2.lower().split() if len(w) > 3)
            if t1_match and t2_match:
                mid = m.get("id")
                if mid:
                    print(f"  Found via currentMatches: {mid}")
                    return mid

    # ── Fallback: cricScore (different ID format, used for score-only tracking) ──
    data = cricapi_get("cricScore", {}, api_key)
    if not data or not data.get("data"): return None
    for m in data["data"]:
        series = (m.get("series") or "").lower()
        if "indian premier league" not in series:
            continue
        date = (m.get("dateTimeGMT") or "")[:10]
        if date != today:
            continue
        # t1/t2 fields may include short codes like "[RCB]" — strip them
        t1 = (m.get("t1") or "").split("[")[0].strip().lower()
        t2 = (m.get("t2") or "").split("[")[0].strip().lower()
        t1_match = any(w in t1 for w in team1.lower().split() if len(w) > 3)
        t2_match = any(w in t2 for w in team2.lower().split() if len(w) > 3)
        if t1_match and t2_match:
            mid = m.get("id")
            print(f"  Found via cricScore (fallback): {mid}")
            return mid
    return None


def get_todays_ipl_matches(api_key=None):
    """Get today's IPL matches from the embedded schedule (time/venue/teams),
    enriched with match IDs from CricAPI cricScore.
    Returns list of dicts: {match_num, date, time, team1, team2, venue, match_id}
    """
    from ipl_schedule_2026 import IPL_2026_SCHEDULE
    today = now_ist().strftime("%Y-%m-%d")

    # Step 1: get today's matches from schedule (correct IST times & venues)
    todays = [dict(m, match_id=None) for m in IPL_2026_SCHEDULE if m["date"] == today]
    if not todays:
        return []

    # Step 2: try currentMatches first (IDs work with matchInfo/matchScore)
    cur_data = cricapi_get("currentMatches", {"offset": 0}, api_key)
    if cur_data and cur_data.get("data"):
        for sm in todays:
            if sm["match_id"]:
                continue   # already found
            for m in cur_data["data"]:
                name   = (m.get("name")   or "").lower()
                series = (m.get("series") or "").lower()
                is_ipl = ("indian premier league" in name or "ipl" in name or
                          "indian premier league" in series or "ipl" in series)
                if not is_ipl:
                    continue
                teams_raw = m.get("teams", [])
                t_str = " ".join(t.lower() for t in teams_raw) if teams_raw else name
                t1_match = any(w in t_str for w in sm["team1"].lower().split() if len(w) > 3)
                t2_match = any(w in t_str for w in sm["team2"].lower().split() if len(w) > 3)
                if t1_match and t2_match:
                    sm["match_id"] = m.get("id")
                    sm["_id_source"] = "currentMatches"
                    break

    # Step 3: fall back to cricScore for any still-unmatched fixtures
    score_data = cricapi_get("cricScore", {}, api_key)
    if score_data and score_data.get("data"):
        for sm in todays:
            if sm["match_id"]:
                continue   # already found above
            for m in score_data["data"]:
                series = (m.get("series") or "").lower()
                if "indian premier league" not in series:
                    continue
                t1 = (m.get("t1") or "").split("[")[0].strip().lower()
                t2 = (m.get("t2") or "").split("[")[0].strip().lower()
                t1_match = any(w in t1 for w in sm["team1"].lower().split() if len(w) > 3)
                t2_match = any(w in t2 for w in sm["team2"].lower().split() if len(w) > 3)
                if t1_match and t2_match:
                    sm["match_id"] = m.get("id")
                    sm["_id_source"] = "cricScore"
                    break

    if not todays:
        return []

    return sorted(todays, key=lambda x: x["time"])


def _norm_venue(v):
    """Normalise a venue string to match the model's expected names."""
    if not v: return v
    if "Chinnaswamy" in v: return "M Chinnaswamy Stadium, Bengaluru"
    if "Eden" in v: return "Eden Gardens, Kolkata"
    if "Wankhede" in v: return "Wankhede Stadium, Mumbai"
    if "Chepauk" in v or "Chidambaram" in v: return "MA Chidambaram Stadium, Chennai"
    if "Feroz" in v or "Arun Jaitley" in v or "Kotla" in v: return "Arun Jaitley Stadium, Delhi"
    if "Rajiv Gandhi" in v: return "Rajiv Gandhi Intl Stadium, Hyderabad"
    if "Sawai" in v: return "Sawai Mansingh Stadium, Jaipur"
    if "Mohali" in v or ("Punjab" in v and "Bindra" in v): return "PCA Stadium, Mohali"
    if "DY Patil" in v: return "DY Patil Stadium, Mumbai"
    if "Brabourne" in v: return "Brabourne Stadium, Mumbai"
    if "Narendra Modi" in v or "Motera" in v: return "Narendra Modi Stadium, Ahmedabad"
    if "Ekana" in v or "Atal Bihari" in v: return "Ekana Stadium, Lucknow"
    if "Himachal" in v or "Dharamsala" in v or "Dharamshala" in v: return "HPCA Stadium, Dharamsala"
    if "Holkar" in v: return "Holkar Stadium, Indore"
    if "Barsapara" in v or ("ACA" in v and "Guwahati" in v): return "Barsapara Cricket Stadium, Guwahati"
    if "Yadavindra" in v or "Mullanpur" in v: return "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur"
    if "Shaheed Veer" in v or "Raipur" in v: return "Shaheed Veer Narayan Singh International Stadium"
    if "Sharjah" in v: return "Sharjah Cricket Stadium"
    if "Dubai" in v: return "Dubai International Cricket Stadium"
    if "Sheikh Zayed" in v or "Abu Dhabi" in v: return "Sheikh Zayed Stadium, Abu Dhabi"
    return v


def make_schedule_from_start(start_h, start_m):
    """Build a MATCH_SCHEDULE-style timing dict from actual match start time (IST)."""
    from datetime import timedelta as td
    base  = datetime(2000, 1, 1, start_h, start_m)
    toss  = base  - td(minutes=15)
    ie    = base  + td(minutes=100)   # inn1 end
    i2s   = base  + td(minutes=120)   # inn2 start
    i2e   = base  + td(minutes=220)   # inn2 end
    return {
        "match_start": (start_h, start_m),
        "toss_time":   (toss.hour,  toss.minute),
        "inn1_start":  (start_h,    start_m),
        "inn1_end":    (ie.hour,    ie.minute),
        "inn2_start":  (i2s.hour,   i2s.minute),
        "inn2_end":    (i2e.hour,   i2e.minute),
    }


def get_match_info(match_id, api_key=None):
    data = cricapi_get("matchInfo", {"id": match_id}, api_key)
    return data.get("data") if data and data.get("data") else None


def extract_xi(match_info, team_name):
    """
    Extract the playing XI for a team from CricAPI matchInfo response.
    CricAPI returns players under info["players"] as a list of dicts with "name".
    Returns a list of data_name strings (e.g. ["RG Sharma", "JJ Bumrah", ...]).
    """
    if not match_info:
        return []

    # CricAPI matchInfo structure: info["players"] is a list of player dicts
    # Each dict has: {"name": "Rohit Sharma", "role": "bat", "battingStyle": ..., "team": "Mumbai Indians"}
    players_raw = match_info.get("players", [])

    # Filter to this team only
    team_players = []
    for p in players_raw:
        p_team = p.get("team", "")
        # Fuzzy match: check if team key words appear in the player's team field
        team_words = set(team_name.lower().split())
        p_words    = set(p_team.lower().split())
        if len(team_words & p_words) >= 2 or p_team == team_name:
            team_players.append(p.get("name", ""))

    if not team_players:
        return []

    return xi_to_data_names(team_players)


def get_match_score(match_id, api_key=None):
    data = cricapi_get("matchScore", {"id": match_id}, api_key)
    return data.get("data") if data and data.get("data") else None


def _parse_cricscore_str(s):
    """Parse '127/3 (14.2)' or '220 (20.0)' -> (runs, wickets, overs) or None."""
    import re
    if not s or not s.strip():
        return None
    s = s.strip()
    # "runs/wkts (overs)"
    m = re.match(r'(\d+)/(\d+)\s*\(([0-9.]+)\)', s)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    # All-out or declared: "runs (overs)"
    m2 = re.match(r'(\d+)\s*\(([0-9.]+)\)', s)
    if m2:
        return int(m2.group(1)), 10, float(m2.group(2))
    return None


def get_cricscore_for_match(team1, team2, api_key=None):
    """Fetch the cricScore entry for today's IPL match.
    Returns the raw cricScore dict or None.
    Fields: t1, t2, t1s, t2s, ms (fixture/live/complete), series, id."""
    data = cricapi_get("cricScore", {}, api_key)
    if not data or not data.get("data"):
        return None
    today = now_ist().strftime("%Y-%m-%d")
    for m in data["data"]:
        series = (m.get("series") or "").lower()
        if "indian premier league" not in series:
            continue
        date = (m.get("dateTimeGMT") or "")[:10]
        if date != today:
            continue
        t1 = (m.get("t1") or "").split("[")[0].strip().lower()
        t2 = (m.get("t2") or "").split("[")[0].strip().lower()
        t1_match = any(w in t1 for w in team1.lower().split() if len(w) > 3)
        t2_match = any(w in t2 for w in team2.lower().split() if len(w) > 3)
        if t1_match and t2_match:
            return m
    return None


def get_score_from_cricscore(team1, team2, api_key=None):
    """Get live scores via cricScore when matchScore fails.
    Returns a matchScore-compatible dict or None.

    t1/t2 are the cricScore team fields (may differ from team1/team2 order).
    We figure out which is which by fuzzy matching.
    """
    m = get_cricscore_for_match(team1, team2, api_key)
    if not m:
        return None

    t1_raw = (m.get("t1") or "").split("[")[0].strip()
    t2_raw = (m.get("t2") or "").split("[")[0].strip()
    t1s_str = m.get("t1s") or ""
    t2s_str = m.get("t2s") or ""
    ms = m.get("ms", "")

    # Build score list in matchScore format
    scores = []
    parsed1 = _parse_cricscore_str(t1s_str)
    parsed2 = _parse_cricscore_str(t2s_str)
    if parsed1:
        r, w, o = parsed1
        scores.append({"inning": f"{t1_raw} Inning 1", "r": r, "w": w, "o": o})
    if parsed2:
        r, w, o = parsed2
        scores.append({"inning": f"{t2_raw} Inning 1", "r": r, "w": w, "o": o})

    return {
        "score": scores,
        "matchEnded": ms == "complete",
        "status": ms,
        "_source": "cricScore",
        "_t1": t1_raw,
        "_t2": t2_raw,
    }


# ======================================================================
# CRICBUZZ SCRAPER (fallback when CricAPI has no live data)
# Cricbuzz's static HTML contains live score + toss info even on the
# Next.js-rendered pages — it's embedded in <title> and <meta description>.
# ======================================================================
_CB_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
CRICBUZZ_SHORT_TO_FULL = {v: k for k, v in TEAM_SHORT.items()}
CRICBUZZ_SHORT_TO_FULL.update({
    "PBKS": "Kings XI Punjab",   # CricAPI uses PBKS, model uses "Kings XI Punjab"
    "GT": "Gujarat Titans",
    "LSG": "Lucknow Super Giants",
    "RCB": "Royal Challengers Bengaluru",
    "SRH": "Sunrisers Hyderabad",
    "MI": "Mumbai Indians",
    "CSK": "Chennai Super Kings",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "RR": "Rajasthan Royals",
})


def find_cricbuzz_match(team1, team2):
    """Scan Cricbuzz live-scores page for today's IPL match.
    Returns (match_id, slug, title_text) or None.
    The title_text contains the toss result e.g. 'RCB opt to bowl'.
    """
    try:
        r = requests.get('https://www.cricbuzz.com/cricket-match/live-scores',
                         headers=_CB_HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        text = r.text
        # title="[TeamA] vs [TeamB], ... - [toss/status]" href="/live-cricket-scores/{id}/..."
        pattern = r'title="([^"]+)" href="/live-cricket-scores/(\d+)/([^"]+)"'
        for title, match_id, slug in re.findall(pattern, text):
            title_lower = title.lower()
            t1_words = [w for w in team1.lower().split() if len(w) > 3]
            t2_words = [w for w in team2.lower().split() if len(w) > 3]
            if any(w in title_lower for w in t1_words) and any(w in title_lower for w in t2_words):
                return match_id, slug, title
    except Exception as e:
        print(f"  [Cricbuzz] find error: {e}")
    return None


def parse_cricbuzz_toss(title_text, team1, team2):
    """Parse toss result from Cricbuzz match title.
    e.g. 'SRH vs RCB, 1st Match - RCB opt to bowl'
    Returns (toss_winner, toss_decision, bat_first, bat_second) or None.
    """
    # "TEAM opt to bat/bowl"
    m = re.search(r'[-–]\s*([A-Z]+)\s+opt\s+to\s+(bat|bowl)', title_text, re.IGNORECASE)
    if m:
        winner_short = m.group(1).upper()
        decision = m.group(2).lower()
        toss_winner = CRICBUZZ_SHORT_TO_FULL.get(winner_short)
        # Fuzzy match to team1/team2
        if toss_winner not in [team1, team2]:
            for t in [team1, team2]:
                if any(w in t.lower() for w in winner_short.lower().split() if len(w) > 2):
                    toss_winner = t
                    break
        if not toss_winner:
            toss_winner = team1  # fallback
        if decision == "bat":
            bat_first = toss_winner
            bat_second = team2 if toss_winner == team1 else team1
        else:
            bat_second = toss_winner
            bat_first = team2 if toss_winner == team1 else team1
        return toss_winner, decision, bat_first, bat_second
    return None


def _cb_short_to_full(short, bat_first=None, bat_second=None):
    """Resolve a Cricbuzz short team name to our model's full name."""
    full = CRICBUZZ_SHORT_TO_FULL.get(short.upper(), short)
    for candidate in [bat_first, bat_second]:
        if candidate and short.upper() in candidate.upper():
            return candidate
    return full


def get_cricbuzz_score(match_id, slug, bat_first=None, bat_second=None):
    """Fetch live score from Cricbuzz match page.
    Returns a matchScore-compatible dict or None.

    Actual Cricbuzz meta description formats observed:
      1st innings live:  "Follow IPL | SRH 49/3 (5.5) (batsmen) | ..."
      2nd innings live:  "Follow IPL | RCB 160/2 (11.5) vs SRH\\n201/9 (batsmen) | ..."
      Result:            "RCB won by 8 wickets | SRH 201/9 (20) RCB 202/2 (18.1) | ..."
    """
    try:
        url = f"https://www.cricbuzz.com/live-cricket-scores/{match_id}/{slug}"
        r = requests.get(url, headers=_CB_HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        text = r.text

        desc_m = re.search(r'<meta name="description" content="([^"]+)"', text)
        if not desc_m:
            return None
        # Normalize whitespace (description has \n and lots of spaces)
        desc = re.sub(r'\s+', ' ', desc_m.group(1))

        scores = []
        match_ended = False
        status_text = ""
        _inferred_bat_first = None    # batting order inferred from score
        _inferred_bat_second = None

        # ── Check for match result ────────────────────────────────────
        result_m = re.search(r'((?:\w+ ){1,4}won\s+by\s+[^|]+)', desc, re.IGNORECASE)
        if result_m:
            status_text = result_m.group(1).strip()
            match_ended = True

        # ── Check for rain / interruption / no result ─────────────────
        if not match_ended:
            _rain_pats = [
                r'(rain[^|]*)',
                r'(play (?:stopped|suspended|interrupted)[^|]*)',
                r'(no result[^|]*)',
                r'(match abandoned[^|]*)',
                r'(bad light[^|]*)',
                r'(wet outfield[^|]*)',
            ]
            for rp in _rain_pats:
                rm = re.search(rp, desc, re.IGNORECASE)
                if rm:
                    status_text = rm.group(1).strip()
                    break

        # ── 2nd innings: "TEAM1 R/W (O) vs TEAM2 R2/W2" format ───────
        # e.g. "RCB 160/2 (11.5) vs SRH 201/9"  — TEAM1 is chasing, TEAM2 batted first
        inn2_m = re.search(
            r'\|\s*([A-Z]{2,5})\s+(\d+)/(\d+)\s*\(([0-9.]+)\)\s+vs\s+([A-Z]{2,5})\s+(\d+)(?:/(\d+))?',
            desc
        )
        if inn2_m:
            # TEAM1 = currently batting (chasing), TEAM2 = set total (batted first)
            chasing_short  = inn2_m.group(1)
            chasing_r      = int(inn2_m.group(2))
            chasing_w      = int(inn2_m.group(3))
            chasing_o      = float(inn2_m.group(4))
            setting_short  = inn2_m.group(5)
            setting_r      = int(inn2_m.group(6))
            setting_w      = int(inn2_m.group(7)) if inn2_m.group(7) else 10
            setting_o      = 20.0   # completed innings

            chasing_full = _cb_short_to_full(chasing_short, bat_first, bat_second)
            setting_full = _cb_short_to_full(setting_short, bat_first, bat_second)

            _inferred_bat_first  = setting_full   # TEAM2 batted first
            _inferred_bat_second = chasing_full   # TEAM1 is chasing

            scores = [
                {"inning": f"{setting_full} Inning 1",
                 "r": setting_r, "w": setting_w, "o": setting_o,
                 "_short": setting_short},
                {"inning": f"{chasing_full} Inning 1",
                 "r": chasing_r, "w": chasing_w, "o": chasing_o,
                 "_short": chasing_short},
            ]
        else:
            # ── 1st innings or result: "TEAM R/W (O)" patterns ───────
            score_pat = re.compile(r'\b([A-Z]{2,5})\s+(\d+)(?:/(\d+))?\s*\(([0-9]+\.?[0-9]*)\)')
            for sm in score_pat.finditer(desc):
                short_name = sm.group(1)
                full_name  = _cb_short_to_full(short_name, bat_first, bat_second)
                r_val = int(sm.group(2))
                w_val = int(sm.group(3)) if sm.group(3) else 10
                o_val = float(sm.group(4))
                exists = any(s.get("_short") == short_name and s.get("r") == r_val for s in scores)
                if not exists:
                    scores.append({
                        "inning": f"{full_name} Inning 1",
                        "r": r_val, "w": w_val, "o": o_val,
                        "_short": short_name,
                    })

        if not scores and not match_ended:
            return None

        result = {
            "score": scores,
            "matchEnded": match_ended,
            "status": status_text or ("complete" if match_ended else "live"),
            "_source": "cricbuzz",
        }
        if _inferred_bat_first:
            result["_inferred_bat_first"]  = _inferred_bat_first
            result["_inferred_bat_second"] = _inferred_bat_second
        return result

    except Exception as e:
        print(f"  [Cricbuzz] score error: {e}")
        return None


# ======================================================================
# CRICBUZZ PLAYING XI (mobile JSON API — no JS needed)
# ======================================================================
def _parse_toss_from_rsc(html, team1, team2):
    """Extract toss winner name from Cricbuzz RSC HTML payload.
    The HTML always contains tossWinnerName once toss has happened."""
    for pat in [
        r'tossWinnerName[^"]{0,10}"([^"]+)"',
        r'tossWinnerName[\\\"]{0,6}([A-Za-z ]{3,30})[\\\"]{0,3}[,}]',
    ]:
        m = re.search(pat, html)
        if m:
            winner_raw = m.group(1).strip().strip('\\"')
            if not winner_raw:
                continue
            winner = match_team_to_ours(winner_raw, [team1, team2])
            if winner:
                return winner
    return None


def get_cricbuzz_playing11(cb_match_id, cb_slug):
    """Fetch playing XI from Cricbuzz. Tries mobile JSON API first, then RSC page scraping
    with multiple escape-level patterns. Returns {team_id_or_name: [player_full_name, ...]} or None."""

    # ── Strategy 1: Mobile JSON API (clean JSON, no regex) ─────────────────
    for mobile_url in [
        f"https://www.cricbuzz.com/api/cricket-match/{cb_match_id}/matchinfo",
        f"https://m.cricbuzz.com/api/cricket-match/{cb_match_id}/matchinfo",
    ]:
        try:
            r = requests.get(mobile_url, headers=_CB_HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                teams_data = data.get("matchInfo", {}).get("team", [])
                result = {}
                for t in teams_data:
                    players = [
                        p["name"] for p in t.get("playerDetails", [])
                        if str(p.get("playing11", "")).lower() == "true"
                    ]
                    if players:
                        result[t.get("name", str(len(result)))] = players
                if result:
                    print(f"  [CB XI] Mobile API success: {list(result.keys())}")
                    return result
        except Exception:
            pass

    # ── Strategy 2: RSC page scraping — try multiple escape levels ──────────
    RSC_PATTERNS = [
        # double-escaped  (\\\\" in raw string = \\\" in actual text = \" after one JSON parse)
        r'id\\\\\":([\d]+)[^}]{0,400}?fullName\\\\\":\\\\\"([^\\\\\\"]+)\\\\\"[^}]{0,400}?substitute\\\\\":(true|false)[^}]{0,400}?teamId\\\\\":([\d]+)',
        # single-escaped  (\\\" in raw string = \" in actual text — one level of JSON)
        r'id\\\":([\d]+)[^}]{0,400}?fullName\\\":\\\"([^\\\"]+)\\\"[^}]{0,400}?substitute\\\":(true|false)[^}]{0,400}?teamId\\\":([\d]+)',
        # unescaped plain JSON embedded in page
        r'"id":([\d]+)[^}]{0,200}?"fullName":"([^"]+)"[^}]{0,200}?"substitute":(true|false)[^}]{0,200}?"teamId":([\d]+)',
    ]

    for page_url in [
        f"https://www.cricbuzz.com/live-cricket-scores/{cb_match_id}/{cb_slug}",
        f"https://www.cricbuzz.com/live-cricket-scorecard/{cb_match_id}/{cb_slug}",
    ]:
        try:
            r = requests.get(page_url, headers=_CB_HEADERS, timeout=15)
            if r.status_code != 200:
                print(f"  [CB XI] HTTP {r.status_code} from {page_url[:60]}")
                continue
            html = r.text
            for pattern in RSC_PATTERNS:
                entries = re.findall(pattern, html, re.DOTALL)
                if entries:
                    from collections import defaultdict
                    teams = defaultdict(list)
                    for _pid, full_name, substitute, tid in entries:
                        if substitute == "false" and tid != "0":
                            teams[tid].append(full_name)
                    result = {}
                    for tid, plist in teams.items():
                        unique = list(dict.fromkeys(plist))
                        if len(unique) >= 6:
                            result[tid] = unique[:11]
                    if result:
                        print(f"  [CB XI] RSC pattern matched ({len(entries)} entries)")
                        # Try to resolve numeric team IDs → short names (CSK, RR, MI...)
                        # by searching for shortName near each teamId in the same HTML
                        tid_to_short = {}
                        for tid in result.keys():
                            for sn_pat in [
                                # teamSName (matchInfo section): "teamId\":65,...,"teamSName\":\"PBKS\"
                                rf'teamId[^\d]{{0,5}}{tid}[^}}]{{0,300}}?teamSName[^A-Z]{{0,5}}([A-Z]{{2,5}})',
                                rf'teamSName[^A-Z]{{0,5}}([A-Z]{{2,5}})[^}}]{{0,300}}?teamId[^\d]{{0,5}}{tid}[^\d]',
                                # shortName near teamId (original patterns)
                                rf'shortName\\\\\":\\\\\"([A-Z]{{2,5}})\\\\\"[^{{}}]{{0,400}}?teamId\\\\\":{tid}[^\\d]',
                                rf'teamId\\\\\":{tid}[^\\d][^{{}}]{{0,400}}?shortName\\\\\":\\\\\"([A-Z]{{2,5}})\\\\\"',
                                rf'shortName\\\":\\\"([A-Z]{{2,5}})\\\"[^{{}}]{{0,400}}?teamId\\\":{tid}[^\\d]',
                                rf'teamId\\\":{tid}[^\\d][^{{}}]{{0,400}}?shortName\\\":\\\"([A-Z]{{2,5}})\\\"',
                                rf'"shortName":"([A-Z]{{2,5}})"[^{{}}]{{0,400}}?"teamId":{tid}[^\\d]',
                                rf'"teamId":{tid}[^\\d][^{{}}]{{0,400}}?"shortName":"([A-Z]{{2,5}})"',
                            ]:
                                m = re.search(sn_pat, html)
                                if m:
                                    tid_to_short[tid] = m.group(1)
                                    break
                        if len(tid_to_short) == len(result):
                            named = {tid_to_short[t]: p for t, p in result.items()}
                            print(f"  [CB XI] Resolved team IDs: {tid_to_short}")
                            return named
                        return result
        except Exception as e:
            print(f"  [CB XI] {page_url[:50]} error: {e}")

    print("  [CB XI] No player entries found in any source")
    return None


# ======================================================================
# CLAUDE LLM (pre-toss and post-toss only)
# ======================================================================
def claude_explain(prompt_text):
    if not ANTHROPIC_API_KEY: return ""
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                      "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt_text}]},
            timeout=30)
        if r.status_code == 200:
            return r.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"  [Claude] Error: {e}")
    return ""


# ======================================================================
# ML API CALLS
# ======================================================================
def ml_prematch(team1, team2, venue, toss_winner=None, toss_decision=None,
                team1_players=None, team2_players=None):
    try:
        r = requests.post(f"{API_BASE}/predict/prematch", json={
            "team1": team1, "team2": team2, "venue": venue,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "team1_players": team1_players or [],
            "team2_players": team2_players or [],
        }, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  [ML] prematch error: {e}")
        return None


def ml_pretoss(team1, team2, venue, is_evening=1, match_hour=19,
               team1_xi=None, team2_xi=None):
    """Call the pre-toss endpoint (reuses post-toss model with last known XI + weather).
    Falls back to ml_prematch() if the pretoss endpoint is unavailable.
    """
    try:
        payload = {
            "team1": team1, "team2": team2, "venue": venue,
            "is_evening": is_evening, "match_hour": match_hour,
        }
        if team1_xi:
            payload["team1_xi"] = team1_xi
        if team2_xi:
            payload["team2_xi"] = team2_xi
        r = requests.post(f"{API_BASE}/predict/pretoss", json=payload, timeout=15)
        result = r.json()
        if "error" in result:
            print(f"  [ML] pretoss unavailable ({result['error']}), falling back to prematch")
            return ml_prematch(team1, team2, venue)
        return result
    except Exception as e:
        print(f"  [ML] pretoss error: {e}, falling back to prematch")
        return ml_prematch(team1, team2, venue)


def ml_posttoss(bat_first, bat_second, venue, toss_winner, toss_decision,
                bf_players=None, bs_players=None, weather=None, is_evening=1,
                bf_impact_player=None, bs_impact_player=None):
    """Call the dedicated post-toss model for bat-first/bat-second prediction."""
    try:
        payload = {
            "bat_first": bat_first, "bat_second": bat_second,
            "venue": venue,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "bf_players": bf_players or [],
            "bs_players": bs_players or [],
            "is_evening": is_evening,
        }
        if weather:
            payload["temperature"] = weather.get("temperature")
            payload["humidity"] = weather.get("humidity")
            payload["cloud_cover"] = weather.get("cloud_cover")
        if bf_impact_player:
            payload["bf_impact_player"] = bf_impact_player
        if bs_impact_player:
            payload["bs_impact_player"] = bs_impact_player
        r = requests.post(f"{API_BASE}/predict/posttoss", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  [ML] posttoss error: {e}")
        return None


def ml_live_inn1(batting_team, bowling_team, runs, wickets, balls, venue,
                 pp_runs=None, pp_wickets=None):
    try:
        payload = {
            "batting_team": batting_team, "bowling_team": bowling_team,
            "runs_scored": runs, "wickets_fallen": wickets,
            "balls_bowled": balls, "venue": venue,
        }
        if pp_runs is not None:
            payload["pp_runs"] = pp_runs
        if pp_wickets is not None:
            payload["pp_wickets"] = pp_wickets
        r = requests.post(f"{API_BASE}/predict/live_inn1", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  [ML] inn1 error: {e}")
        return None


def ml_live_inn2(batting_team, bowling_team, runs, wickets, balls, target,
                 venue="", first_innings_wickets=None, pp_runs=None, pp_wickets=None,
                 max_balls=None):
    try:
        payload = {
            "batting_team": batting_team, "bowling_team": bowling_team,
            "runs_scored": runs, "wickets_fallen": wickets,
            "balls_bowled": balls, "target": target,
            "venue": venue,
        }
        if first_innings_wickets is not None:
            payload["first_innings_wickets"] = first_innings_wickets
        if pp_runs is not None:
            payload["pp_runs"] = pp_runs
        if pp_wickets is not None:
            payload["pp_wickets"] = pp_wickets
        if max_balls is not None and max_balls != 120:
            payload["max_balls"] = max_balls
        r = requests.post(f"{API_BASE}/predict/live", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        print(f"  [ML] inn2 error: {e}")
        return None


def ml_live_unified(innings, bat_first, bat_second, runs, wickets, balls,
                    venue="", target=None, first_innings_wickets=None,
                    pp_runs=None, pp_wickets=None,
                    partnership_runs=None, partnership_balls=None,
                    last_3ov_runs=None, last_3ov_wkts=None,
                    boundary_pct=None, dot_ball_pct=None,
                    max_balls=None, max_partnership=None):
    """Unified live model — single call for both innings.
    Returns bat_first_win_probability (always from bat_first perspective).
    Falls back to separate models if unified model not available.
    """
    try:
        payload = {
            "current_innings": innings,
            "bat_first": bat_first,
            "bat_second": bat_second,
            "runs_scored": runs,
            "wickets_fallen": wickets,
            "balls_bowled": balls,
            "venue": venue,
        }
        if target is not None:
            payload["target"] = target
        if first_innings_wickets is not None:
            payload["first_innings_wickets"] = first_innings_wickets
        if pp_runs is not None:
            payload["pp_runs"] = pp_runs
        if pp_wickets is not None:
            payload["pp_wickets"] = pp_wickets
        if partnership_runs is not None:
            payload["partnership_runs"] = partnership_runs
        if partnership_balls is not None:
            payload["partnership_balls"] = partnership_balls
        if last_3ov_runs is not None:
            payload["last_3ov_runs"] = last_3ov_runs
        if last_3ov_wkts is not None:
            payload["last_3ov_wkts"] = last_3ov_wkts
        if boundary_pct is not None:
            payload["boundary_pct"] = boundary_pct
        if dot_ball_pct is not None:
            payload["dot_ball_pct"] = dot_ball_pct
        if max_balls is not None and max_balls != 120:
            payload["max_balls"] = max_balls
        if max_partnership is not None:
            payload["max_partnership"] = max_partnership
        r = requests.post(f"{API_BASE}/predict/live_unified", json=payload, timeout=10)
        result = r.json()
        if "error" in result:
            print(f"  [Unified] {result['error']} — falling back to separate models")
            return None
        return result
    except Exception as e:
        print(f"  [Unified] error: {e}")
        return None


def fetch_player_scores(players, venue, team=""):
    """Call /player-scores API. Returns list of {data_name, full_name, bat_score, bowl_score}."""
    if not players:
        return []
    try:
        r = requests.post(f"{API_BASE}/player-scores",
                          json={"players": players, "venue": venue, "team": team}, timeout=10)
        if r.status_code != 200:
            print(f"  [PlayerScores] HTTP {r.status_code}: {r.text[:200]}")
            return []
        return r.json().get("players", [])
    except Exception as e:
        print(f"  [PlayerScores] Error: {e}")
        return []


# ======================================================================
# EXCEL HELPERS
# ======================================================================
def _excel_id(match_id, team1, team2):
    """Stable match ID for Excel: prefer CricAPI ID, else generate from teams+date."""
    if match_id:
        return str(match_id)
    return f"ipl2026_{team1[:3].upper()}v{team2[:3].upper()}_{now_ist().strftime('%Y%m%d')}"


def excel_prematch(match_id, team1, team2, venue, toss_winner, toss_decision,
                   ml_result, explanation):
    try:
        mi = ml_result.get("model_inputs", {})
        r = requests.post(f"{API_BASE}/update-excel/prematch", json={
            "match_id": _excel_id(match_id, team1, team2),
            "season": "2026",
            "date": now_ist().strftime("%Y-%m-%d"),
            "team1": team1, "team2": team2, "venue": venue,
            "toss_winner": toss_winner, "toss_decision": toss_decision,
            "team1_win_probability": ml_result["team1_win_probability"],
            "team2_win_probability": ml_result["team2_win_probability"],
            "predicted_winner": ml_result["predicted_winner"],
            "confidence": ml_result["confidence"],
            "elo_diff": mi.get("elo_diff", 0),
            "form_diff": mi.get("form_diff", 0),
            "llm_summary": explanation,
            "risk_factors": ml_result.get("key_factors", []),
        }, timeout=10)
        print(f"  [Excel prematch] {r.json().get('action', r.status_code)}")
    except Exception as e:
        print(f"  [Excel prematch] Error: {e}")


def excel_live(match_id, team1, team2, over_number, batting_team_win_pct,
               score_string, predicted_winner):
    try:
        r = requests.post(f"{API_BASE}/update-excel/live", json={
            "match_id": _excel_id(match_id, team1, team2),
            "over_number": over_number,
            "batting_team_win_pct": batting_team_win_pct,
            "score_string": score_string,
            "predicted_winner": predicted_winner,
        }, timeout=10)
        print(f"  [Excel live o{over_number}] win%={batting_team_win_pct:.1f}")
    except Exception as e:
        print(f"  [Excel live o{over_number}] Error: {e}")


def excel_result(match_id, team1, team2, actual_winner, win_margin,
                 inn1_score, inn2_score):
    try:
        r = requests.post(f"{API_BASE}/update-excel/result", json={
            "match_id": _excel_id(match_id, team1, team2),
            "actual_winner": actual_winner,
            "win_margin": win_margin,
            "first_innings_score": inn1_score,
            "second_innings_score": inn2_score,
        }, timeout=10)
        print(f"  [Excel result] winner={actual_winner}")
    except Exception as e:
        print(f"  [Excel result] Error: {e}")


# ======================================================================
# ======================================================================
# WEATHER HELPERS
# ======================================================================

def get_match_weather(venue: str, match_hour_ist: int = 19) -> dict:
    """
    Fetch Open-Meteo hourly forecast for the venue at the match start hour.
    Returns a dict with temperature, humidity, cloud_cover, wind_speed,
    rain_prob — or None on failure.  No API key required.
    """
    # Fuzzy match venue name to coordinates
    coords = None
    vl = venue.lower()
    for k, v in VENUE_COORDS.items():
        if k.lower() in vl or vl in k.lower():
            coords = v
            break
        # keyword overlap (e.g. "Wankhede" matches "Wankhede Stadium")
        kwords = [w for w in k.lower().split() if len(w) > 4]
        if any(w in vl for w in kwords):
            coords = v
            break
    if not coords:
        print(f"  [Weather] No coordinates found for venue: '{venue}'")
        return None
    lat, lon = coords
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relativehumidity_2m,cloudcover,"
        "windspeed_10m,precipitation_probability"
        "&timezone=Asia%2FKolkata&forecast_days=1"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"  [Weather] API returned {r.status_code}")
            return None
        data = r.json()
        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        # Find index for requested hour (e.g. "2026-04-05T19:00")
        target = f"T{match_hour_ist:02d}:00"
        idx = next((i for i, t in enumerate(times) if t.endswith(target)), match_hour_ist)
        w = {
            "temperature":  hourly["temperature_2m"][idx],
            "humidity":     hourly["relativehumidity_2m"][idx],
            "cloud_cover":  hourly["cloudcover"][idx],
            "wind_speed":   hourly["windspeed_10m"][idx],
            "rain_prob":    hourly["precipitation_probability"][idx],
            "hour":         match_hour_ist,
        }
        print(f"  [Weather] {w['temperature']:.0f}°C  "
              f"Humidity:{w['humidity']:.0f}%  Cloud:{w['cloud_cover']:.0f}%  "
              f"Wind:{w['wind_speed']:.0f}km/h  Rain:{w['rain_prob']:.0f}%")
        return w
    except Exception as e:
        print(f"  [Weather] Fetch error: {e}")
        return None


def compute_weather_adjustment(w: dict) -> float:
    """
    Return a win-probability shift for the *chasing* team based on conditions.
    Positive  →  chasing team gets a bump  (dew / overcast favours them).
    Negative  →  defending team gets a bump (rare; hot & dry conditions).
    Capped at ±0.07 (7 percentage points).

    Applied as:
      inn1 live:  p_bat_first  -= adj   (bat_first loses prob when adj > 0)
      inn2 live:  p_chase      += adj
    """
    if not w:
        return 0.0
    adj = 0.0
    hum   = w["humidity"]
    cloud = w["cloud_cover"]
    temp  = w["temperature"]
    is_evening = w.get("hour", 19) >= 17

    # ── Dew factor (evening T20s only) ───────────────────────────────
    # High humidity in evening = dew on outfield by overs 15-20 of 2nd inn.
    # Grip reduces, spinners and swing bowlers lose effectiveness → chasing easier.
    if is_evening:
        if hum >= 85:
            adj += 0.06
        elif hum >= 75:
            adj += 0.04
        elif hum >= 65:
            adj += 0.02
        elif hum >= 55:
            adj += 0.01

    # ── Overcast / heavy cloud ────────────────────────────────────────
    # Cloud cover aids swing & seam → 1st-innings batting slightly harder.
    # In T20 the effect is modest but real.
    if cloud >= 80:
        adj += 0.025
    elif cloud >= 60:
        adj += 0.015
    elif cloud >= 40:
        adj += 0.005

    # ── Hot & dry day ─────────────────────────────────────────────────
    # Fast, true pitch; large 1st-innings totals are set more easily.
    # Defending team gains marginal advantage over the chasing side.
    if temp >= 36 and hum < 45:
        adj -= 0.015

    return round(max(-0.07, min(0.07, adj)), 3)


def weather_display(w: dict) -> str:
    """Return a formatted Telegram-ready string for match-day weather."""
    if not w:
        return ""
    temp  = w["temperature"]
    hum   = w["humidity"]
    cloud = w["cloud_cover"]
    wind  = w["wind_speed"]
    rain  = w["rain_prob"]
    is_evening = w.get("hour", 19) >= 17

    # Condition icon
    if rain >= 50:
        sky = "🌧"
    elif cloud >= 70:
        sky = "☁️"
    elif cloud >= 30:
        sky = "⛅"
    else:
        sky = "☀️"

    line1 = f"{sky}  🌡 {temp:.0f}°C  💧 {hum:.0f}% humidity  ☁️ {cloud:.0f}% cloud  💨 {wind:.0f} km/h"
    if rain >= 20:
        line1 += f"  🌧 Rain: {rain:.0f}%"

    # Dew / overcast narrative
    notes = []
    if is_evening:
        if hum >= 85:
            notes.append("🌫 <b>Dew: HIGH</b> — significant chasing advantage expected")
        elif hum >= 75:
            notes.append("💧 <b>Dew: Moderate</b> — chasing side likely to benefit")
        elif hum >= 65:
            notes.append("💧 Dew: Low-moderate — slight chasing edge")
    if cloud >= 70:
        notes.append("☁️ Heavy cloud cover — swing &amp; seam conditions early")
    elif cloud >= 45:
        notes.append("⛅ Overcast — some assistance for pace bowlers")

    result = line1
    if notes:
        result += "\n" + "  ".join(notes)
    return result


# ======================================================================
# AUTO-RETRAIN AFTER MATCH
# ======================================================================
def auto_retrain(match_info, score_data, bat_first, bat_second, venue):
    """Automatically retrain models after match ends."""
    try:
        status = score_data.get("status", "")
        winner = normalize_team_name(score_data.get("matchWinner", ""))
        if not winner:
            # Try to extract from status
            for team in [bat_first, bat_second]:
                if short(team) in status or team.split()[-1] in status:
                    winner = team
                    break

        if not winner:
            print("  [Retrain] Could not determine winner, skipping")
            return

        scores = score_data.get("score", [])
        inn1_r, inn1_w, inn2_r, inn2_w = 0, 0, 0, 0
        for s in scores:
            inning = s.get("inning", "")
            if any(w in inning for w in bat_first.split()):
                inn1_r, inn1_w = s.get("r", 0), s.get("w", 0)
            elif any(w in inning for w in bat_second.split()):
                inn2_r, inn2_w = s.get("r", 0), s.get("w", 0)

        toss_winner = normalize_team_name(match_info.get("tossWinner", bat_second))
        toss_decision = (match_info.get("tossChoice", "field")).lower()
        match_id = match_info.get("id", f"auto_{now_ist().strftime('%Y%m%d')}")
        date = now_ist().strftime("%Y-%m-%d")

        cmd = [
            sys.executable, "06_retrain_after_match.py",
            "--match_id", str(match_id),
            "--team1", bat_first, "--team2", bat_second,
            "--winner", winner,
            "--toss_winner", toss_winner, "--toss_decision", toss_decision,
            "--venue", venue, "--date", date,
            "--inn1_score", str(inn1_r), "--inn1_wkts", str(inn1_w),
            "--inn2_score", str(inn2_r), "--inn2_wkts", str(inn2_w),
        ]
        print(f"  [Retrain] Running: {' '.join(cmd[-10:])}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print("  [Retrain] Success - models updated for next match")
            # Also update player database
            try:
                pd_result = subprocess.run(
                    [sys.executable, "08_update_player_db.py"],
                    capture_output=True, text=True, timeout=600,
                )
                if pd_result.returncode == 0:
                    print("  [Retrain] Player database updated")
                else:
                    print(f"  [Retrain] Player DB update failed: {pd_result.stderr[-200:]}")
            except Exception as pe:
                print(f"  [Retrain] Player DB update error: {pe}")
            send_telegram("Models retrained with today's result. Ready for next match.")
        else:
            err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
            print(f"  [Retrain] Failed: {err}")
            send_telegram(f"⚠️ <b>Auto-retrain failed</b> after today's match.\n<code>{err[-200:]}</code>")
    except Exception as e:
        print(f"  [Retrain] Error: {e}")
        send_telegram(f"⚠️ <b>Auto-retrain error:</b> {e}")


# ======================================================================
# MAIN MATCH RUNNER
# ======================================================================
def run_match(match_data, slot="evening", api_key=None, from_schedule=False):
    """Run full prediction pipeline for one match."""
    global api_call_count, _key_fail_streaks, _key_exhausted, _cricapi_newly_exhausted
    # Reset key health state at the start of each match
    _key_fail_streaks = {}
    _key_exhausted = set()
    _cricapi_newly_exhausted = []

    if from_schedule:
        # Schedule dict has exact model names — no CricAPI extraction needed
        team1 = match_data["team1"]
        team2 = match_data["team2"]
        venue = match_data["venue"]
        match_id = match_data.get("match_id")   # pre-enriched if available
        _id_source = match_data.get("_id_source", "cricScore")
        start_h, start_m = map(int, match_data["time"].split(":"))
        schedule = make_schedule_from_start(start_h, start_m)
    else:
        match_id = match_data.get("id", "")
        _id_source = "currentMatches"   # IDs from /matches or matchInfo always work
        schedule = MATCH_SCHEDULE[slot]

        # Extract teams
        teams_raw = match_data.get("teams", [])
        if not teams_raw:
            ti = match_data.get("teamInfo", [])
            teams_raw = [t.get("name", "") for t in ti] if ti else []
        if len(teams_raw) < 2:
            print("Could not find teams in match data")
            return

        try:
            our_teams = requests.get(f"{API_BASE}/teams", timeout=5).json().get("teams", [])
        except:
            our_teams = list(TEAM_SHORT.keys())

        team1 = match_team_to_ours(teams_raw[0], our_teams)
        team2 = match_team_to_ours(teams_raw[1], our_teams)
        venue = match_data.get("venue", "")

    t1s, t2s = short(team1), short(team2)

    print(f"\n{'='*60}")
    print(f"MATCH: {team1} vs {team2} ({slot.upper()} slot)")
    print(f"Venue: {venue}")
    print(f"CricAPI ID: {match_id}")
    print(f"API key: {_key_label(api_key)} ({len(_get_all_cricapi_keys(api_key))} keys available)")
    print(f"{'='*60}")

    # ── Start match logger ──
    mlog.start(team1, team2, venue, slot, match_id)

    # Update shared state for /predict command
    _match_state.update({
        "team1": team1, "team2": team2, "venue": venue,
        "phase": "pre_toss",
        "match_id": match_id,
        "api_key": api_key,
        "bat_first": None, "bat_second": None,
    })

    # ── PHASE 1: Pre-toss prediction ──────────────────────────────────
    mlog.phase("pre_toss")
    _is_eve_pretoss = 1 if schedule["match_start"][0] >= 18 else 0
    _match_hour_pretoss = schedule["match_start"][0]
    if from_schedule:
        print("\n[Phase 1] Pre-toss prediction (from schedule, firing immediately)")
    else:
        pre_toss_time = ist_today_at(*schedule["toss_time"]) - timedelta(minutes=15)
        if now_ist() < pre_toss_time:
            sleep_until(pre_toss_time, "pre-toss")
        print("\n[Phase 1] Pre-toss prediction")
    # Pass last known XI to pretoss so model uses real players, not squad estimate
    _t1_pretoss_xi = _last_xi_map.get(team1, [])
    _t2_pretoss_xi = _last_xi_map.get(team2, [])
    ml = ml_pretoss(team1, team2, venue, _is_eve_pretoss, _match_hour_pretoss,
                    team1_xi=_t1_pretoss_xi, team2_xi=_t2_pretoss_xi)
    mlog.prediction("prematch",
                     {"team1": team1, "team2": team2, "venue": venue},
                     ml)
    if ml:
        p1 = ml["team1_win_probability"] * 100
        p2 = ml["team2_win_probability"] * 100
        winner = ml["predicted_winner"]
        conf = ml["confidence"].upper()
        factors = ml.get("key_factors", [])

        # Scenario breakdown (only available from pretoss endpoint)
        scenarios = ml.get("scenarios")

        explanation = ""
        if ANTHROPIC_API_KEY and factors:
            _scenario_ctx = ""
            if scenarios:
                _s_a = scenarios["team1_bats_first"]
                _s_b = scenarios["team2_bats_first"]
                _scenario_ctx = (
                    f"Scenario analysis: If {team1} bats first → {_s_a['team1_win_prob']*100:.0f}% "
                    f"(weight {_s_a['weight']:.0%}). If {team2} bats first → "
                    f"{_s_b['team1_win_prob']*100:.0f}% (weight {_s_b['weight']:.0%}). "
                )
            explanation = claude_explain(
                f"IPL: {team1} vs {team2} at {venue}. "
                f"ML predicts {winner} ({max(p1,p2):.0f}%). "
                f"{_scenario_ctx}"
                f"Factors: {'; '.join(factors)}. "
                f"2-3 sentences on why. Be IPL-specific. No disclaimers."
            )

        e1, e2 = t_emoji(team1), t_emoji(team2)
        msg = (
            f"🏏 <b>IPL 2026 — MATCH DAY</b>\n\n"
            f"<b>{t1s} vs {t2s}</b>\n"
            f"📍 {venue}\n\n"
            f"{divider()}\n"
            f"📊 <b>PRE-TOSS PREDICTION</b>\n"
            f"{divider()}\n"
            f"{e1} <b>{t1s}</b>  {p1:.1f}%  {prob_bar(p1/100)}\n"
            f"{e2} <b>{t2s}</b>  {p2:.1f}%  {prob_bar(p2/100)}\n\n"
            f"🏆 <b>Predicted: {winner}</b>\n"
            f"⚡ Confidence: {conf_label(conf)}"
        )
        if scenarios:
            _s_a = scenarios["team1_bats_first"]
            _s_b = scenarios["team2_bats_first"]
            msg += (
                f"\n\n{divider()}\n"
                f"🔄 <b>Toss Scenarios</b>\n"
                f"• If {t1s} bats: {t1s} <b>{_s_a['team1_win_prob']*100:.1f}%</b> — {t2s} {_s_a['team2_win_prob']*100:.1f}%\n"
                f"• If {t2s} bats: {t2s} <b>{_s_b['team2_win_prob']*100:.1f}%</b> — {t1s} {_s_b['team1_win_prob']*100:.1f}%"
            )
        # Show which XI source was used
        _xi_note = ""
        if _t1_pretoss_xi or _t2_pretoss_xi:
            _xi_note = f"\n📋 <i>Based on last known XI</i>"
        else:
            _xi_note = f"\n📋 <i>XI estimated from squad data</i>"
        msg += _xi_note
        if factors:
            msg += f"\n\n{divider()}\n🔍 <b>Key Factors</b>\n" + "\n".join(f"• {f}" for f in factors)
        if explanation:
            msg += f"\n\n<i>{explanation}</i>"
        msg += f"\n\n⏳ <i>Toss update to follow...</i>"
        send_telegram(msg)

    # ── PHASE 2: Wait for toss ────────────────────────────────────────
    mlog.phase("toss_detection")
    # Start polling exactly at toss_time: 7:00 PM for evening, 3:00 PM for afternoon
    toss_start = ist_today_at(*schedule["toss_time"])
    if now_ist() < toss_start:
        sleep_until(toss_start, "toss window")

    # If match_id unknown (schedule-based run), try CricAPI once — then proceed anyway.
    # Live tracking uses Cricbuzz (Phase 2+), which doesn't need the CricAPI ID.
    if not match_id:
        print(f"  Trying CricAPI for match ID ({t1s} vs {t2s})...")
        match_id = find_live_ipl_match(team1, team2, api_key)
        if match_id:
            print(f"  Found CricAPI ID: {match_id}")
            _match_state["match_id"] = match_id
        else:
            print("  CricAPI match ID not found — will use Cricbuzz for all live data.")

    print("\n[Phase 2] Polling for toss...")
    toss_winner = None
    toss_decision = None
    bat_first = None
    bat_second = None
    _toss_method = "unknown"
    _toss_attempts = 0
    match_info_cache = None
    _cb_match_id = None    # Cricbuzz match ID
    _cb_slug = None
    # matchInfo/matchScore only work when ID comes from /matches or /currentMatches.
    # IDs from /cricScore are a different format and return "Invalid API requested".
    _matchinfo_works = (_id_source == "currentMatches")
    if not _matchinfo_works:
        print("  Note: match ID from cricScore — will re-check currentMatches closer to toss")
        # Re-try currentMatches now (match appears ~30 min before start, bot starts hours earlier)
        _cm_data = cricapi_get("currentMatches", {"offset": 0}, api_key)
        if _cm_data and _cm_data.get("data"):
            for _cm in _cm_data["data"]:
                _cm_name   = (_cm.get("name")   or "").lower()
                _cm_series = (_cm.get("series") or "").lower()
                if not ("ipl" in _cm_name or "indian premier league" in _cm_name or
                        "ipl" in _cm_series or "indian premier league" in _cm_series):
                    continue
                _cm_teams = _cm.get("teams", [])
                _cm_str = " ".join(t.lower() for t in _cm_teams) if _cm_teams else _cm_name
                if (any(w in _cm_str for w in team1.lower().split() if len(w) > 3) and
                        any(w in _cm_str for w in team2.lower().split() if len(w) > 3)):
                    _cm_id = _cm.get("id")
                    if _cm_id:
                        print(f"  [ID Upgrade] Found match in currentMatches: {_cm_id} (was cricScore)")
                        match_id = _cm_id
                        _id_source = "currentMatches"
                        _matchinfo_works = True
                        break
        if not _matchinfo_works:
            print("  Note: match not in currentMatches yet — using Cricbuzz tracking")
    # Deadline = match start + 4 hours (covers full match including 2nd innings)
    toss_deadline = ist_today_at(*schedule["match_start"]) + timedelta(hours=4)

    # ── Try Cricbuzz first (fastest — toss/score in static HTML) ─────
    print("  Checking Cricbuzz for toss...")
    cb_result = find_cricbuzz_match(team1, team2)
    if cb_result:
        _cb_match_id, _cb_slug, cb_title = cb_result
        _match_state["cb_match_id"] = _cb_match_id
        _match_state["cb_slug"]     = _cb_slug
        print(f"  Cricbuzz match {_cb_match_id}: {cb_title.strip()}")
        toss_parsed = parse_cricbuzz_toss(cb_title, team1, team2)
        if toss_parsed:
            toss_winner, toss_decision, bat_first, bat_second = toss_parsed
            _toss_method = "cricbuzz_title_initial"
            print(f"  TOSS (Cricbuzz): {short(toss_winner)} won, chose to {toss_decision} -> {short(bat_first)} bats first")
        else:
            # Title no longer has toss text (e.g. 2nd innings "Need 62 off 54b")
            # Try to infer batting order from the live score
            cb_score = get_cricbuzz_score(_cb_match_id, _cb_slug)
            if cb_score and cb_score.get("_inferred_bat_first"):
                bat_first  = cb_score["_inferred_bat_first"]
                bat_second = cb_score["_inferred_bat_second"]
                toss_winner = bat_first    # best guess
                toss_decision = "bat"
                _toss_method = "cricbuzz_2nd_inn_inferred"
                print(f"  TOSS (Cricbuzz score): {short(bat_first)} batted first (inferred from 2nd innings score)")
            else:
                # Also try 1st innings score inference (covers innings break case)
                if cb_score and cb_score.get("score"):
                    first_score = cb_score["score"][0]
                    batting_short = first_score.get("_short", "")
                    bat_first_guess = _cb_short_to_full(batting_short, team1, team2)
                    if bat_first_guess in [team1, team2]:
                        bat_first = bat_first_guess
                        bat_second = team2 if bat_first == team1 else team1
                        toss_winner = bat_first
                        toss_decision = "bat"
                        _toss_method = "cricbuzz_1st_inn_inferred"
                        print(f"  TOSS (Cricbuzz 1st inn inferred): {short(bat_first)} batted first")
                if toss_winner is None:
                    print(f"  Cricbuzz title has no toss yet: '{cb_title.strip()}'")

    _toss_rain_notified = False   # send rain alert only once
    _toss_rain_active   = False

    while toss_winner is None and now_ist() < toss_deadline:
        _toss_attempts += 1

        # ── Rain check during toss window ─────────────────────────────
        if _cb_match_id:
            try:
                _toss_cb = get_cricbuzz_score(_cb_match_id, _cb_slug)
                if _toss_cb:
                    _toss_rain, _toss_rain_txt = _check_rain_status(_toss_cb)
                    if _toss_rain and not _toss_rain_notified:
                        _toss_rain_active = True
                        _toss_rain_notified = True
                        send_telegram(
                            f"🌧 <b>TOSS DELAYED — RAIN</b>\n\n"
                            f"📍 {t1s} vs {t2s} at {venue}\n"
                            f"Status: {_toss_rain_txt or 'Play delayed due to rain'}\n\n"
                            f"⏳ <i>Will update when toss happens...</i>"
                        )
                        print(f"  [Rain] Toss delayed: {_toss_rain_txt}")
                    elif not _toss_rain and _toss_rain_active:
                        _toss_rain_active = False
                        send_telegram(
                            f"☀️ <b>PLAY RESUMED — Toss expected shortly</b>\n\n"
                            f"📍 {t1s} vs {t2s}"
                        )
                        print("  [Rain] Toss delay over, play resuming")
            except Exception:
                pass

        # ── Try matchInfo (works when match ID is from currentMatches) ──
        if _matchinfo_works and match_id:
            info = get_match_info(match_id, api_key)
            if info:
                match_info_cache = info
                tw = info.get("tossWinner", "")
                tc = info.get("tossChoice", "")
                if tw and tc:
                    toss_winner = match_team_to_ours(tw, [team1, team2])
                    toss_decision = tc.lower()
                    if toss_decision == "bat":
                        bat_first = toss_winner
                        bat_second = team2 if toss_winner == team1 else team1
                    else:
                        bat_second = toss_winner
                        bat_first = team2 if toss_winner == team1 else team1
                    _toss_method = "matchInfo"
                    print(f"  TOSS (matchInfo): {toss_winner} won, chose to {toss_decision}")
                    break
            else:
                print("  matchInfo unavailable — switching to Cricbuzz/cricScore tracking")
                _matchinfo_works = False

        # ── Try Cricbuzz (no API calls, just a web scrape) ───────────
        if toss_winner is None:
            cb_result = find_cricbuzz_match(team1, team2)
            if cb_result:
                _cb_match_id, _cb_slug, cb_title = cb_result
                _match_state["cb_match_id"] = _cb_match_id
                _match_state["cb_slug"]     = _cb_slug
                toss_parsed = parse_cricbuzz_toss(cb_title, team1, team2)
                if toss_parsed:
                    toss_winner, toss_decision, bat_first, bat_second = toss_parsed
                    _toss_method = "cricbuzz_title_loop"
                    print(f"  TOSS (Cricbuzz): {short(toss_winner)} won, chose to {toss_decision} -> {short(bat_first)} bats first")
                    break
                # If no toss text yet, try to infer from live score on cricbuzz
                # (skip during rain delay — no play means no reliable score to infer from)
                if _cb_match_id and not _toss_rain_active:
                    cb_score = get_cricbuzz_score(_cb_match_id, _cb_slug)
                    if cb_score:
                        # 2nd innings: "_inferred_bat_first" is set reliably
                        if cb_score.get("_inferred_bat_first"):
                            bat_first  = cb_score["_inferred_bat_first"]
                            bat_second = cb_score["_inferred_bat_second"]
                            toss_winner = bat_first
                            toss_decision = "bat"
                            _toss_method = "cricbuzz_2nd_inn_loop"
                            print(f"  TOSS (Cricbuzz 2nd inn): {short(bat_first)} batted first")
                            break
                        # 1st innings: first entry in score list is the batting team
                        elif cb_score.get("score"):
                            first_score = cb_score["score"][0]
                            batting_short = first_score.get("_short", "")
                            bat_first_guess = _cb_short_to_full(batting_short, team1, team2)
                            if bat_first_guess in [team1, team2]:
                                bat_first = bat_first_guess
                                bat_second = team2 if bat_first == team1 else team1
                                toss_winner = bat_first
                                toss_decision = "bat"
                                _toss_method = "cricbuzz_1st_inn_loop"
                                print(f"  TOSS (Cricbuzz 1st inn): {short(bat_first)} batting first")
                                break
                    # RSC fallback: parse tossWinnerName directly from Cricbuzz page HTML
                    if toss_winner is None:
                        try:
                            rsc_r = requests.get(
                                f"https://www.cricbuzz.com/live-cricket-scores/{_cb_match_id}/{_cb_slug}",
                                headers=_CB_HEADERS, timeout=15
                            )
                            if rsc_r.status_code == 200:
                                rsc_winner = _parse_toss_from_rsc(rsc_r.text, team1, team2)
                                if rsc_winner:
                                    toss_winner = rsc_winner
                                    bat_first = rsc_winner
                                    bat_second = team2 if bat_first == team1 else team1
                                    toss_decision = "bat"
                                    _toss_method = "rsc_tossWinnerName"
                                    print(f"  TOSS (RSC tossWinnerName): {short(toss_winner)} won toss")
                                    break
                        except Exception:
                            pass
                print(f"  Cricbuzz: toss not in title yet '{cb_title}'")
            else:
                print(f"  Cricbuzz: match not found yet")

        # ── Final fallback: cricScore (sometimes has scores earlier) ──
        if toss_winner is None:
            cs = get_cricscore_for_match(team1, team2, api_key)
            if cs:
                t1_raw = (cs.get("t1") or "").split("[")[0].strip()
                t2_raw = (cs.get("t2") or "").split("[")[0].strip()
                p1 = _parse_cricscore_str(cs.get("t1s") or "")
                p2 = _parse_cricscore_str(cs.get("t2s") or "")
                if p1 and not p2:
                    bat_first = team1 if any(w in t1_raw.lower() for w in team1.lower().split() if len(w) > 3) else team2
                    bat_second = team2 if bat_first == team1 else team1
                    toss_winner, toss_decision = bat_first, "bat"
                    _toss_method = "cricScore_infer"
                    print(f"  TOSS (cricScore infer): {short(bat_first)} batting first")
                    break
                elif p2:
                    bat_first = team2 if any(w in t2_raw.lower() for w in team2.lower().split() if len(w) > 3) else team1
                    bat_second = team1 if bat_first == team2 else team2
                    toss_winner, toss_decision = bat_first, "bat"
                    _toss_method = "cricScore_infer"
                    print(f"  TOSS (cricScore infer): {short(bat_first)} batting first")
                    break

        if toss_winner is None:
            print(f"  Toss not yet... ({now_ist().strftime('%H:%M')}, calls: {api_call_count})")
            time.sleep(60)   # 60s retry (faster than default 180s for toss window)

    # Validate toss result — all three must be real, known team names
    if toss_winner not in (team1, team2): toss_winner = None
    if bat_first   not in (team1, team2): bat_first   = None
    if bat_second  not in (team1, team2): bat_second  = None

    if toss_winner is None or bat_first is None:
        mlog.error("toss", "Toss not detected within deadline")
        send_telegram(f"⚠️ Toss not detected for {t1s} vs {t2s} — live tracking unavailable.")
        mlog.end("Toss not detected", scores=None)
        return

    # Log successful toss detection
    mlog.toss_detected(
        method=_toss_method, toss_winner=toss_winner, toss_decision=toss_decision,
        bat_first=bat_first, bat_second=bat_second, attempts=_toss_attempts)

    # ── Fetch match-day weather (used in Phase 3 display + live adjustments) ─
    _match_hour = schedule["match_start"][0]
    print(f"\n[Weather] Fetching conditions for {venue} at {_match_hour:02d}:00 IST...")
    _weather = get_match_weather(venue, _match_hour)
    _weather_adj = compute_weather_adjustment(_weather)
    mlog.weather_fetched(_weather, _weather_adj)
    if _weather_adj != 0:
        side = "chasing" if _weather_adj > 0 else "defending"
        print(f"  [Weather] Probability nudge: {_weather_adj:+.1%} favouring {side} team")

    # ── PHASE 3: Post-toss prediction ─────────────────────────────────
    mlog.phase("post_toss")
    print("\n[Phase 3] Post-toss prediction")

    # Extract playing XI — try CricAPI matchInfo first, then Cricbuzz (multi-strategy)
    t1_xi = extract_xi(match_info_cache, team1)
    t2_xi = extract_xi(match_info_cache, team2)
    if (not t1_xi or not t2_xi) and _cb_match_id and _cb_slug:
        print("  Fetching playing XI from Cricbuzz...")
        _cb_xi = None
        # Retry up to 6 times (every 60s = up to 6 minutes) — XI usually announced 15 min before toss
        for _xi_attempt in range(6):
            _cb_xi = get_cricbuzz_playing11(_cb_match_id, _cb_slug)
            if _cb_xi:
                break
            # Also re-try CricAPI matchInfo on each round (in case it has XI now)
            if _matchinfo_works and match_id and not (t1_xi and t2_xi):
                _fresh_info = get_match_info(match_id, api_key)
                if _fresh_info:
                    match_info_cache = _fresh_info
                    _t1_fresh = extract_xi(_fresh_info, team1)
                    _t2_fresh = extract_xi(_fresh_info, team2)
                    if _t1_fresh and not t1_xi:
                        t1_xi = _t1_fresh
                        print(f"  {short(team1)} XI from CricAPI retry ({len(t1_xi)} players)")
                    if _t2_fresh and not t2_xi:
                        t2_xi = _t2_fresh
                        print(f"  {short(team2)} XI from CricAPI retry ({len(t2_xi)} players)")
                    if t1_xi and t2_xi:
                        break
            if _xi_attempt < 5:
                print(f"  XI not available yet (attempt {_xi_attempt+1}/6), retrying in 60s...")
                time.sleep(60)
        if _cb_xi:
            print(f"  Cricbuzz XI: {len(_cb_xi)} teams found, sizes: {[len(v) for v in _cb_xi.values()]}")
            keys = list(_cb_xi.keys())

            # ── Case 1: keys are short names (e.g. "CSK", "RR") — direct match ──
            # This happens when get_cricbuzz_playing11 resolved numeric IDs to short names
            def _key_matches(key, team_name):
                k = key.upper()
                t1_abbr = short(team_name).upper()
                keywords = [w.upper() for w in team_name.split() if len(w) > 3]
                return k == t1_abbr or any(k in w or w in k for w in keywords)

            named_match = any(not k.isdigit() for k in keys)
            if named_match:
                for key, players in _cb_xi.items():
                    if not t1_xi and _key_matches(key, team1):
                        t1_xi = xi_to_data_names(players)
                        print(f"  {short(team1)} XI ({len(t1_xi)} mapped): {players}")
                    elif not t2_xi and _key_matches(key, team2):
                        t2_xi = xi_to_data_names(players)
                        print(f"  {short(team2)} XI ({len(t2_xi)} mapped): {players}")

            # ── Case 2: keys are numeric team IDs — use slug position as fallback ──
            if (not t1_xi or not t2_xi) and len(keys) >= 2:
                slug_lower = (_cb_slug or "").lower()

                def _slug_pos(team_name, slug):
                    for word in team_name.split():
                        if len(word) > 3:
                            p = slug.find(word.lower())
                            if p != -1:
                                return p
                    return slug.find(short(team_name).lower())

                pos1 = _slug_pos(team1, slug_lower)
                pos2 = _slug_pos(team2, slug_lower)
                t1_first = (pos1 < pos2) if (pos1 != -1 and pos2 != -1) else True
                t1_tid, t2_tid = (keys[0], keys[1]) if t1_first else (keys[1], keys[0])
                if not t1_xi:
                    t1_xi = xi_to_data_names(_cb_xi[t1_tid])
                    print(f"  {short(team1)} XI ({len(t1_xi)} mapped, by slug): {_cb_xi[t1_tid]}")
                if not t2_xi:
                    t2_xi = xi_to_data_names(_cb_xi[t2_tid])
                    print(f"  {short(team2)} XI ({len(t2_xi)} mapped, by slug): {_cb_xi[t2_tid]}")
        else:
            print("  Cricbuzz XI not available yet")

    # ── Fallback: use last known XI loaded at startup ─────────────────────
    if not t1_xi:
        t1_xi = _last_xi_map.get(team1, [])
        if t1_xi:
            print(f"  [XI Fallback] {short(team1)}: {len(t1_xi)} players from last known XI")
    if not t2_xi:
        t2_xi = _last_xi_map.get(team2, [])
        if t2_xi:
            print(f"  [XI Fallback] {short(team2)}: {len(t2_xi)} players from last known XI")

    print(f"  XI found: {team1}={len(t1_xi)} players, {team2}={len(t2_xi)} players")
    _xi_src = "cricapi" if (extract_xi(match_info_cache, team1)) else "cricbuzz" if (t1_xi or t2_xi) else "none"
    mlog.xi_detected(_xi_src, len(t1_xi), len(t2_xi), t1_xi[:11], t2_xi[:11])

    # ── Option B: Auto-detect impact player from matches.csv squad data ──
    # The 12th player listed in team1/team2_players is the nominated impact player.
    # We look up the most recent match between these two teams to find the nominee.
    # This fires automatically — no user action required. /impact overrides it.
    if not _match_state.get("bf_impact_player") and not _match_state.get("bs_impact_player"):
        try:
            _mcsv = pd.read_csv("data/matches.csv")
            _mcsv = _mcsv.sort_values("date")
            # Find most recent match involving bat_first and bat_second
            _mask = (
                (_mcsv["team1"].isin([bat_first, bat_second])) &
                (_mcsv["team2"].isin([bat_first, bat_second]))
            )
            _recent_match = _mcsv[_mask].iloc[-1] if _mask.any() else None
            if _recent_match is not None:
                _t1_pl = str(_recent_match.get("team1_players","")).split("|")
                _t2_pl = str(_recent_match.get("team2_players","")).split("|")
                _t1_team = _recent_match["team1"]
                _t2_team = _recent_match["team2"]
                # Map historical team to current bat_first/bat_second
                for _team, _players in [(_t1_team, _t1_pl), (_t2_team, _t2_pl)]:
                    if len(_players) >= 12 and _players[11].strip():
                        _nominee_raw = _players[11].strip()
                        _nominee_mapped = xi_to_data_names([_nominee_raw])
                        _nominee = _nominee_mapped[0] if _nominee_mapped else _nominee_raw
                        if _team == bat_first:
                            _match_state["bf_impact_player"] = _nominee
                            print(f"  [Impact Auto] {short(bat_first)} → {_nominee}")
                        elif _team == bat_second:
                            _match_state["bs_impact_player"] = _nominee
                            print(f"  [Impact Auto] {short(bat_second)} → {_nominee}")
                _match_state["impact_auto_detected"] = True
        except Exception as _e:
            print(f"  [Impact Auto] Error reading squad: {_e}")

    _match_state.update({
        "phase": "post_toss", "bat_first": bat_first, "bat_second": bat_second,
        "toss_winner": toss_winner, "toss_decision": toss_decision,
        "t1_xi": t1_xi, "t2_xi": t2_xi,
    })

    # Determine XI for bat_first / bat_second
    if bat_first == team1:
        bf_xi, bs_xi = t1_xi, t2_xi
    else:
        bf_xi, bs_xi = t2_xi, t1_xi

    # Determine afternoon vs evening
    _is_evening = 1
    if schedule and schedule.get("match_start"):
        _is_evening = 1 if schedule["match_start"][0] >= 18 else 0

    # Try dedicated post-toss model first, fall back to prematch
    ml_pt = ml_posttoss(bat_first, bat_second, venue, toss_winner, toss_decision,
                        bf_players=bf_xi, bs_players=bs_xi,
                        weather=_weather, is_evening=_is_evening,
                        bf_impact_player=_match_state.get("bf_impact_player"),
                        bs_impact_player=_match_state.get("bs_impact_player"))
    mlog.prediction("posttoss",
                     {"bat_first": bat_first, "bat_second": bat_second, "venue": venue,
                      "toss_winner": toss_winner, "toss_decision": toss_decision,
                      "bf_xi_count": len(bf_xi), "bs_xi_count": len(bs_xi),
                      "is_evening": _is_evening, "weather": _weather is not None},
                     ml_pt, weather_adj=0)

    if ml_pt and "error" not in ml_pt:
        # Map bat_first/bat_second probabilities back to team1/team2
        p_bf = ml_pt["batting_first_win_probability"]
        p_bs = ml_pt["batting_second_win_probability"]
        if bat_first == team1:
            p1_pt, p2_pt = p_bf * 100, p_bs * 100
        else:
            p1_pt, p2_pt = p_bs * 100, p_bf * 100

        # Post-toss model stands alone — no blend with pre-match
        p1, p2 = p1_pt, p2_pt
        print(f"  [ML] Post-toss: {t1s}={p1:.1f}% {t2s}={p2:.1f}%")

        winner = team1 if p1 >= p2 else team2
        conf = ml_pt["confidence"].upper()
        factors = ml_pt.get("key_factors", [])
        ps_pt = ml_pt.get("xi_strengths", {})
        # Map strengths to team1/team2 for display
        if bat_first == team1:
            ps = {"team1_bat": ps_pt.get("bf_bat"), "team1_bowl": ps_pt.get("bf_bowl"),
                  "team2_bat": ps_pt.get("bs_bat"), "team2_bowl": ps_pt.get("bs_bowl")}
        else:
            ps = {"team1_bat": ps_pt.get("bs_bat"), "team1_bowl": ps_pt.get("bs_bowl"),
                  "team2_bat": ps_pt.get("bf_bat"), "team2_bowl": ps_pt.get("bf_bowl")}
        xi_used = True
        ml = {"team1_win_probability": p1/100, "team2_win_probability": p2/100,
              "predicted_winner": winner, "confidence": conf,
              "key_factors": factors, "player_strengths": ps, "xi_data_used": xi_used}
        print(f"  [ML] Post-toss model: {short(winner)} ({max(p1,p2):.1f}%) conf={conf}"
              f" [high_conf={ml_pt.get('high_confidence', False)}]")
    else:
        # Fallback to old prematch model
        ml = ml_prematch(team1, team2, venue, toss_winner, toss_decision,
                         team1_players=t1_xi, team2_players=t2_xi)

    if ml:
        p1 = ml["team1_win_probability"] * 100
        p2 = ml["team2_win_probability"] * 100
        winner = ml["predicted_winner"]
        conf = ml["confidence"].upper()
        factors = ml.get("key_factors", [])
        ps = ml.get("player_strengths", {})
        xi_used = ml.get("xi_data_used", False)

        explanation = ""
        if ANTHROPIC_API_KEY:
            ps_text = ""
            if xi_used and ps:
                ps_text = (f"Player strengths (venue-adjusted): "
                           f"{t1s} bat={ps.get('team1_bat','?'):.1f} bowl={ps.get('team1_bowl','?'):.1f} | "
                           f"{t2s} bat={ps.get('team2_bat','?'):.1f} bowl={ps.get('team2_bowl','?'):.1f}. ")
            explanation = claude_explain(
                f"IPL: {team1} vs {team2} at {venue}. "
                f"{toss_winner} won toss, chose to {toss_decision}. "
                f"{bat_first} bats first, {bat_second} chases. "
                f"{ps_text}"
                f"ML predicts {winner} ({max(p1,p2):.0f}%). "
                f"Factors: {'; '.join(factors)}. "
                f"2-3 sentences on toss impact and XI strength. Mention venue/dew if relevant. No disclaimers."
            )

        b1s, b2s = short(bat_first), short(bat_second)
        e1, e2 = t_emoji(team1), t_emoji(team2)
        eb1, eb2 = t_emoji(bat_first), t_emoji(bat_second)
        toss_shift = p1 - (ml_prematch(team1, team2, venue) or {}).get("team1_win_probability", p1/100) * 100
        msg = (
            f"🎯 <b>TOSS UPDATE — {t1s} vs {t2s}</b>\n\n"
            f"🪙 <b>{short(toss_winner)} won toss · chose to {toss_decision.upper()}</b>\n"
            f"{eb1} Batting 1st: <b>{b1s}</b>\n"
            f"{eb2} Chasing: <b>{b2s}</b>\n\n"
            f"{divider()}\n"
            f"📊 <b>UPDATED PREDICTION</b>\n"
            f"{divider()}\n"
            f"{e1} <b>{t1s}</b>  {p1:.1f}%  {prob_bar(p1/100)}\n"
            f"{e2} <b>{t2s}</b>  {p2:.1f}%  {prob_bar(p2/100)}\n\n"
            f"🏆 <b>Predicted: {winner}</b>\n"
            f"⚡ Confidence: {conf_label(conf)}"
        )
        if xi_used and ps:
            b1_bat  = ps.get('team1_bat') or 0
            b1_bowl = ps.get('team1_bowl') or 0
            b2_bat  = ps.get('team2_bat') or 0
            b2_bowl = ps.get('team2_bowl') or 0
            msg += (
                f"\n\n{divider()}\n"
                f"💪 <b>XI Strength (venue-adjusted)</b>\n"
                f"{divider()}\n"
                f"{'':8}{'Bat':>6}  {'Bowl':>6}\n"
                f"{e1} {t1s:<6}  {b1_bat:>5.1f}   {b1_bowl:>5.1f}\n"
                f"{e2} {t2s:<6}  {b2_bat:>5.1f}   {b2_bowl:>5.1f}"
            )
        # Playing XI with per-player scores
        t1_scores = fetch_player_scores(t1_xi, venue, team1)
        t2_scores = fetch_player_scores(t2_xi, venue, team2)
        if t1_xi or t2_xi:
            msg += f"\n\n{divider()}\n📋 <b>XI Analysis (venue-adjusted)</b>\n{divider()}"

            def _xi_section(emoji, team_short, xi_list, scores_list, role_label,
                            team_bat_avg=None, team_bowl_avg=None):
                sec = f"\n\n{emoji} <b>{team_short}</b> ({role_label})\n"
                score_map = {p["data_name"]: p for p in scores_list}
                bat_vals, bowl_vals = [], []
                for i, dn in enumerate(xi_list, 1):
                    p = score_map.get(dn, {})
                    full = p.get("full_name") or DATA_TO_FULL.get(dn, dn)
                    bat  = p.get("bat_score")
                    bowl = p.get("bowl_score")
                    est  = p.get("estimated", False)
                    # Fallback: use team average when individual score is missing
                    if bat is None and team_bat_avg is not None:
                        bat, est = team_bat_avg, True
                    if bowl is None and team_bowl_avg is not None:
                        bowl, est = team_bowl_avg, True
                    bat_str  = f"~{bat:.0f}" if bat  is not None and est  else (f"{bat:.0f}"  if bat  is not None else "--")
                    bowl_str = f"~{bowl:.0f}" if bowl is not None and est else (f"{bowl:.0f}" if bowl is not None else "--")
                    sec += f"  {i:>2}. {full:<22} Bat:{bat_str:>4}  Bowl:{bowl_str:>4}\n"
                    if bat  is not None: bat_vals.append(bat)
                    if bowl is not None: bowl_vals.append(bowl)
                t_bat  = f"{sum(sorted(bat_vals,  reverse=True)[:6])  / min(6, len(bat_vals)):.1f}"  if bat_vals  else "--"
                t_bowl = f"{sum(sorted(bowl_vals, reverse=True)[:4]) / min(4, len(bowl_vals)):.1f}" if bowl_vals else "--"
                sec += f"  {'Team avg':>25} Bat:{t_bat:>3}  Bowl:{t_bowl:>3}"
                return sec

            # IMPORTANT: use e1/t1s (team1 labels) with t1_xi/b1_bat (team1 data),
            # and e2/t2s (team2 labels) with t2_xi/b2_bat (team2 data).
            # eb1/b1s are bat_first labels — mixing them with team1 data causes
            # player lists to appear under the wrong team header when bat_first != team1.
            if t1_xi:
                msg += _xi_section(e1, t1s, t1_xi, t1_scores,
                                   "Batting" if bat_first == team1 else "Chasing",
                                   team_bat_avg=b1_bat or None, team_bowl_avg=b1_bowl or None)
            if t2_xi:
                msg += _xi_section(e2, t2s, t2_xi, t2_scores,
                                   "Batting" if bat_first == team2 else "Chasing",
                                   team_bat_avg=b2_bat or None, team_bowl_avg=b2_bowl or None)
        # Impact player info
        _bf_imp = _match_state.get("bf_impact_player")
        _bs_imp = _match_state.get("bs_impact_player")
        if _bf_imp or _bs_imp:
            _auto = " (auto)" if _match_state.get("impact_auto_detected") else ""
            msg += f"\n\n{divider()}\n🔄 <b>Impact Players{_auto}</b>\n{divider()}\n"
            if _bf_imp:
                msg += f"{eb1} {b1s}: <b>{DATA_TO_FULL.get(_bf_imp, _bf_imp)}</b>\n"
            if _bs_imp:
                msg += f"{eb2} {b2s}: <b>{DATA_TO_FULL.get(_bs_imp, _bs_imp)}</b>\n"
            msg += "<i>Override with /impact TEAM: Player Name</i>"
        # Weather conditions block
        if _weather:
            wd = weather_display(_weather)
            msg += f"\n\n{divider()}\n🌤 <b>Match Conditions</b>\n{divider()}\n{wd}"
            if _weather_adj != 0:
                pct = abs(_weather_adj) * 100
                side = short(bat_second) if _weather_adj > 0 else short(bat_first)
                msg += f"\n📊 Weather edge: <b>+{pct:.0f}% {side}</b>"
        if factors:
            msg += f"\n\n{divider()}\n🔍 <b>Key Factors</b>\n" + "\n".join(f"• {f}" for f in factors)
        if explanation:
            msg += f"\n\n<i>{explanation}</i>"
        send_telegram(msg)
        excel_prematch(match_id, team1, team2, venue, toss_winner, toss_decision, ml, explanation)

    b1s, b2s = short(bat_first), short(bat_second)

    # Convenience: fetch score — CricAPI matchScore -> Cricbuzz -> cricScore
    # Flag: set True once both CricAPI keys are exhausted → skip CricAPI entirely
    _cricapi_skip = False
    _cricapi_skip_notified = False
    _notified_exhausted_keys: set = set()

    def _check_key_exhaustion():
        """Drain the newly-exhausted queue, send Telegram alerts, set skip flag."""
        nonlocal _cricapi_skip, _cricapi_skip_notified
        global _cricapi_newly_exhausted
        if not _cricapi_newly_exhausted:
            return
        for key in _cricapi_newly_exhausted:
            if key not in _notified_exhausted_keys:
                _notified_exhausted_keys.add(key)
                label = _key_label(key)
                send_telegram(f"⚠️ <b>CricAPI {label} exhausted</b> — {_KEY_EXHAUST_THRESHOLD} consecutive failures. Falling back to next key.")
                print(f"  [CRICAPI] TG sent: {label} exhausted")
        _cricapi_newly_exhausted.clear()
        # If all known active keys are exhausted, switch to Cricbuzz-only
        active_keys = _get_all_cricapi_keys(api_key)
        if all(k in _key_exhausted for k in active_keys) and not _cricapi_skip_notified:
            _cricapi_skip = True
            _cricapi_skip_notified = True
            send_telegram("🔴 <b>All CricAPI keys exhausted</b> — switching to Cricbuzz-only mode for live scores.")
            print("  [CRICAPI] All keys exhausted — Cricbuzz-only mode activated")

    def _get_score():
        nonlocal bat_first, bat_second, toss_winner, toss_decision
        # 1. CricAPI matchScore — skip if all keys are exhausted
        if not _cricapi_skip and _matchinfo_works and match_id:
            s = get_match_score(match_id, api_key)
            _check_key_exhaustion()
            if s:
                return s
        # 2. Cricbuzz scrape (no API calls, highly reliable)
        if _cb_match_id and _cb_slug:
            s = get_cricbuzz_score(_cb_match_id, _cb_slug, bat_first, bat_second)
            if s and s.get("score"):
                # If batting order was inferred from score, update local vars
                if s.get("_inferred_bat_first") and not bat_first:
                    bat_first  = s["_inferred_bat_first"]
                    bat_second = s["_inferred_bat_second"]
                    toss_winner   = bat_first
                    toss_decision = "bat"
                return s
        # 3. CricAPI cricScore (last resort — skip if all keys exhausted)
        if not _cricapi_skip:
            return get_score_from_cricscore(team1, team2, api_key)
        return None

    def _get_score_rain_poll():
        """Cricbuzz-only score fetch — zero API calls, used during rain delays."""
        if _cb_match_id and _cb_slug:
            s = get_cricbuzz_score(_cb_match_id, _cb_slug, bat_first, bat_second)
            if s and s.get("score"):
                return s
        return None

    # ── PHASE 4: First innings live ───────────────────────────────────
    mlog.phase("inn1")
    _match_state["phase"] = "inn1"
    # Wait for match to actually start
    inn1_start = ist_today_at(*schedule["inn1_start"])
    if now_ist() < inn1_start:
        sleep_until(inn1_start, "1st innings start")

    print("\n[Phase 4] First innings tracking...")
    last_inn1_over = -1
    _last_inn1_balls = -1    # for ball-by-ball mode
    _stall_polls = 0   # consecutive polls with no over change
    _last_known_overs = -1
    inn1_complete = False
    inn1_final_runs = 0
    inn1_final_wkts = 0
    inn1_final_overs = 0.0
    inn1_r = 0          # live running totals updated each poll; Phase 5 uses as fallback
    inn1_w = 0
    inn1_o = 0.0
    # Safety deadline: match_start + 6h covers rain delays, D/L, etc.
    # Phase 4 exits via state (inn1 complete, inn2 started) — deadline is a last-resort guard.
    _match_start_dt = ist_today_at(*schedule["match_start"])
    inn1_safety_deadline = _match_start_dt + timedelta(hours=6)
    inn1_pp_runs = None   # locked at end of over 6
    inn1_pp_wkts = None
    # Partnership tracking for max_partnership feature
    inn1_max_partnership     = 0   # largest partnership so far (never resets)
    inn1_last_wkt_runs       = 0   # cumulative runs when last wicket fell
    inn1_prev_w_for_part     = 0   # previous wickets count for change detection

    while not inn1_complete and now_ist() < inn1_safety_deadline:
        score = _get_score()
        if score:
            # Rain check
            _is_rain, _rain_text = _check_rain_status(score)
            _handle_rain_change(_is_rain, _rain_text, "inn1")
            if _match_state["rain_active"]:
                # Use Cricbuzz-only poll during rain to avoid burning API quota
                time.sleep(300)  # 5-minute rain poll interval
                _rain_score = _get_score_rain_poll()
                if _rain_score:
                    _r2, _rt = _check_rain_status(_rain_score)
                    _handle_rain_change(_r2, _rt, "inn1")
                continue

            scores = score.get("score", [])
            for s in scores:
                inning = s.get("inning", "")
                if any(w in inning for w in bat_first.split()):
                    inn1_r = s.get("r", 0)
                    inn1_w = s.get("w", 0)
                    inn1_o = s.get("o", 0)
                    inn1_balls = overs_to_balls(inn1_o)
                    inn1_over = int(inn1_o)
                    partial1 = round((inn1_o - inn1_over) * 10)

                    # Lock powerplay stats at end of over 6 only
                    if inn1_over == 6 and partial1 == 0 and inn1_pp_runs is None:
                        inn1_pp_runs = inn1_r
                        inn1_pp_wkts = inn1_w
                        print(f"  [PP Inn1] Powerplay complete: {inn1_pp_runs}/{inn1_pp_wkts}")

                    # Determine if we should predict: per-over (default) or ball-by-ball (/predictASAP)
                    _is_over_boundary = inn1_over > last_inn1_over and partial1 == 0 and inn1_balls > 0
                    _is_new_ball = inn1_balls > _last_inn1_balls and inn1_balls > 0
                    _bbb_active = _match_state.get("ball_by_ball", False)

                    # Update max_partnership tracking
                    if inn1_w > inn1_prev_w_for_part:
                        # Wicket(s) fell — capture partnership that just ended
                        _prev_r = _match_state.get("inn1_runs", inn1_r)
                        ended_part = max(0, _prev_r - inn1_last_wkt_runs)
                        inn1_max_partnership = max(inn1_max_partnership, ended_part)
                        inn1_last_wkt_runs = inn1_r  # new partnership starts ~here
                        inn1_prev_w_for_part = inn1_w
                    _cur_inn1_part = max(0, inn1_r - inn1_last_wkt_runs)
                    inn1_max_partnership = max(inn1_max_partnership, _cur_inn1_part)

                    if _is_over_boundary or (_bbb_active and _is_new_ball):
                        if _is_over_boundary:
                            last_inn1_over = inn1_over
                        _last_inn1_balls = inn1_balls
                        # Try unified model first, fall back to old inn1 model
                        ml = ml_live_unified(1, bat_first, bat_second,
                                             inn1_r, inn1_w, inn1_balls, venue,
                                             pp_runs=inn1_pp_runs, pp_wickets=inn1_pp_wkts,
                                             max_partnership=inn1_max_partnership)
                        if ml is None:
                            ml = ml_live_inn1(bat_first, bat_second, inn1_r, inn1_w, inn1_balls, venue,
                                             pp_runs=inn1_pp_runs, pp_wickets=inn1_pp_wkts)
                        mlog.prediction("inn1_live",
                                         {"over": inn1_over, "runs": inn1_r, "wickets": inn1_w,
                                          "balls": inn1_balls, "pp_runs": inn1_pp_runs, "pp_wkts": inn1_pp_wkts},
                                         ml, weather_adj=_weather_adj)
                        if ml:
                            # Unified model returns bat_first_win_probability
                            # Old model returns batting_team_win_probability (same in Inn1)
                            p_bat  = ml.get("bat_first_win_probability",
                                            ml.get("batting_team_win_probability", 0.5))
                            p_bowl = 1 - p_bat
                            # Apply weather adjustment: positive adj = chasing team benefits
                            # → bat_first win prob decreases
                            if _weather_adj != 0:
                                p_bat  = min(0.95, max(0.05, p_bat - _weather_adj))
                                p_bowl = 1 - p_bat
                            ms     = ml["match_state"]
                            crr    = ms["current_run_rate"]
                            proj   = ms.get("projected_score", inn1_r)
                            eb1, eb2 = t_emoji(bat_first), t_emoji(bat_second)
                            proj_lo = max(0, int(proj) - 7)
                            proj_hi = int(proj) + 7
                            if _bbb_active and not _is_over_boundary:
                                # Ball-by-ball: compact message, no Claude explanation
                                msg = (
                                    f"⚡ <b>{b1s} {inn1_r}/{inn1_w}</b> ({inn1_o} ov)\n"
                                    f"CRR: {crr:.1f} · Proj: {proj_lo}–{proj_hi}\n"
                                    f"{eb1} {b1s} {p_bat*100:.1f}% {prob_bar(p_bat)} "
                                    f"{eb2} {b2s} {p_bowl*100:.1f}%"
                                )
                            else:
                                # Standard per-over message
                                proj_line = f"📈 Projected: <b>{proj_lo}–{proj_hi}</b>\n" if inn1_over < 20 else ""
                                msg = (
                                    f"{eb1} <b>LIVE · Over {inn1_over} · {b1s} {inn1_r}/{inn1_w}</b>\n\n"
                                    f"{proj_line}"
                                    f"⚡ CRR: {crr:.1f}\n\n"
                                    f"{divider()}\n"
                                    f"📊 <b>Win Probability</b>\n"
                                    f"{divider()}\n"
                                    f"{eb1} <b>{b1s}</b>  {p_bat*100:.1f}%  {prob_bar(p_bat)}\n"
                                    f"{eb2} <b>{b2s}</b>  {p_bowl*100:.1f}%  {prob_bar(p_bowl)}"
                                )
                                # Per-over Claude reason (explain probability shift)
                                prev_inn1_prob = _match_state.get("last_inn1_prob")
                                if prev_inn1_prob is not None and ANTHROPIC_API_KEY:
                                    delta = p_bat * 100 - prev_inn1_prob
                                    if abs(delta) >= 2:
                                        _prev_r = _match_state.get("inn1_runs", 0)
                                        _prev_w = _match_state.get("inn1_wkts", 0)
                                        _ov_runs = inn1_r - _prev_r
                                        _ov_wkts = inn1_w - _prev_w
                                        _wkt_txt = f"{_ov_wkts} wicket{'s' if _ov_wkts!=1 else ''} fell" if _ov_wkts > 0 else "no wickets"
                                        reason = claude_explain(
                                            f"IPL T20 1st innings: {short(bat_first)} batting. "
                                            f"End of over {inn1_over}: {inn1_r}/{inn1_w} ({_ov_runs} runs, {_wkt_txt} this over). "
                                            f"CRR: {crr:.1f}, projected: {proj_lo}-{proj_hi}. "
                                            f"Bat-first win prob {'+' if delta>0 else ''}{delta:.1f}pp "
                                            f"({prev_inn1_prob:.1f}% -> {p_bat*100:.1f}%). "
                                            f"1 sentence explaining the shift. Only mention wickets if {_ov_wkts} > 0. "
                                            f"Be specific. No disclaimers."
                                        )
                                        if reason:
                                            msg += f"\n\n<i>{reason}</i>"
                            _match_state["last_inn1_prob"] = p_bat * 100
                            send_telegram(msg)
                            if _is_over_boundary:
                                excel_live(match_id, team1, team2, inn1_over,
                                           p_bat * 100, f"{inn1_r}/{inn1_w}",
                                           ml["predicted_winner"])
                        _match_state.update({"inn1_runs": inn1_r, "inn1_wkts": inn1_w, "inn1_overs": inn1_o})
                        if _is_over_boundary:
                            print(f"  Inn1 Over {inn1_over}: {inn1_r}/{inn1_w} (calls: {api_call_count})")
                        else:
                            print(f"  Inn1 Ball {inn1_o}: {inn1_r}/{inn1_w} [BBB]")

                # Check if 2nd innings has started — require at least 1 ball bowled
                # (score feed sometimes includes a 0/0 placeholder for the upcoming innings)
                if any(w in s.get("inning", "") for w in bat_second.split()):
                    if overs_to_balls(s.get("o", 0)) > 0 or s.get("r", 0) > 0:
                        inn1_complete = True
                        break

            # Also check if innings ended by 20 overs or all out
            if last_inn1_over >= 20:
                inn1_complete = True

            # matchEnded only signals end if we've tracked at least 10 overs
            # (prevents early exit on rain interruptions in first few overs)
            if score.get("matchEnded") and last_inn1_over >= 10:
                inn1_complete = True

            # Stall detection: if overs haven't changed in 5 consecutive polls, likely rain
            _current_overs = max(s.get("o", 0) for s in scores) if scores else -1
            if _current_overs == _last_known_overs and _current_overs > 0:
                _stall_polls += 1
                if _stall_polls >= 5 and not _match_state["rain_active"] and not _match_state.get("stall_notified"):
                    _match_state["stall_notified"] = True
                    _handle_rain_change(True, f"Score stalled at {_current_overs} overs (possible rain delay)", "inn1")
            else:
                _stall_polls = 0
                _last_known_overs = _current_overs
                if _match_state.get("stall_notified"):
                    _match_state["stall_notified"] = False

        if not inn1_complete:
            time.sleep(20 if _match_state.get("ball_by_ball") else POLL_INN1)

    # ── PHASE 5: Innings break ────────────────────────────────────────
    mlog.phase("innings_break")
    # Pre-populate inn1_final_* from Phase 4's last tracked values as a robust fallback.
    # This prevents inn1_final_runs from staying 0 if the innings-break score feed
    # doesn't include bat_first's score in a detectable format (e.g. post-match result page).
    if inn1_r > 0:
        inn1_final_runs = inn1_r
        inn1_final_wkts = inn1_w
        inn1_final_overs = inn1_o
    # Get final 1st innings score — if Phase 4 was skipped (late start), poll until
    # the innings is confirmed finished (20 overs or bat_second has started).
    _inn1_confirmed = False
    _phase5_polls = 0
    while not _inn1_confirmed:
        score = _get_score()
        if score:
            for s in score.get("score", []):
                inning = s.get("inning", "")
                if any(w in inning for w in bat_first.split()):
                    _polled_r = s.get("r", 0)
                    # Guard: only accept polled value if it's >= Phase 4's last tracked value.
                    # This prevents a corrupt/post-match feed entry from overwriting the correct score.
                    if _polled_r >= inn1_final_runs:
                        inn1_final_runs = _polled_r
                        inn1_final_wkts = s.get("w", 0)
                        inn1_final_overs = s.get("o", 0)
                # 2nd innings already has runs → 1st innings definitely done
                if any(w in inning for w in bat_second.split()):
                    if s.get("r", 0) > 0 or overs_to_balls(s.get("o", 0)) > 0:
                        _inn1_confirmed = True
            # 20 overs completed or all out
            if inn1_final_wkts >= 10 or int(inn1_final_overs) >= 20:
                _inn1_confirmed = True
            # matchEnded also confirms first innings is done
            if score.get("matchEnded"):
                _inn1_confirmed = True
        if not _inn1_confirmed:
            _phase5_polls += 1
            if _phase5_polls == 1:
                print(f"  [Phase 5] Inn1 not finished yet ({inn1_final_runs}/{inn1_final_wkts} in {inn1_final_overs} ov) — waiting...")
            time.sleep(30)  # poll every 30s until confirmed
    if _phase5_polls > 0:
        print(f"  [Phase 5] Inn1 confirmed after {_phase5_polls} extra poll(s): {inn1_final_runs}/{inn1_final_wkts} ({inn1_final_overs} ov)")

    target = inn1_final_runs + 1
    max_inn2_balls = 120  # default 20 overs; reduced for D/L matches
    # Try to extract D/L target from status field (rain-affected matches)
    _break_status = (score or {}).get("status", "")
    _dl_target, _dl_max_balls = _parse_dl_target(_break_status, target)
    if _dl_target != target:
        print(f"  [D/L] Revised target detected: {_dl_target} (was {target}), max balls: {_dl_max_balls}")
        target = _dl_target
        max_inn2_balls = _dl_max_balls
    _match_state.update({"phase": "break", "target": target,
                         "inn1_runs": inn1_final_runs, "inn1_wkts": inn1_final_wkts,
                         "max_inn2_balls": max_inn2_balls})
    eb1, eb2 = t_emoji(bat_first), t_emoji(bat_second)

    # Win probability at innings break — use 1st-innings live model final reading
    break_prob_line = ""
    inn1_final_balls = overs_to_balls(inn1_final_overs)
    # Guard: require at least 6 balls (1 over) to make a meaningful break prediction.
    # If we have < 6 balls, the data is corrupted/incomplete — skip the prediction.
    if inn1_final_balls < 6 and inn1_r > inn1_final_runs:
        # inn1_r from Phase 4 tracking is more reliable — use it directly
        inn1_final_runs  = inn1_r
        inn1_final_wkts  = inn1_w
        inn1_final_overs = inn1_o
        inn1_final_balls = overs_to_balls(inn1_final_overs)
        print(f"  [Phase 5 guard] Corrected inn1_final from Phase 4 tracking: "
              f"{inn1_final_runs}/{inn1_final_wkts} ({inn1_final_overs} ov)")
    if inn1_final_balls >= 6:
        ml_break = ml_live_unified(1, bat_first, bat_second,
                                   inn1_final_runs, inn1_final_wkts, inn1_final_balls, venue,
                                   pp_runs=inn1_pp_runs, pp_wickets=inn1_pp_wkts)
        if ml_break is None:
            ml_break = ml_live_inn1(bat_first, bat_second, inn1_final_runs, inn1_final_wkts,
                                    inn1_final_balls, venue,
                                    pp_runs=inn1_pp_runs, pp_wickets=inn1_pp_wkts)
        mlog.prediction("inn1_break",
                         {"runs": inn1_final_runs, "wickets": inn1_final_wkts,
                          "balls": inn1_final_balls, "target": target},
                         ml_break, weather_adj=_weather_adj)
        if ml_break:
            p_def  = ml_break.get("bat_first_win_probability",
                                  ml_break.get("batting_team_win_probability", 0.5))
            # Apply weather adjustment at innings break
            if _weather_adj != 0:
                p_def = min(0.95, max(0.05, p_def - _weather_adj))
            p_chas = 1 - p_def
            break_prob_line = (
                f"\n{divider()}\n"
                f"📊 <b>Chase Prediction</b>\n"
                f"{divider()}\n"
                f"{eb1} <b>{b1s}</b>  {p_def*100:.1f}%  {prob_bar(p_def)}\n"
                f"{eb2} <b>{b2s}</b>  {p_chas*100:.1f}%  {prob_bar(p_chas)}\n"
            )
            _match_state["last_inn1_prob"] = p_def * 100
            # Store as anchor for Inn2 — used to smooth the model switch (overs 1-6)
            _match_state["inn1_anchor_prob"] = p_def  # bat_first win prob as 0-1 float

    msg = (
        f"⏸ <b>INNINGS BREAK</b>\n\n"
        f"{eb1} <b>{b1s}:</b> {inn1_final_runs}/{inn1_final_wkts}  ({inn1_final_overs} ov)\n\n"
        f"{divider()}\n"
        f"🎯 <b>Target for {b2s}: {target}</b>\n"
        f"{divider()}"
        f"{break_prob_line}\n"
        f"⏳ <i>2nd innings begins shortly...</i>"
    )
    send_telegram(msg)
    print(f"\n[Phase 5] Innings break. {b1s}: {inn1_final_runs}/{inn1_final_wkts}. Target: {target}")

    # Sleep through innings break — but skip if 2nd innings already underway
    # (happens when bot is started mid-match)
    _inn2_already_started = False
    if score:
        for s in score.get("score", []):
            if any(w in s.get("inning", "") for w in bat_second.split()):
                if s.get("o", 0) > 0:
                    _inn2_already_started = True
                    break
    if _inn2_already_started:
        print("  2nd innings already in progress — skipping break sleep")
    else:
        print("  Sleeping through innings break (18 min)...")
        time.sleep(18 * 60)

    # ── PHASE 6: Second innings live ──────────────────────────────────
    mlog.phase("inn2")
    _match_state["phase"] = "inn2"
    print("\n[Phase 6] Second innings tracking...")
    last_inn2_over = -1
    _last_inn2_balls = -1    # for ball-by-ball mode
    match_ended = False
    _is_no_result = False
    # Safety deadline: match_start + 6h (same as Phase 4) — exits via state, not time
    inn2_safety_deadline = _match_start_dt + timedelta(hours=6)
    inn2_pp_runs = None   # locked at end of over 6
    inn2_pp_wkts = None
    # Partnership tracking for max_partnership feature
    inn2_max_partnership     = 0
    inn2_last_wkt_runs       = 0
    inn2_prev_w_for_part     = 0
    # Carry forward D/L max_balls from Phase 5 (120 for normal match)
    max_inn2_balls = _match_state.get("max_inn2_balls", 120)
    # Initialise tracking vars — stays 0 if PBKS never bats (abandoned match)
    inn2_r = inn2_w = inn2_over = 0
    inn2_o = 0.0
    inn2_balls = 0
    _stall_polls_2 = 0
    _last_known_overs_2 = -1

    while not match_ended and now_ist() < inn2_safety_deadline:
        score = _get_score()
        if score:
            # Rain check
            _is_rain, _rain_text = _check_rain_status(score)
            _handle_rain_change(_is_rain, _rain_text, "inn2")
            if _match_state["rain_active"]:
                # Use Cricbuzz-only at 5-min intervals — zero API calls during rain
                time.sleep(300)
                _rain_score = _get_score_rain_poll()
                if _rain_score:
                    _r2, _rt = _check_rain_status(_rain_score)
                    _handle_rain_change(_r2, _rt, "inn2")
                continue

            ended = score.get("matchEnded", False)
            scores = score.get("score", [])
            _game_state_winner = None   # reset each poll; status/score detection may set it

            # ── Status-string match-end detection (fires mid-over) ────────────
            # Cricbuzz updates status to "X won by Y" the moment the match ends,
            # even before matchEnded flag is set. Check this every poll so we
            # don't wait until the next over boundary.
            _poll_status = score.get("status", "")
            if not ended and _poll_status and inn2_balls > 0:
                _sl = _poll_status.lower()
                _status_ended = (
                    "won by" in _sl or
                    ("won" in _sl and ("wicket" in _sl or "run" in _sl)) or
                    "match tied" in _sl or "super over" in _sl
                )
                if _status_ended and not _match_state.get("rain_active"):
                    print(f"  [Status-End] Match over detected from status: '{_poll_status}'")
                    ended = True
                    # Determine winner from status text
                    _sl_winner = None
                    for _t in [bat_first, bat_second]:
                        if short(_t).lower() in _sl or _t.split()[-1].lower() in _sl:
                            _sl_winner = _t
                            break
                    if _sl_winner and not _game_state_winner:
                        _game_state_winner = _sl_winner
            _dl_t, _dl_mb = _parse_dl_target(_poll_status, target, current_runs=inn2_r)
            if _dl_t != target and _dl_t > 0:
                print(f"  [D/L] Target updated mid-game: {target} → {_dl_t}, max balls: {_dl_mb}")
                target = _dl_t
                max_inn2_balls = _dl_mb

            for s in scores:
                inning = s.get("inning", "")
                if any(w in inning for w in bat_second.split()):
                    _raw_inn2_r = s.get("r", 0)
                    _raw_inn2_o = s.get("o", 0)
                    # Sanity guard: detect Cricbuzz feed inversion (bat_first's score appearing
                    # under bat_second's innings label). This happens after very short Inn2
                    # chases when the API flips inning labels.
                    # Signal: runs AND overs both exactly match inn1_final values → same innings.
                    _inn2_sanity_ok = True
                    if (inn1_final_runs > 0
                            and _raw_inn2_r == inn1_final_runs
                            and _raw_inn2_o == inn1_final_overs
                            and inn2_r == 0):
                        # This entry is bat_first's completed innings leaking into bat_second's slot
                        _inn2_sanity_ok = False
                        print(f"  [Inn2 guard] Rejecting entry r={_raw_inn2_r}/{_raw_inn2_o}ov "
                              f"— matches inn1 final exactly, likely feed inversion.")
                    elif inn2_r > 0 and _raw_inn2_r < inn2_r - 5:
                        # Runs can't go backwards more than 5 (small corrections allowed)
                        _inn2_sanity_ok = False
                        print(f"  [Inn2 guard] Rejecting inn2_r={_raw_inn2_r} < previous {inn2_r} "
                              f"— stale/inverted feed. Keeping inn2_r={inn2_r}.")
                    if _inn2_sanity_ok:
                        inn2_r = _raw_inn2_r
                    inn2_w = s.get("w", 0)
                    inn2_o = s.get("o", 0)
                    inn2_balls = overs_to_balls(inn2_o)
                    inn2_over = int(inn2_o)
                    partial2 = round((inn2_o - inn2_over) * 10)

                    # Lock powerplay stats at end of over 6 only
                    if inn2_over == 6 and partial2 == 0 and inn2_pp_runs is None:
                        inn2_pp_runs = inn2_r
                        inn2_pp_wkts = inn2_w
                        print(f"  [PP Inn2] Powerplay complete: {inn2_pp_runs}/{inn2_pp_wkts}")

                    # Skip prediction if chase is already complete — match is over.
                    # (Can happen if feed is slow or score arrives post-result.)
                    if inn2_r >= target:
                        print(f"  [Inn2] Chase complete ({inn2_r} >= target {target}), skipping prediction.")
                        break

                    # Determine if we should predict: per-over (default) or ball-by-ball (/predictASAP)
                    _is_over_boundary2 = inn2_over > last_inn2_over and partial2 == 0 and inn2_balls > 0
                    _is_new_ball2 = inn2_balls > _last_inn2_balls and inn2_balls > 0
                    _bbb_active2 = _match_state.get("ball_by_ball", False)

                    # Update max_partnership tracking (Inn2)
                    if inn2_w > inn2_prev_w_for_part:
                        _prev_r2 = _match_state.get("inn2_runs", inn2_r)
                        ended_part2 = max(0, _prev_r2 - inn2_last_wkt_runs)
                        inn2_max_partnership = max(inn2_max_partnership, ended_part2)
                        inn2_last_wkt_runs = inn2_r
                        inn2_prev_w_for_part = inn2_w
                    _cur_inn2_part = max(0, inn2_r - inn2_last_wkt_runs)
                    inn2_max_partnership = max(inn2_max_partnership, _cur_inn2_part)

                    if _is_over_boundary2 or (_bbb_active2 and _is_new_ball2):
                        if _is_over_boundary2:
                            last_inn2_over = inn2_over
                        _last_inn2_balls = inn2_balls
                        # Try unified model first — it integrates Inn1 context natively
                        ml = ml_live_unified(2, bat_first, bat_second,
                                             inn2_r, inn2_w, inn2_balls, venue,
                                             target=target,
                                             first_innings_wickets=inn1_final_wkts,
                                             pp_runs=inn2_pp_runs, pp_wickets=inn2_pp_wkts,
                                             max_balls=max_inn2_balls,
                                             max_partnership=inn2_max_partnership)
                        if ml is None:
                            ml = ml_live_inn2(bat_second, bat_first, inn2_r, inn2_w, inn2_balls, target,
                                             venue=venue, first_innings_wickets=inn1_final_wkts,
                                             pp_runs=inn2_pp_runs, pp_wickets=inn2_pp_wkts,
                                             max_balls=max_inn2_balls)
                        mlog.prediction("inn2_live",
                                         {"over": inn2_over, "runs": inn2_r, "wickets": inn2_w,
                                          "balls": inn2_balls, "target": target,
                                          "runs_needed": max(0, target - inn2_r),
                                          "pp_runs": inn2_pp_runs, "pp_wkts": inn2_pp_wkts},
                                         ml, weather_adj=_weather_adj)
                        if ml:
                            # Clamp probability when target already reached — model doesn't know
                            _runs_needed_now = max(0, target - inn2_r)
                            if _runs_needed_now <= 0:
                                p_chase = 1.0
                                p_bowl  = 0.0
                            else:
                            # Unified: bat_second_win_probability = chase prob
                            # Old Inn2 model: batting_team_win_probability = chase prob
                                p_chase = ml.get("bat_second_win_probability",
                                                 ml.get("batting_team_win_probability", 0.5))
                                p_bowl  = 1 - p_chase
                            # ── Anchor/weather only apply when target not yet reached ──
                            if _runs_needed_now > 0:
                                _using_unified = "bat_second_win_probability" in ml
                                _inn1_anchor = _match_state.get("inn1_anchor_prob")
                                if not _using_unified and _inn1_anchor is not None and inn2_over <= 6:
                                    anchor_weight = (6 - inn2_over) / 6
                                    p_chase_equiv = 1.0 - _inn1_anchor
                                    p_chase = (1 - anchor_weight) * p_chase + anchor_weight * p_chase_equiv
                                    p_bowl  = 1 - p_chase
                                if _weather_adj != 0:
                                    p_chase = min(0.95, max(0.05, p_chase + _weather_adj))
                                    p_bowl  = 1 - p_chase

                            ms      = ml["match_state"]
                            crr     = ms["current_run_rate"]
                            rrr     = ms["required_run_rate"]
                            needed  = ms["runs_needed"]
                            balls_left = ms["balls_remaining"]
                            rrr_gap = round(float(rrr) - float(crr), 1)
                            gap_str = f"+{rrr_gap}" if rrr_gap >= 0 else str(rrr_gap)
                            eb1, eb2 = t_emoji(bat_first), t_emoji(bat_second)

                            # Track score delta for accurate commentary
                            _prev_inn2_r = _match_state.get("inn2_runs", 0)
                            _prev_inn2_w = _match_state.get("inn2_wkts", 0)
                            _over_runs   = inn2_r - _prev_inn2_r
                            _over_wkts   = inn2_w - _prev_inn2_w

                            if _bbb_active2 and not _is_over_boundary2:
                                # Ball-by-ball: compact message, no Claude explanation
                                msg = (
                                    f"⚡ <b>{b2s} {inn2_r}/{inn2_w}</b> ({inn2_o} ov)\n"
                                    f"Need {needed} off {balls_left} · CRR: {crr:.1f} · RRR: {rrr:.1f}\n"
                                    f"{eb1} {b1s} {p_bowl*100:.1f}% {prob_bar(p_bowl)} "
                                    f"{eb2} {b2s} {p_chase*100:.1f}%"
                                )
                            else:
                                # Standard per-over message
                                msg = (
                                    f"{eb2} <b>LIVE · Over {inn2_over} · {b2s} {inn2_r}/{inn2_w}</b>\n\n"
                                    f"🎯 Target: {target}  ·  Need: <b>{needed} off {balls_left} balls</b>\n"
                                    f"⚡ CRR: {crr:.1f}  ·  RRR: <b>{rrr:.1f}</b>  ·  Gap: {gap_str}\n\n"
                                    f"{divider()}\n"
                                    f"📊 <b>Win Probability</b>\n"
                                    f"{divider()}\n"
                                    f"{eb1} <b>{b1s}</b>  {p_bowl*100:.1f}%  {prob_bar(p_bowl)}\n"
                                    f"{eb2} <b>{b2s}</b>  {p_chase*100:.1f}%  {prob_bar(p_chase)}"
                                )
                                # Per-over Claude reason — skip if target already reached (nothing to explain)
                                prev_inn2_prob = _match_state.get("last_inn2_prob")
                                if prev_inn2_prob is not None and ANTHROPIC_API_KEY and _runs_needed_now > 0:
                                    delta = p_chase * 100 - prev_inn2_prob
                                    if abs(delta) >= 2:
                                        _wkt_txt = f"{_over_wkts} wicket{'s' if _over_wkts!=1 else ''} fell" if _over_wkts > 0 else "no wickets"
                                        reason = claude_explain(
                                            f"IPL T20 2nd innings: {short(bat_second)} chasing {target}. "
                                            f"End of over {inn2_over}: {inn2_r}/{inn2_w} ({_over_runs} runs, {_wkt_txt} this over). "
                                            f"Need {needed} off {balls_left} balls. CRR: {crr:.1f}, RRR: {rrr:.1f}. "
                                            f"Chase win prob {'+' if delta>0 else ''}{delta:.1f}pp "
                                            f"({prev_inn2_prob:.1f}% -> {p_chase*100:.1f}%). "
                                            f"1 sentence explaining the shift. Only mention wickets if {_over_wkts} > 0. "
                                            f"Be specific. No disclaimers."
                                        )
                                        if reason:
                                            msg += f"\n\n<i>{reason}</i>"
                            _match_state["last_inn2_prob"] = p_chase * 100
                            send_telegram(msg)
                            if _is_over_boundary2:
                                excel_live(match_id, team1, team2, inn2_over,
                                           p_chase * 100, f"{inn2_r}/{inn2_w}",
                                           ml["predicted_winner"])
                        _match_state.update({"inn2_runs": inn2_r, "inn2_wkts": inn2_w, "inn2_overs": inn2_o})
                        if _is_over_boundary2:
                            print(f"  Inn2 Over {inn2_over}: {inn2_r}/{inn2_w} (calls: {api_call_count})")
                        else:
                            print(f"  Inn2 Ball {inn2_o}: {inn2_r}/{inn2_w} [BBB]")

            # ── Detect match end from score state (Cricbuzz has no matchEnded flag) ──
            # Preserve _game_state_winner if already set by status-string detection above
            if not ended or _game_state_winner is None:
                _game_state_winner = None
            max_inn2_overs = max_inn2_balls // 6  # D/L overs (20 for normal match)
            if not ended and inn2_r > 0:
                if inn2_r >= target:
                    ended = True
                    _game_state_winner = bat_second
                elif inn2_w >= 10:
                    ended = True
                    if inn2_r == target - 1:
                        _game_state_winner = "tied"
                    else:
                        _game_state_winner = bat_first
                elif inn2_over >= max_inn2_overs and partial2 == 0:
                    ended = True
                    if inn2_r == target - 1:
                        _game_state_winner = "tied"
                    else:
                        _game_state_winner = bat_first

        # ── Fallback checks (fire even when score=None, using persisted inn2_* values) ──
        if not match_ended and not ended and inn2_balls > 0:
            if inn2_w >= 10:
                print(f"  [10-wkt fallback] inn2_w={inn2_w} — ending match")
                ended = True
                _game_state_winner = "tied" if inn2_r == target - 1 else bat_first
            elif inn2_r >= target:
                print(f"  [Target fallback] inn2_r={inn2_r} >= target={target} — ending match")
                ended = True
                _game_state_winner = bat_second

        # ── Match-end processing — runs for ANY ended=True (score block OR fallback) ──
        if ended and not match_ended:
            match_ended = True
            status = score.get("status", "") if score else ""

            # If game state conclusively shows a winner, use that — don't trust "live" status
            if _game_state_winner == bat_second:
                if "won" not in status.lower():
                    status = f"{short(bat_second)} won by {10 - inn2_w} wickets"
                _is_no_result = False
            elif _game_state_winner == bat_first:
                if "won" not in status.lower():
                    margin = inn1_final_runs - inn2_r
                    status = f"{short(bat_first)} won by {margin} runs"
                _is_no_result = False
            elif _game_state_winner == "tied":
                _is_no_result = False
                send_telegram(
                    "🏏 <b>MATCH TIED!</b>\n\n"
                    f"{t_emoji(bat_first)} {b1s}: {inn1_final_runs}/{inn1_final_wkts}\n"
                    f"{t_emoji(bat_second)} {b2s}: {inn2_r}/{inn2_w}\n\n"
                    "⚠️ <i>Super Over — prediction may not be accurate due to insufficient data</i>"
                )
                print(f"  [TIED] Scores level at {inn2_r}. Polling for Super Over result...")
                # Poll for Super Over result (up to 15 min)
                _so_winner = None
                for _so_poll in range(30):
                    time.sleep(30)
                    _so_score = _get_score()
                    if _so_score:
                        _so_status = (_so_score.get("status") or "").lower()
                        if "won" in _so_status:
                            for _t in [bat_first, bat_second]:
                                if short(_t).lower() in _so_status or _t.split()[-1].lower() in _so_status:
                                    _so_winner = _t
                                    break
                        if _so_score.get("matchEnded") and not _so_winner:
                            _mw = _so_score.get("matchWinner", "")
                            if _mw:
                                _so_winner = _mw
                        if _so_winner:
                            status = _so_score.get("status") or f"{short(_so_winner)} won (Super Over)"
                            score = _so_score  # update for retrain
                            print(f"  [TIED] Super Over winner: {_so_winner} after {_so_poll+1} polls")
                            break
                if not _so_winner:
                    status = status or "Match Tied (Super Over result unknown)"
                    print("  [TIED] Super Over result not detected after 15 min")
            else:
                # No conclusive game state — check status text and API matchEnded flag
                _status_lower = status.lower()
                _has_valid_result = any(k in _status_lower for k in ["won", "tied"])
                _explicitly_abandoned = any(k in _status_lower for k in
                                            ["no result", "abandoned", "cancelled"])
                if _has_valid_result:
                    _is_no_result = False
                else:
                    # Status "live" or empty with no conclusive play = rain/abandoned
                    _is_no_result = True
                    status = "No Result (rain / abandoned)"

            if not status:
                status = f"{short(bat_second)} won by {10 - inn2_w} wickets"
            print(f"  [ResultCheck] status='{status}' inn2_r={inn2_r} inn2_over={inn2_over} no_result={_is_no_result}")
            # Get final scores
            i1r, i1w, i1o, i2r, i2w, i2o = 0, 0, 0, 0, 0, 0
            for s in scores:
                inning = s.get("inning", "")
                if any(w in inning for w in bat_first.split()):
                    i1r, i1w, i1o = s.get("r",0), s.get("w",0), s.get("o",0)
                elif any(w in inning for w in bat_second.split()):
                    i2r, i2w, i2o = s.get("r",0), s.get("w",0), s.get("o",0)

            eb1, eb2 = t_emoji(bat_first), t_emoji(bat_second)
            msg = (
                f"🏁 <b>MATCH RESULT</b>\n\n"
                f"🏆 <b>{status}</b>\n\n"
                f"{divider()}\n"
                f"{eb1} {b1s}: <b>{i1r}/{i1w}</b>  ({i1o} ov)\n"
                f"{eb2} {b2s}: <b>{i2r}/{i2w}</b>  ({i2o} ov)\n"
                f"{divider()}\n\n"
                f"🤖 <i>{'No retrain — match abandoned' if _is_no_result else 'Models retraining with this result...'}</i>"
            )
            send_telegram(msg)
            _match_state["phase"] = "ended"
            print(f"\n  MATCH ENDED. Status: {status}")
            print(f"  Total CricAPI calls: {api_call_count}")

            if _is_no_result:
                print("  [Retrain] Match abandoned / no result — skipping retrain")
                mlog.phase("ended")
                mlog.end("No Result", scores={"bat_first": f"{i1r}/{i1w}", "bat_second": f"{i2r}/{i2w}"})
                break

            # Excel result update
            actual_winner = score.get("matchWinner", "")
            if not actual_winner:
                # Try to parse winner from status string (e.g. "RCB won by 6 wickets")
                for team in [bat_first, bat_second]:
                    if short(team) in status or team.split()[-1] in status:
                        actual_winner = team
                        break
            # Extract margin from status (everything after "won by")
            win_margin = ""
            if " won by " in status.lower():
                win_margin = status.lower().split(" won by ", 1)[-1].strip()
            excel_result(
                match_id, team1, team2,
                actual_winner or status,
                win_margin,
                f"{i1r}/{i1w}",
                f"{i2r}/{i2w}",
            )

            # ── Log match end and run analysis ──
            mlog.phase("ended")
            _final_scores = {
                "bat_first": f"{i1r}/{i1w} ({i1o} ov)",
                "bat_second": f"{i2r}/{i2w} ({i2o} ov)",
            }
            mlog.end(status, winner=actual_winner or None, scores=_final_scores)
            print("\n[Post-Match] Running automated analysis...")
            mlog.run_analysis(
                anthropic_key=ANTHROPIC_API_KEY,
                send_telegram_fn=send_telegram,
            )

            # Auto-retrain — build minimal match_info if CricAPI matchInfo wasn't available
            print("\n[Phase 7] Auto-retraining models...")
            retrain_info = match_info_cache or {
                "id": match_id or f"ipl2026_{now_ist().strftime('%Y%m%d')}",
                "tossWinner": toss_winner or bat_first,
                "tossChoice": toss_decision or "bat",
            }
            # Build a minimal score dict if Cricbuzz (no matchWinner field)
            retrain_score = dict(score) if score else {}
            if not retrain_score.get("matchWinner"):
                if _game_state_winner == "tied" and _so_winner:
                    retrain_score["matchWinner"] = _so_winner
                elif inn2_r >= target:
                    retrain_score["matchWinner"] = bat_second
                elif inn2_w >= 10 or inn2_over >= max_inn2_overs:
                    retrain_score["matchWinner"] = bat_first
                retrain_score["status"] = status
            auto_retrain(retrain_info, retrain_score, bat_first, bat_second, venue)

        # Stall detection for inn2 — skip once innings is complete (avoids false rain alerts)
        if not match_ended and scores and inn2_over < max_inn2_overs:
            _inn2_scores = [s.get("o", 0) for s in scores if any(w in s.get("inning","") for w in bat_second.split())]
            _curr_ov_2 = max(_inn2_scores) if _inn2_scores else -1
            if _curr_ov_2 == _last_known_overs_2 and _curr_ov_2 > 0:
                _stall_polls_2 += 1
                if _stall_polls_2 >= 5 and not _match_state["rain_active"] and not _match_state.get("stall_notified"):
                    _match_state["stall_notified"] = True
                    _handle_rain_change(True, f"Score stalled at {_curr_ov_2} overs (possible rain delay)", "inn2")
            else:
                _stall_polls_2 = 0
                _last_known_overs_2 = _curr_ov_2
                if _match_state.get("stall_notified"):
                    _match_state["stall_notified"] = False


        if not match_ended:
            time.sleep(20 if _match_state.get("ball_by_ball") else POLL_INN2)

    # ── Deadline expired without match_ended — save whatever we have ──
    if not match_ended:
        print("  [Phase 6] Deadline expired without match end — logging partial session")
        _partial_status = f"Deadline expired — {inn2_r}/{inn2_w} in {inn2_over} ov (target {target})"
        mlog.phase("ended")
        mlog.end(_partial_status, winner=None, scores={
            "bat_first": f"{inn1_final_runs}/{inn1_final_wkts}",
            "bat_second": f"{inn2_r}/{inn2_w} ({inn2_over} ov)",
        })
        print("\n[Post-Match] Running automated analysis (partial match)...")
        mlog.run_analysis(anthropic_key=ANTHROPIC_API_KEY, send_telegram_fn=send_telegram)


# ======================================================================
# ENTRY POINT
# ======================================================================
def main():
    global api_call_count
    parser = argparse.ArgumentParser(description="IPL Match Prediction Bot")
    parser.add_argument("--slot", choices=["afternoon", "evening", "both"], default="both",
                        help="Match slot: afternoon (3:30PM), evening (7:30PM), or both")
    parser.add_argument("--match-id", help="CricAPI match ID (skip auto-detect)")
    parser.add_argument("--poll", type=int, help="Override poll interval (seconds)")
    args = parser.parse_args()

    print("="*60)
    print("IPL Match Prediction Bot v2.0")
    print(f"Time: {now_ist().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"API base: {API_BASE}")
    _test_telegram_connectivity()
    print(f"Claude: {'OK' if ANTHROPIC_API_KEY else 'OFF (set ANTHROPIC_API_KEY)'}")
    _all_keys = _get_all_cricapi_keys()
    print(f"CricAPI keys: {len(_all_keys)} configured ({', '.join(_key_label(k) for k in _all_keys)})")
    print(f"Slot: {args.slot}")
    print("="*60)

    # Start Telegram command listener (background thread)
    start_command_listener()

    # Check API server
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"ML API: {r.json()}")
    except:
        print("ERROR: ML API not running. Start with: python 04_api.py")
        sys.exit(1)

    # Determine which API key to use — first available key
    def get_key(slot=None):
        keys = _get_all_cricapi_keys()
        return keys[0] if keys else ""

    if args.match_id:
        key = get_key(args.slot if args.slot != "both" else "evening")
        info = get_match_info(args.match_id, key)
        if info:
            info["id"] = args.match_id
            slot = args.slot if args.slot != "both" else "evening"
            run_match(info, slot, key)
        else:
            print(f"Could not fetch match: {args.match_id}")
        return

    # ── Try CricAPI cricScore for today's IPL matches ─────────────────
    print("\nFetching today's IPL matches from CricAPI...")
    schedule_matches = get_todays_ipl_matches(get_key("evening"))
    if schedule_matches:
        print(f"Found {len(schedule_matches)} IPL match(es) today:")
        for sm in schedule_matches:
            id_src = sm.get("_id_source", "none")
            mid_str = f"[ID: {sm['match_id'][:8]}... via {id_src}]" if sm.get("match_id") else "[no ID yet]"
            print(f"  {sm['team1']} vs {sm['team2']} at {sm['time']} IST — {sm['venue']} {mid_str}")

        # Filter by --slot if specified
        if args.slot == "afternoon":
            schedule_matches = [m for m in schedule_matches if int(m["time"].split(":")[0]) < 17]
        elif args.slot == "evening":
            schedule_matches = [m for m in schedule_matches if int(m["time"].split(":")[0]) >= 17]

        if not schedule_matches:
            print(f"No {args.slot} matches in schedule today.")
            return

        # Assign API keys: first match always uses afternoon key, second uses evening key
        for i, sm in enumerate(schedule_matches):
            api_call_count = 0
            key = get_key("afternoon") if i == 0 else get_key("evening")
            slot = "afternoon" if i == 0 else "evening"
            print(f"\n{'='*60}")
            print(f"[Match {i+1}/{len(schedule_matches)}] Starting: {sm['team1']} vs {sm['team2']} at {sm['time']} IST")
            print(f"{'='*60}")
            try:
                run_match(sm, slot, key, from_schedule=True)
            except Exception as exc:
                import traceback
                print(f"\n[ERROR] run_match failed for match {i+1} ({sm['team1']} vs {sm['team2']}): {exc}")
                traceback.print_exc()
                send_telegram(f"⚠️ Bot error in match {i+1} ({sm['team1']} vs {sm['team2']}): {exc}")
                # Save whatever logs were captured before the crash
                if not mlog._ended:
                    mlog.end(f"Crashed: {exc}", winner=None, scores=None)
                if mlog._started and not mlog._analysis_run:
                    print("[Post-Match] Running analysis on partial/crashed session...")
                    mlog.run_analysis(anthropic_key=ANTHROPIC_API_KEY, send_telegram_fn=send_telegram)
            print(f"\n[Match {i+1}/{len(schedule_matches)}] Completed: {sm['team1']} vs {sm['team2']}")
        print("\nAll scheduled matches processed. Bot exiting.")
        return

    # ── Fallback: poll /matches endpoint until match appears ──────────
    print("\nNo IPL matches found in cricScore today. Falling back to /matches polling...")
    key = get_key("evening")
    matches = find_todays_ipl_match(key)

    if not matches:
        print("No IPL matches found today. Waiting...")
        while not matches:
            time.sleep(POLL_FIND_MATCH)
            matches = find_todays_ipl_match(key)
            if matches: break
            print(f"  Still no match... ({now_ist().strftime('%H:%M IST')})")

    print(f"Found {len(matches)} match(es) today")

    if len(matches) == 1:
        # Single match — figure out which slot
        m = matches[0]
        match_time = m.get("dateTimeGMT", "")
        # If match starts before 5 PM IST, it's afternoon
        slot = "afternoon" if "10:" in match_time or "09:" in match_time else "evening"
        print(f"  Single match: {m.get('name', '?')} ({slot} slot)")
        run_match(m, slot, get_key(slot))

    elif len(matches) >= 2:
        # Double-header: run afternoon first, then evening
        if args.slot == "afternoon":
            run_match(matches[0], "afternoon", get_key("afternoon"))
        elif args.slot == "evening":
            m = matches[1] if len(matches) > 1 else matches[0]
            run_match(m, "evening", get_key("evening"))
        else:
            # Run both sequentially
            print("  Double-header detected!")
            print(f"  Afternoon: {matches[0].get('name', '?')}")
            print(f"  Evening: {matches[1].get('name', '?')}")

            # Reset counter between matches (using different keys)
            api_call_count = 0
            run_match(matches[0], "afternoon", get_key("afternoon"))

            # Reset for evening
            api_call_count = 0
            run_match(matches[1], "evening", get_key("evening"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nBot stopped. CricAPI calls: {api_call_count}")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        send_telegram(f"Bot crashed: {e}")
