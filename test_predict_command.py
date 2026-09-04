"""
Test the /predict command at every match phase.
Mocks CricAPI so no real API calls are burned.
Requires: API server running (python 04_api.py)
Sends real Telegram messages so you can see the output.
"""
import requests, os, time, math
from unittest.mock import patch
from dotenv import load_dotenv
load_dotenv()

# ── Import match_bot internals ──────────────────────────────────────
import match_bot
from match_bot import (
    _match_state, _handle_predict_command,
    send_telegram, TELEGRAM_CHAT_ID,
)

TEAM1 = "Royal Challengers Bengaluru"
TEAM2 = "Sunrisers Hyderabad"
VENUE = "M Chinnaswamy Stadium, Bengaluru"

def reset_state():
    """Reset _match_state to defaults."""
    _match_state.update({
        "team1": None, "team2": None, "venue": None,
        "bat_first": None, "bat_second": None,
        "toss_winner": None, "toss_decision": None,
        "t1_xi": [], "t2_xi": [],
        "phase": "idle",
        "inn1_runs": 0, "inn1_wkts": 0, "inn1_overs": 0,
        "inn2_runs": 0, "inn2_wkts": 0, "inn2_overs": 0,
        "inn1_final_wkts": 0,
        "target": 0,
        "last_prob": None,
        "match_id": "test_mock_123",
        "api_key": "test_key",
    })

def fake_score_inn1(match_id, api_key=None):
    """Mock CricAPI response — 1st innings: SRH 95/2 (12.3 ov)."""
    return {
        "score": [
            {"inning": "Sunrisers Hyderabad Inning 1", "r": 95, "w": 2, "o": 12.3}
        ],
        "matchEnded": False,
    }

def fake_score_inn2(match_id, api_key=None):
    """Mock CricAPI response — 2nd innings: RCB 112/3 (14.2 ov), SRH 174/6 done."""
    return {
        "score": [
            {"inning": "Sunrisers Hyderabad Inning 1", "r": 174, "w": 6, "o": 20.0},
            {"inning": "Royal Challengers Bengaluru Inning 2", "r": 112, "w": 3, "o": 14.2},
        ],
        "matchEnded": False,
    }

errors = []

print("=" * 60)
print("  TEST: /predict COMMAND (fresh CricAPI calls)")
print("  RCB vs SRH · Chinnaswamy")
print("=" * 60)

# ── Check API health ──────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
try:
    r = requests.get(f"{API_BASE}/health", timeout=5)
    assert r.status_code == 200
    print(f"\n  API: OK")
except:
    print("\n  ERROR: API server not running! Start it with: python 04_api.py")
    exit(1)

# ══════════════════════════════════════════════════════════════════
# TEST 1: /predict during PRE-TOSS
# ══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("  TEST 1: /predict during PRE-TOSS")
print("─" * 50)
reset_state()
_match_state.update({
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "phase": "pre_toss",
})
try:
    _handle_predict_command(TELEGRAM_CHAT_ID)
    print("  PASSED ✓")
except Exception as e:
    print(f"  FAILED ✗ {e}")
    errors.append(f"pre_toss: {e}")
time.sleep(1)

# ══════════════════════════════════════════════════════════════════
# TEST 2: /predict during POST-TOSS
# ══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("  TEST 2: /predict during POST-TOSS")
print("─" * 50)
reset_state()
_match_state.update({
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "phase": "post_toss",
    "toss_winner": TEAM1, "toss_decision": "field",
    "bat_first": TEAM2, "bat_second": TEAM1,
})
try:
    _handle_predict_command(TELEGRAM_CHAT_ID)
    print("  PASSED ✓")
except Exception as e:
    print(f"  FAILED ✗ {e}")
    errors.append(f"post_toss: {e}")
time.sleep(1)

# ══════════════════════════════════════════════════════════════════
# TEST 3: /predict during 1ST INNINGS (with fresh CricAPI mock)
# ══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("  TEST 3: /predict during 1ST INNINGS (fresh score)")
print("─" * 50)
reset_state()
_match_state.update({
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "phase": "inn1",
    "toss_winner": TEAM1, "toss_decision": "field",
    "bat_first": TEAM2, "bat_second": TEAM1,
    "inn1_runs": 40, "inn1_wkts": 1, "inn1_overs": 6,
})
try:
    with patch("match_bot.get_match_score", side_effect=fake_score_inn1):
        _handle_predict_command(TELEGRAM_CHAT_ID)
    # Verify state was updated with fresh score
    assert _match_state["inn1_runs"] == 95, f"Expected 95, got {_match_state['inn1_runs']}"
    assert _match_state["inn1_wkts"] == 2, f"Expected 2, got {_match_state['inn1_wkts']}"
    print(f"  State updated: SRH 95/2 (12.3 ov) ✓")
    print("  PASSED ✓")
except Exception as e:
    print(f"  FAILED ✗ {e}")
    import traceback; traceback.print_exc()
    errors.append(f"inn1_fresh: {e}")
time.sleep(1)

# ══════════════════════════════════════════════════════════════════
# TEST 4: /predict during INNINGS BREAK
# ══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("  TEST 4: /predict during INNINGS BREAK")
print("─" * 50)
reset_state()
_match_state.update({
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "phase": "break",
    "bat_first": TEAM2, "bat_second": TEAM1,
    "inn1_runs": 174, "inn1_wkts": 6, "inn1_overs": 20,
    "inn1_final_wkts": 6,
    "target": 175,
    "inn2_runs": 0, "inn2_wkts": 0, "inn2_overs": 0,
})
try:
    # During break, CricAPI won't have 2nd innings data yet → falls back to break message
    _handle_predict_command(TELEGRAM_CHAT_ID)
    print("  PASSED ✓")
except Exception as e:
    print(f"  FAILED ✗ {e}")
    errors.append(f"break: {e}")
time.sleep(1)

# ══════════════════════════════════════════════════════════════════
# TEST 5: /predict during 2ND INNINGS (with fresh CricAPI mock)
# ══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("  TEST 5: /predict during 2ND INNINGS (fresh score)")
print("─" * 50)
reset_state()
_match_state.update({
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "phase": "inn2",
    "bat_first": TEAM2, "bat_second": TEAM1,
    "inn1_runs": 174, "inn1_wkts": 6, "inn1_overs": 20,
    "inn1_final_wkts": 6,
    "target": 175,
    "inn2_runs": 80, "inn2_wkts": 2, "inn2_overs": 10,
})
try:
    with patch("match_bot.get_match_score", side_effect=fake_score_inn2):
        _handle_predict_command(TELEGRAM_CHAT_ID)
    # Verify state was updated with fresh score
    assert _match_state["inn2_runs"] == 112, f"Expected 112, got {_match_state['inn2_runs']}"
    assert _match_state["inn2_wkts"] == 3, f"Expected 3, got {_match_state['inn2_wkts']}"
    print(f"  State updated: RCB 112/3 (14.2 ov) ✓")
    print("  PASSED ✓")
except Exception as e:
    print(f"  FAILED ✗ {e}")
    import traceback; traceback.print_exc()
    errors.append(f"inn2_fresh: {e}")
time.sleep(1)

# ══════════════════════════════════════════════════════════════════
# TEST 6: /predict when match ENDED
# ══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("  TEST 6: /predict when match ENDED")
print("─" * 50)
reset_state()
_match_state.update({
    "team1": TEAM1, "team2": TEAM2, "venue": VENUE,
    "phase": "ended",
})
try:
    _handle_predict_command(TELEGRAM_CHAT_ID)
    print("  PASSED ✓")
except Exception as e:
    print(f"  FAILED ✗ {e}")
    errors.append(f"ended: {e}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {6 - len(errors)}/6 passed")
if errors:
    for e in errors:
        print(f"    ✗ {e}")
else:
    print("  All tests passed ✓")
print()
print("  Telegram messages sent:")
print("  1. Pre-toss prediction")
print("  2. Post-toss prediction")
print("  3. 1st innings (fresh score: SRH 95/2, 12.3 ov)")
print("  4. Innings break (target 175)")
print("  5. 2nd innings (fresh score: RCB 112/3, 14.2 ov)")
print("  6. Match ended")
print("=" * 60)
