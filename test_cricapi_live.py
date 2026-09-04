"""
Check CricAPI for any live or upcoming matches right now.
"""
import requests, os, json
from dotenv import load_dotenv
load_dotenv()

KEY = os.environ.get("CRICAPI_KEY") or os.environ.get("CRICAPI_KEY_EVENING")

print(f"Using key: {KEY[:8]}...")
print()

# Check current matches
r = requests.get("https://api.cricapi.com/v1/matches", params={"apikey": KEY, "offset": 0}, timeout=15)
data = r.json()

if not data.get("data"):
    print("No data returned:", data)
    exit()

matches = data["data"]
print(f"Total matches found: {len(matches)}\n")

live     = [m for m in matches if m.get("matchStarted") and not m.get("matchEnded")]
upcoming = [m for m in matches if not m.get("matchStarted")]
ended    = [m for m in matches if m.get("matchEnded")]

print(f"=== LIVE NOW ({len(live)}) ===")
for m in live:
    print(f"  [{m['id']}]")
    print(f"  {m.get('name','?')}")
    print(f"  {m.get('venue','?')}")
    scores = m.get("score", [])
    for s in scores:
        print(f"    {s.get('inning','')}: {s.get('r','?')}/{s.get('w','?')} ({s.get('o','?')} ov)")
    print()

print(f"=== UPCOMING ({len(upcoming)}) ===")
for m in upcoming[:10]:
    print(f"  {m.get('name','?')} | {m.get('date','?')[:10]} | {m.get('venue','?')[:40]}")

print(f"\n=== RECENTLY ENDED ({len(ended)}) ===")
for m in ended[:5]:
    print(f"  {m.get('name','?')} | {m.get('status','?')[:60]}")
