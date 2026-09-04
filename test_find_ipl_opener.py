"""
Page through CricAPI to find IPL 2026 Match 1 (March 28 opener).
"""
import requests, os
from dotenv import load_dotenv
load_dotenv()

KEY = os.environ.get("CRICAPI_KEY") or os.environ.get("CRICAPI_KEY_EVENING")

all_matches = []
for offset in range(0, 200, 25):
    r = requests.get("https://api.cricapi.com/v1/matches",
                     params={"apikey": KEY, "offset": offset}, timeout=15)
    data = r.json()
    batch = data.get("data", [])
    if not batch:
        break
    all_matches.extend(batch)
    print(f"Fetched offset {offset}: {len(batch)} matches")

print(f"\nTotal fetched: {len(all_matches)}")
print()

# Find IPL 2026 matches in March
ipl = [m for m in all_matches
       if ("Indian Premier League 2026" in m.get("name","") or
           "IPL" in m.get("series",""))
       and "2026-03" in m.get("date","")]

print(f"IPL 2026 March matches: {len(ipl)}")
for m in sorted(ipl, key=lambda x: x.get("date","")):
    print(f"\n  ID:    {m['id']}")
    print(f"  Name:  {m.get('name','?')}")
    print(f"  Date:  {m.get('date','?')}")
    print(f"  Venue: {m.get('venue','?')}")
    print(f"  Teams: {m.get('teams', m.get('teamInfo','?'))}")
