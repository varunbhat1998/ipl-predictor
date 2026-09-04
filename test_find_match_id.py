"""
Find IPL 2026 Match 1 (March 28) via CricAPI series endpoint.
"""
import requests, os
from dotenv import load_dotenv
load_dotenv()

KEY = os.environ.get("CRICAPI_KEY") or os.environ.get("CRICAPI_KEY_EVENING")

# Step 1: Find IPL 2026 series
print("Step 1: Searching for IPL 2026 series...")
r = requests.get("https://api.cricapi.com/v1/series",
                 params={"apikey": KEY, "offset": 0}, timeout=15)
series_list = r.json().get("data", [])
ipl_series = [s for s in series_list if "Indian Premier League 2026" in s.get("name","")]
print(f"  Found {len(ipl_series)} IPL 2026 series entries")
for s in ipl_series:
    print(f"  ID: {s['id']}  Name: {s['name']}")

if not ipl_series:
    print("  Not found in offset 0, trying offset 25...")
    r2 = requests.get("https://api.cricapi.com/v1/series",
                      params={"apikey": KEY, "offset": 25}, timeout=15)
    series_list2 = r2.json().get("data", [])
    ipl_series = [s for s in series_list2 if "Indian Premier League" in s.get("name","")]
    for s in ipl_series:
        print(f"  ID: {s['id']}  Name: {s['name']}")

# Step 2: Get matches from the IPL series
if ipl_series:
    series_id = ipl_series[0]["id"]
    print(f"\nStep 2: Fetching matches for series {series_id}...")
    r = requests.get("https://api.cricapi.com/v1/series_info",
                     params={"apikey": KEY, "id": series_id}, timeout=15)
    info = r.json().get("data", {})
    match_list = info.get("matchList", [])
    print(f"  Total matches in series: {len(match_list)}")
    print()
    print("March 2026 matches:")
    for m in match_list:
        date = m.get("date","")
        if "2026-03" in date or "2026-04-0" in date:
            print(f"  ID: {m['id']}")
            print(f"  Match: {m.get('name','?')}")
            print(f"  Date:  {date}")
            print(f"  Venue: {m.get('venue','?')}")
            print()
else:
    print("\nTrying currentMatches endpoint...")
    r = requests.get("https://api.cricapi.com/v1/currentMatches",
                     params={"apikey": KEY}, timeout=15)
    data = r.json()
    print(f"Status: {data.get('status')}")
    matches = data.get("data", [])
    ipl = [m for m in matches if "IPL" in m.get("name","") or "Indian Premier" in m.get("name","")]
    print(f"IPL matches in currentMatches: {len(ipl)}")
    for m in ipl:
        print(f"  {m['id']} | {m.get('name','?')} | {m.get('date','?')}")
