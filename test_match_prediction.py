"""
Test prediction for IPL 2026 opener: RCB vs SRH at Chinnaswamy
"""
import requests
import json

BASE = "http://localhost:8000"

print("=" * 60)
print("  IPL 2026 OPENER — PRE-TOSS PREDICTION")
print("  Royal Challengers Bengaluru vs Sunrisers Hyderabad")
print("  M Chinnaswamy Stadium, Bengaluru | 28 Mar 2026")
print("=" * 60)

payload = {
    "team1": "Royal Challengers Bengaluru",
    "team2": "Sunrisers Hyderabad",
    "venue": "M Chinnaswamy Stadium, Bengaluru"
}

r = requests.post(f"{BASE}/predict/prematch", json=payload)
if r.status_code != 200:
    print(f"Error {r.status_code}: {r.text}")
    exit()

d = r.json()
print(f"\n  PRE-TOSS")
print(f"  {'RCB win probability:':<30} {d['team1_win_probability']*100:.1f}%")
print(f"  {'SRH win probability:':<30} {d['team2_win_probability']*100:.1f}%")
print(f"  {'Predicted winner:':<30} {d['predicted_winner']}")
print(f"  {'Confidence:':<30} {d.get('confidence','').upper()}")
print(f"\n  Key factors:")
for f in d.get("key_factors", []):
    print(f"    • {f}")

# ----- Post-toss simulation: RCB wins toss and fields -----
print("\n" + "-" * 60)
print("  SIMULATED: RCB wins toss, elects to FIELD (chase)")
print("-" * 60)

# RCB 2026 likely XI — using data_names from player_database_2026
rcb_xi = [
    "V Kohli",        # Virat Kohli
    "RM Patidar",     # Rajat Patidar
    "PD Salt",        # Phil Salt
    "TH David",       # Tim David
    "JG Bethell",     # Jacob Bethell
    "KH Pandya",      # Krunal Pandya
    "R Shepherd",     # Romario Shepherd
    "VR Iyer",        # Venkatesh Iyer
    "JR Hazlewood",   # Josh Hazlewood
    "Yash Dayal",     # Yash Dayal
    "Vicky Ostwal",   # Vicky Ostwal
]

# SRH 2026 likely XI — using data_names from player_database_2026
srh_xi = [
    "TM Head",              # Travis Head
    "Abhishek Sharma",      # Abhishek Sharma
    "Ishan Kishan",         # Ishan Kishan
    "H Klaasen",            # Heinrich Klaasen
    "Nitish Kumar Reddy",   # Nitish Kumar Reddy
    "BKG Mendis",           # Kamindu Mendis
    "PJ Cummins",           # Pat Cummins
    "LS Livingstone",       # Liam Livingstone
    "HV Patel",             # Harshal Patel
    "B Carse",              # Brydon Carse
    "Harsh Dubey",          # Harsh Dubey
]

payload_toss = {
    "team1": "Royal Challengers Bengaluru",
    "team2": "Sunrisers Hyderabad",
    "venue": "M Chinnaswamy Stadium, Bengaluru",
    "toss_winner": "Royal Challengers Bengaluru",
    "toss_decision": "field",
    "team1_players": rcb_xi,
    "team2_players": srh_xi
}

r2 = requests.post(f"{BASE}/predict/prematch", json=payload_toss)
if r2.status_code != 200:
    print(f"Error {r2.status_code}: {r2.text}")
    exit()

d2 = r2.json()
print(f"\n  POST-TOSS")
print(f"  {'RCB win probability:':<30} {d2['team1_win_probability']*100:.1f}%")
print(f"  {'SRH win probability:':<30} {d2['team2_win_probability']*100:.1f}%")
print(f"  {'Predicted winner:':<30} {d2['predicted_winner']}")
print(f"  {'Confidence:':<30} {d2.get('confidence','').upper()}")
print(f"  {'XI data used:':<30} {d2.get('xi_data_used')}")
print(f"  {'Player source:':<30} {d2.get('player_source')}")

ps = d2.get("player_strengths", {})
if ps:
    print(f"\n  SQUAD STRENGTHS (venue-blended)")
    print(f"  {'RCB bat score:':<30} {ps.get('team1_bat',0):.1f}")
    print(f"  {'SRH bat score:':<30} {ps.get('team2_bat',0):.1f}")
    print(f"  {'RCB bowl score:':<30} {ps.get('team1_bowl',0):.1f}")
    print(f"  {'SRH bowl score:':<30} {ps.get('team2_bowl',0):.1f}")

print(f"\n  Key factors:")
for f in d2.get("key_factors", []):
    print(f"    • {f}")

print("\n" + "=" * 60)
shift = (d2["team1_win_probability"] - d["team1_win_probability"]) * 100
print(f"  Toss impact on RCB: {'+' if shift >= 0 else ''}{shift:.1f}%")
print("=" * 60)
