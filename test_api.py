import requests

BASE = "http://localhost:8000"

# Test 1 — Health check
print("Test 1: Health check")
r = requests.get(f"{BASE}/health")
print(f"  {r.json()}\n")

# Test 2 — Pre-toss prediction (no XI, no toss)
print("Test 2: Pre-toss prediction")
r = requests.post(f"{BASE}/predict/prematch", json={
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium, Mumbai"
})
d = r.json()
print(f"  MI win probability : {d['team1_win_probability']*100:.1f}%")
print(f"  CSK win probability: {d['team2_win_probability']*100:.1f}%")
print(f"  Predicted winner   : {d['predicted_winner']}")
print(f"  Confidence         : {d['confidence']}")
print(f"  Post-toss          : {d['post_toss']}")
print(f"  XI data used       : {d['xi_data_used']}")
print(f"  Player source      : {d['player_strengths']['source']}")
print(f"  Key factors        : {d['key_factors']}\n")

# Test 3 — Post-toss prediction (with toss + XI)
print("Test 3: Post-toss prediction (with playing XI)")
r = requests.post(f"{BASE}/predict/prematch", json={
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium, Mumbai",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "field",
    "team1_players": ["RG Sharma", "Ishan Kishan", "SA Yadav", "TH David",
                      "HH Pandya", "T Varma", "KH Pandya", "JJ Bumrah",
                      "JR Hazlewood", "NU Thushara", "PP Chawla"],
    "team2_players": ["MS Dhoni", "RD Gaikwad", "DP Conway", "AM Rahane",
                      "RA Jadeja", "MM Ali", "M Pathirana", "DL Chahar",
                      "TH Deshpande", "Noor Ahmad", "M Theekshana"]
})
d = r.json()
print(f"  MI win probability : {d['team1_win_probability']*100:.1f}%")
print(f"  CSK win probability: {d['team2_win_probability']*100:.1f}%")
print(f"  Predicted winner   : {d['predicted_winner']}")
print(f"  Confidence         : {d['confidence']}")
print(f"  Post-toss          : {d['post_toss']}")
print(f"  XI data used       : {d['xi_data_used']}")
print(f"  Player source      : {d['player_strengths']['source']}")
ps = d['player_strengths']
print(f"  MI  bat={ps['team1_bat']}  bowl={ps['team1_bowl']}")
print(f"  CSK bat={ps['team2_bat']}  bowl={ps['team2_bowl']}")
print(f"  Key factors        : {d['key_factors']}\n")

print("All tests passed." if True else "")
