"""
07_build_player_db.py
Build IPL 2026 team/player database with per-venue scores,
last-5-game form, and career IPL stats. Uses NA for missing data.

Output:
  data/player_database_2026.csv   — one row per player
  data/player_venue_scores.csv    — one row per (player, venue)
  data/team_profiles_2026.csv     — aggregated team strength per venue
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os, warnings
warnings.filterwarnings("ignore")

DATA = os.path.join(os.path.dirname(__file__), "data")

# ── IPL 2026 Rosters (full name → abbreviated name in data) ──────────────
# Map: full_name -> (team, role, data_name)
# data_name is how the player appears in deliveries.csv / matches.csv squads
# We'll try to auto-match, and manually fix known mismatches

IPL_2026_SQUADS = {
    "Chennai Super Kings": {
        "players": [
            ("Ruturaj Gaikwad", "batter", "RD Gaikwad"),
            ("MS Dhoni", "wk-batter", "MS Dhoni"),
            ("Sanju Samson", "wk-batter", "SV Samson"),
            ("Dewald Brevis", "batter", "D Brevis"),
            ("Ayush Mhatre", "batter", "A Mhatre"),
            ("Kartik Sharma", "wk-batter", "Kartik Sharma"),
            ("Sarfaraz Khan", "batter", "Sarfaraz Khan"),
            ("Urvil Patel", "wk-batter", "Urvil Patel"),
            ("Shivam Dube", "all-rounder", "S Dube"),
            ("Jamie Overton", "all-rounder", "J Overton"),
            ("Ramakrishna Ghosh", "all-rounder", "Ramakrishna Ghosh"),
            ("Prashant Veer", "all-rounder", "Prashant Veer"),
            ("Matthew Short", "all-rounder", "MW Short"),
            ("Aman Khan", "all-rounder", "Aman Khan"),
            ("Zak Foulkes", "all-rounder", "Z Foulkes"),
            ("Anshul Kamboj", "bowler", "A Kamboj"),
            ("Khaleel Ahmed", "bowler", "KK Ahmed"),
            ("Noor Ahmad", "bowler", "Noor Ahmad"),
            ("Mukesh Choudhary", "bowler", "Mukesh Choudhary"),
            ("Nathan Ellis", "bowler", "NT Ellis"),
            ("Shreyas Gopal", "bowler", "S Gopal"),
            ("Gurjapneet Singh", "bowler", "Gurjapneet Singh"),
            ("Akeal Hosein", "bowler", "AJ Hosein"),
            ("Matt Henry", "bowler", "MJ Henry"),
            ("Rahul Chahar", "bowler", "RD Chahar"),
        ],
        "home_venue": "MA Chidambaram Stadium, Chennai",
    },
    "Kolkata Knight Riders": {
        "players": [
            ("Ajinkya Rahane", "batter", "AM Rahane"),
            ("Angkrish Raghuvanshi", "batter", "A Raghuvanshi"),
            ("Manish Pandey", "batter", "MK Pandey"),
            ("Rinku Singh", "batter", "RK Singh"),
            ("Sunil Narine", "all-rounder", "SP Narine"),
            ("Varun Chakaravarthy", "bowler", "CV Varun"),
            ("Harshit Rana", "bowler", "Harshit Rana"),
            ("Umran Malik", "bowler", "Umran Malik"),
            ("Vaibhav Arora", "bowler", "VG Arora"),
            ("Ramandeep Singh", "all-rounder", "Ramandeep Singh"),
            ("Rovman Powell", "batter", "R Powell"),
            ("Cameron Green", "all-rounder", "C Green"),
            ("Finn Allen", "batter", "FA Allen"),
            ("Matheesha Pathirana", "bowler", "M Pathirana"),
            ("Anukul Roy", "all-rounder", "Anukul Roy"),
            ("Tejasvi Singh", "batter", "Tejasvi Singh"),
            ("Kartik Tyagi", "bowler", "Kartik Tyagi"),
            ("Prashant Solanki", "bowler", "PH Solanki"),
            ("Rahul Tripathi", "batter", "RA Tripathi"),
            ("Tim Seifert", "wk-batter", "TL Seifert"),
            ("Sarthak Ranjan", "batter", "Sarthak Ranjan"),
            ("Daksh Kamra", "batter", "Daksh Kamra"),
            ("Rachin Ravindra", "all-rounder", "R Ravindra"),
            ("Akash Deep", "bowler", "Akash Deep"),
            ("Blessing Muzarabani", "bowler", "Blessing Muzarabani"),
        ],
        "home_venue": "Eden Gardens, Kolkata",
    },
    "Mumbai Indians": {
        "players": [
            ("Rohit Sharma", "batter", "RG Sharma"),
            ("Suryakumar Yadav", "batter", "SA Yadav"),
            ("Hardik Pandya", "all-rounder", "HH Pandya"),
            ("Tilak Varma", "batter", "Tilak Varma"),
            ("Jasprit Bumrah", "bowler", "JJ Bumrah"),
            ("Trent Boult", "bowler", "TA Boult"),
            ("Will Jacks", "all-rounder", "WG Jacks"),
            ("Ryan Rickelton", "batter", "RD Rickelton"),
            ("Robin Minz", "wk-batter", "R Minz"),
            ("Raj Bawa", "all-rounder", "RA Bawa"),
            ("Raghu Sharma", "bowler", "Raghu Sharma"),
            ("Mitchell Santner", "all-rounder", "MJ Santner"),
            ("Corbin Bosch", "all-rounder", "C Bosch"),
            ("Naman Dhir", "all-rounder", "Naman Dhir"),
            ("Allah Ghazanfar", "bowler", "Allah Ghazanfar"),
            ("Ashwani Kumar", "bowler", "Ashwani Kumar"),
            ("Deepak Chahar", "bowler", "DL Chahar"),
            ("Sherfane Rutherford", "all-rounder", "SE Rutherford"),
            ("Mayank Markande", "bowler", "M Markande"),
            ("Shardul Thakur", "bowler", "SN Thakur"),
            ("Quinton de Kock", "wk-batter", "Q de Kock"),
            ("Danish Malewar", "batter", "Danish Malewar"),
            ("Mohammad Izhar", "bowler", "Mohammad Izhar"),
            ("Atharva Ankolekar", "all-rounder", "Atharva Ankolekar"),
            ("Mayank Rawat", "wk-batter", "M Rawat"),
        ],
        "home_venue": "Wankhede Stadium, Mumbai",
    },
    "Royal Challengers Bengaluru": {
        "players": [
            ("Rajat Patidar", "batter", "RM Patidar"),
            ("Virat Kohli", "batter", "V Kohli"),
            ("Devdutt Padikkal", "batter", "D Padikkal"),
            ("Phil Salt", "wk-batter", "PD Salt"),
            ("Jitesh Sharma", "wk-batter", "Jitesh Sharma"),
            ("Krunal Pandya", "all-rounder", "KH Pandya"),
            ("Swapnil Singh", "all-rounder", "Swapnil Singh"),
            ("Tim David", "all-rounder", "TH David"),
            ("Romario Shepherd", "all-rounder", "R Shepherd"),
            ("Jacob Bethell", "all-rounder", "JG Bethell"),
            ("Josh Hazlewood", "bowler", "JR Hazlewood"),
            ("Yash Dayal", "bowler", "Yash Dayal"),
            ("Bhuvneshwar Kumar", "bowler", "B Kumar"),
            ("Nuwan Thushara", "bowler", "N Thushara"),
            ("Rasikh Salam", "bowler", "Rasikh Salam"),
            ("Abhinandan Singh", "bowler", "Abhinandan Singh"),
            ("Suyash Sharma", "bowler", "Suyash Sharma"),
            ("Venkatesh Iyer", "all-rounder", "VR Iyer"),
            ("Jacob Duffy", "bowler", "Jacob Duffy"),
            ("Satvik Deswal", "batter", "Satvik Deswal"),
            ("Mangesh Yadav", "bowler", "Mangesh Yadav"),
            ("Jordan Cox", "batter", "Jordan Cox"),
            ("Vicky Ostwal", "bowler", "Vicky Ostwal"),
            ("Vihaan Malhotra", "batter", "Vihaan Malhotra"),
            ("Kanishk Chouhan", "bowler", "Kanishk Chouhan"),
        ],
        "home_venue": "M Chinnaswamy Stadium, Bengaluru",
    },
    "Gujarat Titans": {
        "players": [
            ("Shubman Gill", "batter", "Shubman Gill"),
            ("Sai Sudharsan", "batter", "B Sai Sudharsan"),
            ("Jos Buttler", "wk-batter", "JC Buttler"),
            ("Kumar Kushagra", "wk-batter", "Kumar Kushagra"),
            ("Anuj Rawat", "wk-batter", "Anuj Rawat"),
            ("Nishant Sindhu", "all-rounder", "N Sindhu"),
            ("Washington Sundar", "all-rounder", "Washington Sundar"),
            ("Glenn Phillips", "all-rounder", "GD Phillips"),
            ("Arshad Khan", "all-rounder", "Arshad Khan"),
            ("Shahrukh Khan", "batter", "M Shahrukh Khan"),
            ("Rahul Tewatia", "all-rounder", "R Tewatia"),
            ("Kagiso Rabada", "bowler", "K Rabada"),
            ("Mohammed Siraj", "bowler", "Mohammed Siraj"),
            ("Prasidh Krishna", "bowler", "M Prasidh Krishna"),
            ("Ishant Sharma", "bowler", "I Sharma"),
            ("Gurnoor Singh Brar", "bowler", "Gurnoor Brar"),
            ("Rashid Khan", "bowler", "Rashid Khan"),
            ("Manav Suthar", "bowler", "MJ Suthar"),
            ("Sai Kishore", "bowler", "R Sai Kishore"),
            ("Jayant Yadav", "bowler", "JD Yadav"),
            ("Ashok Sharma", "all-rounder", "Ashok Sharma"),
            ("Jason Holder", "all-rounder", "JO Holder"),
            ("Tom Banton", "batter", "T Banton"),
            ("Prithvi Raj Yarra", "batter", "Prithvi Raj Yarra"),
            ("Luke Wood", "bowler", "L Wood"),
        ],
        "home_venue": "Narendra Modi Stadium, Ahmedabad",
    },
    "Rajasthan Royals": {
        "players": [
            ("Yashasvi Jaiswal", "batter", "YBK Jaiswal"),
            ("Riyan Parag", "all-rounder", "R Parag"),
            ("Dhruv Jurel", "wk-batter", "Dhruv Jurel"),
            ("Shimron Hetmyer", "batter", "SO Hetmyer"),
            ("Ravindra Jadeja", "all-rounder", "RA Jadeja"),
            ("Sam Curran", "all-rounder", "SM Curran"),
            ("Donovan Ferreira", "all-rounder", "D Ferreira"),
            ("Vaibhav Suryavanshi", "batter", "Vaibhav Suryavanshi"),
            ("Sandeep Sharma", "bowler", "Sandeep Sharma"),
            ("Shubham Dubey", "batter", "Shubham Dubey"),
            ("Lhuan-Dre Pretorius", "all-rounder", "L Pretorius"),
            ("Jofra Archer", "bowler", "JC Archer"),
            ("Tushar Deshpande", "bowler", "TU Deshpande"),
            ("Kwena Maphaka", "bowler", "K Maphaka"),
            ("Nandre Burger", "bowler", "N Burger"),
            ("Ravi Bishnoi", "bowler", "Ravi Bishnoi"),
            ("Sushant Mishra", "bowler", "Sushant Mishra"),
            ("Yudhvir Singh Charak", "all-rounder", "Yudhvir Singh Charak"),
            ("Yash Raj Punja", "bowler", "Yash Raj Punja"),
            ("Vignesh Puthur", "all-rounder", "Vignesh Puthur"),
            ("Ravi Singh", "all-rounder", "Ravi Singh"),
            ("Aman Rao", "batter", "Aman Rao"),
            ("Brijesh Sharma", "batter", "Brijesh Sharma"),
            ("Adam Milne", "bowler", "AF Milne"),
            ("Kuldeep Sen", "bowler", "Kuldeep Sen"),
        ],
        "home_venue": "Sawai Mansingh Stadium, Jaipur",
    },
    "Sunrisers Hyderabad": {
        "players": [
            ("Pat Cummins", "bowler", "PJ Cummins"),
            ("Travis Head", "batter", "TM Head"),
            ("Abhishek Sharma", "all-rounder", "Abhishek Sharma"),
            ("Ishan Kishan", "wk-batter", "Ishan Kishan"),
            ("Heinrich Klaasen", "wk-batter", "H Klaasen"),
            ("Nitish Kumar Reddy", "all-rounder", "Nitish Kumar Reddy"),
            ("Kamindu Mendis", "all-rounder", "BKG Mendis"),
            ("Harshal Patel", "bowler", "HV Patel"),
            ("Brydon Carse", "all-rounder", "B Carse"),
            ("Jaydev Unadkat", "bowler", "JD Unadkat"),
            ("Aniket Verma", "batter", "Aniket Verma"),
            ("R Smaran", "batter", "R Smaran"),
            ("Harsh Dubey", "bowler", "Harsh Dubey"),
            ("Eshan Malinga", "bowler", "E Malinga"),
            ("Zeeshan Ansari", "bowler", "Zeeshan Ansari"),
            ("Shivang Kumar", "bowler", "Shivang Kumar"),
            ("Salil Arora", "bowler", "Salil Arora"),
            ("Sakib Hussain", "bowler", "Sakib Hussain"),
            ("Onkar Tarmale", "bowler", "Onkar Tarmale"),
            ("Amit Kumar", "batter", "Amit Kumar"),
            ("Praful Hinge", "batter", "Praful Hinge"),
            ("Krains Fuletra", "batter", "Krains Fuletra"),
            ("Liam Livingstone", "all-rounder", "LS Livingstone"),
            ("Shivam Mavi", "bowler", "Shivam Mavi"),
            ("Jack Edwards", "all-rounder", "Jack Edwards"),
        ],
        "home_venue": "Rajiv Gandhi Intl Cricket Stadium, Hyderabad",
    },
    "Delhi Capitals": {
        "players": [
            ("KL Rahul", "wk-batter", "KL Rahul"),
            ("Karun Nair", "batter", "K Nair"),
            ("Axar Patel", "all-rounder", "AR Patel"),
            ("Mitchell Starc", "bowler", "MA Starc"),
            ("Kuldeep Yadav", "bowler", "Kuldeep Yadav"),
            ("Tristan Stubbs", "batter", "T Stubbs"),
            ("Abishek Porel", "wk-batter", "Abishek Porel"),
            ("Sameer Rizvi", "batter", "Sameer Rizvi"),
            ("T Natarajan", "bowler", "T Natarajan"),
            ("Nitish Rana", "batter", "N Rana"),
            ("Ashutosh Sharma", "all-rounder", "Ashutosh Sharma"),
            ("Madhav Tiwari", "batter", "Madhav Tiwari"),
            ("Tripurana Vijay", "batter", "Tripurana Vijay"),
            ("Vipraj Nigam", "all-rounder", "Vipraj Nigam"),
            ("Ajay Mandal", "bowler", "Ajay Mandal"),
            ("David Miller", "batter", "DA Miller"),
            ("Ben Duckett", "batter", "BM Duckett"),
            ("Auqib Nabi", "bowler", "Auqib Nabi"),
            ("Pathum Nissanka", "batter", "P Nissanka"),
            ("Lungi Ngidi", "bowler", "L Ngidi"),
            ("Dushmantha Chameera", "bowler", "PVD Chameera"),
            ("Sahil Parakh", "bowler", "Sahil Parakh"),
            ("Prithvi Shaw", "batter", "PP Shaw"),
            ("Kyle Jamieson", "bowler", "KA Jamieson"),
        ],
        "home_venue": "Arun Jaitley Stadium, Delhi",
    },
    "Lucknow Super Giants": {
        "players": [
            ("Rishabh Pant", "wk-batter", "RR Pant"),
            ("Aiden Markram", "batter", "AK Markram"),
            ("Nicholas Pooran", "wk-batter", "N Pooran"),
            ("Mitchell Marsh", "all-rounder", "MR Marsh"),
            ("Ayush Badoni", "batter", "A Badoni"),
            ("Abdul Samad", "all-rounder", "Abdul Samad"),
            ("Matthew Breetzke", "batter", "M Breetzke"),
            ("Himmat Singh", "batter", "Himmat Singh"),
            ("Shahbaz Ahmed", "all-rounder", "Shahbaz Ahmed"),
            ("Arshin Kulkarni", "all-rounder", "Arshin Kulkarni"),
            ("Mayank Yadav", "bowler", "Mayank Yadav"),
            ("Avesh Khan", "bowler", "Avesh Khan"),
            ("Mohsin Khan", "bowler", "Mohsin Khan"),
            ("Manimaran Siddharth", "bowler", "M Siddharth"),
            ("Mohammed Shami", "bowler", "Mohammed Shami"),
            ("Anrich Nortje", "bowler", "A Nortje"),
            ("Wanindu Hasaranga", "all-rounder", "PHKD Hasaranga"),
            ("Digvesh Rathi", "bowler", "DS Rathi"),
            ("Prince Yadav", "bowler", "Prince Yadav"),
            ("Akash Singh", "bowler", "Akash Singh"),
            ("Arjun Tendulkar", "bowler", "Arjun Tendulkar"),
            ("Mukul Choudhary", "bowler", "Mukul Choudhary"),
            ("Naman Tiwari", "all-rounder", "Naman Tiwari"),
            ("Akshat Raghuwanshi", "batter", "Akshat Raghuwanshi"),
            ("Josh Inglis", "wk-batter", "J Inglis"),
        ],
        "home_venue": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    },
    "Punjab Kings": {
        "players": [
            ("Shreyas Iyer", "batter", "SS Iyer"),
            ("Prabhsimran Singh", "wk-batter", "Prabhsimran Singh"),
            ("Priyansh Arya", "batter", "Priyansh Arya"),
            ("Shashank Singh", "all-rounder", "Shashank Singh"),
            ("Nehal Wadhera", "batter", "Nehal Wadhera"),
            ("Marcus Stoinis", "all-rounder", "MP Stoinis"),
            ("Azmatullah Omarzai", "all-rounder", "Azmatullah Omarzai"),
            ("Marco Jansen", "all-rounder", "M Jansen"),
            ("Harpreet Brar", "all-rounder", "Harpreet Brar"),
            ("Yuzvendra Chahal", "bowler", "YS Chahal"),
            ("Arshdeep Singh", "bowler", "Arshdeep Singh"),
            ("Musheer Khan", "all-rounder", "Musheer Khan"),
            ("Pyala Avinash", "batter", "Pyala Avinash"),
            ("Harnoor Pannu", "batter", "Harnoor Pannu"),
            ("Suryansh Shedge", "all-rounder", "Suryansh Shedge"),
            ("Mitchell Owen", "all-rounder", "Mitchell Owen"),
            ("Xavier Bartlett", "bowler", "X Bartlett"),
            ("Lockie Ferguson", "bowler", "LH Ferguson"),
            ("Vyshak Vijaykumar", "bowler", "V Vyshak"),
            ("Yash Thakur", "bowler", "Yash Thakur"),
            ("Vishnu Vinod", "wk-batter", "Vishnu Vinod"),
            ("Cooper Connolly", "all-rounder", "C Connolly"),
            ("Ben Dwarshuis", "bowler", "B Dwarshuis"),
            ("Pravin Dubey", "bowler", "Pravin Dubey"),
            ("Vishal Nishad", "bowler", "Vishal Nishad"),
        ],
        "home_venue": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    },
}

# ── Venue normalization (same as 02_features.py) ──────────────────────────
def norm_venue(v):
    if not isinstance(v, str): return v
    if "Chinnaswamy" in v: return "M Chinnaswamy Stadium, Bengaluru"
    if "Eden" in v: return "Eden Gardens, Kolkata"
    if "Wankhede" in v: return "Wankhede Stadium, Mumbai"
    if "Chepauk" in v or "Chidambaram" in v: return "MA Chidambaram Stadium, Chennai"
    if "Feroz" in v or "Arun Jaitley" in v or "Kotla" in v: return "Arun Jaitley Stadium, Delhi"
    if "Rajiv Gandhi" in v and "Hyderabad" in v: return "Rajiv Gandhi Intl Cricket Stadium, Hyderabad"
    if "Sawai" in v: return "Sawai Mansingh Stadium, Jaipur"
    if "Ekana" in v or ("Lucknow" in v and "Atal" in v): return "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"
    if "Narendra Modi" in v or ("Motera" in v): return "Narendra Modi Stadium, Ahmedabad"
    if "Punjab" in v or "Mohali" in v or "Bindra" in v or "Mullanpur" in v or "Chandigarh" in v:
        return "Punjab Cricket Association IS Bindra Stadium, Mohali"
    if "DY Patil" in v: return "Dr DY Patil Sports Academy, Mumbai"
    if "Brabourne" in v: return "Brabourne Stadium, Mumbai"
    if "Holkar" in v: return "Holkar Cricket Stadium, Indore"
    if "Himachal" in v or "Dharamsala" in v: return "Himachal Pradesh Cricket Association Stadium, Dharamsala"
    if "Barabati" in v: return "Barabati Stadium, Cuttack"
    if "JSCA" in v or "Ranchi" in v: return "JSCA International Stadium Complex, Ranchi"
    if "Greenfield" in v or "Trivandrum" in v or "Thiruvananthapuram" in v:
        return "Greenfield International Stadium, Thiruvananthapuram"
    if "Uppal" in v and "Hyderabad" in v: return "Rajiv Gandhi Intl Cricket Stadium, Hyderabad"
    if "Visakhapatnam" in v or "VDCA" in v or "Rajasekhara" in v:
        return "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam"
    if "Barsapara" in v or "Guwahati" in v: return "Barsapara Cricket Stadium, Guwahati"
    if "Green Park" in v: return "Green Park, Kanpur"
    return v


# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
deliveries = pd.read_csv(os.path.join(DATA, "deliveries.csv"))

matches["venue"] = matches["venue"].apply(norm_venue)
deliveries["venue"] = deliveries["venue"].apply(norm_venue)
matches["date"] = pd.to_datetime(matches["date"])
deliveries["date"] = pd.to_datetime(deliveries["date"])

# Get all IPL venues used in recent years (2019+) for venue scoring
RECENT_VENUES = sorted(matches[matches["season"] >= 2019]["venue"].unique())
print(f"Active venues (2019+): {len(RECENT_VENUES)}")

# ── Collect all 2026 players ─────────────────────────────────────────────
all_players = []
for team, info in IPL_2026_SQUADS.items():
    for full_name, role, data_name in info["players"]:
        all_players.append({
            "full_name": full_name,
            "data_name": data_name,
            "team_2026": team,
            "role": role,
            "home_venue": info["home_venue"],
        })

print(f"Total squad players: {len(all_players)}")

# ── Auto-discover players from deliveries.csv not in the squad list ───────
# Catches mid-season replacements, overseas players added late, and anyone
# missed when IPL_2026_SQUADS was written.
print("Auto-discovering players from deliveries...")
_known_data_names = {p["data_name"] for p in all_players}

# Build team lookup from matches.csv: data_name -> most-recent team
_matches_df = pd.read_csv("data/matches.csv")
_matches_df["date"] = pd.to_datetime(_matches_df["date"])
_matches_df = _matches_df.sort_values("date")

_player_team_map = {}   # data_name -> team
_team_venue_map  = {}   # team -> home_venue (from squad dict)
for team, info in IPL_2026_SQUADS.items():
    _team_venue_map[team] = info["home_venue"]

for _, mr in _matches_df.iterrows():
    for col, team in [("team1_players", mr["team1"]), ("team2_players", mr["team2"])]:
        raw = str(mr.get(col, ""))
        if raw in ("nan", ""):
            continue
        for p in raw.split("|"):
            p = p.strip()
            if p:
                _player_team_map[p] = team   # last occurrence wins (most recent)

# Scan all deliveries for unknown players
_all_del_players = (
    set(deliveries["batter"].dropna()) | set(deliveries["bowler"].dropna())
)
_new_count = 0
for dn in sorted(_all_del_players):
    if dn in _known_data_names:
        continue
    # Infer role from career delivery stats
    bat_inns  = deliveries[deliveries["batter"] == dn]["file_id"].nunique()
    bowl_inns = deliveries[deliveries["bowler"] == dn]["file_id"].nunique()
    if bowl_inns == 0:
        inferred_role = "batter"
    elif bat_inns == 0:
        inferred_role = "bowler"
    elif bowl_inns / (bat_inns + bowl_inns) >= 0.5:
        inferred_role = "all-rounder" if bat_inns >= 3 else "bowler"
    else:
        inferred_role = "all-rounder" if bowl_inns >= 3 else "batter"

    team_2026  = _player_team_map.get(dn, "Unknown")
    home_venue = _team_venue_map.get(team_2026, "")

    all_players.append({
        "full_name":  dn,          # best we can do without a name map
        "data_name":  dn,
        "team_2026":  team_2026,
        "role":       inferred_role,
        "home_venue": home_venue,
    })
    _known_data_names.add(dn)
    _new_count += 1

print(f"  Auto-discovered {_new_count} additional players "
      f"(not in IPL_2026_SQUADS) — total: {len(all_players)}")

# ── Compute career batting stats per player ──────────────────────────────
print("Computing batting stats...")

def compute_bat_stats(player_name, venue=None, last_n=None, seasons=None):
    """Compute batting stats for a player, optionally filtered by venue/recency."""
    mask = deliveries["batter"] == player_name
    if venue:
        mask &= deliveries["venue"] == venue
    if seasons:
        mask &= deliveries["season"].isin(seasons)

    df = deliveries[mask].copy()

    if last_n:
        # Last N matches
        match_ids = df["file_id"].unique()
        if len(match_ids) > last_n:
            # Sort by date, take last N
            match_dates = df.groupby("file_id")["date"].first().sort_values()
            recent_ids = match_dates.tail(last_n).index
            df = df[df["file_id"].isin(recent_ids)]

    if len(df) == 0:
        return {
            "innings": pd.NA, "runs": pd.NA, "balls": pd.NA,
            "avg": pd.NA, "sr": pd.NA, "fours": pd.NA, "sixes": pd.NA,
            "dot_pct": pd.NA, "boundary_pct": pd.NA,
        }

    innings = df["file_id"].nunique()
    # Only count legal balls faced (exclude wides)
    legal = df[df["is_wide"] == 0]
    balls = len(legal)
    runs = legal["runs_batter"].sum()
    fours = (legal["runs_batter"] == 4).sum()
    sixes = (legal["runs_batter"] == 6).sum()
    dots = (legal["runs_batter"] == 0).sum()

    avg = runs / innings if innings > 0 else pd.NA
    sr = (runs / balls * 100) if balls > 0 else pd.NA
    dot_pct = (dots / balls * 100) if balls > 0 else pd.NA
    boundary_pct = ((fours + sixes) / balls * 100) if balls > 0 else pd.NA

    return {
        "innings": innings, "runs": int(runs), "balls": int(balls),
        "avg": round(avg, 2) if pd.notna(avg) else pd.NA,
        "sr": round(sr, 2) if pd.notna(sr) else pd.NA,
        "fours": int(fours), "sixes": int(sixes),
        "dot_pct": round(dot_pct, 1) if pd.notna(dot_pct) else pd.NA,
        "boundary_pct": round(boundary_pct, 1) if pd.notna(boundary_pct) else pd.NA,
    }


def compute_bowl_stats(player_name, venue=None, last_n=None, seasons=None):
    """Compute bowling stats for a player, optionally filtered by venue/recency."""
    mask = deliveries["bowler"] == player_name
    if venue:
        mask &= deliveries["venue"] == venue
    if seasons:
        mask &= deliveries["season"].isin(seasons)

    df = deliveries[mask].copy()

    if last_n:
        match_ids = df["file_id"].unique()
        if len(match_ids) > last_n:
            match_dates = df.groupby("file_id")["date"].first().sort_values()
            recent_ids = match_dates.tail(last_n).index
            df = df[df["file_id"].isin(recent_ids)]

    if len(df) == 0:
        return {
            "bowl_innings": pd.NA, "wickets": pd.NA, "runs_conceded": pd.NA,
            "balls_bowled": pd.NA, "econ": pd.NA, "bowl_sr": pd.NA,
            "bowl_dot_pct": pd.NA,
        }

    innings = df["file_id"].nunique()
    legal = df[(df["is_wide"] == 0) & (df["is_noball"] == 0)]
    balls = len(legal)
    runs_conceded = df["runs_total"].sum()
    wickets = df["is_wicket"].sum()
    dots = (legal["runs_total"] == 0).sum()

    overs = balls / 6
    econ = (runs_conceded / overs) if overs > 0 else pd.NA
    bowl_sr = (balls / wickets) if wickets > 0 else pd.NA
    dot_pct = (dots / balls * 100) if balls > 0 else pd.NA

    return {
        "bowl_innings": innings, "wickets": int(wickets),
        "runs_conceded": int(runs_conceded), "balls_bowled": int(balls),
        "econ": round(econ, 2) if pd.notna(econ) else pd.NA,
        "bowl_sr": round(bowl_sr, 1) if pd.notna(bowl_sr) else pd.NA,
        "bowl_dot_pct": round(dot_pct, 1) if pd.notna(dot_pct) else pd.NA,
    }


# ── Build player database ────────────────────────────────────────────────
print("Building player profiles...")
player_rows = []

for p in all_players:
    dn = p["data_name"]

    # Career stats (all IPL)
    career_bat = compute_bat_stats(dn)
    career_bowl = compute_bowl_stats(dn)

    # Last 5 matches form
    form_bat = compute_bat_stats(dn, last_n=5)
    form_bowl = compute_bowl_stats(dn, last_n=5)

    # Recent seasons (2023-2025)
    recent_bat = compute_bat_stats(dn, seasons=[2023, 2024, 2025])
    recent_bowl = compute_bowl_stats(dn, seasons=[2023, 2024, 2025])

    row = {
        "full_name": p["full_name"],
        "data_name": dn,
        "team_2026": p["team_2026"],
        "role": p["role"],
        "home_venue": p["home_venue"],
        # Career batting
        "career_bat_innings": career_bat["innings"],
        "career_runs": career_bat["runs"],
        "career_balls_faced": career_bat["balls"],
        "career_bat_avg": career_bat["avg"],
        "career_bat_sr": career_bat["sr"],
        "career_fours": career_bat["fours"],
        "career_sixes": career_bat["sixes"],
        # Career bowling
        "career_bowl_innings": career_bowl["bowl_innings"],
        "career_balls_bowled": career_bowl["balls_bowled"],
        "career_wickets": career_bowl["wickets"],
        "career_runs_conceded": career_bowl["runs_conceded"],
        "career_econ": career_bowl["econ"],
        "career_bowl_sr": career_bowl["bowl_sr"],
        # Last 5 match form — batting
        "form5_bat_innings": form_bat["innings"],
        "form5_runs": form_bat["runs"],
        "form5_bat_avg": form_bat["avg"],
        "form5_bat_sr": form_bat["sr"],
        "form5_boundary_pct": form_bat["boundary_pct"],
        # Last 5 match form — bowling
        "form5_bowl_innings": form_bowl["bowl_innings"],
        "form5_wickets": form_bowl["wickets"],
        "form5_econ": form_bowl["econ"],
        "form5_bowl_sr": form_bowl["bowl_sr"],
        "form5_bowl_dot_pct": form_bowl["bowl_dot_pct"],
        # Recent (2023-2025) batting
        "recent_bat_innings": recent_bat["innings"],
        "recent_runs": recent_bat["runs"],
        "recent_bat_avg": recent_bat["avg"],
        "recent_bat_sr": recent_bat["sr"],
        # Recent (2023-2025) bowling
        "recent_bowl_innings": recent_bowl["bowl_innings"],
        "recent_wickets": recent_bowl["wickets"],
        "recent_econ": recent_bowl["econ"],
        "recent_bowl_sr": recent_bowl["bowl_sr"],
    }
    player_rows.append(row)

player_db = pd.DataFrame(player_rows)

# ── Compute composite scores ─────────────────────────────────────────────
print("Computing composite scores...")

def bat_score(row):
    """Composite batting score: blend of career + recent + form."""
    # Non-batters: fewer than 5 career innings → too small a sample, return low score
    career_bi = pd.to_numeric(row.get("career_bat_innings"), errors="coerce")
    if pd.isna(career_bi) or int(career_bi) < 5:
        return 5.0

    parts = []
    weights = []

    # Career component (weight 0.3)
    if pd.notna(row["career_bat_avg"]) and pd.notna(row["career_bat_sr"]):
        career = (row["career_bat_avg"] / 40) * 0.5 + (row["career_bat_sr"] / 150) * 0.5
        parts.append(career)
        weights.append(0.3)

    # Recent component (weight 0.4)
    if pd.notna(row["recent_bat_avg"]) and pd.notna(row["recent_bat_sr"]):
        recent = (row["recent_bat_avg"] / 40) * 0.5 + (row["recent_bat_sr"] / 150) * 0.5
        parts.append(recent)
        weights.append(0.4)

    # Form component (weight 0.3)
    if pd.notna(row["form5_bat_avg"]) and pd.notna(row["form5_bat_sr"]):
        form = (row["form5_bat_avg"] / 40) * 0.5 + (row["form5_bat_sr"] / 150) * 0.5
        parts.append(form)
        weights.append(0.3)

    if not parts:
        return pd.NA

    total_w = sum(weights)
    score = sum(p * w for p, w in zip(parts, weights)) / total_w
    return round(score * 100, 1)  # 0-100 scale (50 = avg IPL batter)


def bowl_score(row):
    """Composite bowling score: blend of career + recent + form.

    Non-bowlers (< 60 career balls = < 10 overs total) return 0.0 explicitly
    so they do not inflate team bowl-strength averages.

    Economy denominator uses /4 (range 6–10 RPO) so elite IPL bowlers (7 econ)
    score ~0.75 on that component instead of the old ~0.5 from /6.
    """
    # Non-bowlers: fewer than 60 career balls → return 0 (not league average)
    career_bb = pd.to_numeric(row.get("career_balls_bowled"), errors="coerce")
    if pd.isna(career_bb) or int(career_bb) < 60:
        return 0.0

    parts = []
    weights = []

    # Career component
    if pd.notna(row["career_econ"]) and pd.notna(row["career_wickets"]):
        # Lower econ is better; /4 gives realistic spread for T20 (6–10 RPO range)
        econ_score = min(1.0, max(0, (10 - row["career_econ"]) / 4))
        wkt_rate = row["career_wickets"] / max(row["career_bowl_innings"], 1)
        career = econ_score * 0.5 + min(wkt_rate / 2, 1) * 0.5
        parts.append(career)
        weights.append(0.3)

    # Recent component
    if pd.notna(row["recent_econ"]) and pd.notna(row["recent_wickets"]):
        econ_score = min(1.0, max(0, (10 - row["recent_econ"]) / 4))
        wkt_rate = row["recent_wickets"] / max(row["recent_bowl_innings"], 1)
        recent = econ_score * 0.5 + min(wkt_rate / 2, 1) * 0.5
        parts.append(recent)
        weights.append(0.4)

    # Form component
    if pd.notna(row["form5_econ"]) and pd.notna(row["form5_wickets"]):
        econ_score = min(1.0, max(0, (10 - row["form5_econ"]) / 4))
        wkt_rate = row["form5_wickets"] / max(row["form5_bowl_innings"], 1)
        form = econ_score * 0.5 + min(wkt_rate / 2, 1) * 0.5
        parts.append(form)
        weights.append(0.3)

    if not parts:
        return 0.0  # has career balls but no scoreable stats → treat as non-bowler

    total_w = sum(weights)
    score = sum(p * w for p, w in zip(parts, weights)) / total_w
    return round(score * 100, 1)


player_db["bat_score"] = pd.to_numeric(player_db.apply(bat_score, axis=1), errors="coerce")
player_db["bowl_score"] = pd.to_numeric(player_db.apply(bowl_score, axis=1), errors="coerce")

# Overall player impact score
def overall_score(row):
    bs = row["bat_score"]
    bw = row["bowl_score"]
    role = row["role"]

    if pd.isna(bs) and pd.isna(bw):
        return pd.NA

    if role in ("batter", "wk-batter"):
        if pd.notna(bs) and pd.notna(bw):
            return round(bs * 0.8 + bw * 0.2, 1)
        return bs if pd.notna(bs) else pd.NA
    elif role == "bowler":
        if pd.notna(bs) and pd.notna(bw):
            return round(bs * 0.2 + bw * 0.8, 1)
        return bw if pd.notna(bw) else pd.NA
    else:  # all-rounder
        if pd.notna(bs) and pd.notna(bw):
            return round(bs * 0.5 + bw * 0.5, 1)
        return bs if pd.notna(bs) else (bw if pd.notna(bw) else pd.NA)

player_db["overall_score"] = pd.to_numeric(player_db.apply(overall_score, axis=1), errors="coerce")

# Save player database
player_db.to_csv(os.path.join(DATA, "player_database_2026.csv"), index=False)
print(f"Saved player_database_2026.csv ({len(player_db)} players)")

# ── Build per-venue scores ────────────────────────────────────────────────
print("\nBuilding per-venue player scores...")
venue_rows = []

for p in all_players:
    dn = p["data_name"]
    for venue in RECENT_VENUES:
        vbat = compute_bat_stats(dn, venue=venue)
        vbowl = compute_bowl_stats(dn, venue=venue)

        # Only create row if player has ANY data at this venue
        has_bat = pd.notna(vbat["innings"])
        has_bowl = pd.notna(vbowl["bowl_innings"])

        # Compute venue-specific scores
        v_bat_score = pd.NA
        if has_bat and pd.notna(vbat["avg"]) and pd.notna(vbat["sr"]):
            v_bat_score = round(((vbat["avg"] / 40) * 0.5 + (vbat["sr"] / 150) * 0.5) * 100, 1)

        v_bowl_score = pd.NA
        venue_balls = pd.to_numeric(vbowl.get("balls_bowled"), errors="coerce") if has_bowl else pd.NA
        if (has_bowl and pd.notna(vbowl["econ"]) and vbowl["wickets"] is not pd.NA
                and pd.notna(venue_balls) and int(venue_balls) >= 30):
            # /4 denominator consistent with career formula; require ≥ 30 venue balls (5 overs)
            econ_s = min(1.0, max(0, (10 - vbowl["econ"]) / 4))
            wkt_r = int(vbowl["wickets"]) / max(int(vbowl["bowl_innings"]), 1)
            v_bowl_score = round((econ_s * 0.5 + min(wkt_r / 2, 1) * 0.5) * 100, 1)

        venue_rows.append({
            "full_name": p["full_name"],
            "data_name": dn,
            "team_2026": p["team_2026"],
            "role": p["role"],
            "venue": venue,
            "venue_bat_innings": vbat["innings"],
            "venue_runs": vbat["runs"],
            "venue_balls": vbat["balls"],
            "venue_bat_avg": vbat["avg"],
            "venue_bat_sr": vbat["sr"],
            "venue_bowl_innings": vbowl["bowl_innings"],
            "venue_wickets": vbowl["wickets"],
            "venue_econ": vbowl["econ"],
            "venue_bowl_sr": vbowl["bowl_sr"],
            "venue_bat_score": v_bat_score,
            "venue_bowl_score": v_bowl_score,
        })

venue_db = pd.DataFrame(venue_rows)
venue_db.to_csv(os.path.join(DATA, "player_venue_scores.csv"), index=False)
print(f"Saved player_venue_scores.csv ({len(venue_db)} rows)")

# ── Build team profiles per venue ─────────────────────────────────────────
print("\nBuilding team profiles...")
team_rows = []

for team, info in IPL_2026_SQUADS.items():
    for venue in RECENT_VENUES:
        team_venue = venue_db[(venue_db["team_2026"] == team) & (venue_db["venue"] == venue)]

        # Aggregate: average scores of players who have data at this venue
        bat_scores = pd.to_numeric(team_venue["venue_bat_score"], errors="coerce").dropna()
        bowl_scores = pd.to_numeric(team_venue["venue_bowl_score"], errors="coerce").dropna()

        team_rows.append({
            "team": team,
            "venue": venue,
            "players_with_bat_data": len(bat_scores),
            "players_with_bowl_data": len(bowl_scores),
            "avg_bat_score": round(bat_scores.mean(), 1) if len(bat_scores) >= 2 else pd.NA,
            "avg_bowl_score": round(bowl_scores.mean(), 1) if len(bowl_scores) >= 2 else pd.NA,
            "max_bat_score": round(bat_scores.max(), 1) if len(bat_scores) > 0 else pd.NA,
            "max_bowl_score": round(bowl_scores.max(), 1) if len(bowl_scores) > 0 else pd.NA,
            "top3_bat_avg": round(bat_scores.nlargest(3).mean(), 1) if len(bat_scores) >= 3 else pd.NA,
            "top3_bowl_avg": round(bowl_scores.nlargest(3).mean(), 1) if len(bowl_scores) >= 3 else pd.NA,
        })

team_profiles = pd.DataFrame(team_rows)
team_profiles.to_csv(os.path.join(DATA, "team_profiles_2026.csv"), index=False)
print(f"Saved team_profiles_2026.csv ({len(team_profiles)} rows)")

# ── Summary report ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("IPL 2026 PLAYER DATABASE SUMMARY")
print("=" * 70)

for team in sorted(IPL_2026_SQUADS.keys()):
    tp = player_db[player_db["team_2026"] == team].copy()
    has_data = tp["overall_score"].notna().sum()
    no_data = tp["overall_score"].isna().sum()

    print(f"\n{team} ({has_data} with data, {no_data} NA):")

    # Top players by overall score
    top = tp.dropna(subset=["overall_score"]).nlargest(5, "overall_score")
    for _, r in top.iterrows():
        bat_s = f"bat:{r['bat_score']}" if pd.notna(r["bat_score"]) else "bat:NA"
        bowl_s = f"bowl:{r['bowl_score']}" if pd.notna(r["bowl_score"]) else "bowl:NA"
        print(f"  {r['full_name']:25s} {r['role']:15s} overall:{r['overall_score']:5.1f}  {bat_s}  {bowl_s}")

print(f"\n{'=' * 70}")
print(f"Total players: {len(player_db)}")
print(f"Players with data: {player_db['overall_score'].notna().sum()}")
print(f"Players with NA (no IPL history): {player_db['overall_score'].isna().sum()}")
print(f"Venues tracked: {len(RECENT_VENUES)}")
print(f"{'=' * 70}")
