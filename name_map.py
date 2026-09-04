"""
name_map.py
Maps CricAPI full player names → data_name abbreviations used in
player_database_2026.csv and historical deliveries.csv.

Usage:
    from name_map import to_data_name, xi_to_data_names

    # Single player
    data_name = to_data_name("Rohit Sharma")  # → "RG Sharma"

    # Full XI list from CricAPI matchInfo
    data_names = xi_to_data_names(["Rohit Sharma", "Virat Kohli", ...])

IMPORTANT: target data_names must exactly match player_database_2026.csv.
Run: python -c "from name_map import verify_map; verify_map()" to check.
"""

# Full CricAPI name  →  data_name (Cricsheet abbreviated format)
NAME_MAP = {
    # ── Mumbai Indians ────────────────────────────────────────────────
    "Rohit Sharma":             "RG Sharma",
    "Ishan Kishan":             "Ishan Kishan",
    "Suryakumar Yadav":         "SA Yadav",
    "Hardik Pandya":            "HH Pandya",
    "Tilak Varma":              "Tilak Varma",
    "Naman Dhir":               "Naman Dhir",
    "Kieron Pollard":           "KA Pollard",
    "Tim David":                "TH David",
    "Jasprit Bumrah":           "JJ Bumrah",
    "Trent Boult":              "TA Boult",
    "Deepak Chahar":            "DL Chahar",
    "Rahul Chahar":             "RD Chahar",
    "Piyush Chawla":            "PP Chawla",
    "Krunal Pandya":            "KH Pandya",
    "Arjun Tendulkar":          "Arjun Tendulkar",
    "Nuwan Thushara":           "N Thushara",
    "Will Jacks":               "WG Jacks",
    "Robin Minz":               "R Minz",
    "Allah Ghazanfar":          "Allah Ghazanfar",
    "Reece Topley":             "RJW Topley",
    "Mitchell Santner":         "MJ Santner",
    "Raj Bawa":                 "RA Bawa",

    # ── Chennai Super Kings ───────────────────────────────────────────
    "MS Dhoni":                 "MS Dhoni",
    "Ruturaj Gaikwad":          "RD Gaikwad",
    "Devon Conway":             "DP Conway",
    "Ajinkya Rahane":           "AM Rahane",
    "Ambati Rayudu":            "AT Rayudu",
    "Ravindra Jadeja":          "RA Jadeja",
    "Moeen Ali":                "MM Ali",
    "Tushar Deshpande":         "TU Deshpande",
    "Matheesha Pathirana":      "M Pathirana",
    "Noor Ahmad":               "Noor Ahmad",
    "Khaleel Ahmed":            "KK Ahmed",
    "Anshul Kamboj":            "A Kamboj",
    "Rachin Ravindra":          "R Ravindra",
    "Shaik Rasheed":            "SK Rasheed",
    "Jamie Overton":            "J Overton",
    "Mukesh Choudhary":         "Mukesh Choudhary",
    "Sameer Rizvi":             "Sameer Rizvi",
    "Nathan Ellis":             "NT Ellis",

    # ── Royal Challengers Bengaluru ───────────────────────────────────
    "Virat Kohli":              "V Kohli",
    "Faf du Plessis":           "F du Plessis",
    "Glenn Maxwell":            "GJ Maxwell",
    "Rajat Patidar":            "RM Patidar",
    "Cameron Green":            "CJ Green",
    "Mahipal Lomror":           "MK Lomror",
    "Dinesh Karthik":           "KD Karthik",
    "Suyash Prabhudessai":      "SS Prabhudessai",
    "Wanindu Hasaranga":        "PHKD Hasaranga",
    "WD Hasaranga":             "PHKD Hasaranga",
    "Shahbaz Ahmed":            "Shahbaz Ahmed",
    "Mohammed Siraj":           "Mohammed Siraj",
    "Josh Hazlewood":           "JR Hazlewood",
    "Karn Sharma":              "KS Sharma",
    "Yash Dayal":               "Yash Dayal",
    "Mayank Agarwal":           "MA Agarwal",
    "Srikar Bharat":            "KS Bharat",
    "Phil Salt":                "PD Salt",
    "Jacob Bethell":            "JG Bethell",
    "Liam Livingstone":         "LS Livingstone",
    "Tim Southee":              "TG Southee",
    "Rasikh Dar":               "Rasikh Salam",
    "Lungi Ngidi":              "L Ngidi",
    "Swapnil Singh":            "Swapnil Singh",
    "Romario Shepherd":         "R Shepherd",
    "Manoj Bhandage":           "MS Bhandage",

    # ── Kolkata Knight Riders ─────────────────────────────────────────
    "Shreyas Iyer":             "SS Iyer",
    "Venkatesh Iyer":           "VR Iyer",
    "Nitish Rana":              "N Rana",
    "Rinku Singh":              "RK Singh",
    "Andre Russell":            "AD Russell",
    "Sunil Narine":             "SP Narine",
    "Manish Pandey":            "MK Pandey",
    "Shardul Thakur":           "SN Thakur",
    "Varun Chakravarthy":       "CV Varun",
    "Umesh Yadav":              "UT Yadav",
    "Mitchell Starc":           "MA Starc",
    "Harshit Rana":             "Harshit Rana",
    "Rahmanullah Gurbaz":       "Rahmanullah Gurbaz",
    "Angkrish Raghuvanshi":     "A Raghuvanshi",
    "Spencer Johnson":          "SH Johnson",
    "Anrich Nortje":            "A Nortje",
    "Rovman Powell":            "R Powell",

    # ── Delhi Capitals ────────────────────────────────────────────────
    "David Warner":             "DA Warner",
    "Prithvi Shaw":             "PP Shaw",
    "Axar Patel":               "AR Patel",
    "Rishabh Pant":             "RR Pant",
    "Mitchell Marsh":           "MR Marsh",
    "Tristan Stubbs":           "T Stubbs",
    "Kuldeep Yadav":            "Kuldeep Yadav",
    "Mustafizur Rahman":        "Mustafizur Rahman",
    "Ishant Sharma":            "I Sharma",
    "Shai Hope":                "SD Hope",
    "Jake Fraser-McGurk":       "J Fraser-McGurk",
    "Harry Brook":              "HC Brook",
    "Ashutosh Sharma":          "Ashutosh Sharma",
    "Mohit Sharma":             "MM Sharma",
    "Dushmantha Chameera":      "PVD Chameera",
    "KL Rahul":                 "KL Rahul",
    "Vipraj Nigam":             "V Nigam",
    "Tripurana Vijay":          "Tripurana Vijay",
    "Donovan Ferreira":         "D Ferreira",

    # ── Rajasthan Royals ─────────────────────────────────────────────
    "Sanju Samson":             "SV Samson",
    "Jos Buttler":              "JC Buttler",
    "Yashasvi Jaiswal":         "YBK Jaiswal",
    "Devdutt Padikkal":         "D Padikkal",
    "Riyan Parag":              "R Parag",
    "Shimron Hetmyer":          "SO Hetmyer",
    "Ravichandran Ashwin":      "R Ashwin",
    "Yuzvendra Chahal":         "YS Chahal",
    "Prasidh Krishna":          "M Prasidh Krishna",
    "Sandeep Sharma":           "Sandeep Sharma",
    "Jason Holder":             "JO Holder",
    "Dhruv Jurel":              "Dhruv Jurel",
    "Kunal Rathore":            "KS Rathore",
    "Kunal Singh Rathore":      "KS Rathore",
    "Maheesh Theekshana":       "M Theekshana",
    "Shubham Dubey":            "Shubham Dubey",
    "Akash Madhwal":            "Akash Madhwal",
    "Tanush Kotian":            "Tanush Kotian",

    # ── Sunrisers Hyderabad ───────────────────────────────────────────
    "Pat Cummins":              "PJ Cummins",
    "Heinrich Klaasen":         "H Klaasen",
    "Aiden Markram":            "AK Markram",
    "Abhishek Sharma":          "Abhishek Sharma",
    "Travis Head":              "TM Head",
    "Rahul Tripathi":           "RA Tripathi",
    "Washington Sundar":        "Washington Sundar",
    "Bhuvneshwar Kumar":        "B Kumar",
    "T Natarajan":              "T Natarajan",
    "Shaheen Afridi":           "Shaheen Shah Afridi",
    "Adam Zampa":               "A Zampa",
    "Marco Jansen":             "M Jansen",
    "Mayank Markande":          "M Markande",
    "Nitish Kumar Reddy":       "Nitish Kumar Reddy",
    "Sanvir Singh":             "Sanvir Singh",
    "Zeeshan Ansari":           "Zeeshan Ansari",
    "Jaydev Unadkat":           "JD Unadkat",
    "Aniket Verma":             "Aniket Verma",
    "Simarjeet Singh":          "Simarjeet Singh",
    "Harshal Patel":            "HV Patel",
    "Kamindu Mendis":           "PHKD Mendis",

    # ── Kings XI Punjab (PBKS) ────────────────────────────────────────
    "Shikhar Dhawan":           "S Dhawan",
    "Jonny Bairstow":           "JM Bairstow",
    "Sam Curran":               "SM Curran",
    "Kagiso Rabada":            "K Rabada",
    "Arshdeep Singh":           "Arshdeep Singh",
    "Harpreet Brar":            "Harpreet Brar",
    "Rishi Dhawan":             "R Dhawan",
    "Prabhsimran Singh":        "Prabhsimran Singh",
    "Shashank Singh":           "Shashank Singh",
    "Marcus Stoinis":           "MP Stoinis",
    "Josh Inglis":              "JP Inglis",
    "Azmatullah Omarzai":       "Azmatullah Omarzai",
    "Suryansh Shedge":          "Suryansh Shedge",
    "Priyansh Arya":            "Priyansh Arya",
    "Nehal Wadhera":            "N Wadhera",
    "Vishwanath Pratap Singh":  "V Pratap Singh",
    "Shreyas Gopal":            "S Gopal",
    "Yudhvir Charak":           "Yudhvir Singh Charak",
    "Musheer Khan":             "Musheer Khan",
    "Aaron Hardie":             "AJ Hardie",
    "Xavier Bartlett":          "XC Bartlett",

    # ── Gujarat Titans ────────────────────────────────────────────────
    "Shubman Gill":             "Shubman Gill",
    "David Miller":             "DA Miller",
    "Wriddhiman Saha":          "WP Saha",
    "Rashid Khan":              "Rashid Khan",
    "Mohammed Shami":           "Mohammed Shami",
    "Lockie Ferguson":          "LH Ferguson",
    "Alzarri Joseph":           "AS Joseph",
    "Sai Sudharsan":            "B Sai Sudharsan",
    "Vijay Shankar":            "V Shankar",
    "Rahul Tewatia":            "R Tewatia",
    "Abhinav Manohar":          "A Manohar",
    "Kane Williamson":          "KS Williamson",
    "Darshan Nalkande":         "DG Nalkande",
    "Shahrukh Khan":            "M Shahrukh Khan",
    "Matthew Wade":             "MS Wade",
    "Sherfane Rutherford":      "SE Rutherford",
    "Anuj Rawat":               "Anuj Rawat",
    "Manav Suthar":             "MJ Suthar",
    "Karim Janat":              "Karim Janat",
    "Gerald Coetzee":           "G Coetzee",
    "Urvil Patel":              "Urvil Patel",
    "Sai Kishore":              "R Sai Kishore",
    "Jayant Yadav":             "J Yadav",
    "Ishant Sharma":            "I Sharma",

    # ── Lucknow Super Giants ──────────────────────────────────────────
    "Quinton de Kock":          "Q de Kock",
    "Nicholas Pooran":          "N Pooran",
    "Deepak Hooda":             "DJ Hooda",
    "Ravi Bishnoi":             "R Bishnoi",
    "Mark Wood":                "MA Wood",
    "Avesh Khan":               "Avesh Khan",
    "Mohsin Khan":              "Mohsin Khan",
    "Ayush Badoni":             "A Badoni",
    "Kyle Mayers":              "KR Mayers",
    "Prerak Mankad":            "PN Mankad",
    "Shamar Joseph":            "S Joseph",
    "David Willey":             "DJ Willey",
    "Yash Thakur":              "Yash Thakur",
    "Aryan Juyal":              "Aryan Juyal",
    "Manan Vohra":              "M Vohra",
    "Digvesh Rathi":            "DS Rathi",
    "Matt Henry":               "MJ Henry",
    "Akash Singh":              "Akash Singh",

    # ── Common alternate spellings / CricAPI variations ───────────────
    "Mohammed Shami":           "Mohammed Shami",
    "Mohammed Siraj":           "Mohammed Siraj",
    "Arshdeep Singh":           "Arshdeep Singh",
    "Noor Ahmad":               "Noor Ahmad",
    "Avesh Khan":               "Avesh Khan",
    "Mohsin Khan":              "Mohsin Khan",
    "Prabhsimran Singh":        "Prabhsimran Singh",
    "Sarfaraz Khan":            "Sarfaraz Khan",
    "Vaibhav Suryavanshi":      "V Suryavanshi",
    "Ayush Mhatre":             "A Mhatre",
    "Ryan Rickelton":           "RD Rickelton",
    "Matthew Short":            "MW Short",
    "Tim Seifert":              "TL Seifert",
    "Fabian Allen":             "FA Allen",
    "Akeal Hosein":             "AJ Hosein",
    "Eshan Malinga":            "E Malinga",
    "Ishan Kishan":             "Ishan Kishan",
    "Kuldeep Yadav":            "Kuldeep Yadav",
    "Rashid Khan":              "Rashid Khan",
    "T Natarajan":              "T Natarajan",
    "Ramandeep Singh":          "Ramandeep Singh",
    "Naman Dhir":               "Naman Dhir",
    "Rinku Singh":              "RK Singh",

    # ── 2026 IPL / corrected mappings ────────────────────────────────────
    "Shivam Dube":              "S Dube",
    "Vijaykumar Vyshak":        "V Vyshak",
    "Cooper Connolly":          "C Connolly",
    "Yuzvendra Chahal":         "YS Chahal",
    "Sanju Samson":             "SV Samson",
    "Kartik Sharma":            "Kartik Sharma",
    "Prashant Veer":            "Prashant Veer",
    "Harpreet Singh Bhatia":    "Harpreet Singh Bhatia",
}

# Reverse map: data_name → full name (for display)
DATA_TO_FULL = {v: k for k, v in NAME_MAP.items()}


def to_data_name(full_name: str) -> str:
    """
    Convert a CricAPI full player name to data_name format.
    Falls back to the original name if not in map (handles new players).
    """
    return NAME_MAP.get(full_name, full_name)


def xi_to_data_names(cricapi_players: list) -> list:
    """
    Convert a list of CricAPI player names to data_name format.
    Handles both string names and dict objects {"name": "..."}
    """
    result = []
    for p in cricapi_players:
        name = p["name"] if isinstance(p, dict) else str(p)
        result.append(to_data_name(name))
    return result


def verify_map():
    """Check all NAME_MAP targets exist in player_database_2026.csv."""
    import pandas as pd, os
    db_path = os.path.join(os.path.dirname(__file__), "data", "player_database_2026.csv")
    if not os.path.exists(db_path):
        print("player_database_2026.csv not found")
        return
    known = set(pd.read_csv(db_path)["data_name"].dropna())
    misses = [(f, d) for f, d in NAME_MAP.items() if d not in known]
    if misses:
        print(f"{len(misses)} targets not in DB:")
        for f, d in sorted(misses):
            print(f"  {f!r:45s} -> {d!r}")
    else:
        print(f"All {len(NAME_MAP)} mappings verified OK.")
