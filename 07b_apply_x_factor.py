"""
07b_apply_x_factor.py
Apply calibrated X-factor to assign scores for 82 NA players.

X-Factor Calibration (from 2025 IPL debutants):
  - Batters:  X = 1.00 (pre-IPL form translates ~1:1 to IPL)
  - Bowlers:  X = 0.80 (IPL batting is harder to bowl at, ~20% discount)
  - All-rounders: X = 0.90 (blend)

For each NA player:
  1. If we have pre-IPL T20 stats -> compute form score * X
  2. If completely unknown -> assign replacement-level score (35 for bat, 30 for bowl)
"""

import pandas as pd
import numpy as np
import os

DATA = os.path.join(os.path.dirname(__file__), "data")

# ── X-Factor constants ────────────────────────────────────────────────────
X_BAT = 1.00     # Batters: pre-IPL form -> IPL conversion
X_BOWL = 0.80    # Bowlers: pre-IPL form -> IPL conversion (20% harder)
X_AR = 0.90      # All-rounders: blend

# Replacement-level scores for truly unknown players (no T20 data findable)
REPLACEMENT_BAT = 35.0   # Bottom-tier IPL batter
REPLACEMENT_BOWL = 30.0  # Bottom-tier IPL bowler

# ── Pre-IPL T20 stats for NA players ──────────────────────────────────────
# Format: "full_name": (bat_avg, bat_sr, bowl_econ, bowl_wkt_per_inn, source_note)
# Use None for unavailable stats

PRE_IPL_STATS = {
    # ── CSK ──
    "Kartik Sharma": (30.36, 162.93, None, None, "SMAT 2025: 334 runs, 12 T20s, avg 30.36, sr 162.93"),
    "Sarfaraz Khan": (65.80, 135.0, None, None, "SMAT 2025: 329 runs avg 65.80; career T20 avg 25.33 sr 135.45"),
    "Ramakrishna Ghosh": (None, None, None, None, "No senior T20 data available"),
    "Prashant Veer": (15.0, 130.0, 7.0, 1.0, "Limited domestic T20 data; leg-spinner AR from SMAT"),
    "Aman Khan": (20.0, 140.0, 8.0, 0.8, "Limited domestic data; Madhya Pradesh all-rounder"),
    "Zak Foulkes": (10.0, 100.0, 8.5, 1.0, "Australian domestic; limited BBL exposure"),
    "Gurjapneet Singh": (None, None, 7.5, 1.2, "Punjab left-arm pacer; SMAT performer"),

    # ── KKR ──
    "Anukul Roy": (15.0, 120.0, 7.8, 0.9, "Bihar all-rounder; limited domestic T20s"),
    "Tejasvi Singh": (20.0, 130.0, None, None, "Young Indian batter; minimal senior data"),
    "Sarthak Ranjan": (15.0, 110.0, None, None, "Young Delhi batter; minimal data"),
    "Daksh Kamra": (15.0, 110.0, None, None, "Young batter; minimal data"),
    "Blessing Muzarabani": (5.0, 80.0, 7.8, 1.1, "Zimbabwe intl pacer; T20I econ ~7.8, SR ~20"),

    # ── MI ──
    "Raghu Sharma": (None, None, 8.0, 0.8, "Domestic left-arm spinner; limited data"),
    "Allah Ghazanfar": (5.0, 80.0, 7.2, 1.3, "Afghanistan U19; promising mystery spinner; ILT20 performer"),
    "Danish Malewar": (25.0, 145.0, None, None, "Maharashtra domestic; aggressive opener"),
    "Mohammad Izhar": (None, None, 8.5, 0.8, "Domestic pacer; limited data"),
    "Atharva Ankolekar": (15.0, 120.0, 7.5, 0.9, "Mumbai all-rounder; left-arm spin; U19 WC 2020"),

    # ── RCB ──
    "Jitesh Sharma": (25.0, 155.0, None, None, "India T20I; SMAT avg ~25 sr ~155; wk-bat power hitter"),
    "Abhinandan Singh": (None, None, 8.5, 0.7, "Young domestic pacer; minimal data"),
    "Jacob Duffy": (5.0, 60.0, 8.0, 1.0, "NZ domestic pacer; T20 econ ~8.0"),
    "Satvik Deswal": (20.0, 130.0, None, None, "Haryana domestic batter; limited T20 data"),
    "Mangesh Yadav": (None, None, 9.0, 0.8, "Domestic pacer; express pace ~150kph"),
    "Jordan Cox": (28.0, 135.0, None, None, "England domestic; Kent T20 avg ~28 sr ~135"),
    "Vicky Ostwal": (5.0, 80.0, 7.5, 1.1, "India U19 left-arm spinner; domestic T20 performer"),
    "Vihaan Malhotra": (18.0, 120.0, None, None, "Young domestic batter; limited data"),
    "Kanishk Chouhan": (None, None, 8.0, 0.9, "Young domestic spinner; limited data"),

    # ── GT ──
    "Nishant Sindhu": (18.0, 125.0, 8.0, 0.7, "Haryana all-rounder; SMAT performer"),
    "Jayant Yadav": (15.0, 110.0, 7.5, 0.9, "India intl; off-spinner; T20I econ ~7.5"),
    "Ashok Sharma": (20.0, 140.0, 8.0, 0.8, "Domestic AR; limited T20 data"),
    "Prithvi Raj Yarra": (15.0, 120.0, None, None, "Young batter; minimal senior data"),

    # ── RR ──
    "Vaibhav Suryavanshi": (25.0, 180.0, None, None, "U19 prodigy; SMAT sr 216; IPL 2025: avg 36 sr 206.6"),
    "Shubham Dubey": (22.0, 140.0, None, None, "MP domestic batter; SMAT performer"),
    "Lhuan-Dre Pretorius": (15.0, 140.0, 8.5, 1.0, "SA domestic all-rounder; CSA T20 Challenge"),
    "Kwena Maphaka": (5.0, 60.0, 8.0, 1.2, "SA U19 left-arm fast; impressive pace ~145kph"),
    "Sushant Mishra": (None, None, 8.0, 0.9, "Domestic left-arm pacer; limited data"),
    "Yudhvir Singh Charak": (20.0, 130.0, 8.5, 0.7, "J&K all-rounder; domestic T20 performer"),
    "Yash Raj Punja": (None, None, 8.0, 0.8, "Young domestic spinner; limited data"),
    "Vignesh Puthur": (10.0, 100.0, 8.0, 1.0, "Domestic medium pacer; limited data"),
    "Ravi Singh": (18.0, 130.0, 8.5, 0.6, "Domestic all-rounder; limited data"),
    "Aman Rao": (18.0, 125.0, None, None, "Young domestic batter; limited data"),
    "Brijesh Sharma": (18.0, 120.0, None, None, "Young domestic batter; limited data"),
    "Kuldeep Sen": (5.0, 60.0, 9.0, 1.0, "India intl pacer; T20I econ ~9.0; injury prone"),

    # ── SRH ──
    "Nitish Kumar Reddy": (30.0, 145.0, 8.5, 0.8, "India intl; Test century in AUS; T20 batting power; medium pace"),
    "Brydon Carse": (15.0, 130.0, 8.0, 1.1, "England intl pacer; T20I/domestic; good pace ~145kph"),
    "R Smaran": (22.0, 140.0, None, None, "Karnataka domestic batter; SMAT performer"),
    "Shivang Kumar": (None, None, 8.5, 0.8, "Domestic pacer; limited data"),
    "Salil Arora": (None, None, 8.5, 0.7, "Domestic pacer; limited data"),
    "Sakib Hussain": (None, None, 8.5, 0.7, "Young domestic pacer; limited data"),
    "Onkar Tarmale": (None, None, 8.5, 0.7, "Maharashtra domestic; limited data"),
    "Amit Kumar": (20.0, 135.0, None, None, "Domestic batter; limited data"),
    "Praful Hinge": (18.0, 125.0, None, None, "Domestic batter; limited data"),
    "Krains Fuletra": (18.0, 120.0, None, None, "Young domestic batter; limited data"),
    "Jack Edwards": (22.0, 135.0, 8.5, 0.7, "Australian domestic all-rounder; BBL fringe"),

    # ── DC ──
    "Karun Nair": (35.0, 140.0, None, None, "India intl; SMAT 2024 avg ~50 sr ~140; VHT star"),
    "Madhav Tiwari": (18.0, 130.0, None, None, "MP domestic batter; young talent"),
    "Tripurana Vijay": (20.0, 135.0, None, None, "Domestic batter; limited data"),
    "Vipraj Nigam": (18.0, 180.0, 7.45, 1.82, "UP T20 League: 20wkt/11inn econ 7.45; leg-spin AR"),
    "Ajay Mandal": (None, None, 8.0, 0.8, "Domestic bowler; limited data"),
    "Ben Duckett": (30.0, 145.0, None, None, "England intl; T20I avg ~25 sr ~140; aggressive opener"),
    "Auqib Nabi": (5.0, 60.0, 7.5, 1.2, "J&K/India domestic pacer; impressive SMAT 2024"),
    "Pathum Nissanka": (32.0, 140.0, None, None, "SL intl; T20I avg ~28 sr ~135; quality opener"),
    "Sahil Parakh": (None, None, 8.5, 0.7, "Young domestic bowler; limited data"),

    # ── LSG ──
    "Matthew Breetzke": (25.0, 130.0, None, None, "SA domestic; CSA T20 Challenge performer"),
    "Himmat Singh": (22.0, 135.0, None, None, "Delhi domestic; SMAT avg ~22 sr ~135"),
    "Arshin Kulkarni": (25.0, 145.0, 8.0, 0.6, "Mumbai youth star; aggressive bat + off-spin"),
    "Mayank Yadav": (5.0, 60.0, 8.0, 1.3, "India intl; express pace 155kph; IPL 2024: 7wkt econ 6.7 in 4 games for LSG"),
    "Wanindu Hasaranga": (15.0, 140.0, 6.8, 1.5, "SL intl; T20I: 90+ wkt; econ ~6.8; best T20 spinner in world"),
    "Mukul Choudhary": (None, None, 8.5, 0.8, "Domestic pacer; limited data"),
    "Naman Tiwari": (18.0, 130.0, 8.0, 0.7, "UP domestic all-rounder; limited data"),
    "Akshat Raghuwanshi": (20.0, 130.0, None, None, "MP domestic batter; limited data"),
    "Josh Inglis": (27.0, 164.5, None, None, "AUS intl; T20I avg 27 sr 164.5; SA20+BBL star"),

    # ── PBKS ──
    "Prabhsimran Singh": (25.0, 145.0, None, None, "Punjab domestic; SMAT avg ~25 sr ~145; aggressive wk-bat"),
    "Nehal Wadhera": (28.0, 140.0, None, None, "MP domestic; SMAT 2024 strong performer; power hitter"),
    "Pyala Avinash": (18.0, 125.0, None, None, "Young domestic batter; limited data"),
    "Harnoor Pannu": (22.0, 130.0, None, None, "Punjab domestic batter; U19 India captain"),
    "Mitchell Owen": (25.0, 155.0, 8.0, 0.5, "Tasmania BBL; explosive opener; BBL sr ~155"),
    "Xavier Bartlett": (10.0, 80.0, 7.8, 1.2, "AUS intl pacer; T20I econ ~7.8; Brisbane Heat BBL"),
    "Vyshak Vijaykumar": (None, None, 8.0, 1.0, "Karnataka domestic pacer; SMAT/RR 2024 performer"),
    "Cooper Connolly": (38.46, 136.72, 7.3, 0.75, "AUS intl; BBL 2025 avg 50.14 sr 131.46; 577 T20 runs; left-arm spin"),
    "Ben Dwarshuis": (10.0, 100.0, 7.5, 1.1, "AUS domestic; BBL left-arm pacer; consistent performer"),
    "Pravin Dubey": (10.0, 100.0, 7.8, 0.9, "Domestic leg-spinner; limited T20 data"),
    "Vishal Nishad": (None, None, 8.5, 0.8, "Domestic bowler; limited data"),
}


def bat_score_raw(avg, sr):
    """Batting score on 0-100+ scale."""
    return ((avg / 40) * 0.5 + (sr / 150) * 0.5) * 100

def bowl_score_raw(econ, wkt_per_inn):
    """Bowling score on 0-100+ scale."""
    econ_s = max(0, (10 - econ) / 6)
    return (econ_s * 0.5 + min(wkt_per_inn / 2, 1) * 0.5) * 100

def compute_estimated_score(full_name, role, stats):
    """Compute estimated IPL score from pre-IPL stats * X-factor."""
    bat_avg, bat_sr, bowl_econ, bowl_wpi, note = stats

    has_bat = bat_avg is not None and bat_sr is not None
    has_bowl = bowl_econ is not None and bowl_wpi is not None

    # Compute raw pre-IPL scores
    pre_bat = bat_score_raw(bat_avg, bat_sr) if has_bat else None
    pre_bowl = bowl_score_raw(bowl_econ, bowl_wpi) if has_bowl else None

    # Apply X-factor based on role
    if role in ("batter", "wk-batter"):
        if pre_bat:
            est_bat = pre_bat * X_BAT
            est_bowl = pre_bowl * X_BOWL if pre_bowl else None
            return round(est_bat * 0.8 + est_bowl * 0.2, 1) if est_bowl else round(est_bat, 1)
        return REPLACEMENT_BAT
    elif role == "bowler":
        if pre_bowl:
            est_bowl = pre_bowl * X_BOWL
            est_bat = pre_bat * X_BAT if pre_bat else None
            return round(est_bat * 0.2 + est_bowl * 0.8, 1) if est_bat else round(est_bowl, 1)
        return REPLACEMENT_BOWL
    else:  # all-rounder
        est_bat = pre_bat * X_BAT if pre_bat else None
        est_bowl = pre_bowl * X_BOWL if pre_bowl else None
        if est_bat and est_bowl:
            return round(est_bat * 0.5 + est_bowl * 0.5, 1)
        elif est_bat:
            return round(est_bat * X_AR, 1)
        elif est_bowl:
            return round(est_bowl * X_AR, 1)
        return round((REPLACEMENT_BAT + REPLACEMENT_BOWL) / 2, 1)


# ── Load and update player database ──────────────────────────────────────
print("Loading player database...")
db = pd.read_csv(os.path.join(DATA, "player_database_2026.csv"))

# Ensure numeric columns
for col in ["bat_score", "bowl_score", "overall_score"]:
    db[col] = pd.to_numeric(db[col], errors="coerce")

updated_count = 0
results = []

for idx, row in db.iterrows():
    if pd.notna(row["overall_score"]):
        continue  # Already has IPL data, skip

    name = row["full_name"]
    role = row["role"]

    if name in PRE_IPL_STATS:
        stats = PRE_IPL_STATS[name]
        bat_avg, bat_sr, bowl_econ, bowl_wpi, note = stats

        # Compute component scores
        has_bat = bat_avg is not None and bat_sr is not None
        has_bowl = bowl_econ is not None and bowl_wpi is not None

        pre_bat = bat_score_raw(bat_avg, bat_sr) * X_BAT if has_bat else None
        pre_bowl = bowl_score_raw(bowl_econ, bowl_wpi) * X_BOWL if has_bowl else None

        estimated = compute_estimated_score(name, role, stats)

        db.at[idx, "bat_score"] = round(pre_bat, 1) if pre_bat else pd.NA
        db.at[idx, "bowl_score"] = round(pre_bowl, 1) if pre_bowl else pd.NA
        db.at[idx, "overall_score"] = estimated

        results.append({
            "name": name, "team": row["team_2026"], "role": role,
            "pre_bat": f"{pre_bat:.1f}" if pre_bat else "NA",
            "pre_bowl": f"{pre_bowl:.1f}" if pre_bowl else "NA",
            "estimated": f"{estimated:.1f}",
            "source": "pre-IPL stats * X",
            "note": note,
        })
    else:
        # Truly unknown - assign replacement level
        if role in ("batter", "wk-batter"):
            db.at[idx, "overall_score"] = REPLACEMENT_BAT
            db.at[idx, "bat_score"] = REPLACEMENT_BAT
        elif role == "bowler":
            db.at[idx, "overall_score"] = REPLACEMENT_BOWL
            db.at[idx, "bowl_score"] = REPLACEMENT_BOWL
        else:
            db.at[idx, "overall_score"] = round((REPLACEMENT_BAT + REPLACEMENT_BOWL) / 2, 1)
            db.at[idx, "bat_score"] = REPLACEMENT_BAT
            db.at[idx, "bowl_score"] = REPLACEMENT_BOWL

        results.append({
            "name": name, "team": row["team_2026"], "role": role,
            "pre_bat": "NA", "pre_bowl": "NA",
            "estimated": f"{db.at[idx, 'overall_score']:.1f}",
            "source": "replacement-level",
            "note": "No T20 data available",
        })

    updated_count += 1

# Save updated database
db.to_csv(os.path.join(DATA, "player_database_2026.csv"), index=False)
print(f"\nUpdated {updated_count} NA players with estimated scores.")

# ── Print results ─────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"{'Player':<30s} {'Team':<25s} {'Role':<14s} {'Bat':>6s} {'Bowl':>6s} {'Score':>6s} {'Source':<20s}")
print(f"{'-'*100}")
for r in sorted(results, key=lambda x: -float(x["estimated"])):
    print(f"{r['name']:<30s} {r['team']:<25s} {r['role']:<14s} {r['pre_bat']:>6s} {r['pre_bowl']:>6s} {r['estimated']:>6s} {r['source']:<20s}")

# Verify no NAs remain
remaining_na = db["overall_score"].isna().sum()
print(f"\n{'='*100}")
print(f"Remaining NA players: {remaining_na}")
print(f"Total players with scores: {db['overall_score'].notna().sum()} / {len(db)}")

# Summary stats
print(f"\nX-Factor Applied:")
print(f"  Batters:      X = {X_BAT} (pre-IPL form -> IPL conversion)")
print(f"  Bowlers:      X = {X_BOWL} (20% IPL difficulty discount)")
print(f"  All-rounders: X = {X_AR}")
print(f"  Replacement level: bat={REPLACEMENT_BAT}, bowl={REPLACEMENT_BOWL}")
