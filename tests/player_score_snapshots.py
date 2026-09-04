"""
Snapshot test for player bat/bowl score sanity.

Run AFTER rebuilding player_database_2026.csv (07_build_player_db.py).

Usage:
    python tests/player_score_snapshots.py

Exit code 0 = all assertions passed.
Exit code 1 = one or more failures (fix scoring before retraining).

Score scale context (IPL T20):
  Bowl 65+  : elite (e.g. Bumrah at 7 RPO)
  Bowl 40-65: good spinner / effective seamer
  Bowl 25-40: average IPL bowler (8.5-9 RPO)
  Bowl  0-25: expensive / part-timer
  Bowl 0    : non-bowler (<60 career balls)

  Bat  90+  : elite (avg 40+, SR 150+)
  Bat  70-90: good batter
  Bat  50-70: useful
  Bat   5   : non-batter default
"""

import sys
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "player_database_2026.csv")


def load_scores():
    df = pd.read_csv(DB_PATH)
    df["bat_score"]  = pd.to_numeric(df["bat_score"],  errors="coerce")
    df["bowl_score"] = pd.to_numeric(df["bowl_score"], errors="coerce")
    scores = {}
    for _, row in df.iterrows():
        scores[row["data_name"]] = {
            "bat":  row["bat_score"],
            "bowl": row["bowl_score"],
        }
    return scores


# ── Expected score bands ────────────────────────────────────────────────────
# Format: data_name → (bat_lo, bat_hi, bowl_lo, bowl_hi)
# None means "don't assert this discipline"
EXPECTATIONS = {
    # ── Elite pace bowler ──
    "JJ Bumrah":          (None, 50,  65, 100),   # bat: bats some; bowl: elite

    # ── Specialist spinners — IPL economies 7-8 RPO ──
    "YS Chahal":          (None, 30,  35,  65),
    "AR Patel":           (40,   85,  35,  65),   # genuine all-rounder

    # ── Pace bowlers — IPL economies tend to be 8.5-9.5 RPO ──
    "Mohammed Siraj":     (None, 50,  20,  50),
    "T Natarajan":        (None, 15,  20,  50),

    # ── Elite batters: bowl must be low ──
    "V Kohli":            (75, 115, None,  25),   # occasional part-timer; bowl < 25
    "YBK Jaiswal":        (75, 115,    0,   5),   # never bowls
    "Shubman Gill":       (70, 115,    0,   5),   # never bowls
    "SA Yadav":           (80, 115,    0,   5),   # Suryakumar Yadav — never bowls

    # ── Known non-bowlers must score 0 ──
    "AM Rahane":          (None, None,   0,   5),
    "MS Dhoni":           (None, None,   0,   5),
    "RK Singh":           (None, None,   0,   5),  # Rinku Singh

    # ── All-rounders: both disciplines meaningful ──
    "RA Jadeja":          (50,  85,  35,  65),
    "SP Narine":          (55,  90,  45,  75),
    "KH Pandya":          (40,  80,  35,  65),    # Krunal Pandya — more reliable bowler than Hardik
    "HH Pandya":          (50,  85,  10,  40),    # Hardik — recent IPL bowling expensive

    # ── Bowlers who bat at tail — bat must be low ──
    "Mohammed Shami":     (None, 50,  25,  55),   # limited data post-injury
    "B Kumar":            (None, 45,  25,  55),   # Bhuvneshwar Kumar
}

# ── Ordering checks ─────────────────────────────────────────────────────────
# Each tuple: (player_a, player_b, discipline) asserts score[a] > score[b]
ORDERINGS = [
    # Best bowler outranks non-bowlers on bowl
    ("JJ Bumrah",      "V Kohli",       "bowl"),
    ("JJ Bumrah",      "AM Rahane",     "bowl"),
    ("YS Chahal",      "YBK Jaiswal",   "bowl"),
    ("Mohammed Siraj", "Shubman Gill",  "bowl"),

    # Best batters outrank tail-enders on bat
    ("V Kohli",        "JJ Bumrah",     "bat"),
    ("YBK Jaiswal",    "Mohammed Siraj","bat"),
    ("Shubman Gill",   "T Natarajan",   "bat"),

    # Elite bowler outranks average on bowl
    ("JJ Bumrah",      "Mohammed Siraj","bowl"),
    ("JJ Bumrah",      "T Natarajan",   "bowl"),

    # All-rounder: non-zero on both disciplines
    ("RA Jadeja",      "AM Rahane",     "bowl"),   # all-rounder > pure batter on bowl
    ("RA Jadeja",      "JJ Bumrah",     "bat"),    # all-rounder > tail bowler on bat
]


def run():
    if not os.path.exists(DB_PATH):
        print(f"[FAIL] Database not found: {DB_PATH}")
        print("       Run 07_build_player_db.py first.")
        sys.exit(1)

    scores = load_scores()
    failures = []

    # Band checks
    for player, (bat_lo, bat_hi, bowl_lo, bowl_hi) in EXPECTATIONS.items():
        if player not in scores:
            failures.append(f"  MISSING player: {player}  (check data_name in player_database_2026.csv)")
            continue
        s = scores[player]
        bat, bowl = s["bat"], s["bowl"]

        if bat_lo is not None or bat_hi is not None:
            lo = bat_lo if bat_lo is not None else 0
            hi = bat_hi if bat_hi is not None else 200
            if pd.isna(bat) or not (lo <= bat <= hi):
                failures.append(
                    f"  BAT FAIL  {player:35s}: got {bat:.1f if not pd.isna(bat) else 'NaN':>6}, expected [{lo}, {hi}]"
                )

        if bowl_lo is not None or bowl_hi is not None:
            lo = bowl_lo if bowl_lo is not None else 0
            hi = bowl_hi if bowl_hi is not None else 200
            if pd.isna(bowl) or not (lo <= bowl <= hi):
                failures.append(
                    f"  BOWL FAIL {player:35s}: got {bowl:.1f if not pd.isna(bowl) else 'NaN':>6}, expected [{lo}, {hi}]"
                )

    # Ordering checks
    for player_a, player_b, disc in ORDERINGS:
        sa = scores.get(player_a, {}).get(disc)
        sb = scores.get(player_b, {}).get(disc)
        if sa is None or sb is None or pd.isna(sa) or pd.isna(sb):
            failures.append(
                f"  ORDER SKIP {player_a} vs {player_b} ({disc}): player missing"
            )
        elif sa <= sb:
            failures.append(
                f"  ORDER FAIL {player_a} ({sa:.1f}) should > {player_b} ({sb:.1f}) on {disc}"
            )

    # Print summary
    print(f"\n{'='*65}")
    print(f"Player score snapshot test")
    print(f"{'='*65}")
    print(f"Players checked : {len(EXPECTATIONS)}")
    print(f"Ordering checks : {len(ORDERINGS)}")
    print()

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f)
        print(f"\n[FAIL] {len(failures)} check(s) failed.")
        sys.exit(1)
    else:
        print("Sample scores:")
        sample = [
            "JJ Bumrah", "YS Chahal", "RA Jadeja", "SP Narine",
            "V Kohli", "YBK Jaiswal", "AM Rahane", "HH Pandya",
            "Mohammed Siraj", "T Natarajan",
        ]
        for player in sample:
            if player in scores:
                s = scores[player]
                print(f"  {player:35s}  bat={s['bat']:6.1f}  bowl={s['bowl']:6.1f}")
        print(f"\n[PASS] All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    run()
