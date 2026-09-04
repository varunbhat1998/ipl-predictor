"""
Test that the XI Analysis section in match_bot.py uses consistent
team1/team2 labels (e1/t1s) rather than bat_first/bat_second labels
(eb1/b1s) when pairing labels with player arrays.

The bug: when bat_first != team1, using eb1 (bat_first emoji) with
t1_xi (team1 players) causes the wrong team name to appear above
the wrong player list.

This test reads match_bot.py source and verifies the correct variable
names are used in the XI section. It does NOT require the bot to run.

Usage:
    python tests/xi_label_invariant.py
"""

import sys
import os
import re

BOT_PATH = os.path.join(os.path.dirname(__file__), "..", "match_bot.py")


def run():
    if not os.path.exists(BOT_PATH):
        print(f"[FAIL] match_bot.py not found at {BOT_PATH}")
        sys.exit(1)

    with open(BOT_PATH, "r", encoding="utf-8") as f:
        source = f.read()

    failures = []

    # ── Check 1: The XI _xi_section calls must use e1/t1s not eb1/b1s ────────
    # Find the region around "_xi_section" calls in the toss message block.
    # We look for the specific pattern that was the bug: eb1/b1s used with t1_xi.
    bad_patterns = [
        (r"_xi_section\(eb1,\s*b1s,\s*t1_xi",
         "XI section for t1_xi still uses eb1/b1s (bat_first labels) — should use e1/t1s (team1 labels)"),
        (r"_xi_section\(eb2,\s*b2s,\s*t2_xi",
         "XI section for t2_xi still uses eb2/b2s (bat_second labels) — should use e2/t2s (team2 labels)"),
    ]

    for pattern, description in bad_patterns:
        if re.search(pattern, source):
            failures.append(f"  BAD PATTERN: {description}")

    # ── Check 2: The correct pattern must be present ──────────────────────────
    good_patterns = [
        (r"_xi_section\(e1,\s*t1s,\s*t1_xi",
         "XI section for t1_xi should use e1/t1s (team1 labels)"),
        (r"_xi_section\(e2,\s*t2s,\s*t2_xi",
         "XI section for t2_xi should use e2/t2s (team2 labels)"),
    ]

    for pattern, description in good_patterns:
        if not re.search(pattern, source):
            failures.append(f"  MISSING: {description}")

    # ── Check 3: Role label logic must stay correct ───────────────────────────
    # "Batting" if bat_first == team1 else "Chasing" is the correct formula
    role_pattern = r'"Batting"\s+if\s+bat_first\s*==\s*team1\s+else\s+"Chasing"'
    count = len(re.findall(role_pattern, source))
    if count < 1:
        failures.append(
            "  MISSING: role label 'Batting if bat_first == team1 else Chasing' "
            "not found in XI section"
        )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("XI label invariant test")
    print(f"{'='*60}")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f)
        print(f"\n[FAIL] team label swap bug is back — check match_bot.py")
        sys.exit(1)
    else:
        print("\n[PASS] XI section uses e1/t1s and e2/t2s labels consistently.")
        print("       No bat_first/bat_second label mixing with team1/team2 player arrays.")
        sys.exit(0)


if __name__ == "__main__":
    run()
