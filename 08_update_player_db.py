"""
08_update_player_db.py
Update player database after each match.
Called automatically by match_bot.py after auto-retrain.

Usage:
  python 08_update_player_db.py

Rebuilds all player stats from the latest deliveries.csv data,
preserving the 2026 roster structure.
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def update_player_db():
    """Re-run the full player database builder with latest data."""
    builder = os.path.join(SCRIPT_DIR, "07_build_player_db.py")
    print("[update_player_db] Rebuilding player database with latest data...")
    result = subprocess.run(
        [sys.executable, builder],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    if result.returncode == 0:
        print("[update_player_db] Player database updated successfully.")
        # Print last few lines of output (summary)
        lines = result.stdout.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")
    else:
        print(f"[update_player_db] ERROR: {result.stderr[-500:]}")
        return False

    # Step 2: Apply X-factor estimates for players still without IPL data
    x_factor = os.path.join(SCRIPT_DIR, "07b_apply_x_factor.py")
    print("[update_player_db] Applying X-factor for estimated scores...")
    result2 = subprocess.run(
        [sys.executable, x_factor],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
    )
    if result2.returncode == 0:
        print("[update_player_db] X-factor estimates applied.")
        lines2 = result2.stdout.strip().split("\n")
        for line in lines2[-5:]:
            print(f"  {line}")
    else:
        print(f"[update_player_db] X-factor WARNING: {result2.stderr[-300:]}")

    return True


if __name__ == "__main__":
    success = update_player_db()
    sys.exit(0 if success else 1)
