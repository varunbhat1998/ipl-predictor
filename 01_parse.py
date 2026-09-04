"""
01_parse.py  —  Parse Cricsheet IPL JSON files → matches.csv + deliveries.csv

Handles two layouts:
  A) Folder of individual match JSONs  (standard Cricsheet ZIP)
  B) One big JSON file that is a list of matches
  C) One big JSON file that is a single match (like your sample)

Usage:
  python 01_parse.py --data_dir ./raw_json
  python 01_parse.py --data_dir ./raw_json --out_dir ./data
"""

import json, glob, os, argparse
from pathlib import Path
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_one_match(match: dict, file_id: str = "") -> tuple[dict, list]:
    """Returns (match_row dict, list of delivery rows)"""
    info = match.get("info", {})
    teams = info.get("teams", [])
    toss  = info.get("toss", {})
    outcome = info.get("outcome", {})

    # Season normalise: "2007/08" → "2008", "2023" → "2023"
    raw_season = str(info.get("season", ""))
    if "/" in raw_season:
        season = raw_season.split("/")[1]          # "2007/08" → "08" — pad below
        season = "20" + season if len(season) == 2 else season
    else:
        season = raw_season

    winner = outcome.get("winner")
    if "result" in outcome and outcome["result"] in ("tie", "no result"):
        winner = None

    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    toss_winner   = toss.get("winner")
    toss_decision = toss.get("decision")          # "bat" or "field"
    toss_won      = int(toss_winner == winner) if (toss_winner and winner) else None

    match_row = {
        "file_id":          file_id,
        "match_number":     info.get("event", {}).get("match_number"),
        "season":           season,
        "date":             info.get("dates", [None])[0],
        "venue":            info.get("venue"),
        "city":             info.get("city"),
        "team1":            team1,
        "team2":            team2,
        "toss_winner":      toss_winner,
        "toss_decision":    toss_decision,
        "toss_winner_won":  toss_won,
        "winner":           winner,
        "win_by_runs":      outcome.get("by", {}).get("runs"),
        "win_by_wickets":   outcome.get("by", {}).get("wickets"),
        "player_of_match":  (info.get("player_of_match") or [None])[0],
        "team1_players":    "|".join(info.get("players", {}).get(team1, [])),
        "team2_players":    "|".join(info.get("players", {}).get(team2, [])),
    }

    # Innings summary
    innings = match.get("innings", [])
    for idx in range(2):
        pfx = f"inn{idx+1}"
        if idx < len(innings):
            inn = innings[idx]
            deliveries_flat = [
                d for ov in inn.get("overs", [])
                for d in ov.get("deliveries", [])
            ]
            total_runs = sum(d.get("runs", {}).get("total", 0) for d in deliveries_flat)
            total_wkts = sum(len(d.get("wickets", [])) for d in deliveries_flat)
            match_row[f"{pfx}_team"]    = inn.get("team")
            match_row[f"{pfx}_runs"]    = total_runs
            match_row[f"{pfx}_wickets"] = total_wkts
        else:
            match_row[f"{pfx}_team"] = match_row[f"{pfx}_runs"] = match_row[f"{pfx}_wickets"] = None

    # ── Deliveries ────────────────────────────────────────────────────────────
    delivery_rows = []
    for inn_idx, inn in enumerate(innings[:2]):
        batting_team = inn.get("team")
        bowling_team = next((t for t in teams if t != batting_team), None)
        cum_runs = 0
        cum_wkts = 0
        legal_balls = 0

        for ov in inn.get("overs", []):
            over_num = ov.get("over", 0)
            for ball_idx, d in enumerate(ov.get("deliveries", [])):
                extras = d.get("extras", {})
                is_wide  = int(bool(extras.get("wides")))
                is_noball = int(bool(extras.get("noballs")))
                is_legal = int(not is_wide and not is_noball)
                runs_total = d.get("runs", {}).get("total", 0)
                wickets    = d.get("wickets", [])

                delivery_rows.append({
                    "file_id":        file_id,
                    "season":         season,
                    "date":           info.get("dates", [None])[0],
                    "venue":          info.get("venue"),
                    "innings":        inn_idx + 1,
                    "over":           over_num,
                    "ball_in_over":   ball_idx + 1,
                    "legal_ball_num": legal_balls + is_legal,
                    "batting_team":   batting_team,
                    "bowling_team":   bowling_team,
                    "batter":         d.get("batter"),
                    "bowler":         d.get("bowler"),
                    "non_striker":    d.get("non_striker"),
                    "runs_batter":    d.get("runs", {}).get("batter", 0),
                    "runs_extras":    d.get("runs", {}).get("extras", 0),
                    "runs_total":     runs_total,
                    "is_wide":        is_wide,
                    "is_noball":      is_noball,
                    "is_wicket":      int(len(wickets) > 0),
                    "dismissal_kind": wickets[0].get("kind") if wickets else None,
                    "player_out":     wickets[0].get("player_out") if wickets else None,
                    "cum_runs":       cum_runs,
                    "cum_wickets":    cum_wkts,
                    "winner":         winner,
                })
                cum_runs  += runs_total
                cum_wkts  += len(wickets)
                legal_balls += is_legal

    return match_row, delivery_rows


def load_and_parse(data_dir: str):
    json_files = sorted(glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    print(f"Found {len(json_files)} JSON file(s) in {data_dir}")

    all_match_rows = []
    all_delivery_rows = []
    skipped = 0

    for fpath in json_files:
        fid = Path(fpath).stem
        with open(fpath) as f:
            raw = json.load(f)

        # Could be list of matches or single match
        matches = raw if isinstance(raw, list) else [raw]

        for match in matches:
            # Skip non-IPL or non-T20
            info = match.get("info", {})
            if info.get("match_type") != "T20":
                skipped += 1
                continue
            event = info.get("event", {}).get("name", "")
            if event and "premier league" not in event.lower() and "ipl" not in event.lower():
                # Allow through if no event name (some older files)
                if event:
                    skipped += 1
                    continue

            mrow, drows = parse_one_match(match, fid)
            all_match_rows.append(mrow)
            all_delivery_rows.extend(drows)

    print(f"Parsed {len(all_match_rows)} matches, {len(all_delivery_rows)} deliveries. Skipped {skipped}.")
    return pd.DataFrame(all_match_rows), pd.DataFrame(all_delivery_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", help="Folder containing JSON files")
    parser.add_argument("--out_dir",  default="./data")
    args = parser.parse_args()

    matches_df, deliveries_df = load_and_parse(args.data_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(f"{args.out_dir}/matches.csv", index=False)
    deliveries_df.to_csv(f"{args.out_dir}/deliveries.csv", index=False)
    print(f"Saved {args.out_dir}/matches.csv  ({len(matches_df)} rows)")
    print(f"Saved {args.out_dir}/deliveries.csv  ({len(deliveries_df)} rows)")
    print("\nSample match:")
    print(matches_df.head(2).to_string())
