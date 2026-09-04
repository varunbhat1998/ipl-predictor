"""
04_api.py  —  IPL Win Predictor API
Endpoints:
  GET  /health
  GET  /teams
  GET  /venues
  POST /predict/prematch
  POST /predict/live
  POST /reload-models          <- hot-reload after retraining
  POST /update-excel/prematch
  POST /update-excel/live
  POST /update-excel/result
  GET  /excel/status
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pickle, numpy as np, pandas as pd, os, sys, threading

app = FastAPI(title="IPL Win Predictor API", version="4.0")

sys.path.insert(0, os.path.dirname(__file__))
from excel_writer import router as excel_router
app.include_router(excel_router)
from name_map import DATA_TO_FULL
from model_classes import EnsemblePreMatchModel  # needed for unpickling ensemble bundles

# ── Model loader (called at startup and on /reload-models) ─────────────────
_models = {}
_lock   = threading.Lock()
GLOBAL_AVG_FIRST_INNINGS = 160.0  # overwritten at startup from venue data

def load_models():
    with _lock:
        with open("models/prematch_model.pkl", "rb") as f:
            _models["pre"] = pickle.load(f)
        with open("models/live_model.pkl", "rb") as f:
            _models["live"] = pickle.load(f)
        with open("models/inn1_live_model.pkl", "rb") as f:
            _models["inn1"] = pickle.load(f)
        pt_path = "models/posttoss_model.pkl"
        if os.path.exists(pt_path):
            with open(pt_path, "rb") as f:
                _models["posttoss"] = pickle.load(f)
            print("Models loaded (including posttoss).")
        else:
            print("Models loaded (posttoss not found).")
        ul_path = "models/unified_live_model.pkl"
        if os.path.exists(ul_path):
            with open(ul_path, "rb") as f:
                _models["unified"] = pickle.load(f)
            print("  Unified live model loaded.")
        else:
            print("  Unified live model not found (run 11_unified_live_model.py).")

load_models()

# ── Load lookup tables ─────────────────────────────────────────────────────
def load_lookups():
    mf = pd.read_csv("data/match_features.csv")
    mf = mf.sort_values("date")

    # Team ELO and form (latest values)
    elo_map, form_map, form3_map, form10_map, formw_map = {}, {}, {}, {}, {}
    for _, r in mf.iterrows():
        elo_map[r["team1"]]    = r["team1_elo"]
        elo_map[r["team2"]]    = r["team2_elo"]
        form_map[r["team1"]]   = r["team1_form"]
        form_map[r["team2"]]   = r["team2_form"]
        if "team1_form_3" in r:
            form3_map[r["team1"]]  = r["team1_form_3"]
            form3_map[r["team2"]]  = r["team2_form_3"]
        if "team1_form_10" in r:
            form10_map[r["team1"]] = r["team1_form_10"]
            form10_map[r["team2"]] = r["team2_form_10"]

    # Venue stats (latest values per venue)
    venue_stats = {}
    for _, r in mf.iterrows():
        v = r["venue"]
        venue_stats[v] = {
            "venue_toss_win_rate": r.get("venue_toss_win_rate", 0.5),
            "venue_bat_first_win_rate": r.get("venue_bat_first_win_rate", 0.5),
            "venue_avg_first_innings": r.get("venue_avg_first_innings", 160),
            "venue_chase_win_rate": r.get("venue_chase_win_rate", 0.5),
            "venue_matches": r.get("venue_matches", 10),
        }

    # Dynamic global average: used as fallback for unknown/new venues
    _all_avgs = [v["venue_avg_first_innings"] for v in venue_stats.values()
                 if v.get("venue_avg_first_innings") and v.get("venue_matches", 0) >= 3]
    global GLOBAL_AVG_FIRST_INNINGS
    GLOBAL_AVG_FIRST_INNINGS = round(sum(_all_avgs) / len(_all_avgs), 1) if _all_avgs else 160.0
    print(f"  [Startup] Global avg 1st innings score: {GLOBAL_AVG_FIRST_INNINGS} (from {len(_all_avgs)} venues)")

    # Venue-team win rates (latest values)
    venue_team_map = {}
    for _, r in mf.iterrows():
        v = r["venue"]
        if "team1_venue_win_rate" in r:
            venue_team_map[(v, r["team1"])] = r["team1_venue_win_rate"]
            venue_team_map[(v, r["team2"])] = r["team2_venue_win_rate"]

    # H2H
    h2h_map = {}
    for _, r in mf.iterrows():
        key = tuple(sorted([r["team1"], r["team2"]]))
        h2h_map[key] = r["h2h_win_rate_team1"]

    # Player stats (legacy — still used if 2026 DB not found)
    bat_df  = pd.read_csv("data/player_bat_stats.csv")
    bowl_df = pd.read_csv("data/player_bowl_stats.csv")
    latest_season = str(mf["season"].max())

    # Match count in latest season
    latest_match_num = int(mf[mf["season"].astype(str) == latest_season].shape[0])

    # ── 2026 player database (career + X-factor scores) ─────────────────────
    player_score_map = {}   # data_name -> {bat_score, bowl_score}
    player_db_path = "data/player_database_2026.csv"
    if os.path.exists(player_db_path):
        pdb = pd.read_csv(player_db_path)
        for _, r in pdb.iterrows():
            dn = r["data_name"]
            player_score_map[dn] = {
                "bat_score":  pd.to_numeric(r.get("bat_score"),  errors="coerce"),
                "bowl_score": pd.to_numeric(r.get("bowl_score"), errors="coerce"),
            }

    # ── Venue-specific player scores ─────────────────────────────────────────
    venue_player_map = {}   # (data_name, venue) -> {bat_score, bowl_score, bat_innings, bowl_innings}
    vp_path = "data/player_venue_scores.csv"
    if os.path.exists(vp_path):
        vpdb = pd.read_csv(vp_path)
        for _, r in vpdb.iterrows():
            key = (r["data_name"], r["venue"])
            vbs  = pd.to_numeric(r.get("venue_bat_score"),    errors="coerce")
            vws  = pd.to_numeric(r.get("venue_bowl_score"),   errors="coerce")
            vbi  = pd.to_numeric(r.get("venue_bat_innings"),  errors="coerce")
            vwi  = pd.to_numeric(r.get("venue_bowl_innings"), errors="coerce")
            if pd.notna(vbs) or pd.notna(vws):
                venue_player_map[key] = {
                    "bat_score": vbs, "bowl_score": vws,
                    "bat_innings": vbi, "bowl_innings": vwi,
                }

    # ── Team-level fallback strengths (when no XI provided) ──────────────────
    team_strength_map = {}  # (venue, team) -> {bat_strength, bowl_strength}
    tp_path = "data/team_profiles_2026.csv"
    if os.path.exists(tp_path):
        tpdb = pd.read_csv(tp_path)
        for _, r in tpdb.iterrows():
            key = (r["venue"], r["team"])
            team_strength_map[key] = {
                "bat_strength":  pd.to_numeric(r.get("bat_strength"),  errors="coerce"),
                "bowl_strength": pd.to_numeric(r.get("bowl_strength"), errors="coerce"),
            }
        # Also store overall (no venue) strengths under key (None, team)
        if "bat_strength" in tpdb.columns:
            for team, grp in tpdb.groupby("team"):
                team_strength_map[(None, team)] = {
                    "bat_strength":  grp["bat_strength"].mean(),
                    "bowl_strength": grp["bowl_strength"].mean(),
                }

    # ── Per-team chase win rate (latest value per team) ──────────────────────
    chase_wr_map = {}   # team -> chase win rate
    if "team1_chase_wr" in mf.columns:
        for _, r in mf.iterrows():
            chase_wr_map[r["team1"]] = float(r["team1_chase_wr"])
            chase_wr_map[r["team2"]] = float(r["team2_chase_wr"])

    # ── Toss decision frequency per venue (for pre-toss blending) ─────────
    venue_toss_field_pct = {}   # venue -> P(toss winner chooses field)
    for v, grp in mf.groupby("venue"):
        total = len(grp)
        field_cnt = len(grp[grp["toss_decision"].isin(["field", "bowl"])])
        venue_toss_field_pct[v] = field_cnt / total if total > 0 else 0.65
    _avg_field = sum(venue_toss_field_pct.values()) / len(venue_toss_field_pct) if venue_toss_field_pct else 0.65
    print(f"  [Startup] Toss-field-pct: {len(venue_toss_field_pct)} venues (avg {_avg_field:.0%} choose field)")

    # ── Last-known XI per team (from matches.csv) ─────────────────────────
    last_xi_map = {}   # team -> list[str] (data_name format)
    matches_path = "data/matches.csv"
    if os.path.exists(matches_path):
        matches_df = pd.read_csv(matches_path)
        matches_df = matches_df.sort_values("date")
        for _, r in matches_df.iterrows():
            if pd.notna(r.get("team1_players")):
                last_xi_map[r["team1"]] = str(r["team1_players"]).split("|")
            if pd.notna(r.get("team2_players")):
                last_xi_map[r["team2"]] = str(r["team2_players"]).split("|")
        print(f"  [Startup] Last-known XI: {len(last_xi_map)} teams loaded")

    # ── Squad fallback: top-11 by overall_score from player_database_2026 ──
    squad_xi_map = {}   # team -> list[str] (data_name format, top 11)
    if os.path.exists(player_db_path):
        pdb2 = pd.read_csv(player_db_path)
        for team, grp in pdb2.groupby("team_2026"):
            top11 = grp.nlargest(11, "overall_score")["data_name"].tolist()
            squad_xi_map[team] = top11
        print(f"  [Startup] Squad XI fallback: {len(squad_xi_map)} teams")

    # ── PP/phase wicket win rates (for unified live model) ──────────────────
    pp_wkt_wr_map = {}  # (team, phase, wicket_bucket, role) -> win_rate
    pp_wr_path = "data/pp_wicket_win_rates.csv"
    if os.path.exists(pp_wr_path):
        pp_wr_df = pd.read_csv(pp_wr_path)
        for _, r in pp_wr_df.iterrows():
            key = (r["team"], r["phase"], int(r["wicket_bucket"]), r["role"])
            pp_wkt_wr_map[key] = r["win_rate"] if pd.notna(r["win_rate"]) else 0.0
        print(f"  [Startup] PP wicket win rates: {len(pp_wkt_wr_map)} entries")

    # ── Batter-Bowler H2H matchup lookup (for post-toss model) ─────────────
    # Use latest expanding-window H2H stats per (batter, bowler) pair.
    h2h_bvb_map = {}  # (batter, bowler) -> matchup_advantage (0-1, 0.5 = neutral)
    h2h_matrix_path = "data/h2h_matchup_matrix.csv"
    if os.path.exists(h2h_matrix_path):
        _h2h_df = pd.read_csv(h2h_matrix_path)
        # Keep last row per (batter, bowler) = most recent prior stats
        _h2h_latest = _h2h_df.sort_values("date").groupby(["batter", "bowler"]).last().reset_index()
        for _, r in _h2h_latest.iterrows():
            if pd.notna(r["matchup_adv_final"]):
                h2h_bvb_map[(r["batter"], r["bowler"])] = float(r["matchup_adv_final"])
        print(f"  [Startup] H2H batter-bowler pairs: {len(h2h_bvb_map)}")
    else:
        print("  [Startup] H2H matrix not found — run 13_h2h_matchups.py")

    return (elo_map, form_map, form3_map, form10_map, formw_map,
            venue_stats, venue_team_map, h2h_map,
            bat_df, bowl_df, latest_season, latest_match_num,
            player_score_map, venue_player_map, team_strength_map,
            chase_wr_map, venue_toss_field_pct, last_xi_map, squad_xi_map,
            pp_wkt_wr_map, h2h_bvb_map)

(elo_map, form_map, form3_map, form10_map, formw_map,
 venue_stats, venue_team_map, h2h_map,
 bat_df, bowl_df, latest_season, latest_match_num,
 player_score_map, venue_player_map, team_strength_map,
 chase_wr_map, venue_toss_field_pct, last_xi_map, squad_xi_map,
 pp_wkt_wr_map, h2h_bvb_map) = load_lookups()

# ── Player scoring helpers ─────────────────────────────────────────────────

def _blended_player_scores(data_name: str, venue: str):
    """Return (bat_score, bowl_score) blended career + venue for one player."""
    career = player_score_map.get(data_name, {})
    c_bat  = career.get("bat_score",  np.nan)
    c_bowl = career.get("bowl_score", np.nan)

    vk = (data_name, venue)
    if vk in venue_player_map:
        vs = venue_player_map[vk]
        v_bat, v_bowl = vs["bat_score"], vs["bowl_score"]
        v_bi,  v_wi   = vs["bat_innings"], vs["bowl_innings"]

        bat = (0.6 * c_bat + 0.4 * v_bat
               if pd.notna(v_bat) and pd.notna(v_bi) and v_bi >= 2 and pd.notna(c_bat)
               else (v_bat if pd.notna(v_bat) and pd.notna(v_bi) and v_bi >= 2 else c_bat))
        bowl = (0.6 * c_bowl + 0.4 * v_bowl
                if pd.notna(v_bowl) and pd.notna(v_wi) and v_wi >= 2 and pd.notna(c_bowl)
                else (v_bowl if pd.notna(v_bowl) and pd.notna(v_wi) and v_wi >= 2 else c_bowl))
    else:
        bat, bowl = c_bat, c_bowl

    return (float(bat) if pd.notna(bat) else np.nan,
            float(bowl) if pd.notna(bowl) else np.nan)


def get_xi_strengths(players: list[str], team: str, venue: str):
    """
    Compute bat_strength (top-6 mean) and bowl_strength (top-4 mean)
    from the announced playing XI using player_database_2026.

    Falls back to team_profiles_2026 averages when:
      - players list is empty (pre-toss / not yet announced)
    Unknown players in the XI receive the team average score so all 11
    always contribute (no blank "--" scores).
    """
    # Get team fallback scores for unknown players
    ts = team_strength_map.get((venue, team)) or team_strength_map.get((None, team))
    default_bat  = float(ts["bat_strength"])  if ts and pd.notna(ts.get("bat_strength"))  else 45.0
    default_bowl = float(ts["bowl_strength"]) if ts and pd.notna(ts.get("bowl_strength")) else 40.0

    if players:
        bat_scores, bowl_scores = [], []
        for p in players:
            bat, bowl = _blended_player_scores(p, venue)
            bat_scores.append(bat  if not np.isnan(bat)  else default_bat)
            bowl_scores.append(bowl if not np.isnan(bowl) else default_bowl)

        bat_scores.sort(reverse=True)
        bowl_scores.sort(reverse=True)
        bat_str  = float(np.mean(bat_scores[:6]))
        bowl_str = float(np.mean(bowl_scores[:4]))
        return bat_str, bowl_str, True   # True = XI data used

    # Fallback: team profile average for this venue (or overall) — no XI provided
    ts = team_strength_map.get((venue, team)) or team_strength_map.get((None, team))
    if ts:
        bat_str  = ts.get("bat_strength",  np.nan)
        bowl_str = ts.get("bowl_strength", np.nan)
        bat_str  = float(bat_str)  if pd.notna(bat_str)  else 50.0
        bowl_str = float(bowl_str) if pd.notna(bowl_str) else 45.0
        return bat_str, bowl_str, False  # False = fallback used

    # Last resort: season-aggregate legacy lookup
    scores = bat_df[(bat_df["season"].astype(str) == latest_season) &
                    bat_df["batter"].isin(players or [])]["bat_score"].nlargest(5)
    b = float(scores.mean()) if len(scores) > 0 else 50.0
    scores2 = bowl_df[(bowl_df["season"].astype(str) == latest_season) &
                      bowl_df["bowler"].isin(players or [])]["bowl_score"].nlargest(3)
    bw = float(scores2.mean()) if len(scores2) > 0 else 45.0
    return b, bw, False


def get_xi_strengths_extended(players: list[str], team: str, venue: str):
    """Like get_xi_strengths but also returns depth (std) and max bat score."""
    ts = team_strength_map.get((venue, team)) or team_strength_map.get((None, team))
    default_bat  = float(ts["bat_strength"])  if ts and pd.notna(ts.get("bat_strength"))  else 45.0
    default_bowl = float(ts["bowl_strength"]) if ts and pd.notna(ts.get("bowl_strength")) else 40.0

    if not players:
        return default_bat, default_bowl, 0.0, default_bat, False

    bat_scores, bowl_scores = [], []
    for p in players:
        bat, bowl = _blended_player_scores(p, venue)
        bat_scores.append(bat if not np.isnan(bat) else default_bat)
        bowl_scores.append(bowl if not np.isnan(bowl) else default_bowl)

    bat_sorted = sorted(bat_scores, reverse=True)
    bowl_sorted = sorted(bowl_scores, reverse=True)
    top6_bat  = float(np.mean(bat_sorted[:6]))
    top4_bowl = float(np.mean(bowl_sorted[:4]))
    depth = float(np.std(bat_sorted)) if len(bat_sorted) > 1 else 0.0
    max_bat = bat_sorted[0] if bat_sorted else default_bat
    return top6_bat, top4_bowl, depth, max_bat, True


# ── Pre-toss estimation helpers ───────────────────────────────────────────

def estimate_likely_xi(team: str, venue: str) -> list[str]:
    """Estimate a team's likely playing XI for pre-toss prediction.

    Strategy:
    1. Last known XI from matches.csv (most recent match with XI data)
    2. Fallback: top-11 by overall_score from player_database_2026.csv
    3. Last resort: empty list (get_xi_strengths_extended uses team profile)
    """
    # Try last known XI first (usually from previous season or earlier in season)
    xi = last_xi_map.get(team, [])
    if xi:
        return xi[:11]  # cap at 11 (matches.csv may have 12 with impact player)

    # Fallback: top-11 from squad by overall_score
    squad = squad_xi_map.get(team, [])
    if squad:
        return squad[:11]

    return []


def estimate_toss_probabilities(venue: str) -> tuple:
    """Estimate P(bat_first) and P(field_first) for a venue based on historical
    toss decision frequency.

    Returns (p_bat_first, p_field_first) — how likely the toss winner will
    choose each option. NOT who wins the toss.
    """
    p_field = venue_toss_field_pct.get(venue, 0.65)  # default: 65% choose field (IPL-wide trend)
    p_bat = 1 - p_field
    return (p_bat, p_field)


# ── Venue coordinate lookup for weather (shared with match_bot) ───────────

VENUE_COORDS = {
    "Wankhede Stadium":                                             (18.9388,  72.8258),
    "Wankhede Stadium, Mumbai":                                     (18.9388,  72.8258),
    "M.Chinnaswamy Stadium":                                        (12.9790,  77.5995),
    "M Chinnaswamy Stadium, Bengaluru":                             (12.9790,  77.5995),
    "MA Chidambaram Stadium":                                       (13.0629,  80.2792),
    "MA Chidambaram Stadium, Chennai":                              (13.0629,  80.2792),
    "Eden Gardens":                                                 (22.5646,  88.3433),
    "Eden Gardens, Kolkata":                                        (22.5646,  88.3433),
    "Narendra Modi Stadium":                                        (23.0900,  72.0830),
    "Narendra Modi Stadium, Ahmedabad":                             (23.0900,  72.0830),
    "Rajiv Gandhi International Stadium":                           (17.4046,  78.5481),
    "Rajiv Gandhi Intl Stadium, Hyderabad":                         (17.4046,  78.5481),
    "Punjab Cricket Association Stadium":                           (30.6943,  76.8601),
    "PCA Stadium, Mohali":                                          (30.6943,  76.8601),
    "Sawai Mansingh Stadium":                                       (26.8869,  75.8063),
    "Sawai Mansingh Stadium, Jaipur":                               (26.8869,  75.8063),
    "BRSABV Ekana Cricket Stadium":                                 (26.9034,  80.9450),
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": (26.9034,  80.9450),
    "Dr DY Patil Sports Academy":                                   (19.0443,  73.0168),
    "DY Patil Stadium, Mumbai":                                     (19.0443,  73.0168),
    "Arun Jaitley Stadium":                                         (28.6376,  77.2209),
    "Arun Jaitley Stadium, Delhi":                                  (28.6376,  77.2209),
    "Holkar Cricket Stadium":                                       (22.7196,  75.8577),
    "Himachal Pradesh Cricket Association Stadium":                 (31.8350,  76.9430),
}

import requests as _requests

def _fetch_weather_for_pretoss(venue: str, match_hour: int = 19) -> dict:
    """Fetch current/forecast weather from Open-Meteo for a venue.
    Returns dict with temperature, humidity, cloud_cover or defaults on failure.
    """
    coords = None
    vl = venue.lower()
    for k, v in VENUE_COORDS.items():
        if k.lower() in vl or vl in k.lower():
            coords = v
            break
        kwords = [w for w in k.lower().split() if len(w) > 4]
        if any(w in vl for w in kwords):
            coords = v
            break
    if not coords:
        return {"temperature": 30.0, "humidity": 55.0, "cloud_cover": 30.0}

    lat, lon = coords
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relativehumidity_2m,cloudcover"
        "&timezone=Asia%2FKolkata&forecast_days=1"
    )
    try:
        r = _requests.get(url, timeout=8)
        if r.status_code != 200:
            return {"temperature": 30.0, "humidity": 55.0, "cloud_cover": 30.0}
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        target = f"T{match_hour:02d}:00"
        idx = next((i for i, t in enumerate(times) if t.endswith(target)), match_hour)
        return {
            "temperature": hourly["temperature_2m"][idx],
            "humidity": hourly["relativehumidity_2m"][idx],
            "cloud_cover": hourly["cloudcover"][idx],
        }
    except Exception:
        return {"temperature": 30.0, "humidity": 55.0, "cloud_cover": 30.0}


# ── Projection helper ─────────────────────────────────────────────────────

def phase_aware_projection(runs, balls, crr, venue_avg, total_balls=120):
    """Phase-aware projected score. Blends CRR with venue mean,
    with trust in CRR growing as more overs are bowled.

    Powerplay (1-6):  CRR barely trusted — early run rate is noisy
    Middle   (7-15):  Growing trust in CRR
    Death    (16-20): Fully trust CRR
    """
    balls_rem = max(0, total_balls - balls)
    if balls_rem == 0:
        return float(runs)
    venue_rr = venue_avg / 20  # venue average run rate (runs per over)
    if balls <= 36:  # powerplay
        blend = balls / 120  # 0.05 at over 1, 0.30 at over 6
        proj_rr = blend * crr + (1 - blend) * venue_rr
    elif balls <= 90:  # middle overs
        blend = 0.3 + 0.7 * (balls - 36) / 54  # 0.30 at over 6, 1.0 at over 15
        proj_rr = blend * crr + (1 - blend) * venue_rr
    else:  # death overs
        proj_rr = crr
    projected = runs + proj_rr * balls_rem / 6
    return round(projected, 1)


# ── Schemas ────────────────────────────────────────────────────────────────
class PreMatchRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: Optional[str] = None
    toss_decision: Optional[str] = None
    team1_players: list[str] = []
    team2_players: list[str] = []

class PreTossRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    is_evening: int = 1        # 1 for 19:30 games, 0 for 15:30
    match_hour: int = 19       # IST hour for weather fetch
    team1_xi: list[str] = []   # last known XI — if provided, skips estimate_likely_xi
    team2_xi: list[str] = []

class PostTossRequest(BaseModel):
    bat_first: str
    bat_second: str
    venue: str
    toss_winner: str
    toss_decision: str         # "bat" or "field"
    bf_players: list[str] = []
    bs_players: list[str] = []
    # Impact player (12th player — nominated but not in starting XI)
    bf_impact_player: Optional[str] = None   # bat_first team's impact nominee
    bs_impact_player: Optional[str] = None   # bat_second team's impact nominee
    # Weather (from get_match_weather in match_bot or Open-Meteo)
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    cloud_cover: Optional[float] = None
    is_evening: int = 1        # 1 for 19:30 games, 0 for 15:30

class LiveRequest(BaseModel):
    batting_team: str
    bowling_team: str
    runs_scored: int
    wickets_fallen: int
    balls_bowled: int
    target: int
    # Optional momentum features — match_bot passes these when tracking live
    partnership_runs: Optional[int] = None
    partnership_balls: Optional[int] = None
    last_3ov_runs: Optional[int] = None
    last_3ov_wkts: Optional[int] = None
    boundary_pct: Optional[float] = None
    dot_ball_pct: Optional[float] = None
    first_innings_wickets: Optional[int] = None
    pp_runs: Optional[int] = None
    pp_wickets: Optional[int] = None
    venue: str = ""
    max_balls: Optional[int] = None   # D/L: total balls in reduced innings (default 120)

class Inn1LiveRequest(BaseModel):
    batting_team: str
    bowling_team: str
    runs_scored: int
    wickets_fallen: int
    balls_bowled: int
    pp_runs: Optional[int] = None
    pp_wickets: Optional[int] = None
    venue: str = ""

class UnifiedLiveRequest(BaseModel):
    current_innings: int            # 1 or 2
    bat_first: str                  # team batting first (constant label reference)
    bat_second: str                 # team batting second
    # Current innings state
    runs_scored: int
    wickets_fallen: int
    balls_bowled: int
    venue: str = ""
    # Inn2 specific (ignored for innings=1)
    target: Optional[int] = None
    first_innings_wickets: Optional[int] = None
    # Momentum (optional)
    partnership_runs: Optional[int] = None
    partnership_balls: Optional[int] = None
    last_3ov_runs: Optional[int] = None
    last_3ov_wkts: Optional[int] = None
    boundary_pct: Optional[float] = None
    dot_ball_pct: Optional[float] = None
    pp_runs: Optional[int] = None
    pp_wickets: Optional[int] = None
    max_balls: Optional[int] = None  # D/L reduced innings
    max_partnership: Optional[int] = None  # largest partnership runs so far this innings

class ReloadRequest(BaseModel):
    secret: str = "ipl2026"

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": "4.0", "latest_season": latest_season}

@app.get("/teams")
def list_teams():
    return {"teams": sorted(elo_map.keys())}

@app.get("/venues")
def list_venues():
    return {"venues": sorted(venue_stats.keys())}

@app.post("/reload-models")
def reload_models(req: ReloadRequest):
    if req.secret != "ipl2026":
        return {"status": "error", "message": "wrong secret"}
    global elo_map, form_map, form3_map, form10_map, formw_map
    global venue_stats, venue_team_map, h2h_map
    global bat_df, bowl_df, latest_season, latest_match_num
    global player_score_map, venue_player_map, team_strength_map, chase_wr_map
    global venue_toss_field_pct, last_xi_map, squad_xi_map
    global pp_wkt_wr_map, h2h_bvb_map
    load_models()
    (elo_map, form_map, form3_map, form10_map, formw_map,
     venue_stats, venue_team_map, h2h_map,
     bat_df, bowl_df, latest_season, latest_match_num,
     player_score_map, venue_player_map, team_strength_map,
     chase_wr_map, venue_toss_field_pct, last_xi_map, squad_xi_map,
     pp_wkt_wr_map, h2h_bvb_map) = load_lookups()
    return {"status": "ok", "message": "Models and lookups reloaded", "latest_season": latest_season}

@app.post("/predict/prematch")
def predict_prematch(req: PreMatchRequest):
    pre = _models["pre"]
    post_toss = bool(req.toss_winner)  # True once toss is known
    t1won_toss = int(req.toss_winner == req.team1) if req.toss_winner else 0
    toss_bat   = 1 if req.toss_decision and req.toss_decision.lower() == "bat" else 0

    e1 = elo_map.get(req.team1, 1500)
    e2 = elo_map.get(req.team2, 1500)
    f1 = form_map.get(req.team1, 0.5)
    f2 = form_map.get(req.team2, 0.5)
    f1_3  = form3_map.get(req.team1, 0.5)
    f2_3  = form3_map.get(req.team2, 0.5)
    f1_10 = form10_map.get(req.team1, 0.5)
    f2_10 = form10_map.get(req.team2, 0.5)

    key = tuple(sorted([req.team1, req.team2]))
    raw_h2h = h2h_map.get(key, 0.5)
    h2h = raw_h2h if req.team1 <= req.team2 else 1 - raw_h2h

    vs  = venue_stats.get(req.venue, {})
    vt1 = venue_team_map.get((req.venue, req.team1), 0.5)
    vt2 = venue_team_map.get((req.venue, req.team2), 0.5)

    # ── Player XI strengths ───────────────────────────────────────────────
    # Uses player_database_2026 + venue blending when XI is provided (post-toss).
    # Falls back to team_profiles_2026 averages when XI is unknown (pre-toss).
    b1,  bw1, t1_xi_used = get_xi_strengths(req.team1_players, req.team1, req.venue)
    b2,  bw2, t2_xi_used = get_xi_strengths(req.team2_players, req.team2, req.venue)
    xi_data_used = t1_xi_used or t2_xi_used

    # ── Toss-venue alignment (post-toss only) ─────────────────────────────
    venue_chase_wr = vs.get("venue_chase_win_rate", 0.5)
    if post_toss and req.toss_decision:
        chose_field       = req.toss_decision.lower() == "field"
        venue_favors_chase = venue_chase_wr > 0.5
        aligned = (chose_field and venue_favors_chase) or (not chose_field and not venue_favors_chase)
        toss_venue_aligned = int(aligned if t1won_toss else not aligned)
    else:
        toss_venue_aligned = 0

    # team1 bats second?
    if post_toss and req.toss_winner and req.toss_decision:
        toss_winner_bats = req.toss_decision.lower() == "bat"
        t1_bats_second = int(
            (req.toss_winner == req.team1 and not toss_winner_bats) or
            (req.toss_winner == req.team2 and toss_winner_bats)
        )
    else:
        t1_bats_second = 0

    # ── Chase win rates (per-team historical, drive the #2-#4 features) ──────
    t1_chase_wr = chase_wr_map.get(req.team1, 0.5)
    t2_chase_wr = chase_wr_map.get(req.team2, 0.5)
    if post_toss:
        # Post-toss: only the chasing team gets its chase advantage activated
        t1_chase_adv = t1_bats_second * t1_chase_wr
        t2_chase_adv = (1 - t1_bats_second) * t2_chase_wr
    else:
        # Pre-toss: neutral — neither team is assigned as chaser yet
        t1_chase_adv = 0.0
        t2_chase_adv = 0.0
    chase_adv_diff = t1_chase_adv - t2_chase_adv
    early_season   = int(latest_match_num + 1 <= 25)
    early_chase_boost = early_season * t1_bats_second

    feats = {
        "elo_diff": e1 - e2,
        "team1_form": f1, "team2_form": f2, "form_diff": f1 - f2,
        "form_3_diff": f1_3 - f2_3, "form_10_diff": f1_10 - f2_10,
        "form_weighted_diff": f1 - f2,
        "h2h_win_rate_team1": h2h,
        "team1_won_toss": t1won_toss, "toss_chose_bat": toss_bat,
        "venue_toss_win_rate": vs.get("venue_toss_win_rate", 0.5),
        "venue_bat_first_win_rate": vs.get("venue_bat_first_win_rate", 0.5),
        "venue_avg_first_innings": vs.get("venue_avg_first_innings", GLOBAL_AVG_FIRST_INNINGS),
        "venue_chase_win_rate": venue_chase_wr,
        "team1_venue_win_rate": vt1, "team2_venue_win_rate": vt2,
        "match_num_in_season": latest_match_num + 1,
        "is_playoff": int(latest_match_num + 1 > 56),
        "bat_diff": (b1 or 0) - (b2 or 0),
        "bowl_diff": (bw1 or 0) - (bw2 or 0),
        "team1_bat_strength": b1 or 0, "team2_bat_strength": b2 or 0,
        "team1_bowl_strength": bw1 or 0, "team2_bowl_strength": bw2 or 0,
        # Post-toss features
        "team1_bats_second": t1_bats_second,
        "toss_venue_aligned": toss_venue_aligned,
        "venue_chase_batting_second": venue_chase_wr if t1_bats_second else (1 - venue_chase_wr),
        # Chase advantage features (most important missing features)
        "team1_chase_wr": t1_chase_wr,
        "team2_chase_wr": t2_chase_wr,
        "chase_wr_diff": t1_chase_wr - t2_chase_wr,
        "team1_chase_advantage": t1_chase_adv,
        "team2_chase_advantage": t2_chase_adv,
        "chase_advantage_diff": chase_adv_diff,
        "early_season": early_season,
        "early_chase_boost": early_chase_boost,
    }

    tm = pre["train_median"]
    X = pd.DataFrame([[feats.get(f, tm.get(f, 0)) for f in pre["features"]]], columns=pre["features"])
    raw = float(pre["model"].predict_proba(X)[0, 1])

    # ── Feature-based confidence scoring ─────────────────────────────────
    # Instead of a uniform 3x amplifier, modulate amplification based on
    # how many independent feature signals agree with the model's direction.
    # When signals align (correct predictions) → 65-75% confidence.
    # When signals conflict (likely wrong) → stay near 50-55%.
    raw_favors_t1 = raw >= 0.5

    agreement_score = 0.0
    weight_sum = 0.0

    # Signal 1: ELO difference (strongest predictor, weight 3)
    elo_d = feats["elo_diff"]
    if abs(elo_d) > 10:
        w = min(3.0, 1.0 + abs(elo_d) / 50.0)  # stronger ELO gap → more weight
        weight_sum += w
        if (elo_d > 0) == raw_favors_t1:
            agreement_score += w

    # Signal 2: Head-to-head (weight 2)
    h2h_d = feats["h2h_win_rate_team1"] - 0.5
    if abs(h2h_d) > 0.05:
        w = 2.0
        weight_sum += w
        if (h2h_d > 0) == raw_favors_t1:
            agreement_score += w

    # Signal 3: Venue advantage differential (weight 1.5)
    venue_d = feats.get("team1_venue_win_rate", 0.5) - feats.get("team2_venue_win_rate", 0.5)
    if abs(venue_d) > 0.05:
        w = 1.5
        weight_sum += w
        if (venue_d > 0) == raw_favors_t1:
            agreement_score += w

    # Signal 4: Form difference (weight 1)
    form_d = feats["form_diff"]
    if abs(form_d) > 0.05:
        w = 1.0
        weight_sum += w
        if (form_d > 0) == raw_favors_t1:
            agreement_score += w

    # Signal 5: Player strength (weight 1.5, only when XI data available)
    bat_d = feats.get("bat_diff", 0)
    bowl_d = feats.get("bowl_diff", 0)
    if xi_data_used and (abs(bat_d) > 2 or abs(bowl_d) > 2):
        w = 1.5
        weight_sum += w
        strength_favors_t1 = (bat_d + bowl_d) > 0
        if strength_favors_t1 == raw_favors_t1:
            agreement_score += w

    # Agreement ratio: 0 (all disagree) to 1 (all agree)
    if weight_sum > 0:
        agreement_ratio = agreement_score / weight_sum
    else:
        agreement_ratio = 0.5  # no meaningful signals → neutral

    # Amplification: scales from 0.5 (conflicting) to 4.5 (full agreement)
    amp = 0.5 + 4.0 * agreement_ratio

    dev = raw - 0.5
    direction = 1 if dev >= 0 else -1
    abs_dev = abs(dev)
    amplified_dev = abs_dev * amp

    # When signals strongly agree, enforce a minimum confidence floor
    # Prevents 50.1% when all evidence points one way (raw model too compressed)
    if agreement_ratio >= 0.7:
        min_dev = 0.10 + 0.10 * agreement_ratio  # 0.17-0.20 → 67-70%
        amplified_dev = max(amplified_dev, min_dev)
    elif agreement_ratio >= 0.5:
        min_dev = 0.05 + 0.05 * agreement_ratio  # 0.075-0.085 → 58-59%
        amplified_dev = max(amplified_dev, min_dev)

    final = 0.5 + direction * amplified_dev
    final = max(0.10, min(0.90, final))

    factors = []
    if abs(feats["elo_diff"]) > 50:
        factors.append(f"{req.team1 if feats['elo_diff']>0 else req.team2} has higher ELO rating")
    if abs(feats["form_diff"]) > 0.3:
        factors.append(f"{req.team1 if feats['form_diff']>0 else req.team2} in better recent form")
    if feats["h2h_win_rate_team1"] > 0.6:
        factors.append(f"{req.team1} leads head-to-head")
    elif feats["h2h_win_rate_team1"] < 0.4:
        factors.append(f"{req.team2} leads head-to-head")
    if post_toss and toss_venue_aligned:
        factors.append(f"Toss decision well-aligned with {req.venue} conditions")
    if not post_toss and vs.get("venue_toss_win_rate", 0.5) > 0.55:
        factors.append(f"Toss is highly influential at {req.venue}")
    if vt1 > 0.6:
        factors.append(f"{req.team1} has strong record at this venue")
    elif vt2 > 0.6:
        factors.append(f"{req.team2} has strong record at this venue")
    if xi_data_used and abs(feats["bat_diff"]) > 5:
        stronger = req.team1 if feats["bat_diff"] > 0 else req.team2
        factors.append(f"{stronger} has stronger batting XI at this venue")
    if xi_data_used and abs(feats["bowl_diff"]) > 3:
        stronger = req.team1 if feats["bowl_diff"] > 0 else req.team2
        factors.append(f"{stronger} has stronger bowling attack")

    return {
        "team1": req.team1, "team2": req.team2, "venue": req.venue,
        "post_toss": post_toss,
        "xi_data_used": xi_data_used,
        "team1_win_probability": round(final, 4),
        "team2_win_probability": round(1 - final, 4),
        "predicted_winner": req.team1 if final >= 0.5 else req.team2,
        "confidence": "high" if abs(final - 0.5) > 0.15 else "medium" if abs(final - 0.5) > 0.07 else "low",
        "model_inputs": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in feats.items()},
        "player_strengths": {
            "team1_bat": round(b1, 1) if b1 else None,
            "team1_bowl": round(bw1, 1) if bw1 else None,
            "team2_bat": round(b2, 1) if b2 else None,
            "team2_bowl": round(bw2, 1) if bw2 else None,
            "source": "playing_xi" if xi_data_used else "team_profile_average",
        },
        "key_factors": factors,
    }

@app.post("/predict/posttoss")
def predict_posttoss(req: PostTossRequest):
    """Post-toss prediction using dedicated bat-first/bat-second model.

    Requires: toss result, playing XI for both teams.
    Optionally: weather data (temperature, humidity, cloud_cover).
    Returns P(bat_first_wins) and P(bat_second_wins).
    """
    if "posttoss" not in _models:
        return {"error": "Post-toss model not loaded. Run 10_post_toss_model.py first."}

    pt = _models["posttoss"]

    # Lookup ELO, form for each team
    bf_elo = elo_map.get(req.bat_first, 1500)
    bs_elo = elo_map.get(req.bat_second, 1500)
    bf_form = form_map.get(req.bat_first, 0.5)
    bs_form = form_map.get(req.bat_second, 0.5)
    bf_form3 = form3_map.get(req.bat_first, 0.5)
    bs_form3 = form3_map.get(req.bat_second, 0.5)
    bf_form10 = form10_map.get(req.bat_first, 0.5)
    bs_form10 = form10_map.get(req.bat_second, 0.5)

    # H2H from bat_first perspective
    key = tuple(sorted([req.bat_first, req.bat_second]))
    raw_h2h = h2h_map.get(key, 0.5)
    h2h_bf = raw_h2h if req.bat_first <= req.bat_second else 1 - raw_h2h

    # XI strengths (extended: includes depth and max)
    bf_bat, bf_bowl, bf_depth, bf_max, _ = get_xi_strengths_extended(
        req.bf_players, req.bat_first, req.venue)
    bs_bat, bs_bowl, bs_depth, bs_max, _ = get_xi_strengths_extended(
        req.bs_players, req.bat_second, req.venue)

    # H2H batter-bowler matchup advantage (Phase 4 feature)
    def _compute_matchup_adv(batters, bowlers, top_bat=6, top_bowl=4):
        """Mean H2H advantage for top batters vs top bowlers; 0.5 = neutral."""
        advs = []
        for b in batters[:top_bat]:
            for w in bowlers[:top_bowl]:
                adv = h2h_bvb_map.get((b, w), 0.5)
                advs.append(adv)
        return float(np.mean(advs)) if advs else 0.5

    _bf_batters = req.bf_players if req.bf_players else []
    _bs_bowlers = req.bs_players if req.bs_players else []
    matchup_adv_bf   = _compute_matchup_adv(_bf_batters, _bs_bowlers)
    matchup_adv_diff = matchup_adv_bf - 0.5

    # Impact player bowling strength (Phase 6)
    def _impact_bowl_score(player_name: Optional[str]) -> float:
        """Return bowl_score for impact player from player_database_2026."""
        if not player_name:
            return 40.0  # league-average default
        _, bowl = _blended_player_scores(player_name, req.venue)
        return float(bowl) if not np.isnan(bowl) else 40.0

    bf_impact_bowl = _impact_bowl_score(req.bf_impact_player)
    bs_impact_bowl = _impact_bowl_score(req.bs_impact_player)
    impact_bowl_diff = bf_impact_bowl - bs_impact_bowl

    # Venue stats
    vs = venue_stats.get(req.venue, {})
    bf_venue_wr = venue_team_map.get((req.venue, req.bat_first), 0.5)
    bs_venue_wr = venue_team_map.get((req.venue, req.bat_second), 0.5)
    venue_chase_wr = vs.get("venue_chase_win_rate", 0.5)

    # Toss features
    toss_chose_field = 1 if req.toss_decision.lower() == "field" else 0
    tw_bats_first = int(
        (req.toss_winner == req.bat_first and req.toss_decision.lower() == "bat") or
        (req.toss_winner == req.bat_second and req.toss_decision.lower() == "field")
    )
    toss_venue_aligned = int(
        (toss_chose_field == 1 and venue_chase_wr > 0.5) or
        (toss_chose_field == 0 and venue_chase_wr <= 0.5)
    )

    # Chase WR for batting-second team
    bs_chase_wr = chase_wr_map.get(req.bat_second, 0.5)

    # Weather
    temp = req.temperature if req.temperature is not None else 30.0
    hum = req.humidity if req.humidity is not None else 55.0
    cloud = req.cloud_cover if req.cloud_cover is not None else 30.0
    is_eve = req.is_evening
    dew = max(0, min(1, (hum - 65) / 35)) if hum >= 65 and is_eve else 0.0
    heat = 1 if temp >= 35 else 0

    feats = {
        "elo_diff_bf": bf_elo - bs_elo,
        "bf_elo": bf_elo, "bs_elo": bs_elo,
        "form_diff_bf": bf_form - bs_form,
        "bf_form": bf_form, "bs_form": bs_form,
        "form_3_diff_bf": bf_form3 - bs_form3,
        "form_10_diff_bf": bf_form10 - bs_form10,
        "h2h_bf": h2h_bf,
        "bf_xi_bat": bf_bat, "bs_xi_bat": bs_bat, "xi_bat_diff": bf_bat - bs_bat,
        "bf_xi_bowl": bf_bowl, "bs_xi_bowl": bs_bowl, "xi_bowl_diff": bf_bowl - bs_bowl,
        "bf_xi_depth": bf_depth, "bs_xi_depth": bs_depth,
        "bf_xi_max_bat": bf_max, "bs_xi_max_bat": bs_max,
        "toss_chose_field": toss_chose_field,
        "toss_winner_bats_first": tw_bats_first,
        "toss_venue_aligned_bf": toss_venue_aligned,
        "matchup_advantage_bf":   matchup_adv_bf,
        "matchup_advantage_diff": matchup_adv_diff,
        "bf_impact_bowl":   bf_impact_bowl,
        "bs_impact_bowl":   bs_impact_bowl,
        "impact_bowl_diff": impact_bowl_diff,
        "venue_chase_win_rate": venue_chase_wr,
        "venue_bat_first_win_rate": vs.get("venue_bat_first_win_rate", 0.5),
        "venue_avg_first_innings": vs.get("venue_avg_first_innings", GLOBAL_AVG_FIRST_INNINGS),
        "venue_matches": vs.get("venue_matches", 10) if "venue_matches" in vs else 10,
        "bf_venue_wr": bf_venue_wr, "bs_venue_wr": bs_venue_wr,
        "venue_toss_win_rate": vs.get("venue_toss_win_rate", 0.5),
        "bs_chase_wr": bs_chase_wr,
        "temperature": temp, "humidity": hum, "cloud_cover": cloud,
        "dew_factor": dew, "is_evening": is_eve, "heat_factor": heat,
        "dew_chase_advantage": dew * venue_chase_wr,
        "humidity_x_evening": hum * is_eve / 100,
        "match_num_in_season": latest_match_num + 1,
        "is_playoff": int(latest_match_num + 1 > 56),
    }

    tm = pt.get("train_median", {})
    X = pd.DataFrame([[feats.get(f, tm.get(f, 0)) for f in pt["features"]]], columns=pt["features"])
    p_bf = float(pt["model"].predict_proba(X)[0, 1])  # P(bat_first wins)
    p_bf = max(0.05, min(0.95, p_bf))
    p_bs = 1 - p_bf

    conf_thr = pt.get("confidence_threshold", 0.68)
    is_high_conf = abs(p_bf - 0.5) >= (conf_thr - 0.5)
    confidence = "high" if is_high_conf else ("medium" if abs(p_bf - 0.5) > 0.10 else "low")

    factors = []
    if abs(feats["elo_diff_bf"]) > 50:
        fav = req.bat_first if feats["elo_diff_bf"] > 0 else req.bat_second
        factors.append(f"{fav} has higher ELO rating")
    if abs(feats["xi_bat_diff"]) > 5:
        fav = req.bat_first if feats["xi_bat_diff"] > 0 else req.bat_second
        factors.append(f"{fav} has stronger batting XI")
    if abs(feats["xi_bowl_diff"]) > 3:
        fav = req.bat_first if feats["xi_bowl_diff"] > 0 else req.bat_second
        factors.append(f"{fav} has stronger bowling attack")
    if toss_venue_aligned:
        factors.append(f"Toss decision matches {req.venue} conditions")
    if dew > 0.3:
        factors.append(f"Dew factor favors {req.bat_second} (chasing)")
    if venue_chase_wr > 0.55:
        factors.append(f"{req.venue} favors chasing ({venue_chase_wr:.0%})")
    elif venue_chase_wr < 0.45:
        factors.append(f"{req.venue} favors batting first ({1-venue_chase_wr:.0%})")

    return {
        "bat_first": req.bat_first,
        "bat_second": req.bat_second,
        "venue": req.venue,
        "batting_first_win_probability": round(p_bf, 4),
        "batting_second_win_probability": round(p_bs, 4),
        "predicted_winner": req.bat_first if p_bf >= 0.5 else req.bat_second,
        "confidence": confidence,
        "high_confidence": is_high_conf,
        "weather": {"temperature": temp, "humidity": hum, "cloud_cover": cloud,
                    "dew_factor": round(dew, 3), "is_evening": is_eve},
        "xi_strengths": {
            "bf_bat": round(bf_bat, 1), "bf_bowl": round(bf_bowl, 1),
            "bs_bat": round(bs_bat, 1), "bs_bowl": round(bs_bowl, 1),
        },
        "key_factors": factors,
    }


@app.post("/predict/pretoss")
def predict_pretoss(req: PreTossRequest):
    """Pre-toss prediction by reusing the post-toss model.

    Strategy: estimate XI and weather, then run the post-toss model TWICE
    (team1 bats first vs team2 bats first) and blend by venue toss-decision
    probability.  This avoids the weak 53% pre-match model entirely.
    """
    if "posttoss" not in _models:
        return {"error": "Post-toss model not loaded. Run 10_post_toss_model.py first."}

    # ── Estimate playing XI — prefer caller-supplied XI (last known from bot) ─
    xi_t1 = req.team1_xi if req.team1_xi else estimate_likely_xi(req.team1, req.venue)
    xi_t2 = req.team2_xi if req.team2_xi else estimate_likely_xi(req.team2, req.venue)
    xi_source = "last_known_xi" if (req.team1_xi or req.team2_xi) else "squad_estimate"

    # ── Fetch weather ─────────────────────────────────────────────────────
    weather = _fetch_weather_for_pretoss(req.venue, req.match_hour)
    temp = weather["temperature"]
    hum  = weather["humidity"]
    cloud = weather["cloud_cover"]

    # ── Toss-decision probabilities for this venue ────────────────────────
    p_bat, p_field = estimate_toss_probabilities(req.venue)

    # ── Run 4 scenarios (2 toss winners × 2 decisions) ────────────────────
    # s1: team1 wins toss (50%), chooses bat  (p_bat)   → team1 bats first
    # s2: team1 wins toss (50%), chooses field (p_field) → team2 bats first
    # s3: team2 wins toss (50%), chooses bat  (p_bat)   → team2 bats first
    # s4: team2 wins toss (50%), chooses field (p_field) → team1 bats first
    w1 = 0.5 * p_bat     # team1 wins toss, bats
    w2 = 0.5 * p_field   # team1 wins toss, fields
    w3 = 0.5 * p_bat     # team2 wins toss, bats
    w4 = 0.5 * p_field   # team2 wins toss, fields

    s1 = _run_posttoss_internal(
        bat_first=req.team1, bat_second=req.team2, venue=req.venue,
        toss_winner=req.team1, toss_decision="bat",
        bf_players=xi_t1, bs_players=xi_t2,
        temperature=temp, humidity=hum, cloud_cover=cloud, is_evening=req.is_evening,
    )
    s2 = _run_posttoss_internal(
        bat_first=req.team2, bat_second=req.team1, venue=req.venue,
        toss_winner=req.team1, toss_decision="field",
        bf_players=xi_t2, bs_players=xi_t1,
        temperature=temp, humidity=hum, cloud_cover=cloud, is_evening=req.is_evening,
    )
    s3 = _run_posttoss_internal(
        bat_first=req.team2, bat_second=req.team1, venue=req.venue,
        toss_winner=req.team2, toss_decision="bat",
        bf_players=xi_t2, bs_players=xi_t1,
        temperature=temp, humidity=hum, cloud_cover=cloud, is_evening=req.is_evening,
    )
    s4 = _run_posttoss_internal(
        bat_first=req.team1, bat_second=req.team2, venue=req.venue,
        toss_winner=req.team2, toss_decision="field",
        bf_players=xi_t1, bs_players=xi_t2,
        temperature=temp, humidity=hum, cloud_cover=cloud, is_evening=req.is_evening,
    )

    # P(team1 wins) for each scenario
    p_t1_s1 = s1["batting_first_win_probability"]    # team1 bats, team1 won toss
    p_t1_s2 = s2["batting_second_win_probability"]   # team2 bats, team1 won toss (team1 chases)
    p_t1_s3 = s3["batting_second_win_probability"]   # team2 bats, team2 won toss (team1 chases)
    p_t1_s4 = s4["batting_first_win_probability"]    # team1 bats, team2 won toss

    # Weighted blend across all 4 scenarios
    p_t1 = w1 * p_t1_s1 + w2 * p_t1_s2 + w3 * p_t1_s3 + w4 * p_t1_s4
    p_t1 = max(0.05, min(0.95, p_t1))
    p_t2 = 1 - p_t1

    # Aggregate for display: "if team1 bats" = weighted avg of s1,s4; "if team2 bats" = s2,s3
    p_t1_wins_if_bats = (w1 * p_t1_s1 + w4 * p_t1_s4) / (w1 + w4) if (w1 + w4) > 0 else 0.5
    p_t1_wins_if_chases = (w2 * p_t1_s2 + w3 * p_t1_s3) / (w2 + w3) if (w2 + w3) > 0 else 0.5

    confidence = "high" if abs(p_t1 - 0.5) > 0.15 else "medium" if abs(p_t1 - 0.5) > 0.07 else "low"

    # ── Key factors ───────────────────────────────────────────────────────
    factors = []
    e1 = elo_map.get(req.team1, 1500)
    e2 = elo_map.get(req.team2, 1500)
    if abs(e1 - e2) > 50:
        fav = req.team1 if e1 > e2 else req.team2
        factors.append(f"{fav} has higher ELO rating ({max(e1,e2):.0f} vs {min(e1,e2):.0f})")
    if abs(p_t1_wins_if_bats - p_t1_wins_if_chases) > 0.05:
        better = "batting first" if p_t1_wins_if_bats > p_t1_wins_if_chases else "chasing"
        factors.append(f"{req.team1} stronger when {better}")
    vs = venue_stats.get(req.venue, {})
    venue_chase_wr = vs.get("venue_chase_win_rate", 0.5)
    if venue_chase_wr > 0.55:
        factors.append(f"{req.venue.split(',')[0]} favors chasing ({venue_chase_wr:.0%})")
    elif venue_chase_wr < 0.45:
        factors.append(f"{req.venue.split(',')[0]} favors batting first ({1-venue_chase_wr:.0%})")
    if p_field > 0.75:
        factors.append(f"Teams overwhelmingly choose to field here ({p_field:.0%})")
    if xi_source == "last_known_xi":
        factors.append("Using last known playing XIs")
    else:
        factors.append("Using estimated XIs from squad data")

    return {
        "team1": req.team1,
        "team2": req.team2,
        "venue": req.venue,
        "team1_win_probability": round(p_t1, 4),
        "team2_win_probability": round(p_t2, 4),
        "predicted_winner": req.team1 if p_t1 >= 0.5 else req.team2,
        "confidence": confidence,
        "xi_source": xi_source,
        "weather": {"temperature": temp, "humidity": hum, "cloud_cover": cloud},
        "toss_decision_probs": {"p_bat_first": round(p_bat, 3), "p_field_first": round(p_field, 3)},
        "scenarios": {
            "team1_bats_first": {
                "team1_win_prob": round(p_t1_wins_if_bats, 4),
                "team2_win_prob": round(1 - p_t1_wins_if_bats, 4),
                "weight": 0.5,
            },
            "team2_bats_first": {
                "team1_win_prob": round(p_t1_wins_if_chases, 4),
                "team2_win_prob": round(1 - p_t1_wins_if_chases, 4),
                "weight": 0.5,
            },
        },
        "key_factors": factors,
    }


def _run_posttoss_internal(bat_first, bat_second, venue, toss_winner, toss_decision,
                           bf_players, bs_players, temperature, humidity, cloud_cover,
                           is_evening, bf_impact_player=None, bs_impact_player=None):
    """Internal version of predict_posttoss that takes raw args instead of a request object.
    Returns the same dict as the /predict/posttoss endpoint.
    """
    pt = _models["posttoss"]

    bf_elo = elo_map.get(bat_first, 1500)
    bs_elo = elo_map.get(bat_second, 1500)
    bf_form = form_map.get(bat_first, 0.5)
    bs_form = form_map.get(bat_second, 0.5)
    bf_form3 = form3_map.get(bat_first, 0.5)
    bs_form3 = form3_map.get(bat_second, 0.5)
    bf_form10 = form10_map.get(bat_first, 0.5)
    bs_form10 = form10_map.get(bat_second, 0.5)

    key = tuple(sorted([bat_first, bat_second]))
    raw_h2h = h2h_map.get(key, 0.5)
    h2h_bf = raw_h2h if bat_first <= bat_second else 1 - raw_h2h

    bf_bat, bf_bowl, bf_depth, bf_max, _ = get_xi_strengths_extended(
        bf_players, bat_first, venue)
    bs_bat, bs_bowl, bs_depth, bs_max, _ = get_xi_strengths_extended(
        bs_players, bat_second, venue)

    # H2H matchup advantage (Phase 4 feature)
    _adv_list = []
    for _b in (bf_players or [])[:6]:
        for _w in (bs_players or [])[:4]:
            _adv_list.append(h2h_bvb_map.get((_b, _w), 0.5))
    matchup_adv_bf   = float(np.mean(_adv_list)) if _adv_list else 0.5
    matchup_adv_diff = matchup_adv_bf - 0.5

    # Impact player bowling strength (Phase 6)
    def _imp_bowl(player_name):
        if not player_name:
            return 40.0
        _, bowl = _blended_player_scores(player_name, venue)
        return float(bowl) if not np.isnan(bowl) else 40.0

    bf_impact_bowl = _imp_bowl(bf_impact_player)
    bs_impact_bowl = _imp_bowl(bs_impact_player)
    impact_bowl_diff = bf_impact_bowl - bs_impact_bowl

    vs = venue_stats.get(venue, {})
    bf_venue_wr = venue_team_map.get((venue, bat_first), 0.5)
    bs_venue_wr = venue_team_map.get((venue, bat_second), 0.5)
    venue_chase_wr = vs.get("venue_chase_win_rate", 0.5)

    toss_chose_field = 1 if toss_decision.lower() == "field" else 0
    tw_bats_first = int(
        (toss_winner == bat_first and toss_decision.lower() == "bat") or
        (toss_winner == bat_second and toss_decision.lower() == "field")
    )
    toss_venue_aligned = int(
        (toss_chose_field == 1 and venue_chase_wr > 0.5) or
        (toss_chose_field == 0 and venue_chase_wr <= 0.5)
    )

    bs_chase_wr = chase_wr_map.get(bat_second, 0.5)

    temp = temperature if temperature is not None else 30.0
    hum = humidity if humidity is not None else 55.0
    cloud = cloud_cover if cloud_cover is not None else 30.0
    is_eve = is_evening
    dew = max(0, min(1, (hum - 65) / 35)) if hum >= 65 and is_eve else 0.0
    heat = 1 if temp >= 35 else 0

    feats = {
        "elo_diff_bf": bf_elo - bs_elo,
        "bf_elo": bf_elo, "bs_elo": bs_elo,
        "form_diff_bf": bf_form - bs_form,
        "bf_form": bf_form, "bs_form": bs_form,
        "form_3_diff_bf": bf_form3 - bs_form3,
        "form_10_diff_bf": bf_form10 - bs_form10,
        "h2h_bf": h2h_bf,
        "bf_xi_bat": bf_bat, "bs_xi_bat": bs_bat, "xi_bat_diff": bf_bat - bs_bat,
        "bf_xi_bowl": bf_bowl, "bs_xi_bowl": bs_bowl, "xi_bowl_diff": bf_bowl - bs_bowl,
        "bf_xi_depth": bf_depth, "bs_xi_depth": bs_depth,
        "bf_xi_max_bat": bf_max, "bs_xi_max_bat": bs_max,
        "toss_chose_field": toss_chose_field,
        "toss_winner_bats_first": tw_bats_first,
        "toss_venue_aligned_bf": toss_venue_aligned,
        "matchup_advantage_bf":   matchup_adv_bf,
        "matchup_advantage_diff": matchup_adv_diff,
        "bf_impact_bowl":   bf_impact_bowl,
        "bs_impact_bowl":   bs_impact_bowl,
        "impact_bowl_diff": impact_bowl_diff,
        "venue_chase_win_rate": venue_chase_wr,
        "venue_bat_first_win_rate": vs.get("venue_bat_first_win_rate", 0.5),
        "venue_avg_first_innings": vs.get("venue_avg_first_innings", GLOBAL_AVG_FIRST_INNINGS),
        "venue_matches": vs.get("venue_matches", 10) if "venue_matches" in vs else 10,
        "bf_venue_wr": bf_venue_wr, "bs_venue_wr": bs_venue_wr,
        "venue_toss_win_rate": vs.get("venue_toss_win_rate", 0.5),
        "bs_chase_wr": bs_chase_wr,
        "temperature": temp, "humidity": hum, "cloud_cover": cloud,
        "dew_factor": dew, "is_evening": is_eve, "heat_factor": heat,
        "dew_chase_advantage": dew * venue_chase_wr,
        "humidity_x_evening": hum * is_eve / 100,
        "match_num_in_season": latest_match_num + 1,
        "is_playoff": int(latest_match_num + 1 > 56),
    }

    tm = pt.get("train_median", {})
    X = pd.DataFrame([[feats.get(f, tm.get(f, 0)) for f in pt["features"]]], columns=pt["features"])
    p_bf = float(pt["model"].predict_proba(X)[0, 1])
    p_bf = max(0.05, min(0.95, p_bf))

    return {
        "batting_first_win_probability": p_bf,
        "batting_second_win_probability": 1 - p_bf,
    }


@app.post("/predict/live")
def predict_live(req: LiveRequest):
    live = _models["live"]
    total_balls = req.max_balls if req.max_balls else 120  # D/L support
    balls_remaining = max(0, total_balls - req.balls_bowled)
    runs_needed     = max(0, req.target - req.runs_scored)
    wickets_left    = max(0, 10 - req.wickets_fallen)
    crr = (req.runs_scored / req.balls_bowled * 6) if req.balls_bowled > 0 else 0
    rrr = (runs_needed / balls_remaining * 6) if balls_remaining > 0 else 99.0
    rrr = min(rrr, 99.0)

    first_inn_rr = (req.target - 1) / total_balls * 6
    overs_bowled = max(req.balls_bowled / 6, 0.1)

    # ── Smart momentum estimates when not provided ────────────────────
    # Instead of 0 (which badly distorts the scaler), estimate from match state.
    # These are overridden by real values when match_bot passes them.

    # Partnership: estimate from average interval between wickets
    if req.partnership_runs is not None:
        p_runs  = req.partnership_runs
        p_balls = req.partnership_balls or max(1, int(p_runs / max(crr / 6, 0.5)))
    elif req.wickets_fallen == 0:
        p_runs  = req.runs_scored        # no wicket yet — entire innings is one partnership
        p_balls = req.balls_bowled
    else:
        # Estimate: runs since last wicket ~ runs per wicket interval
        avg_interval = req.runs_scored / req.wickets_fallen
        p_runs  = int(avg_interval * 0.4)   # typically last partnership is shorter
        p_balls = max(1, int(p_runs / max(crr / 6, 0.5)))

    # Last 3 overs: extrapolate from current run rate
    if req.last_3ov_runs is not None:
        l3_runs = req.last_3ov_runs
        l3_wkts = req.last_3ov_wkts or 0
    else:
        l3_runs = round(crr * 3)           # CRR scaled to 3 overs
        wkt_rate = req.wickets_fallen / overs_bowled
        l3_wkts = round(min(wkt_rate * 3, 3))  # cap at 3 wickets in 3 overs

    # Boundary and dot percentages: use T20 averages as fallback
    bdry_pct = req.boundary_pct if req.boundary_pct is not None else 0.25
    dot_pct  = req.dot_ball_pct if req.dot_ball_pct is not None else 0.42

    # First innings wickets
    fi_wkts = req.first_innings_wickets if req.first_innings_wickets is not None else 6

    # Venue average for target context
    venue_avg = GLOBAL_AVG_FIRST_INNINGS
    if req.venue:
        vs_lookup = venue_stats.get(req.venue, {})
        venue_avg = vs_lookup.get("venue_avg_first_innings", GLOBAL_AVG_FIRST_INNINGS) or GLOBAL_AVG_FIRST_INNINGS

    # Powerplay features (overs 1-6 = balls 1-36)
    balls = req.balls_bowled
    is_pp = int(balls <= 36)
    if balls > 36:
        pp_r = req.pp_runs if req.pp_runs is not None else req.runs_scored  # fallback: current runs
        pp_w = req.pp_wickets if req.pp_wickets is not None else req.wickets_fallen
    else:
        pp_r, pp_w = 0, 0
    pp_run_rate = (pp_r / 36 * 6) if balls > 36 else 0.0
    pp_req_rate = ((req.target - pp_r) / 84 * 6) if balls > 36 else 0.0
    pp_rate_gap = pp_run_rate - pp_req_rate

    X_raw = {
        "ball_num": balls, "balls_remaining": balls_remaining,
        "balls_pct": balls / total_balls, "cum_runs": req.runs_scored,
        "runs_needed": runs_needed, "cum_wickets": req.wickets_fallen,
        "wickets_left": wickets_left, "wickets_pct": req.wickets_fallen / 10,
        "crr": crr, "rrr": rrr, "rrr_diff": crr - rrr,
        "run_rate_ratio": crr / max(rrr, 0.01),
        # Momentum features — smart estimates or real values from match_bot
        "partnership_runs": p_runs, "partnership_balls": p_balls,
        "last_3ov_runs": l3_runs, "last_3ov_wkts": l3_wkts,
        "boundary_pct": bdry_pct, "dot_ball_pct": dot_pct,
        # First innings context
        "first_innings_run_rate": first_inn_rr,
        "target_vs_venue_avg": req.target / venue_avg,
        "first_innings_wickets": fi_wkts,
        # Powerplay features
        "is_pp": is_pp, "pp_runs": pp_r, "pp_wickets": pp_w,
        "pp_run_rate": pp_run_rate, "pp_req_rate": pp_req_rate, "pp_rate_gap": pp_rate_gap,
    }

    X = pd.DataFrame([[X_raw.get(f, 0) for f in live["features"]]], columns=live["features"])
    X_s = pd.DataFrame(live["scaler"].transform(X), columns=live["features"])
    prob = float(live["model"].predict_proba(X_s)[0, 1])

    over = f"{req.balls_bowled // 6}.{req.balls_bowled % 6}"
    status = "on track" if rrr <= crr * 1.1 else "behind"
    if runs_needed <= 0:
        status = "won"
    if wickets_left == 0:
        status = "lost"

    return {
        "batting_team": req.batting_team, "bowling_team": req.bowling_team, "over": over,
        "match_state": {
            "runs_scored": req.runs_scored, "wickets_fallen": req.wickets_fallen,
            "balls_bowled": req.balls_bowled, "runs_needed": runs_needed,
            "balls_remaining": balls_remaining, "wickets_left": wickets_left,
            "current_run_rate": round(crr, 2), "required_run_rate": round(rrr, 2),
        },
        "batting_team_win_probability": round(prob, 4),
        "bowling_team_win_probability": round(1 - prob, 4),
        "predicted_winner": req.batting_team if prob >= 0.5 else req.bowling_team,
        "status": status,
        "confidence": "high" if abs(prob - 0.5) > 0.2 else "medium" if abs(prob - 0.5) > 0.1 else "low",
    }

@app.post("/predict/live_inn1")
def predict_live_inn1(req: Inn1LiveRequest):
    """First innings live prediction: will batting-first team win?"""
    inn1 = _models["inn1"]
    balls = req.balls_bowled
    crr = (req.runs_scored / balls * 6) if balls > 0 else 0
    balls_remaining = max(0, 120 - balls)

    vs = venue_stats.get(req.venue, {})
    venue_avg = vs.get("venue_avg_first_innings", GLOBAL_AVG_FIRST_INNINGS) or GLOBAL_AVG_FIRST_INNINGS
    expected_at = venue_avg * (balls / 120) if balls > 0 else 0
    projected = phase_aware_projection(req.runs_scored, balls, crr, venue_avg)

    overs_bowled = max(balls / 6, 0.1)

    # Smart momentum estimates (same logic as 2nd innings)
    if req.wickets_fallen == 0:
        p_runs_e  = req.runs_scored
        p_balls_e = balls
    else:
        avg_int    = req.runs_scored / req.wickets_fallen
        p_runs_e   = int(avg_int * 0.4)
        p_balls_e  = max(1, int(p_runs_e / max(crr / 6, 0.5)))

    l3_runs_e = round(crr * 3)
    wkt_rate  = req.wickets_fallen / overs_bowled
    l3_wkts_e = round(min(wkt_rate * 3, 3))

    # Acceleration: run rate in last phase vs first phase
    accel = 0.0
    if balls > 36:  # after powerplay, estimate acceleration
        pp_rate = min(crr * 0.85, crr)  # powerplay is typically slightly slower
        recent_rate = crr * 1.1          # middle/death overs typically faster
        accel = recent_rate - pp_rate

    X_raw = {
        "cum_runs": req.runs_scored,
        "cum_wickets": req.wickets_fallen,
        "crr": crr,
        "balls_remaining": balls_remaining,
        "balls_pct": balls / 120,
        "wickets_pct": req.wickets_fallen / 10,
        "projected_score": projected,
        "venue_avg": venue_avg,
        "score_vs_expected": req.runs_scored - expected_at,
        "score_vs_expected_pct": req.runs_scored / expected_at if expected_at > 0 else 1.0,
        "partnership_runs": p_runs_e,
        "partnership_balls": p_balls_e,
        "last_3ov_runs": l3_runs_e,
        "last_3ov_wkts": l3_wkts_e,
        "boundary_pct": 0.25,
        "dot_pct": 0.42,
        "acceleration": accel,
        "elo_diff": elo_map.get(req.batting_team, 1500) - elo_map.get(req.bowling_team, 1500),
        "form_diff": form_map.get(req.batting_team, 0.5) - form_map.get(req.bowling_team, 0.5),
        # Powerplay features
        "is_pp": int(balls <= 36),
        "pp_runs": (req.pp_runs if req.pp_runs is not None else req.runs_scored) if balls > 36 else 0,
        "pp_wickets": (req.pp_wickets if req.pp_wickets is not None else req.wickets_fallen) if balls > 36 else 0,
        "pp_vs_venue_avg": (
            ((req.pp_runs or req.runs_scored) / (venue_avg * 6 / 20))
            if balls > 36 and venue_avg > 0 else 0.0
        ),
    }

    X = pd.DataFrame([[X_raw.get(f, 0) for f in inn1["features"]]], columns=inn1["features"])
    X_s = pd.DataFrame(inn1["scaler"].transform(X), columns=inn1["features"])
    prob = float(inn1["model"].predict_proba(X_s)[0, 1])

    over = f"{balls // 6}.{balls % 6}"
    return {
        "batting_team": req.batting_team,
        "bowling_team": req.bowling_team,
        "innings": 1,
        "over": over,
        "match_state": {
            "runs_scored": req.runs_scored,
            "wickets_fallen": req.wickets_fallen,
            "balls_bowled": balls,
            "current_run_rate": round(crr, 2),
            "projected_score": round(projected),
            "venue_avg": round(venue_avg),
        },
        "batting_team_win_probability": round(prob, 4),
        "bowling_team_win_probability": round(1 - prob, 4),
        "predicted_winner": req.batting_team if prob >= 0.5 else req.bowling_team,
        "confidence": "high" if abs(prob - 0.5) > 0.2 else "medium" if abs(prob - 0.5) > 0.1 else "low",
    }

def _get_team_phase_wkt_wr(batting_team: str, over_num: int, pp_wickets: int, role: str) -> float:
    """Look up team's historical win rate for the most recently completed phase.
    over 1-6:  PP in progress → 0
    over 7-15: PP complete → PP wicket win rate
    over 16+:  middle complete → middle wicket win rate
    """
    if over_num <= 6:
        return 0.0
    wkt_bucket = min(pp_wickets, 3)  # 0, 1, 2, 3+
    if over_num <= 15:
        return pp_wkt_wr_map.get((batting_team, "pp", wkt_bucket, role), 0.0)
    else:
        return pp_wkt_wr_map.get((batting_team, "middle", wkt_bucket, role), 0.0)

@app.post("/predict/live_unified")
def predict_live_unified(req: UnifiedLiveRequest):
    """Unified live predictor — single model for both innings.
    Always returns P(bat_first wins) as the canonical output.
    Inn1: batting_team = bat_first, bowling_team = bat_second.
    Inn2: batting_team = bat_second (chaser), bowling_team = bat_first.
    """
    if "unified" not in _models:
        return {"error": "Unified live model not loaded. Run 11_unified_live_model.py first."}

    uni = _models["unified"]
    balls = req.balls_bowled
    total_balls = req.max_balls or 120
    vs = venue_stats.get(req.venue, {})
    venue_avg = vs.get("venue_avg_first_innings", GLOBAL_AVG_FIRST_INNINGS) or GLOBAL_AVG_FIRST_INNINGS

    # Pre-match context (bat_first perspective)
    elo_diff  = elo_map.get(req.bat_first, 1500) - elo_map.get(req.bat_second, 1500)
    form_diff = form_map.get(req.bat_first, 0.5) - form_map.get(req.bat_second, 0.5)

    # Shared momentum (estimate if not provided)
    crr = (req.runs_scored / balls * 6) if balls > 0 else 0.0
    p_runs_e  = req.runs_scored // max(req.wickets_fallen, 1) if req.wickets_fallen else req.runs_scored
    l3r = req.last_3ov_runs  if req.last_3ov_runs  is not None else round(crr * 3)
    l3w = req.last_3ov_wkts if req.last_3ov_wkts is not None else 0
    par = req.partnership_runs  if req.partnership_runs  is not None else p_runs_e
    pab = req.partnership_balls if req.partnership_balls is not None else max(1, balls // max(req.wickets_fallen + 1, 1))
    max_par = req.max_partnership if req.max_partnership is not None else par
    bpct = req.boundary_pct  if req.boundary_pct  is not None else 0.25
    dpct = req.dot_ball_pct  if req.dot_ball_pct  is not None else 0.42
    pp_r = req.pp_runs    if req.pp_runs    is not None else (req.runs_scored    if balls <= 36 else 0)
    pp_w = req.pp_wickets if req.pp_wickets is not None else (req.wickets_fallen if balls <= 36 else 0)
    pp_run_rate = pp_r / 36 * 6 if balls > 36 else 0.0

    if req.current_innings == 1:
        expected_at = venue_avg * (balls / 120) if balls > 0 else 0
        projected   = phase_aware_projection(req.runs_scored, balls, crr, venue_avg)
        accel       = (l3r / 18 * 6 - crr) if balls >= 36 else 0.0
        feats = {
            "current_innings":     1,
            "innings_balls":       balls,
            "innings_balls_rem":   120 - balls,
            "innings_balls_pct":   balls / 120,
            "inn1_runs":           req.runs_scored,
            "inn1_wickets":        req.wickets_fallen,
            "inn1_crr":            crr,
            "inn1_projected":      projected,
            "inn1_vs_avg":         req.runs_scored - expected_at,
            "inn1_vs_avg_pct":     req.runs_scored / expected_at if expected_at > 0 else 1.0,
            "inn1_balls_pct":      balls / 120,
            "inn1_acceleration":   accel,
            "inn2_runs": 0, "inn2_wickets": 0, "inn2_crr": 0.0,
            "inn2_rrr": 0.0, "inn2_rrr_diff": 0.0, "inn2_run_rate_ratio": 0.0,
            "inn2_runs_needed": 0, "inn2_balls_rem": 0, "inn2_balls_pct": 0.0,
            "first_innings_wickets": 0,
            "target": 0, "target_vs_venue_avg": 0.0,
            "pp_req_rate": 0.0, "pp_rate_gap": 0.0,
        }
    else:
        target = req.target or 150
        inn1_runs_final = target - 1
        balls_rem   = max(0, total_balls - balls)
        runs_needed = max(0, target - req.runs_scored)
        rrr         = runs_needed / balls_rem * 6 if balls_rem > 0 else 99.0
        rrr_diff    = rrr - crr
        rr_ratio    = min(crr / rrr if rrr > 0 else 1.0, 3.0)
        inn1_wkts   = req.first_innings_wickets or 0
        pp_req_rate = (target - pp_r) / 84 * 6 if balls > 36 else 0.0
        pp_rate_gap = pp_run_rate - pp_req_rate if balls > 36 else 0.0
        feats = {
            "current_innings":     2,
            "innings_balls":       balls,
            "innings_balls_rem":   balls_rem,
            "innings_balls_pct":   balls / 120,
            "inn1_runs":           inn1_runs_final,
            "inn1_wickets":        inn1_wkts,
            "inn1_crr":            inn1_runs_final / 120 * 6,
            "inn1_projected":      0.0,
            "inn1_vs_avg":         inn1_runs_final - venue_avg,
            "inn1_vs_avg_pct":     inn1_runs_final / venue_avg if venue_avg > 0 else 1.0,
            "inn1_balls_pct":      1.0,
            "inn1_acceleration":   0.0,
            "inn2_runs":           req.runs_scored,
            "inn2_wickets":        req.wickets_fallen,
            "inn2_crr":            crr,
            "inn2_rrr":            rrr,
            "inn2_rrr_diff":       rrr_diff,
            "inn2_run_rate_ratio": rr_ratio,
            "inn2_runs_needed":    runs_needed,
            "inn2_balls_rem":      balls_rem,
            "inn2_balls_pct":      balls / 120,
            "first_innings_wickets": inn1_wkts,
            "target":              target,
            "target_vs_venue_avg": target / venue_avg if venue_avg > 0 else 1.0,
            "pp_req_rate":         pp_req_rate,
            "pp_rate_gap":         pp_rate_gap,
        }

    # Shared features (same for both innings)
    partnership_quality = par / max(venue_avg / 20, 1.0)
    feats.update({
        "partnership_runs":    par,
        "partnership_balls":   pab,
        "partnership_quality": partnership_quality,
        "max_partnership":     max_par,
        "last_3ov_runs":       l3r,
        "last_3ov_wkts":       l3w,
        "boundary_pct":        bpct,
        "dot_pct":             dpct,
        "is_pp":               int(balls <= 36),
        "pp_runs":             pp_r if balls > 36 else 0,
        "pp_wickets":          pp_w if balls > 36 else 0,
        "pp_run_rate":         pp_run_rate,
        "elo_diff":            elo_diff,
        "form_diff":           form_diff,
        "venue_avg":           venue_avg,
        "venue_bat_first_win_rate": vs.get("venue_bat_first_win_rate", 0.5),
        "venue_chase_win_rate":     vs.get("venue_chase_win_rate", 0.5),
        # Team-specific phase wicket win rate
        "team_phase_wkt_wr": _get_team_phase_wkt_wr(
            batting_team=req.bat_first if req.current_innings == 1 else req.bat_second,
            over_num=balls // 6,
            pp_wickets=pp_w if balls > 36 else req.wickets_fallen,
            role="bat_first" if req.current_innings == 1 else "bat_second",
        ),
    })

    X = pd.DataFrame([[feats.get(f, 0) for f in uni["features"]]], columns=uni["features"])
    X_s = pd.DataFrame(uni["scaler"].transform(X), columns=uni["features"])
    bat_first_prob = float(uni["model"].predict_proba(X_s)[0, 1])

    # Derived quantities for response
    if req.current_innings == 2:
        target    = req.target or 150
        balls_rem = max(0, total_balls - balls)
        runs_need = max(0, target - req.runs_scored)
        rrr_val   = runs_need / balls_rem * 6 if balls_rem > 0 else 99.0
        match_state = {
            "runs_scored": req.runs_scored, "wickets_fallen": req.wickets_fallen,
            "balls_bowled": balls, "balls_remaining": balls_rem,
            "target": target, "runs_needed": runs_need,
            "current_run_rate": round(crr, 2),
            "required_run_rate": round(rrr_val, 2),
        }
    else:
        projected = phase_aware_projection(req.runs_scored, balls, crr, venue_avg)
        match_state = {
            "runs_scored": req.runs_scored, "wickets_fallen": req.wickets_fallen,
            "balls_bowled": balls, "balls_remaining": 120 - balls,
            "current_run_rate": round(crr, 2),
            "projected_score": round(projected),
            "venue_avg": round(venue_avg),
        }

    return {
        "current_innings":          req.current_innings,
        "bat_first":                req.bat_first,
        "bat_second":               req.bat_second,
        "bat_first_win_probability":  round(bat_first_prob, 4),
        "bat_second_win_probability": round(1 - bat_first_prob, 4),
        "predicted_winner": req.bat_first if bat_first_prob >= 0.5 else req.bat_second,
        "confidence": "high" if abs(bat_first_prob - 0.5) > 0.2 else "medium" if abs(bat_first_prob - 0.5) > 0.1 else "low",
        "match_state": match_state,
    }


class PlayerScoresRequest(BaseModel):
    players: list[str]   # data_name format
    venue: str
    team: str = ""       # optional — used to fill default scores for unknowns

@app.post("/player-scores")
def get_player_scores(req: PlayerScoresRequest):
    """Return per-player venue-adjusted bat/bowl scores for a list of data_names."""
    # Get team defaults for unknown players
    ts = (team_strength_map.get((req.venue, req.team)) or
          team_strength_map.get((None, req.team))) if req.team else None
    default_bat  = float(ts["bat_strength"])  if ts and pd.notna(ts.get("bat_strength"))  else 45.0
    default_bowl = float(ts["bowl_strength"]) if ts and pd.notna(ts.get("bowl_strength")) else 40.0

    results = []
    for dn in req.players:
        bat, bowl = _blended_player_scores(dn, req.venue)
        results.append({
            "data_name": dn,
            "full_name": DATA_TO_FULL.get(dn, dn),
            "bat_score":  round(float(bat),  1) if not np.isnan(bat)  else round(default_bat,  1),
            "bowl_score": round(float(bowl), 1) if not np.isnan(bowl) else round(default_bowl, 1),
            "estimated":  bool(np.isnan(bat) or np.isnan(bowl)),  # flag: True = used team default
        })
    return {"players": results, "venue": req.venue}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("04_api:app", host="0.0.0.0", port=8000, reload=False)
