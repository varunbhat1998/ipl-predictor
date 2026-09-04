"""
10_post_toss_model.py - Post-toss prediction model (toss + XI + weather known)

Framing: predict P(batting_first_team_wins) given post-toss information.
Training data: 2008-2025 matches (1146 with deliveries).
Player scores: expanding-window per-match stats from deliveries.csv.
Weather: Open-Meteo archive API (cached to data/weather_cache.csv).
Target: >=80% accuracy at a confidence threshold.

Run:
    python 10_post_toss_model.py
"""

import pickle, warnings, time, os, re, sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import optuna

from model_classes import EnsemblePreMatchModel

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
Path("models").mkdir(exist_ok=True)

# =====================================================================
# VENUE COORDINATES (for weather lookups)
# =====================================================================
VENUE_COORDS = {
    "Wankhede Stadium, Mumbai":                         (18.9388,  72.8258),
    "M Chinnaswamy Stadium, Bengaluru":                 (12.9790,  77.5995),
    "MA Chidambaram Stadium, Chennai":                  (13.0629,  80.2792),
    "Eden Gardens, Kolkata":                            (22.5646,  88.3433),
    "Narendra Modi Stadium, Ahmedabad":                 (23.0900,  72.0830),
    "Rajiv Gandhi Intl Stadium, Hyderabad":             (17.4046,  78.5481),
    "PCA Stadium, Mohali":                              (30.6943,  76.8601),
    "Sawai Mansingh Stadium, Jaipur":                   (26.8869,  75.8063),
    "Ekana Stadium, Lucknow":                           (26.9034,  80.9450),
    "DY Patil Stadium, Mumbai":                         (19.0443,  73.0168),
    "Arun Jaitley Stadium, Delhi":                      (28.6376,  77.2209),
    "Holkar Stadium, Indore":                           (22.7196,  75.8577),
    "HPCA Stadium, Dharamsala":                         (31.8350,  76.9430),
    "Brabourne Stadium, Mumbai":                        (18.9390,  72.8260),
    "MCA Stadium, Pune":                                (18.6770,  73.8752),
    "Maharashtra Cricket Association Stadium":          (18.6770,  73.8752),
    "Subrata Roy Sahara Stadium, Pune":                 (18.6770,  73.8752),
    "Barsapara Cricket Stadium, Guwahati":              (26.1535,  91.7890),
    "JSCA International Stadium Complex":               (23.3335,  85.3210),
    "Saurashtra Cricket Association Stadium":           (22.2742,  70.7564),
    "Barabati Stadium, Cuttack":                        (20.4715,  85.8767),
    "Green Park":                                       (26.4499,  80.3319),
    "Nehru Stadium":                                    (13.0750,  80.2792),
    "Shaheed Veer Narayan Singh International Stadium": (21.2514,  81.6296),
    "Vidarbha Cricket Association Stadium, Jamtha":     (21.0612,  79.0685),
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": (17.7215, 83.2175),
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": (17.7215, 83.2175),
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": (30.7908, 76.7337),
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": (30.7908, 76.7337),
    # UAE/SA venues
    "Dubai International Cricket Stadium":              (25.0478,  55.2250),
    "Sharjah Cricket Stadium":                          (25.3373,  55.4210),
    "Sheikh Zayed Stadium, Abu Dhabi":                  (24.4539,  54.6064),
    "Newlands":                                         (-33.9270, 18.4377),
    "St George's Park":                                 (-33.9638, 25.5994),
    "SuperSport Park":                                  (-25.7467, 28.2122),
    "Kingsmead":                                        (-29.8579, 31.0292),
    "New Wanderers Stadium":                            (-26.1653, 28.0566),
    "OUTsurance Oval":                                  (-29.1073, 26.2149),
    "De Beers Diamond Oval":                            (-28.7300, 24.7600),
    "Buffalo Park":                                     (-32.8711, 27.6322),
}

def _fuzzy_venue_coords(venue):
    """Match venue string to VENUE_COORDS key."""
    if venue in VENUE_COORDS:
        return VENUE_COORDS[venue]
    vl = venue.lower()
    for k, v in VENUE_COORDS.items():
        if k.lower() in vl or vl in k.lower():
            return v
    # Keyword search
    for word in venue.replace(",", "").split():
        if len(word) > 4:
            for k, v in VENUE_COORDS.items():
                if word.lower() in k.lower():
                    return v
    return None


# =====================================================================
# PHASE 1: FETCH & CACHE HISTORICAL WEATHER
# =====================================================================
print("=" * 62)
print("PHASE 1: Historical weather data")
print("=" * 62)

WEATHER_CACHE_V2 = "data/weather_cache_v2.csv"

def fetch_weather_for_matches(match_df):
    """Fetch weather for each match at the correct hour (afternoon or evening)."""
    if os.path.exists(WEATHER_CACHE_V2):
        cache = pd.read_csv(WEATHER_CACHE_V2)
        cached_keys = set(cache['file_id'].astype(str))
        print(f"  Loaded {len(cache)} cached weather records (v2)")
    else:
        cache = pd.DataFrame()
        cached_keys = set()

    needed = match_df[['file_id', 'date', 'venue', 'match_hour']].drop_duplicates()
    needed['file_id'] = needed['file_id'].astype(str)
    to_fetch = needed[~needed['file_id'].isin(cached_keys)]

    if len(to_fetch) == 0:
        print("  All weather data cached.")
        return cache

    print(f"  Need to fetch weather for {len(to_fetch)} matches...")
    new_rows = []
    for idx, row in to_fetch.iterrows():
        coords = _fuzzy_venue_coords(row['venue'])
        if coords is None:
            continue
        lat, lon = coords
        date_str = str(row['date'])[:10]
        target_hour = int(row['match_hour'])
        try:
            url = (f"https://archive-api.open-meteo.com/v1/archive"
                   f"?latitude={lat}&longitude={lon}"
                   f"&start_date={date_str}&end_date={date_str}"
                   f"&hourly=temperature_2m,relative_humidity_2m,cloud_cover,"
                   f"wind_speed_10m,precipitation"
                   f"&timezone=Asia%2FKolkata")
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            h = r.json().get('hourly', {})
            times = h.get('time', [])
            temps = h.get('temperature_2m', [])
            hums = h.get('relative_humidity_2m', [])
            clouds = h.get('cloud_cover', [])
            winds = h.get('wind_speed_10m', [])
            precips = h.get('precipitation', [])

            for i, t in enumerate(times):
                hr = int(t.split('T')[1].split(':')[0])
                if hr == target_hour:
                    new_rows.append({
                        'file_id': row['file_id'],
                        'date': date_str,
                        'venue': row['venue'],
                        'weather_hour': target_hour,
                        'temperature': temps[i] if i < len(temps) else None,
                        'humidity': hums[i] if i < len(hums) else None,
                        'cloud_cover': clouds[i] if i < len(clouds) else None,
                        'wind_speed': winds[i] if i < len(winds) else None,
                        'precipitation': precips[i] if i < len(precips) else None,
                    })
                    break

        except Exception:
            pass

        if len(new_rows) % 50 == 0 and len(new_rows) > 0:
            print(f"    ... fetched {len(new_rows)}/{len(to_fetch)}")
            time.sleep(1)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True) if len(cache) else new_df
        cache.to_csv(WEATHER_CACHE_V2, index=False)
        print(f"  Fetched {len(new_rows)} new weather records. Total cached: {len(cache)}")
    else:
        print("  No new weather records fetched.")

    return cache


# =====================================================================
# PHASE 2: EXTRACT PLAYING XI PER MATCH FROM DELIVERIES
# =====================================================================
print("\n" + "=" * 62)
print("PHASE 2: Extract playing XI per match")
print("=" * 62)

del_df = pd.read_csv("data/deliveries.csv")
del_df['file_id'] = del_df['file_id'].astype(str)
del_df['date'] = pd.to_datetime(del_df['date'])

# Normalize team names in deliveries to match match_features conventions
TEAM_NAME_MAP = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils":            "Delhi Capitals",
    "Deccan Chargers":             "Deccan Chargers",  # no mapping, will be dropped
    "Rising Pune Supergiants":     "Rising Pune Supergiant",
    "Punjab Kings":                "Kings XI Punjab",
}
for col in ['batting_team', 'bowling_team', 'winner']:
    del_df[col] = del_df[col].replace(TEAM_NAME_MAP)

mf = pd.read_csv("data/match_features.csv")
mf['file_id'] = mf['file_id'].astype(str)
mf['date'] = pd.to_datetime(mf['date'])
mf = mf[mf['team1_won'].notna()].copy()

# Only keep matches with deliveries
valid_fids = set(del_df['file_id'].unique()) & set(mf['file_id'].unique())
mf = mf[mf['file_id'].isin(valid_fids)].copy()
print(f"Matches with deliveries + features: {len(mf)}")

# Determine batting first team from deliveries
inn1_teams = (del_df[del_df['innings'] == 1]
              .groupby('file_id')['batting_team'].first()
              .reset_index()
              .rename(columns={'batting_team': 'bat_first_team'}))

mf = mf.merge(inn1_teams, on='file_id', how='left')

# Determine bat_second_team
inn2_teams = (del_df[del_df['innings'] == 2]
              .groupby('file_id')['batting_team'].first()
              .reset_index()
              .rename(columns={'batting_team': 'bat_second_team'}))
mf = mf.merge(inn2_teams, on='file_id', how='left')

# bat_first_won: compare with BOTH deliveries winner and features winner
# Use deliveries winner (already name-normalized)
del_winners = del_df.groupby('file_id')['winner'].first().reset_index().rename(columns={'winner': 'del_winner'})
mf = mf.merge(del_winners, on='file_id', how='left')
mf['bat_first_won'] = (mf['bat_first_team'] == mf['del_winner']).astype(int)
print(f"Bat-first win rate: {mf['bat_first_won'].mean():.3f}")

# Extract XI per match
def extract_xi(fid):
    """Get playing XI for each team from deliveries."""
    match = del_df[del_df['file_id'] == fid]
    inn1 = match[match['innings'] == 1]
    inn2 = match[match['innings'] == 2]

    bf_team = inn1['batting_team'].iloc[0] if len(inn1) else None
    bs_team = inn1['bowling_team'].iloc[0] if len(inn1) else None

    # Batting first team: batted in inn1, bowled in inn2
    bf_players = set()
    bf_players.update(inn1['batter'].unique())
    bf_players.update(inn1['non_striker'].unique())
    if len(inn2):
        bf_players.update(inn2['bowler'].unique())
    out1 = inn1.loc[inn1['player_out'].notna(), 'player_out'].unique()
    bf_players.update(out1)

    # Batting second team: bowled in inn1, batted in inn2
    bs_players = set()
    bs_players.update(inn1['bowler'].unique())
    if len(inn2):
        bs_players.update(inn2['batter'].unique())
        bs_players.update(inn2['non_striker'].unique())
        out2 = inn2.loc[inn2['player_out'].notna(), 'player_out'].unique()
        bs_players.update(out2)

    # Remove NaN / empty strings
    bf_players = [p for p in bf_players if pd.notna(p) and p.strip()]
    bs_players = [p for p in bs_players if pd.notna(p) and p.strip()]

    return bf_team, bs_team, bf_players, bs_players

print("Extracting XI for all matches...")
xi_data = {}
for fid in mf['file_id'].unique():
    xi_data[fid] = extract_xi(fid)

# Verify XI counts
xi_lens = [(len(xi_data[f][2]), len(xi_data[f][3])) for f in xi_data]
avg_bf = np.mean([x[0] for x in xi_lens])
avg_bs = np.mean([x[1] for x in xi_lens])
print(f"  Avg players detected: bat_first={avg_bf:.1f}, bat_second={avg_bs:.1f}")


# =====================================================================
# PHASE 3: COMPUTE EXPANDING-WINDOW PLAYER SCORES
# =====================================================================
print("\n" + "=" * 62)
print("PHASE 3: Expanding-window player scores")
print("=" * 62)

# Sort deliveries by date for expanding windows
del_sorted = del_df.sort_values(['date', 'file_id', 'innings', 'over', 'ball_in_over']).copy()

# Batting: per-innings stats (one row per player per innings)
# Filter to legal deliveries for balls faced (exclude wides)
bat_legal = del_sorted[del_sorted['is_wide'] == 0].copy()
bat_inns = (bat_legal.groupby(['batter', 'file_id', 'date'])
            .agg(
                runs=('runs_batter', 'sum'),
                balls=('runs_batter', 'count'),
                fours=('runs_batter', lambda x: (x == 4).sum()),
                sixes=('runs_batter', lambda x: (x == 6).sum()),
            )
            .reset_index()
            .sort_values(['batter', 'date', 'file_id']))

# Cumulative stats per player (BEFORE current match = shift by 1)
bat_inns['cum_runs'] = bat_inns.groupby('batter')['runs'].cumsum()
bat_inns['cum_balls'] = bat_inns.groupby('batter')['balls'].cumsum()
bat_inns['cum_inns'] = bat_inns.groupby('batter').cumcount() + 1

# Shift to get stats BEFORE this match
bat_inns['prev_runs'] = bat_inns.groupby('batter')['cum_runs'].shift(1, fill_value=0)
bat_inns['prev_balls'] = bat_inns.groupby('batter')['cum_balls'].shift(1, fill_value=0)
bat_inns['prev_inns'] = bat_inns.groupby('batter')['cum_inns'].shift(1, fill_value=0)

# Career avg and SR at match time
bat_inns['career_avg'] = np.where(
    bat_inns['prev_inns'] > 0,
    bat_inns['prev_runs'] / bat_inns['prev_inns'].clip(1),
    0
)
bat_inns['career_sr'] = np.where(
    bat_inns['prev_balls'] > 0,
    bat_inns['prev_runs'] / bat_inns['prev_balls'].clip(1) * 100,
    0
)

# Form: rolling last 5 innings BEFORE this match
bat_inns['form5_avg'] = (bat_inns.groupby('batter')['runs']
                         .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
bat_inns['form5_sr_num'] = (bat_inns.groupby('batter')['runs']
                            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()))
bat_inns['form5_sr_den'] = (bat_inns.groupby('batter')['balls']
                            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()))
bat_inns['form5_sr'] = np.where(
    bat_inns['form5_sr_den'] > 0,
    bat_inns['form5_sr_num'] / bat_inns['form5_sr_den'] * 100,
    0
)

# Current-season stats BEFORE this match (expanding window within season)
bat_inns_s = bat_inns.copy()
bat_inns_s['season'] = bat_inns_s['date'].dt.year
bat_inns_s = bat_inns_s.sort_values(['batter', 'season', 'date', 'file_id'])
bat_inns_s['s_cum_runs']  = bat_inns_s.groupby(['batter', 'season'])['runs'].cumsum()
bat_inns_s['s_cum_balls'] = bat_inns_s.groupby(['batter', 'season'])['balls'].cumsum()
bat_inns_s['s_cum_inns']  = bat_inns_s.groupby(['batter', 'season']).cumcount() + 1
bat_inns_s['s_prev_runs']  = bat_inns_s.groupby(['batter', 'season'])['s_cum_runs'].shift(1, fill_value=0)
bat_inns_s['s_prev_balls'] = bat_inns_s.groupby(['batter', 'season'])['s_cum_balls'].shift(1, fill_value=0)
bat_inns_s['s_prev_inns']  = bat_inns_s.groupby(['batter', 'season'])['s_cum_inns'].shift(1, fill_value=0)
bat_inns_s['season_avg'] = np.where(
    bat_inns_s['s_prev_inns'] >= 2,
    bat_inns_s['s_prev_runs'] / bat_inns_s['s_prev_inns'].clip(1), np.nan)
bat_inns_s['season_sr'] = np.where(
    bat_inns_s['s_prev_balls'] >= 10,
    bat_inns_s['s_prev_runs'] / bat_inns_s['s_prev_balls'].clip(1) * 100, np.nan)
# Merge season stats back into bat_inns
bat_inns = bat_inns.merge(
    bat_inns_s[['batter', 'file_id', 'season_avg', 'season_sr']],
    on=['batter', 'file_id'], how='left')

# Compute bat_score: 0.35 career + 0.30 form5 + 0.35 current-season
# component = (avg/40)*0.5 + (sr/150)*0.5
def compute_bat_score(row):
    if row['prev_inns'] < 1:
        return np.nan
    career = (row['career_avg'] / 40) * 0.5 + (row['career_sr'] / 150) * 0.5
    form_avg = row['form5_avg'] if pd.notna(row['form5_avg']) else row['career_avg']
    form_sr  = row['form5_sr']  if pd.notna(row['form5_sr']) and row['form5_sr'] > 0 else row['career_sr']
    form = (form_avg / 40) * 0.5 + (form_sr / 150) * 0.5
    # Current-season component (if available)
    if pd.notna(row.get('season_avg')) and pd.notna(row.get('season_sr')):
        season = (row['season_avg'] / 40) * 0.5 + (row['season_sr'] / 150) * 0.5
        score = (0.35 * career + 0.30 * form + 0.35 * season) * 100
    else:
        score = (0.50 * career + 0.50 * form) * 100
    return max(0, score)

bat_inns['bat_score'] = bat_inns.apply(compute_bat_score, axis=1)

# Build lookup: player -> file_id -> bat_score
bat_score_lookup = {}
for _, row in bat_inns[['batter', 'file_id', 'bat_score']].dropna(subset=['bat_score']).iterrows():
    bat_score_lookup[(row['batter'], row['file_id'])] = row['bat_score']

print(f"  Batting scores computed: {len(bat_score_lookup)} player-match pairs")

# Bowling: per-innings stats
bowl_legal = del_sorted.copy()  # include wides/noballs for bowling
bowl_inns = (bowl_legal.groupby(['bowler', 'file_id', 'date'])
             .agg(
                 runs_conceded=('runs_total', 'sum'),
                 balls_bowled=('is_wide', lambda x: (x == 0).sum()),  # legal deliveries
                 wickets=('is_wicket', 'sum'),
             )
             .reset_index()
             .sort_values(['bowler', 'date', 'file_id']))

bowl_inns['cum_runs_c'] = bowl_inns.groupby('bowler')['runs_conceded'].cumsum()
bowl_inns['cum_balls_b'] = bowl_inns.groupby('bowler')['balls_bowled'].cumsum()
bowl_inns['cum_wkts'] = bowl_inns.groupby('bowler')['wickets'].cumsum()
bowl_inns['cum_bowl_inns'] = bowl_inns.groupby('bowler').cumcount() + 1

# Shift
bowl_inns['prev_runs_c'] = bowl_inns.groupby('bowler')['cum_runs_c'].shift(1, fill_value=0)
bowl_inns['prev_balls_b'] = bowl_inns.groupby('bowler')['cum_balls_b'].shift(1, fill_value=0)
bowl_inns['prev_wkts'] = bowl_inns.groupby('bowler')['cum_wkts'].shift(1, fill_value=0)
bowl_inns['prev_bowl_inns'] = bowl_inns.groupby('bowler')['cum_bowl_inns'].shift(1, fill_value=0)

# Career economy and wicket rate
bowl_inns['career_econ'] = np.where(
    bowl_inns['prev_balls_b'] > 0,
    bowl_inns['prev_runs_c'] / (bowl_inns['prev_balls_b'] / 6),
    12.0  # default high economy for unknown bowlers
)
bowl_inns['career_wkt_rate'] = np.where(
    bowl_inns['prev_bowl_inns'] > 0,
    bowl_inns['prev_wkts'] / bowl_inns['prev_bowl_inns'],
    0
)

# Form: rolling 5 innings
bowl_inns['form5_econ_runs'] = (bowl_inns.groupby('bowler')['runs_conceded']
                                .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()))
bowl_inns['form5_econ_balls'] = (bowl_inns.groupby('bowler')['balls_bowled']
                                 .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()))
bowl_inns['form5_econ'] = np.where(
    bowl_inns['form5_econ_balls'] > 0,
    bowl_inns['form5_econ_runs'] / (bowl_inns['form5_econ_balls'] / 6),
    bowl_inns['career_econ']
)
bowl_inns['form5_wkts'] = (bowl_inns.groupby('bowler')['wickets']
                           .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

# Current-season bowling stats BEFORE this match
bowl_inns_s = bowl_inns.copy()
bowl_inns_s['season'] = bowl_inns_s['date'].dt.year
bowl_inns_s = bowl_inns_s.sort_values(['bowler', 'season', 'date', 'file_id'])
bowl_inns_s['s_cum_rc']   = bowl_inns_s.groupby(['bowler','season'])['runs_conceded'].cumsum()
bowl_inns_s['s_cum_bb']   = bowl_inns_s.groupby(['bowler','season'])['balls_bowled'].cumsum()
bowl_inns_s['s_cum_wk']   = bowl_inns_s.groupby(['bowler','season'])['wickets'].cumsum()
bowl_inns_s['s_cum_inns'] = bowl_inns_s.groupby(['bowler','season']).cumcount() + 1
bowl_inns_s['s_prev_rc']   = bowl_inns_s.groupby(['bowler','season'])['s_cum_rc'].shift(1, fill_value=0)
bowl_inns_s['s_prev_bb']   = bowl_inns_s.groupby(['bowler','season'])['s_cum_bb'].shift(1, fill_value=0)
bowl_inns_s['s_prev_wk']   = bowl_inns_s.groupby(['bowler','season'])['s_cum_wk'].shift(1, fill_value=0)
bowl_inns_s['s_prev_inns'] = bowl_inns_s.groupby(['bowler','season'])['s_cum_inns'].shift(1, fill_value=0)
bowl_inns_s['season_econ'] = np.where(
    bowl_inns_s['s_prev_bb'] >= 6,
    bowl_inns_s['s_prev_rc'] / (bowl_inns_s['s_prev_bb'] / 6), np.nan)
bowl_inns_s['season_wkt_rate'] = np.where(
    bowl_inns_s['s_prev_inns'] >= 2,
    bowl_inns_s['s_prev_wk'] / bowl_inns_s['s_prev_inns'].clip(1), np.nan)
bowl_inns = bowl_inns.merge(
    bowl_inns_s[['bowler', 'file_id', 'season_econ', 'season_wkt_rate']],
    on=['bowler', 'file_id'], how='left')

# Compute bowl_score: 0.35 career + 0.30 form5 + 0.35 current-season
def compute_bowl_score(row):
    # Non-bowlers: fewer than 60 career balls → NaN (excluded from lookup,
    # they receive DEFAULT_BOWL which top-4 strategy naturally deprioritises
    # once real bowlers score above default with the /4 denominator)
    if row['prev_balls_b'] < 60:
        return np.nan
    # Career component — /4 gives realistic spread for T20 (6–10 RPO range)
    c_econ = min(1.0, max(0, (10 - row['career_econ']) / 4))
    c_wkt  = min(row['career_wkt_rate'] / 2, 1)
    career = c_econ * 0.5 + c_wkt * 0.5
    # Form component
    f_econ = min(1.0, max(0, (10 - row['form5_econ']) / 4)) if pd.notna(row['form5_econ']) else c_econ
    f_wkt  = min(row['form5_wkts'] / 2, 1) if pd.notna(row['form5_wkts']) else c_wkt
    form = f_econ * 0.5 + f_wkt * 0.5
    # Current-season component
    if pd.notna(row.get('season_econ')) and pd.notna(row.get('season_wkt_rate')):
        s_econ = min(1.0, max(0, (10 - row['season_econ']) / 4))
        s_wkt  = min(row['season_wkt_rate'] / 2, 1)
        season = s_econ * 0.5 + s_wkt * 0.5
        score = (0.35 * career + 0.30 * form + 0.35 * season) * 100
    else:
        score = (0.50 * career + 0.50 * form) * 100
    return max(0, score)

bowl_inns['bowl_score'] = bowl_inns.apply(compute_bowl_score, axis=1)

bowl_score_lookup = {}
for _, row in bowl_inns[['bowler', 'file_id', 'bowl_score']].dropna(subset=['bowl_score']).iterrows():
    bowl_score_lookup[(row['bowler'], row['file_id'])] = row['bowl_score']

print(f"  Bowling scores computed: {len(bowl_score_lookup)} player-match pairs")

# ---- Venue-specific player scores ----
# For venue-specific: compute career stats at this venue BEFORE this match
print("  Computing venue-specific batting scores...")

# Add venue info to batting innings
bat_venue_info = del_sorted[['file_id', 'venue']].drop_duplicates()
bat_inns_v = bat_inns.merge(bat_venue_info, on='file_id', how='left')

# Venue cumulative stats
bat_inns_v = bat_inns_v.sort_values(['batter', 'venue', 'date', 'file_id'])
bat_inns_v['v_cum_runs'] = bat_inns_v.groupby(['batter', 'venue'])['runs'].cumsum()
bat_inns_v['v_cum_balls'] = bat_inns_v.groupby(['batter', 'venue'])['balls'].cumsum()
bat_inns_v['v_cum_inns'] = bat_inns_v.groupby(['batter', 'venue']).cumcount() + 1

bat_inns_v['v_prev_runs'] = bat_inns_v.groupby(['batter', 'venue'])['v_cum_runs'].shift(1, fill_value=0)
bat_inns_v['v_prev_balls'] = bat_inns_v.groupby(['batter', 'venue'])['v_cum_balls'].shift(1, fill_value=0)
bat_inns_v['v_prev_inns'] = bat_inns_v.groupby(['batter', 'venue'])['v_cum_inns'].shift(1, fill_value=0)

bat_inns_v['v_career_avg'] = np.where(bat_inns_v['v_prev_inns'] >= 2,
    bat_inns_v['v_prev_runs'] / bat_inns_v['v_prev_inns'].clip(1), np.nan)
bat_inns_v['v_career_sr'] = np.where(bat_inns_v['v_prev_balls'] >= 10,
    bat_inns_v['v_prev_runs'] / bat_inns_v['v_prev_balls'].clip(1) * 100, np.nan)

def compute_venue_bat_score(row):
    if pd.isna(row['v_career_avg']) or pd.isna(row['v_career_sr']):
        return np.nan
    return ((row['v_career_avg'] / 40) * 0.5 + (row['v_career_sr'] / 150) * 0.5) * 100

bat_inns_v['venue_bat_score'] = bat_inns_v.apply(compute_venue_bat_score, axis=1)

venue_bat_lookup = {}
for _, row in bat_inns_v[['batter', 'file_id', 'venue_bat_score']].dropna(subset=['venue_bat_score']).iterrows():
    venue_bat_lookup[(row['batter'], row['file_id'])] = row['venue_bat_score']

print(f"    Venue batting scores: {len(venue_bat_lookup)} player-match-venue pairs")


# =====================================================================
# PHASE 3b: COMPUTE XI-LEVEL FEATURES PER MATCH
# =====================================================================
print("\nComputing XI-level features per match...")

# Dynamic defaults: median of computed scores (50th percentile = league average)
_bat_scores_all  = bat_inns['bat_score'].dropna()
DEFAULT_BAT  = float(_bat_scores_all.median())  if len(_bat_scores_all)  > 0 else 50.0
_bowl_scores_all = bowl_inns['bowl_score'].dropna()
DEFAULT_BOWL = float(_bowl_scores_all.median()) if len(_bowl_scores_all) > 0 else 40.0
print(f"  Dynamic defaults: bat={DEFAULT_BAT:.1f}, bowl={DEFAULT_BOWL:.1f}")

def xi_features_for_match(fid, players, team_label):
    """Compute aggregated XI batting/bowling strength for a team."""
    bat_scores = []
    bowl_scores = []

    for p in players:
        # Batting: blend career + venue (0.6/0.4)
        career_bat = bat_score_lookup.get((p, fid), np.nan)
        venue_bat = venue_bat_lookup.get((p, fid), np.nan)

        if not np.isnan(career_bat) and not np.isnan(venue_bat):
            blended_bat = 0.6 * career_bat + 0.4 * venue_bat
        elif not np.isnan(venue_bat):
            blended_bat = venue_bat
        elif not np.isnan(career_bat):
            blended_bat = career_bat
        else:
            blended_bat = DEFAULT_BAT

        # Bowling
        career_bowl = bowl_score_lookup.get((p, fid), np.nan)
        if np.isnan(career_bowl):
            career_bowl = DEFAULT_BOWL

        bat_scores.append(blended_bat)
        bowl_scores.append(career_bowl)

    if not bat_scores:
        return DEFAULT_BAT, DEFAULT_BOWL, 0, 0, 0

    bat_sorted = sorted(bat_scores, reverse=True)
    bowl_sorted = sorted(bowl_scores, reverse=True)

    top6_bat = np.mean(bat_sorted[:min(6, len(bat_sorted))])
    top4_bowl = np.mean(bowl_sorted[:min(4, len(bowl_sorted))])
    bat_depth = np.std(bat_sorted) if len(bat_sorted) > 1 else 0
    max_bat = bat_sorted[0] if bat_sorted else DEFAULT_BAT
    n_players = len(players)

    return top6_bat, top4_bowl, bat_depth, max_bat, n_players

# Compute for all matches
xi_features = []
for _, row in mf.iterrows():
    fid = row['file_id']
    xi = xi_data.get(fid)
    if xi is None:
        continue

    bf_team, bs_team, bf_players, bs_players = xi

    bf_bat, bf_bowl, bf_depth, bf_max, bf_n = xi_features_for_match(fid, bf_players, bf_team)
    bs_bat, bs_bowl, bs_depth, bs_max, bs_n = xi_features_for_match(fid, bs_players, bs_team)

    xi_features.append({
        'file_id': fid,
        'bf_xi_bat': bf_bat,
        'bf_xi_bowl': bf_bowl,
        'bf_xi_depth': bf_depth,
        'bf_xi_max_bat': bf_max,
        'bf_xi_n': bf_n,
        'bs_xi_bat': bs_bat,
        'bs_xi_bowl': bs_bowl,
        'bs_xi_depth': bs_depth,
        'bs_xi_max_bat': bs_max,
        'bs_xi_n': bs_n,
        'xi_bat_diff': bf_bat - bs_bat,
        'xi_bowl_diff': bf_bowl - bs_bowl,
    })

xi_df = pd.DataFrame(xi_features)
print(f"  XI features computed for {len(xi_df)} matches")
print(f"  bf_xi_bat mean={xi_df['bf_xi_bat'].mean():.1f}, bs_xi_bat mean={xi_df['bs_xi_bat'].mean():.1f}")

# ── Phase 3b2: Impact Player features (IPL 2023+) ─────────────────────
# Load the impact player log built by 15_impact_player_analysis.py.
# For each match: the 12th player in each team's list = nominated impact player.
# We score them on bowling (the dominant use case: 96% used, always as bowler).
# For pre-2023 matches, impact_bowl features default to 0 (neutral).
print("\n  Loading impact player features...")
IMPACT_LOG_PATH = "data/impact_player_log.csv"

def _get_impact_bowl_score(player, fid):
    """Return bowl_score for this player at this match (expanding-window, no leakage)."""
    if not player or pd.isna(player):
        return DEFAULT_BOWL
    score = bowl_score_lookup.get((player, fid), np.nan)
    return score if not np.isnan(score) else DEFAULT_BOWL

# Extract nominated impact player from matches.csv (12th in list)
matches_raw = pd.read_csv("data/matches.csv")
matches_raw["file_id"] = matches_raw["file_id"].astype(str)
matches_raw["date"] = pd.to_datetime(matches_raw["date"])

# Build map: file_id -> (team1_impact, team2_impact)
impact_nominee_map = {}  # file_id -> {team: impact_player}
for _, mr in matches_raw.iterrows():
    fid = str(mr["file_id"])
    t1_pl = str(mr.get("team1_players","")).split("|")
    t2_pl = str(mr.get("team2_players","")).split("|")
    imp = {}
    if len(t1_pl) >= 12:
        imp[mr["team1"]] = t1_pl[11].strip()
    if len(t2_pl) >= 12:
        imp[mr["team2"]] = t2_pl[11].strip()
    impact_nominee_map[fid] = imp

# Add impact bowl scores to xi_df
impact_features = []
for _, row in mf.iterrows():
    fid = row["file_id"]
    xi  = xi_data.get(fid)
    if xi is None:
        impact_features.append({"file_id": fid, "bf_impact_bowl": DEFAULT_BOWL,
                                  "bs_impact_bowl": DEFAULT_BOWL, "impact_bowl_diff": 0.0})
        continue
    bf_team, bs_team, _, _ = xi
    nominees = impact_nominee_map.get(fid, {})
    bf_imp = nominees.get(bf_team) or nominees.get(next(iter(nominees), ""), None)
    bs_imp = nominees.get(bs_team) or None
    # Find the right team from nominees dict
    bf_imp = nominees.get(bf_team, None)
    bs_imp = nominees.get(bs_team, None)
    bf_score = _get_impact_bowl_score(bf_imp, fid)
    bs_score = _get_impact_bowl_score(bs_imp, fid)
    impact_features.append({
        "file_id":          fid,
        "bf_impact_bowl":   bf_score,
        "bs_impact_bowl":   bs_score,
        "impact_bowl_diff": bf_score - bs_score,
    })

impact_df = pd.DataFrame(impact_features)
xi_df = xi_df.merge(impact_df, on="file_id", how="left")
xi_df["bf_impact_bowl"]   = xi_df["bf_impact_bowl"].fillna(DEFAULT_BOWL)
xi_df["bs_impact_bowl"]   = xi_df["bs_impact_bowl"].fillna(DEFAULT_BOWL)
xi_df["impact_bowl_diff"] = xi_df["impact_bowl_diff"].fillna(0.0)
print(f"  Impact bowl: bf_mean={xi_df['bf_impact_bowl'].mean():.1f}, "
      f"bs_mean={xi_df['bs_impact_bowl'].mean():.1f}")

# ── Phase 3c: H2H matchup features ────────────────────────────────────
# Load pre-computed expanding-window batter vs bowler matchup advantages.
# Generated by 13_h2h_matchups.py — run that script first if the file doesn't exist.
H2H_PATH = "data/h2h_team_matchup.csv"
if Path(H2H_PATH).exists():
    print(f"\n  Loading H2H team matchup features from {H2H_PATH}...")
    h2h_df = pd.read_csv(H2H_PATH)
    h2h_df["file_id"] = h2h_df["file_id"].astype(str)
    h2h_df = h2h_df[["file_id", "matchup_advantage_bf", "matchup_advantage_diff"]].copy()
    xi_df = xi_df.merge(h2h_df, on="file_id", how="left")
    # Fill missing H2H values with neutral (0.5 / 0.0)
    xi_df["matchup_advantage_bf"]   = xi_df["matchup_advantage_bf"].fillna(0.5)
    xi_df["matchup_advantage_diff"] = xi_df["matchup_advantage_diff"].fillna(0.0)
    print(f"  H2H coverage: {h2h_df['matchup_advantage_bf'].notna().sum()} / {len(xi_df)} matches")
    print(f"  matchup_advantage_bf mean={xi_df['matchup_advantage_bf'].mean():.3f}, "
          f"std={xi_df['matchup_advantage_bf'].std():.3f}")
else:
    print(f"\n  WARNING: {H2H_PATH} not found — H2H features will be missing.")
    print("  Run 13_h2h_matchups.py first to generate this file.")
    xi_df["matchup_advantage_bf"]   = 0.5
    xi_df["matchup_advantage_diff"] = 0.0


# =====================================================================
# PHASE 4: FETCH WEATHER
# =====================================================================
print("\n" + "=" * 62)
print("PHASE 4: Fetching historical weather")
print("=" * 62)

# Determine afternoon vs evening: if 2+ matches on same date, first is afternoon (15), rest evening (19)
# Single-match days are evening (19)
mf['date_str'] = mf['date'].dt.strftime('%Y-%m-%d')
date_counts = mf.groupby('date_str')['file_id'].transform('count')
date_ranks = mf.groupby('date_str')['file_id'].transform('rank', method='first')
mf['is_evening'] = np.where((date_counts >= 2) & (date_ranks == 1), 0, 1)
mf['match_hour'] = np.where(mf['is_evening'] == 1, 19, 15)

print(f"  Evening matches: {mf['is_evening'].sum()}, Afternoon: {(1 - mf['is_evening']).sum()}")

weather_input = mf[['file_id', 'date_str', 'venue', 'match_hour']].rename(columns={'date_str': 'date'})
weather_df = fetch_weather_for_matches(weather_input)

if len(weather_df):
    weather_df['file_id'] = weather_df['file_id'].astype(str)
    # Deduplicate (keep first per file_id)
    weather_df = weather_df.drop_duplicates(subset='file_id', keep='first')
    print(f"  Weather records available: {len(weather_df)}")
else:
    weather_df = pd.DataFrame(columns=['file_id', 'temperature', 'humidity', 'cloud_cover', 'wind_speed', 'precipitation'])

# Compute dew and heat factors from weather
# Dew factor uses is_evening from match data (already in mf), not from weather_hour
if len(weather_df):
    weather_df['heat_factor'] = np.where(
        weather_df['temperature'].fillna(30) >= 35, 1, 0
    )
else:
    weather_df['heat_factor'] = 0


# =====================================================================
# PHASE 5: BUILD FEATURE MATRIX
# =====================================================================
print("\n" + "=" * 62)
print("PHASE 5: Build feature matrix")
print("=" * 62)

# Merge everything onto mf
df = mf.merge(xi_df, on='file_id', how='left')
weather_cols = ['file_id', 'temperature', 'humidity', 'cloud_cover', 'wind_speed', 'heat_factor']
weather_cols = [c for c in weather_cols if c in weather_df.columns]
if weather_cols:
    df = df.merge(weather_df[weather_cols], on='file_id', how='left')

# Compute dew_factor from humidity + is_evening (is_evening already in df from Phase 4)
df['humidity'] = df['humidity'].fillna(df['humidity'].median() if df['humidity'].notna().any() else 55)
df['dew_factor'] = np.where(
    (df['humidity'] >= 65) & (df['is_evening'] == 1),
    np.clip((df['humidity'] - 65) / 35, 0, 1),
    0
)

# Re-orient features: batting_first perspective
# Need ELO/form for batting-first and batting-second teams
# In match_features.csv: team1 and team2 columns
# Need to map team1/team2 -> bat_first/bat_second

df['bf_is_team1'] = (df['bat_first_team'] == df['team1']).astype(int)

# ELO features
df['bf_elo'] = np.where(df['bf_is_team1'] == 1, df['team1_elo'], df['team2_elo'])
df['bs_elo'] = np.where(df['bf_is_team1'] == 1, df['team2_elo'], df['team1_elo'])
df['elo_diff_bf'] = df['bf_elo'] - df['bs_elo']

# Form features
df['bf_form'] = np.where(df['bf_is_team1'] == 1, df['team1_form'], df['team2_form'])
df['bs_form'] = np.where(df['bf_is_team1'] == 1, df['team2_form'], df['team1_form'])
df['form_diff_bf'] = df['bf_form'] - df['bs_form']

# Short-window form
for w in ['form_3', 'form_10']:
    df[f'bf_{w}'] = np.where(df['bf_is_team1'] == 1, df[f'team1_{w}'], df[f'team2_{w}'])
    df[f'bs_{w}'] = np.where(df['bf_is_team1'] == 1, df[f'team2_{w}'], df[f'team1_{w}'])
    df[f'{w}_diff_bf'] = df[f'bf_{w}'] - df[f'bs_{w}']

# H2H
df['h2h_bf'] = np.where(df['bf_is_team1'] == 1,
                         df['h2h_win_rate_team1'],
                         1 - df['h2h_win_rate_team1'])

# Venue features
df['bf_venue_wr'] = np.where(df['bf_is_team1'] == 1,
                              df['team1_venue_win_rate'], df['team2_venue_win_rate'])
df['bs_venue_wr'] = np.where(df['bf_is_team1'] == 1,
                              df['team2_venue_win_rate'], df['team1_venue_win_rate'])

# Chase-related features
# The toss winner CHOSE to bat or field:
# toss_chose_bat = 1 means toss winner chose to bat first
# Did toss winner bat first?
# If team1_won_toss=1 and toss_chose_bat=1 -> team1 bats first
# If team1_won_toss=1 and toss_chose_bat=0 -> team1 fields first -> team2 bats first
# If team1_won_toss=0 and toss_chose_bat=1 -> team2 chose bat -> team2 bats first
# If team1_won_toss=0 and toss_chose_bat=0 -> team2 chose field -> team1 bats first
df['toss_winner_bats_first'] = np.where(
    ((df['team1_won_toss'] == 1) & (df['toss_chose_bat'] == 1)) |
    ((df['team1_won_toss'] == 0) & (df['toss_chose_bat'] == 0)),
    1, 0
)
# Simpler: toss winner chose to FIELD = chase (strong signal)
df['toss_chose_field'] = 1 - df['toss_chose_bat']

# Toss-venue alignment: did toss winner's choice match venue bias?
df['toss_venue_aligned_bf'] = np.where(
    (df['toss_chose_field'] == 1) & (df['venue_chase_win_rate'] > 0.5), 1,
    np.where((df['toss_chose_bat'] == 1) & (df['venue_bat_first_win_rate'] > 0.5), 1, 0)
)

# Chase WR for each team
if 'team1_chase_wr' in df.columns:
    df['bf_chase_wr'] = np.where(df['bf_is_team1'] == 1,
                                  df['team1_chase_wr'], df['team2_chase_wr'])
    df['bs_chase_wr'] = np.where(df['bf_is_team1'] == 1,
                                  df['team2_chase_wr'], df['team1_chase_wr'])

# Venue × toss-decision win rate (bat-first perspective)
# venue_bat_wr: P(toss winner wins | chose to BAT at this venue)
# venue_field_wr: P(toss winner wins | chose to FIELD at this venue)
# Re-orient to bat-first perspective:
#   toss_winner_bats_first=1 → bat-first team IS the toss winner → use venue_bat_wr
#   toss_winner_bats_first=0 → toss winner chose field (bats second) → bat-first team = loser of toss
#       From bat-first perspective: win prob = 1 - venue_field_wr
if 'venue_bat_wr' in df.columns and 'venue_field_wr' in df.columns:
    df['venue_decision_wr_bf'] = np.where(
        df['toss_winner_bats_first'] == 1,
        df['venue_bat_wr'],
        1.0 - df['venue_field_wr']
    )
else:
    df['venue_decision_wr_bf'] = 0.5

# Weather interaction features
df['dew_chase_advantage'] = df['dew_factor'] * df['venue_chase_win_rate']
df['humidity_x_evening'] = df['humidity'].fillna(60) * df['is_evening'].fillna(0) / 100

# Season features
df['season_int'] = df['season'].astype(int)

# Fill NaN weather with venue medians or defaults
for col in ['temperature', 'humidity', 'cloud_cover', 'wind_speed', 'dew_factor', 'is_evening', 'heat_factor']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

# ---- Feature selection ----
POST_TOSS_FEATURES = [
    # Team strength (ELO)
    "elo_diff_bf", "bf_elo", "bs_elo",
    # Team form
    "form_diff_bf", "bf_form", "bs_form",
    "form_3_diff_bf", "form_10_diff_bf",
    # Head-to-head
    "h2h_bf",
    # XI player strength (expanding-window scores)
    "bf_xi_bat", "bs_xi_bat", "xi_bat_diff",
    "bf_xi_bowl", "bs_xi_bowl", "xi_bowl_diff",
    "bf_xi_depth", "bs_xi_depth",
    "bf_xi_max_bat", "bs_xi_max_bat",
    # Impact player bowling strength (Phase 6)
    "bf_impact_bowl", "bs_impact_bowl", "impact_bowl_diff",
    # Batter-bowler H2H matchup advantage (Phase 4)
    "matchup_advantage_bf", "matchup_advantage_diff",
    # Toss
    "toss_chose_field", "toss_winner_bats_first",
    "toss_venue_aligned_bf",
    "venue_decision_wr_bf",   # venue × decision win rate (bat-first perspective)
    # Venue
    "venue_chase_win_rate", "venue_bat_first_win_rate",
    "venue_avg_first_innings", "venue_matches",
    "bf_venue_wr", "bs_venue_wr",
    "venue_toss_win_rate",
    # Chase rates
    "bs_chase_wr",
    # Weather
    "temperature", "humidity", "cloud_cover", "dew_factor",
    "is_evening", "heat_factor",
    # Weather interactions
    "dew_chase_advantage", "humidity_x_evening",
    # Season
    "match_num_in_season", "is_playoff",
]

# Only keep features that exist
POST_TOSS_FEATURES = [f for f in POST_TOSS_FEATURES if f in df.columns]
print(f"Post-toss features: {len(POST_TOSS_FEATURES)}")

feature_median = df[POST_TOSS_FEATURES].median()
X = df[POST_TOSS_FEATURES].fillna(feature_median).values
y = df['bat_first_won'].values
seasons = df['season'].astype(str).values

print(f"Training samples: {len(X)}")
print(f"Bat-first win rate: {y.mean():.3f}")
print(f"Seasons: {sorted(df['season'].astype(str).unique())}")

# Check feature variance
zero_var = [f for i, f in enumerate(POST_TOSS_FEATURES) if X[:, i].std() == 0]
if zero_var:
    print(f"  WARNING: Zero-variance features: {zero_var}")
    POST_TOSS_FEATURES = [f for f in POST_TOSS_FEATURES if f not in zero_var]
    X = df[POST_TOSS_FEATURES].fillna(feature_median).values


# =====================================================================
# PHASE 6: TRAIN ENSEMBLE
# =====================================================================
print("\n" + "=" * 62)
print("PHASE 6: Train post-toss ensemble")
print("=" * 62)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Sample weights: upweight recent seasons, actively discount pre-2019
# 2026: 5.0x | 2024-2025: 4.0x | 2022-2023: 2.5x | 2021: 1.5x | 2019-2020: 1.0x | pre-2019: 0.5x
sample_weight = np.ones(len(y))
for i, s in enumerate(seasons):
    s_int = int(s)
    if s_int >= 2026:
        sample_weight[i] = 5.0
    elif s_int >= 2024:
        sample_weight[i] = 4.0
    elif s_int >= 2022:
        sample_weight[i] = 2.5
    elif s_int == 2021:
        sample_weight[i] = 1.5
    elif s_int >= 2019:
        sample_weight[i] = 1.0
    else:
        sample_weight[i] = 0.5

w_counts = {w: int((sample_weight == w).sum()) for w in sorted(set(sample_weight))}
print(f"Sample weight distribution: {w_counts}")
print(f"  Total effective samples: {sample_weight.sum():.0f} (raw: {len(y)})")

# ---- Tune XGBoost ----
print("\nTuning XGBoost with Optuna (100 trials)...")
def objective_xgb(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
        "max_depth":         trial.suggest_int("max_depth", 2, 5),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0, 5.0),
    }
    m = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
    oof = cross_val_predict(m, X, y, cv=skf, method="predict_proba",
                            params={"sample_weight": sample_weight})[:, 1]
    return log_loss(y, oof)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=100)
bp_xgb = study_xgb.best_params
print(f"  XGB best CV log-loss: {study_xgb.best_value:.4f}")
print(f"  depth={bp_xgb['max_depth']}  lr={bp_xgb['learning_rate']:.4f}  "
      f"n_est={bp_xgb['n_estimators']}  gamma={bp_xgb['gamma']:.2f}")

# ---- Tune LightGBM ----
print("\nTuning LightGBM with Optuna (80 trials)...")
def objective_lgb(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
        "max_depth":         trial.suggest_int("max_depth", 2, 6),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 8, 40),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
    }
    m = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
    oof = cross_val_predict(m, X, y, cv=skf, method="predict_proba",
                            params={"sample_weight": sample_weight})[:, 1]
    return log_loss(y, oof)

study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lgb, n_trials=80)
bp_lgb = study_lgb.best_params
print(f"  LGB best CV log-loss: {study_lgb.best_value:.4f}")

# ---- Generate OOF predictions ----
print("\nGenerating OOF predictions...")
xgb_clf = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss", random_state=42, verbosity=0)
lgb_clf = lgb.LGBMClassifier(**bp_lgb, random_state=42, verbosity=-1)
lr_clf = LogisticRegression(C=0.1, max_iter=3000, random_state=42)

scaler_lr = StandardScaler()
X_scaled = scaler_lr.fit_transform(X)

oof_xgb = cross_val_predict(xgb_clf, X, y, cv=skf, method="predict_proba",
                             params={"sample_weight": sample_weight})[:, 1]
oof_lgb = cross_val_predict(lgb_clf, X, y, cv=skf, method="predict_proba",
                             params={"sample_weight": sample_weight})[:, 1]
oof_lr = cross_val_predict(lr_clf, X_scaled, y, cv=skf, method="predict_proba",
                            params={"sample_weight": sample_weight})[:, 1]
oof_ens = (oof_xgb + oof_lgb + oof_lr) / 3

# Isotonic calibration on OOF ensemble predictions
print("\nFitting isotonic calibration on OOF ensemble predictions...")
iso_calibrator = IsotonicRegression(y_min=0.05, y_max=0.95, out_of_bounds="clip")
iso_calibrator.fit(oof_ens, y)
oof_calibrated = iso_calibrator.predict(oof_ens)

print("\n--- 5-fold Stratified CV accuracies ---")
for name, oof in [("XGBoost", oof_xgb), ("LightGBM", oof_lgb), ("LogReg", oof_lr),
                   ("Ensemble", oof_ens), ("Ens+Isotonic", oof_calibrated)]:
    acc = accuracy_score(y, (oof >= 0.5).astype(int))
    ll = log_loss(y, np.clip(oof, 1e-7, 1 - 1e-7))
    brier = brier_score_loss(y, oof)
    print(f"  {name:12s}: acc={acc:.3f}  ll={ll:.4f}  brier={brier:.4f}")

# ---- Recent-season accuracy ----
print("\nAccuracy on recent seasons (2024-2025):")
recent_mask = np.array([int(s) >= 2024 for s in seasons])
if recent_mask.sum() > 0:
    for name, oof in [("XGBoost", oof_xgb), ("LightGBM", oof_lgb), ("Ensemble", oof_ens)]:
        acc = accuracy_score(y[recent_mask], (oof[recent_mask] >= 0.5).astype(int))
        print(f"  {name:12s}: {acc:.3f} ({recent_mask.sum()} matches)")

# ---- Confidence threshold analysis ----
print("\n--- Confidence threshold analysis (ensemble OOF) ---")
print(f" {'Threshold':>10}  {'Coverage':>10}  {'Matches':>8}  {'Accuracy':>9}")
print("-" * 45)

best_thr = 0.50
best_cov = 1.0
found_80 = False

for thr in np.round(np.arange(0.50, 0.82, 0.01), 2):
    mask = (oof_ens >= thr) | (oof_ens <= 1 - thr)
    n = mask.sum()
    if n < 15:
        break
    cov = mask.mean()
    acc = accuracy_score(y[mask], (oof_ens[mask] >= 0.5).astype(int))
    print(f"  {thr:8.2f}  {cov*100:>9.1f}%  {n:>7d}  {acc*100:>8.1f}%")

    if acc >= 0.80 and not found_80:
        best_thr = thr
        best_cov = cov
        found_80 = True

    # Track best even if 80% not reached
    if not found_80 and acc >= best_cov:
        best_thr = thr

if found_80:
    print(f"\n>> 80% threshold FOUND: {best_thr:.2f} (coverage {best_cov*100:.1f}%)")
else:
    # Fall back to the threshold with highest accuracy
    best_acc = 0
    for thr in np.round(np.arange(0.50, 0.82, 0.01), 2):
        mask = (oof_ens >= thr) | (oof_ens <= 1 - thr)
        if mask.sum() < 15:
            break
        acc = accuracy_score(y[mask], (oof_ens[mask] >= 0.5).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
            best_cov = mask.mean()
    print(f"\n>> 80% not reached. Best: thr={best_thr:.2f} acc={best_acc:.1%} cov={best_cov:.1%}")

# ---- Check per-season accuracy at threshold ----
print("\nPer-season accuracy at threshold {:.2f}:".format(best_thr))
for s in sorted(df['season'].astype(str).unique()):
    s_mask = (seasons == s)
    conf_mask = (oof_ens >= best_thr) | (oof_ens <= 1 - best_thr)
    both = s_mask & conf_mask
    if both.sum() > 0:
        acc = accuracy_score(y[both], (oof_ens[both] >= 0.5).astype(int))
        print(f"  {s}: {acc:.1%} ({both.sum()}/{s_mask.sum()} matches)")

# =====================================================================
# PHASE 7: TRAIN FINAL MODEL & SAVE
# =====================================================================
print("\n" + "=" * 62)
print("PHASE 7: Train final model and save")
print("=" * 62)

# Train on all data
xgb_final = xgb.XGBClassifier(**bp_xgb, eval_metric="logloss", random_state=42, verbosity=0)
lgb_final = lgb.LGBMClassifier(**bp_lgb, random_state=42, verbosity=-1)
lr_final = LogisticRegression(C=0.1, max_iter=3000, random_state=42)
scaler_final = StandardScaler()

xgb_final.fit(X, y, sample_weight=sample_weight)
lgb_final.fit(X, y, sample_weight=sample_weight)
X_s = scaler_final.fit_transform(X)
lr_final.fit(X_s, y, sample_weight=sample_weight)

ensemble = EnsemblePreMatchModel(xgb_final, lgb_final, lr_final, scaler_final, calibrator=iso_calibrator)

# Feature importances
fi_xgb = pd.Series(xgb_final.feature_importances_, index=POST_TOSS_FEATURES)
fi_lgb = pd.Series(lgb_final.feature_importances_, index=POST_TOSS_FEATURES)
fi_avg = ((fi_xgb / fi_xgb.sum()) + (fi_lgb / fi_lgb.sum())) / 2
print("\nTop 15 features (importance):")
for feat, imp in fi_avg.sort_values(ascending=False).head(15).items():
    print(f"  {feat:35s} {imp:.4f}")

# Compute overall CV accuracy for the bundle
cv_acc = accuracy_score(y, (oof_ens >= 0.5).astype(int))

# Save bundle
posttoss_bundle = {
    "model":                ensemble,
    "features":             POST_TOSS_FEATURES,
    "train_median":         {f: float(feature_median[f]) for f in POST_TOSS_FEATURES if f in feature_median.index},
    "confidence_threshold": float(best_thr),
    "cv_accuracy":          round(float(cv_acc), 4),
    "bat_first_framing":    True,  # Flag: model predicts P(bat_first_wins)
    "seasons":              sorted(df['season'].astype(str).unique()),
}

with open("models/posttoss_model.pkl", "wb") as f:
    pickle.dump(posttoss_bundle, f)

print(f"\nSaved models/posttoss_model.pkl")
print(f"  Features          : {len(POST_TOSS_FEATURES)}")
print(f"  CV accuracy (all) : {cv_acc:.3f}")
print(f"  80% threshold     : {best_thr:.2f}")
print(f"  Training matches  : {len(X)}")

print("\n" + "=" * 62)
print("Post-toss model training complete.")
print("  Use /reload-models or restart API to pick up new model.")
print("=" * 62)
