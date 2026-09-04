# IPL Prediction System — Developer Guide
**Version:** 4.0 | **Last Updated:** April 2026

---

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [File-by-File Reference](#3-file-by-file-reference)
4. [Data Files](#4-data-files)
5. [Models](#5-models)
6. [API Endpoints](#6-api-endpoints)
7. [Match Bot — Phase-by-Phase Flow](#7-match-bot--phase-by-phase-flow)
8. [Recent Updates (April 2026)](#8-recent-updates-april-2026)
9. [Environment Variables](#9-environment-variables)
10. [Running the System](#10-running-the-system)
11. [Retraining After a Match](#11-retraining-after-a-match)

---

## 1. System Overview

The IPL Prediction System is a fully automated pipeline that:

- **Ingests** historical IPL match data (Cricsheet JSON, 2008–2025)
- **Engineers features** — ELO ratings, form, H2H, venue stats, player strength (no data leakage)
- **Trains 3 prediction models** — pre-match, post-toss, and live (1st and 2nd innings)
- **Serves predictions** via a FastAPI REST server
- **Runs a Telegram bot** that monitors matches live via CricAPI, sends predictions, and auto-retrains after each match

The system uses approximately 60 API calls per match (well under the 100-call daily limit per CricAPI key).

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA PIPELINE (one-time + after each season)                        │
│                                                                      │
│  Cricsheet JSONs → 01_parse.py → matches.csv + deliveries.csv        │
│                                       ↓                              │
│                               02_features.py → match_features.csv   │
│                                       ↓                              │
│                    ┌──────────────────┼────────────────────┐         │
│                    ↓                  ↓                    ↓         │
│              03_train.py    10_post_toss_model.py   03_live_model.py │
│                    ↓                  ↓                    ↓         │
│          prematch_model.pkl  posttoss_model.pkl   live_model.pkl     │
│                             inn1_live_model.pkl                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  LIVE MATCH DAY                                                       │
│                                                                      │
│  python 04_api.py  ←─────────── loads all .pkl models               │
│         ↑                                                            │
│  python match_bot.py ──→ CricAPI (live scores)                       │
│         ↓                                                            │
│  Telegram Bot → Users                                                │
│         ↓                                                            │
│  06_retrain_after_match.py (auto-retrain after result)               │
│         ↓                                                            │
│  POST /reload-models (hot-reload API without restart)                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. File-by-File Reference

### `01_parse.py` — Data Parser
**Purpose:** Converts Cricsheet IPL JSON files into structured CSVs.

**Run:** `python 01_parse.py --data_dir path/to/jsons`

**Outputs:**
- `data/matches.csv` — one row per match (teams, toss, result, innings scores, player lists)
- `data/deliveries.csv` — one row per ball (cumulative runs/wickets, batting/bowling info)

**Key notes:**
- Normalises season format ("2007/08" → "2008")
- Recovers Super Over winners from a hardcoded lookup for all 15 tied matches
- Handles IPL team name changes (Delhi DD→DC, KXIP→PBKS, RCB Bangalore→Bengaluru)

---

### `02_features.py` — Feature Engineering
**Purpose:** Builds the full feature matrix for model training. All features use expanding windows (no future data leakage).

**Run:** `python 02_features.py`

**Input:** `data/matches.csv`, `data/deliveries.csv`
**Output:** `data/match_features.csv` (60+ columns)

**Key features computed:**
| Feature | Description |
|---|---|
| `elo_diff` | ELO rating difference (K=24, init=1500) |
| `form_diff` | Win rate difference over last 5 games |
| `form_3_diff` | Last 3 games win rate difference |
| `form_10_diff` | Last 10 games win rate difference |
| `h2h_win_rate_team1` | Historical head-to-head win rate |
| `venue_bat_first_win_rate` | Venue batting-first win % (expanding) |
| `venue_avg_first_innings` | Average 1st innings score at venue |
| `venue_chase_win_rate` | Venue chasing win % |
| `team1_venue_win_rate` | Team 1 win % at specific venue |
| `bat_diff` | Batting strength differential (top-6 average) |
| `bowl_diff` | Bowling strength differential (top-4 average) |
| `chase_advantage_diff` | Chase ability differential |
| `toss_venue_aligned` | Toss decision aligned with venue tendency |

**Super Over recovery (Fix 5):** Tied matches are recovered using a hardcoded `_SUPER_OVER_WINNERS` dict. Training data grew from 1,154 to 1,169 rows (+15 matches).

**Dynamic venue fallback (Fix 4):** Instead of hardcoded `160`, uses a running global mean from all venues with 3+ matches.

---

### `03_train.py` — Model Training
**Purpose:** Trains all three prediction models using Optuna hyperparameter tuning.

**Run:** `python 03_train.py`

**Models trained:**
1. **Pre-match model** (`prematch_model.pkl`) — XGBoost + LightGBM + Logistic Regression soft-voting ensemble. Optuna-tuned (60 trials). OOF isotonic calibrator fitted but stored separately.
2. **Live 2nd innings model** (`live_model.pkl`) — LightGBM with momentum features (partnership runs/balls, last-3-over rates, boundary %, dot ball %, powerplay stats). GroupKFold prevents leakage across same-match snapshots.
3. **Live 1st innings model** (`inn1_live_model.pkl`) — LightGBM using projected score, run rate, wickets, acceleration.

**Test/train split:** Pre-2023 for training, 2023–2025 for test.

**Backtest output:** `data/backtest_results.csv`

---

### `10_post_toss_model.py` — Post-Toss Model
**Purpose:** Trains a dedicated post-toss model that knows batting/fielding order, playing XI, and weather.

**Run:** `python 10_post_toss_model.py`

**Output:** `models/posttoss_model.pkl`

**Additional features vs pre-match:**
- Playing XI bat/bowl strength (top-6 bat avg, top-4 bowl avg, depth, max)
- Weather: temperature, humidity, cloud cover, dew factor
- Dew factor formula: `max(0, min(1, (humidity - 65) / 35))` for evening games only
- Toss-venue alignment signal

**Calibration:** OOF-based isotonic calibration. Brier score improved from 0.2074 → 0.2029.

---

### `model_classes.py` — Shared Model Class
**Purpose:** Defines `EnsemblePreMatchModel` — imported by both training scripts and the API so pickle can deserialise the model correctly.

```python
class EnsemblePreMatchModel:
    # Soft-voting: XGBoost + LightGBM + Logistic Regression
    # predict_proba() averages the 3 models equally
    # Optional calibrator stored but NOT applied (API uses feature-based confidence instead)
```

**Important:** This file must be importable from wherever `04_api.py` runs. It is imported at the top of `04_api.py` via `sys.path.insert`.

---

### `04_api.py` — Prediction API
**Purpose:** FastAPI server that loads all models and serves predictions.

**Run:** `python 04_api.py` (starts on port 8000)

**Endpoints:** See Section 6.

**Feature-based confidence scoring (latest update):**
The pre-match endpoint no longer uses a uniform 3x amplifier. Instead, it counts how many independent signals (ELO, H2H, venue, form, player strength) agree with the model's direction:

- **All signals agree** → amplification 4.5x + minimum confidence floor of 67–70%
- **Signals mixed** → amplification 2–3x, confidence 55–60%
- **Signals conflict** → amplification 0.5x, stays near 50–55%

This means correct predictions (where features align) show 65–75% confidence; uncertain/wrong predictions show 50–58%.

**Dynamic global venue average:** At startup, computes `GLOBAL_AVG_FIRST_INNINGS` from all venues with 3+ matches (currently 165.4). Used as fallback for new/unknown venues instead of the old hardcoded `160`.

---

### `match_bot.py` — Live Match Bot
**Purpose:** Fully automated match-day bot. Polls CricAPI, calls the prediction API, and sends updates to Telegram.

**Run:** `python match_bot.py` (auto-detects today's match)

See Section 7 for full phase-by-phase flow.

**Key recent updates:**
- 3 CricAPI keys (`CRICAPI_KEY_1/2/3`) with round-robin rotation
- Ball-by-ball mode (`/predictASAP` Telegram command)
- Phase transitions use safety deadlines (`match_start + 6h`) instead of fixed clock times
- D/L target detection handles 4 different Cricbuzz format patterns
- Anchor blend: post-toss prediction = 60% post-toss + 40% pre-toss

---

### `06_retrain_after_match.py` — Post-Match Retrain
**Purpose:** CLI tool to append match result and trigger full retrain.

**Run:** `python 06_retrain_after_match.py --team1 "MI" --team2 "CSK" --winner "MI" --venue "Wankhede" ...`

**`--winner` is optional** (omit for tied/abandoned matches — retrain skips gracefully).

**Flow:**
1. Appends result to `data/matches.csv`
2. Runs `02_features.py` to rebuild features
3. Runs `03_train.py` to retrain all models
4. POSTs to `/reload-models` for hot-reload without API restart

---

### `backtest_2026.py` — Season Backtest
**Purpose:** Backtests all 4 model stages against 2026 IPL matches.

**Run:** Requires API running. `python backtest_2026.py`

**Stages tested:**
1. Pre-match (before toss)
2. Post-toss (with 60/40 anchor blend)
3. Live Inn1 at over 10 (estimated as 55% of final score, 25% of wickets)
4. Live Inn2 at over 10 (same estimation)

**2026 results (7 matches):**
| Model | Accuracy |
|---|---|
| Pre-match | 5/7 (71.4%) |
| Post-toss (60/40 blend) | 6/7 (85.7%) |
| Live Inn1 @Over10 | 3/7 (42.9%) |
| Live Inn2 @Over10 | 6/7 (85.7%) |

---

### `name_map.py` — Player Name Mapper
**Purpose:** Maps CricAPI player names (full form) to Cricsheet abbreviated format (e.g., "Virat Kohli" → "V Kohli"). 300+ entries covering all 10 IPL franchises.

**Used by:** `match_bot.py` to convert playing XI from CricAPI before passing to the prediction API.

---

### `07_build_player_db.py` / `07b_apply_x_factor.py` / `08_update_player_db.py`
**Purpose:** Builds and maintains `data/player_database_2026.csv` — career bat/bowl scores with X-factor adjustments for impact players.

---

### `ipl_schedule_2026.py`
**Purpose:** Stores the full 2026 IPL fixture list. Used by `match_bot.py` to find today's match ID.

---

## 4. Data Files

| File | Description |
|---|---|
| `data/matches.csv` | One row per match, 2008–present. Source of truth for all training. |
| `data/deliveries.csv` | Ball-by-ball data. Used for live model training. |
| `data/match_features.csv` | Engineered features. Rebuilt by `02_features.py` after each match. |
| `data/player_database_2026.csv` | Career bat/bowl scores per player with X-factor. |
| `data/player_venue_scores.csv` | Player performance at specific venues. |
| `data/team_profiles_2026.csv` | Team-level bat/bowl strength fallback (used when XI not announced). |
| `data/backtest_results.csv` | Walk-forward backtest on 2023–2025 test set. |
| `data/backtest_2026.csv` | 2026 season backtest results. |
| `data/weather_cache_v2.csv` | Cached weather data (Open-Meteo API) to avoid re-fetching. |

---

## 5. Models

| File | Algorithm | Purpose |
|---|---|---|
| `models/prematch_model.pkl` | XGB + LGB + LR ensemble | Pre-toss win prediction |
| `models/posttoss_model.pkl` | XGB + LGB + LR ensemble | Post-toss win prediction (with XI + weather) |
| `models/live_model.pkl` | LightGBM | 2nd innings live win probability |
| `models/inn1_live_model.pkl` | LightGBM | 1st innings live win probability |

All `.pkl` files are loaded at API startup. Hot-reload is available via `POST /reload-models` without restarting the server.

---

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status check |
| GET | `/teams` | List all known teams |
| GET | `/venues` | List all known venues |
| POST | `/predict/prematch` | Pre-match prediction (works pre- and post-toss) |
| POST | `/predict/posttoss` | Dedicated post-toss prediction |
| POST | `/predict/live` | Live 2nd innings prediction |
| POST | `/predict/live_inn1` | Live 1st innings prediction |
| POST | `/reload-models` | Hot-reload models after retrain |
| POST | `/update-excel/prematch` | Log pre-match prediction to Excel |
| POST | `/update-excel/live` | Log live prediction to Excel |
| POST | `/update-excel/result` | Log match result to Excel |
| GET | `/excel/status` | Excel tracker status |

### Key Request Fields

**`/predict/prematch`**
```json
{
  "team1": "Mumbai Indians",
  "team2": "Delhi Capitals",
  "venue": "Wankhede Stadium, Mumbai",
  "toss_winner": "Mumbai Indians",    // optional
  "toss_decision": "bat",             // optional
  "team1_players": ["R Sharma", ...], // optional
  "team2_players": ["D Warner", ...]  // optional
}
```

**`/predict/live`**
```json
{
  "batting_team": "Delhi Capitals",
  "bowling_team": "Mumbai Indians",
  "runs_scored": 87,
  "wickets_fallen": 3,
  "balls_bowled": 78,
  "target": 163,
  "partnership_runs": 45,
  "last_3ov_runs": 28,
  "venue": "Wankhede Stadium, Mumbai"
}
```

---

## 7. Match Bot — Phase-by-Phase Flow

```
Phase 0: Find match
  → Searches ipl_schedule_2026.py for today's fixture
  → Resolves CricAPI match ID
  → Sends "Match found" Telegram message

Phase 1: Pre-match prediction
  → Calls /predict/prematch with team1, team2, venue
  → Sends Telegram: team names, win %, key factors, Claude explanation

Phase 2: Sleep until toss window
  → Sleeps to 30 min before toss

Phase 3: Toss detection (poll every 3 min)
  → Detects toss result from CricAPI score feed
  → Calls /predict/posttoss (with playing XI if announced)
  → Blends: final = 60% post-toss + 40% pre-toss (anchor blend)
  → Sends Telegram: batting order, blended probabilities, Claude explanation

Phase 4: 1st innings tracking (poll every 60s)
  → Calls /predict/live_inn1 at each over boundary
  → Sends Telegram after each over: score, projected total, win %
  → Ball-by-ball mode: sends after every delivery if /predictASAP active
  → Safety deadline: match_start + 6h (not fixed clock time)

Phase 5: Innings break
  → Waits for 2nd innings to start (inn2 overs > 0.1 or runs > 0)
  → Reads final 1st innings score once inn2 has started (authoritative)
  → Computes target

Phase 6: 2nd innings tracking (poll every 60s)
  → Calls /predict/live at each over boundary
  → Detects D/L revised targets from status string (4 regex patterns)
  → Sends Telegram: chase progress, required rate, win %
  → Ball-by-ball mode: sends after every delivery if /predictASAP active
  → Safety deadline: match_start + 6h

Phase 7: Match end detection
  → inn2_r >= target → batting second wins
  → inn2_w >= 10 OR (inn2_over >= max_overs AND partial == 0) → batting first wins
  → inn2_r == target - 1 at over completion → TIED (Super Over)
  → Sends final result + accuracy report to Telegram

Phase 8: Auto-retrain
  → Calls 06_retrain_after_match.py with match result
  → API hot-reloads updated models
```

### Telegram Commands

| Command | Description |
|---|---|
| `/predict` | Force a prediction at current game state |
| `/predictASAP` | Toggle ball-by-ball prediction mode on/off |
| `/status` | Show current match state and API call count |
| `/help` | List available commands |

---

## 8. Recent Updates (April 2026)

### Fix 1: Match End Detection (Tied / Super Over)
**Problem:** Bot declared bat-first winner when scores were level at 20 overs. Also the `inn2_balls == 0` check was always false for complete innings.

**Fix:** Changed condition to use `partial2 == 0` (zero at 20.0 overs, non-zero mid-over). Added `"tied"` state that sends a Super Over alert instead of declaring a winner.

---

### Fix 2: Phase Transitions — Safety Deadlines
**Problem:** Fixed time-based inn1/inn2 deadlines caused the bot to skip innings for rain-delayed matches (e.g., "Inn1 deadline already passed — skipping").

**Fix:** Replaced fixed deadlines with `match_start + 6 hours`. Phase 4 never exits early. Phase 5 waits for inn2 overs > 0.1 as the authoritative inn1-complete signal.

---

### Fix 3: D/L Target Detection
**Problem:** `_parse_dl_target()` only matched `"target revised to X"`. The most common format (`"MI need 61 runs off 30 balls"`) was skipped.

**Fix:** Added `current_runs` parameter. Now handles 4 patterns:
1. `"target revised to X"` / `"target: X"`
2. `"need X runs off Y balls"` → target = current_runs + X
3. `"X to win off Y overs/balls"`
4. `"need X more runs"`

---

### Fix 4: Dynamic Venue Average
**Problem:** Hardcoded `160.0` / `165.0` fallback for unknown venues. Inconsistent between files.

**Fix:** `GLOBAL_AVG_FIRST_INNINGS` computed at API startup from all venues with 3+ matches. Currently **165.4** across 35 venues.

---

### Fix 5: Training Data — Tied Matches
**Problem:** All 23 tied matches were dropped from training data (`winner.notna()` filter).

**Fix:** `_SUPER_OVER_WINNERS` dict in `02_features.py` recovers the Super Over winner for all 15 resolvable tied matches. Training data grew from 1,154 to **1,169 rows**.

Also: `--winner` in `06_retrain_after_match.py` is now optional (graceful skip for abandoned/tied matches).

---

### Fix 6: Ball-by-Ball Prediction Mode (`/predictASAP`)
**New feature:** Telegram command `/predictASAP` toggles per-delivery predictions.

- When active: polls every 20s, sends compact probability message after each ball
- Normal mode: predictions only at each over boundary
- `/predictASAP` again turns it off

---

### Fix 7: Pre-Match Confidence — Feature-Based Scoring
**Problem:** Uniform 3x amplifier produced similar confidence for all predictions (52–57%).

**Fix:** Amplification now depends on signal agreement:
- ELO difference (weight up to 3.0)
- Head-to-head win rate (weight 2.0)
- Venue win rate differential (weight 1.5)
- Form difference (weight 1.0)
- Player XI strength (weight 1.5, when XI available)

Agreement ratio drives amplification (0.5× to 4.5×) plus a minimum confidence floor at high agreement (67–70%). Result: correct predictions average **60–70%** confidence; uncertain calls stay near **50–55%**.

---

### Fix 8: Post-Toss Anchor Blend (60/40)
**Problem:** Pre-toss and post-toss models could diverge by 40+ percentage points (e.g., 63.8% → 18.4%).

**Fix:** Post-toss displayed probability is blended: `60% × post-toss + 40% × pre-toss`. Smooths transitions and prevents extreme swings.

---

### Fix 9: Three CricAPI Keys
**Change:** Environment variables changed from `CRICAPI_KEY_AFTERNOON` / `CRICAPI_KEY_EVENING` to:
- `CRICAPI_KEY_1`
- `CRICAPI_KEY_2`
- `CRICAPI_KEY_3`

Keys are tried in sequence. Exhaustion of any key triggers a Telegram alert. Legacy variable names (`CRICAPI_KEY`, `CRICAPI_KEY_AFTERNOON`, `CRICAPI_KEY_EVENING`) still work as fallback.

---

## 9. Environment Variables

Create a `.env` file in the project root:

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=chat_id_1,chat_id_2   # comma-separated for multiple recipients

# CricAPI keys (3 keys for double-headers / key rotation)
CRICAPI_KEY_1=your_key_1
CRICAPI_KEY_2=your_key_2
CRICAPI_KEY_3=your_key_3

# Claude API (for match explanation messages)
ANTHROPIC_API_KEY=your_anthropic_key

# API server (default: localhost:8000)
API_BASE=http://localhost:8000
```

---

## 10. Running the System

### First-time setup
```bash
# 1. Parse raw data (only needed when adding new seasons)
python 01_parse.py --data_dir path/to/cricsheet_jsons

# 2. Build features
python 02_features.py

# 3. Build player database (2026 squad data)
python 07_build_player_db.py
python 07b_apply_x_factor.py

# 4. Train all models
python 03_train.py
python 10_post_toss_model.py
```

### Match day
```bash
# Terminal 1: Start the API
python 04_api.py

# Terminal 2: Start the bot (auto-detects today's match)
python match_bot.py

# Or for specific slot:
python match_bot.py --slot afternoon   # 3:30 PM game
python match_bot.py --slot evening     # 7:30 PM game
```

### After a match
```bash
# Auto-retrain is triggered automatically by match_bot.py.
# To retrain manually:
python 06_retrain_after_match.py \
  --team1 "Mumbai Indians" \
  --team2 "Delhi Capitals" \
  --winner "Delhi Capitals" \
  --venue "Arun Jaitley Stadium, Delhi" \
  --toss-winner "Delhi Capitals" \
  --toss-decision "bowl" \
  --inn1-runs 162 --inn1-wickets 6 \
  --inn2-runs 164 --inn2-wickets 4
```

---

## 11. Retraining After a Match

The retrain pipeline:
1. `06_retrain_after_match.py` appends the result to `data/matches.csv`
2. Calls `python 02_features.py` — rebuilds `match_features.csv` with new ELO, form, venue stats
3. Calls `python 03_train.py` — retrains XGBoost + LightGBM ensemble
4. Calls `python 10_post_toss_model.py` — retrains post-toss model
5. POSTs to `http://localhost:8000/reload-models` — hot-reloads the API

The API does **not** need to be restarted. All model objects are replaced in memory within the running process.

---

*For questions or issues, refer to the inline comments in each file. Each major function has a docstring describing inputs, outputs, and any edge cases handled.*
