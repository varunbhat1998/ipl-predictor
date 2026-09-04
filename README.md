# IPL Match Predictor

A fully automated IPL prediction system that monitors live matches via CricAPI, sends predictions to Telegram at every key phase (pre-match, post-toss, every over), and retrains all models after each match. Built on historical ball-by-ball data from 2008–2026.

---

## How it works

The bot runs as a single process on match day, moving through 8 phases per match:

```
Pre-match → Toss detection → 1st innings (per over) → 2nd innings (per over) → Result + auto-retrain
```

### Model pipeline

| Stage | Model | Trigger | Accuracy |
|---|---|---|---|
| Pre-match | XGBoost + LightGBM + LR ensemble (`prematch_model.pkl`) | Bot startup | ~70% (5-year walkforward) |
| Post-toss | XGBoost + LightGBM + LR ensemble (`posttoss_model.pkl`) | Toss detected | ~80% (1,161 matches) |
| 1st innings live | LightGBM (`inn1_live_model.pkl`) | Each over boundary | ~63% |
| 2nd innings live | LightGBM (`live_model.pkl`) | Each over boundary | ~72% |
| Unified live | LightGBM (`unified_live_model.pkl`) | Both innings combined | ~68% |

Post-toss prediction uses an **anchor blend**: `60% × post-toss + 40% × pre-toss`, smoothing extreme swings when the model and toss diverge.

Pre-match confidence is **feature-based** — ELO, H2H, venue, form, and player XI strength signals are weighted. When all signals agree, confidence reaches 67–70%; when signals conflict it stays near 50–55%.

---

## Architecture

```
DATA PIPELINE
Cricsheet JSONs (raw_json/) → 01_parse.py → matches.csv + deliveries.csv
                                                    ↓
                                            02_features.py → match_features.csv
                                                    ↓
                        ┌───────────────────────────┼──────────────────────┐
                        ↓                           ↓                      ↓
                  03_train.py            10_post_toss_model.py     03_live_model.py
                        ↓                           ↓                      ↓
              prematch_model.pkl         posttoss_model.pkl         live_model.pkl
                                                                  inn1_live_model.pkl

MATCH DAY
  python 04_api.py  ←── loads all .pkl models at startup
        ↑
  python match_bot.py ──→ CricAPI (live scores)
        ↓
  Telegram → Users
        ↓
  06_retrain_after_match.py (auto-triggers after result)
        ↓
  POST /reload-models (hot-reloads API without restart)
```

---

## Project structure

```
V1/
├── match_bot.py                  # Main bot — all 8 match phases end-to-end
├── match_logger.py               # Structured per-match logging
├── model_classes.py              # EnsemblePreMatchModel class (required for pickle)
├── name_map.py                   # CricAPI → Cricsheet player name mapping (300+ entries)
├── ipl_schedule_2026.py          # 2026 fixture list with CricAPI match IDs
│
├── 04_api.py                     # FastAPI server (port 8000) — serves predictions
│
├── Data pipeline
│   ├── 01_parse.py               # Cricsheet JSON → matches.csv + deliveries.csv
│   ├── 02_features.py            # Feature engineering (ELO, form, H2H, venue, player XI)
│   ├── 03_train.py               # Train pre-match ensemble + live models (Optuna-tuned)
│   ├── 10_post_toss_model.py     # Train post-toss model (adds XI + weather features)
│   ├── 11_unified_live_model.py  # Train unified live model
│   ├── 06_retrain_after_match.py # Append result + trigger full retrain (called by bot)
│   ├── 07_build_player_db.py     # Build player batting/bowling database
│   ├── 07b_apply_x_factor.py     # Apply X-factor adjustments for impact players
│   └── 08_update_player_db.py    # Update player database with new season data
│
├── Backtesting
│   ├── backtest_2026.py          # 2026 season backtest (all 4 model stages)
│   ├── backtest_5yr.py           # 5-year (2021–2025) walkforward backtest
│   ├── backtest_posttoss.py      # Post-toss model standalone backtest
│   └── 10_backtest_full.py       # Full pipeline walkforward backtest
│
├── data/                         # Processed CSVs — match history, features, backtest results
├── models/                       # Trained .pkl model files
├── raw_json/                     # Cricsheet ball-by-ball JSON (1,170 matches, 2008–2026)
├── graphify-out/                 # Codebase knowledge graph (590 nodes, 784 edges)
│
├── start_bot.bat                 # Double-click launcher for match day (Windows)
├── requirements.txt
├── .env.sample                   # Copy to .env and fill in your keys
└── IPL_Predictor_Developer_Guide.md  # Full developer reference
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/varunbhat1998/ipl-predictor.git
cd ipl-predictor
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.sample .env
# Fill in your Telegram, CricAPI, and Anthropic keys
```

### 3. Build data (first time only — pre-built CSVs included)

```bash
python 01_parse.py --data_dir raw_json/
python 02_features.py
python 07_build_player_db.py
python 07b_apply_x_factor.py
```

### 4. Train models (pre-trained .pkl files included)

```bash
python 03_train.py
python 10_post_toss_model.py
python 11_unified_live_model.py
```

### 5. Match day

Start the API in one terminal:

```bash
python 04_api.py
```

Start the bot in another (auto-detects today's match):

```bash
python match_bot.py
```

Or double-click `start_bot.bat` on Windows. Output logs to `start_bot.log`.

### 6. Manual retrain (bot does this automatically after each match)

```bash
python 06_retrain_after_match.py \
  --team1 "Mumbai Indians" --team2 "Delhi Capitals" \
  --winner "Delhi Capitals" \
  --venue "Arun Jaitley Stadium, Delhi" \
  --toss-winner "Delhi Capitals" --toss-decision "bowl" \
  --inn1-runs 162 --inn1-wickets 6 \
  --inn2-runs 164 --inn2-wickets 4
```

The API hot-reloads models after retrain — no restart needed.

---

## Telegram commands (during a live match)

| Command | Description |
|---|---|
| `/predict` | Force a prediction at current game state |
| `/predictASAP` | Toggle ball-by-ball mode (prediction after every delivery) |
| `/status` | Show current match state and API call count |
| `/help` | List available commands |

---

## API endpoints (`04_api.py` on port 8000)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status check |
| POST | `/predict/prematch` | Pre-match prediction (works pre- and post-toss) |
| POST | `/predict/posttoss` | Post-toss prediction (with XI + weather) |
| POST | `/predict/live` | Live 2nd innings prediction |
| POST | `/predict/live_inn1` | Live 1st innings prediction |
| POST | `/reload-models` | Hot-reload all models after retrain |

---

## Features engineered (`02_features.py`)

ELO ratings (K=24), form over last 3/5/10 games, head-to-head win rate, venue batting-first/chase win rates, venue average first innings score, team-specific venue win rate, batting and bowling strength differentials (top-6 bat avg, top-4 bowl avg), chase ability differential, toss-venue alignment signal. All computed with expanding windows — no future data leakage.

---

## Data sources

- **Historical data:** [Cricsheet](https://cricsheet.org/) — ball-by-ball IPL JSON, 2008–2026 (`raw_json/`)
- **Live scores:** [CricAPI](https://cricapi.com/) — ~60 API calls per match (within free tier daily limit)
- **Weather:** Open-Meteo API — temperature, humidity, cloud cover, dew factor (no key required)

---

## Requirements

- Python 3.10+
- CricAPI key (free tier supports 1–2 matches/day; bot supports 3 rotating keys via `CRICAPI_KEY_1/2/3`)
- Telegram bot token + chat ID
- Anthropic API key (used for natural-language match summary messages)
