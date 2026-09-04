# IPL Match Predictor

A Telegram bot that predicts IPL match outcomes in real time using a 5-stage machine learning pipeline. It monitors live matches via CricAPI, fires predictions at each key phase (pre-match, post-toss, every over), auto-retrains models after each match, and sends everything to Telegram.

---

## How it works

The bot runs as a single process on match days. For each scheduled match it moves through 5 prediction stages:

```
Pre-match → Post-toss → 1st innings live → 2nd innings live → Result + retrain
```

| Stage | Model | Trigger | Accuracy |
|---|---|---|---|
| Pre-match | EnsemblePreMatchModel (stacked) | Bot startup | ~70% (5-year backtest) |
| Post-toss | LightGBM (`posttoss_model.pkl`) | Toss result detected | ~80% (1161 matches) |
| 1st innings live | LightGBM (`inn1_live_model.pkl`) | Every over (inn1) | ~63% |
| 2nd innings live | LightGBM (`live_model.pkl`) | Every over (inn2) | ~72% |
| Unified live | LightGBM (`unified_live_model.pkl`) | Both innings | ~68% overall |

After the match ends, `06_retrain_after_match.py` runs automatically and updates all models with the new result. A Telegram message confirms success or reports failure.

---

## Project structure

```
V1/
├── match_bot.py              # Main bot — runs all match phases end-to-end
├── match_logger.py           # Structured per-match logging
├── model_classes.py          # EnsemblePreMatchModel class definition
├── name_map.py               # Team name normalisation across data sources
├── ipl_schedule_2026.py      # 2026 fixture list with CricAPI match IDs
├── 04_api.py                 # FastAPI server (port 8000) — serves model predictions
│
├── Training pipeline
│   ├── 01_parse.py           # Parse Cricsheet JSON → matches.csv + deliveries.csv
│   ├── 02_features.py        # Feature engineering
│   ├── 03_train.py           # Train pre-match ensemble
│   ├── 10_post_toss_model.py # Train post-toss LightGBM
│   ├── 11_unified_live_model.py  # Train unified live model
│   ├── 06_retrain_after_match.py # Nightly retrain (called automatically by bot)
│
├── Backtesting
│   ├── backtest_2026.py      # 2026 season backtest across all models
│   ├── backtest_5yr.py       # 5-year (2021–2025) out-of-sample backtest
│   ├── backtest_posttoss.py  # Post-toss model standalone backtest
│   ├── 10_backtest_full.py   # Full pipeline walkforward backtest
│
├── data/                     # Processed CSVs — match history, features, backtest results
├── models/                   # Trained model files (.pkl)
├── raw_json/                 # Raw Cricsheet ball-by-ball JSON (1170 matches, 2008–2026)
├── graphify-out/             # Knowledge graph of the codebase (590 nodes, 784 edges)
│
├── start_bot.bat             # Double-click to run the bot on match day
├── requirements.txt
└── .env.sample               # Copy to .env and fill in your keys
```

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/varunbhat1998/ipl-predictor.git
cd ipl-predictor
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.sample .env
# Edit .env with your keys (Telegram, CricAPI, Anthropic)
```

**3. Prepare data** (skip if using the included CSVs)
```bash
python 01_parse.py        # Parse raw_json/ → data/matches.csv + deliveries.csv
python 02_features.py     # Build features
python 03_train.py        # Train pre-match model
python 10_post_toss_model.py
python 11_unified_live_model.py
```

**4. Start the API server** (in a separate terminal)
```bash
python 04_api.py
```

**5. Run the bot on match day**

Double-click `start_bot.bat`, or:
```bash
python match_bot.py
```

The bot finds today's matches from `ipl_schedule_2026.py`, processes them sequentially, and exits when done. Output is logged to `start_bot.log`.

---

## Data sources

- **Historical match data:** [Cricsheet](https://cricsheet.org/) — ball-by-ball JSON for IPL matches 2008–2026 (`raw_json/`)
- **Live scores:** [CricAPI](https://cricapi.com/) — real-time match state during bot execution
- **Weather:** Open-Meteo API (no key required)

---

## Requirements

- Python 3.10+
- CricAPI key (free tier works for 1–2 matches/day)
- Telegram bot token + chat ID
- Anthropic API key (used for match review summaries)
