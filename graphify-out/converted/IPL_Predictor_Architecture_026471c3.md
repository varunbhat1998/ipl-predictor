<!-- converted from IPL_Predictor_Architecture.docx -->


IPL Match Prediction Engine
Technical Architecture & Model Intelligence
AI-Powered. Real-Time. Cricket-Native.


v4.0  |  Season 2026
CONFIDENTIAL — For Authorized Recipients Only
This document contains proprietary technical information. Do not distribute.

Table of Contents


1. Executive Summary
2. System Architecture Overview
3. The Three ML Models
3.1  Model A — Pre-Match Win Predictor (XGBoost)
3.2  Model B — Second Innings Live Predictor (LightGBM)
3.3  Model C — First Innings Live Predictor (LightGBM)
4. Feature Engineering Intelligence
4.1  ELO Rating System
4.2  Exponential Form Weighting
4.3  Venue-Month Interaction
4.4  Player Strength Scoring
4.5  Zero Data Leakage Guarantee
5. Live Match Bot — Automated Delivery
5.1  Six-Phase Match Lifecycle
5.2  Data Source Priority
5.3  Powerplay Lock Logic
5.4  Dual API Key Architecture
5.5  Auto-Retrain After Every Match
6. Accuracy & Validation
7. Outputs & Deliverables
7.1  Telegram Messages
7.2  Excel Tracker
7.3  Model Artifacts
8. Why This System Is Different
9. Technical Specifications


# 1. Executive Summary


The IPL Match Prediction Engine is a fully automated, end-to-end machine learning system that delivers real-time win probability updates throughout every IPL match. Built on 18 years of ball-by-ball data and three purpose-built ML models, this system provides actionable intelligence at every stage of a match — from before the toss to the final ball.

- Real-time win probability updates delivered throughout every IPL match via Telegram
- Three distinct ML models cover pre-match, first innings, and second innings stages — no single model forced to handle all phases
- Automated Telegram delivery with per-over updates — zero manual intervention required
- Self-retraining after every match — models continuously improve throughout the IPL season
- 63% pre-match prediction accuracy (random baseline = 50%)
- 70% first-innings prediction accuracy, improving to 75% by the final over
- 75% second-innings (live chase) accuracy — the highest-stakes prediction stage
- Built on 18 years of IPL data (2008-2026), covering 900+ matches across all seasons

At 63% pre-match accuracy against a 50% random baseline, the system captures 26% more correct predictions than chance — a statistically significant edge across a full 74-match IPL season.


# 2. System Architecture Overview


The system operates as a linear data pipeline, transforming raw cricket data into real-time, actionable predictions. Every component has a single responsibility, and data flows deterministically from stage to stage.

Raw JSON ball-by-ball data from Cricsheet is first parsed into structured tabular format by 01_parse.py, producing two foundational datasets: matches.csv (match-level metadata) and deliveries.csv (ball-by-ball records). From there, 02_features.py applies sophisticated feature engineering — computing ELO ratings, rolling form windows, venue statistics, and player strength scores — to produce match_features.csv, the single input to all three models.

03_train.py reads match_features.csv and trains three separate XGBoost and LightGBM models using Optuna-tuned hyperparameters and walk-forward cross-validation. The resulting .pkl files are served by 04_api.py, a FastAPI server running on port 8000 that exposes prediction endpoints for each model stage.

match_bot.py is the orchestration layer — it polls live match data from CricAPI and Cricbuzz, calls the prediction API at every over, formats results into rich Telegram messages, and logs everything to an Excel tracker. At match end, it triggers an automatic retraining cycle, ensuring the models incorporate the latest match before the next game begins.



# 3. The Three ML Models

The system uses three purpose-built models, each trained and calibrated for a specific phase of an IPL match. This phase-aware design means each model is optimized for the information available at that stage — no single model is forced to handle all scenarios.

## 3.1  Model A — Pre-Match Win Predictor (XGBoost)
- Algorithm: XGBoost Classifier, Optuna-tuned (60 trials)
- Purpose: Predict match winner before toss and post-toss
- Training: All seasons before test year (walk-forward, no data leakage)
- Accuracy: 63% on 2023-2025 holdout | After toss: 65%
- Features: 33 features across 7 categories


Toss-Venue Alignment is the single most powerful post-toss signal. Teams that make the venue-optimal toss decision win at 58% vs 42% — a 16-percentage-point edge that the model captures and quantifies.


## 3.2  Model B — Second Innings Live Predictor (LightGBM)
- Algorithm: LightGBM + CalibratedClassifierCV (isotonic, GroupKFold CV=5)
- Purpose: Predict chasing team win probability ball-by-ball throughout the second innings
- Calibration: Isotonic regression ensures probabilities are reliable — not just rankings
- Group CV: Grouped by match file_id to prevent same-match data leakage across train/test folds
- Accuracy: 75% on 2023-2025 holdout
- Features: 26 features across 4 groups


The powerplay (overs 1-6) is the single most predictive phase. pp_runs is the #2 most important feature overall, and pp_rate_gap (how far ahead or behind the chasing team was at the end of over 6) ranks #9. A chase that starts strong in the powerplay wins 67% of the time.


## 3.3  Model C — First Innings Live Predictor (LightGBM)
- Algorithm: LightGBM (300 estimators, max_depth=5, learning_rate=0.05)
- Purpose: Predict whether the batting-first team will post a match-winning total
- Target: Was batting-first team's final score higher than the chasing team's?
- Accuracy: 70% overall, improving to 75% by over 20
- Features: 23 features across 6 groups


Per-Over Accuracy Progression:


Accuracy improves steadily through the innings as more data accumulates. By over 10 (halfway), the model achieves 72% accuracy — well above the 50% random baseline — and peaks at 75% by the final over.


# 4. Feature Engineering Intelligence

The quality of a predictive model is determined entirely by the quality of its features. Every feature in this system is constructed from first principles, with rigorous chronological discipline to guarantee zero data leakage.

## 4.1  ELO Rating System
- Baseline: 1500 for all teams at the start of data history
- K-factor: 24 (same as standard chess K-factor, calibrated empirically to IPL volatility)
- Formula: Expected = 1 / (1 + 10^((Rb - Ra) / 400))
- ELO updated after every single match — ratings reflect the most recent form
- Captures long-run team quality more robustly than simple win/loss counts
ELO provides a continuous, self-correcting measure of team strength. A team that beats a strong opponent gains more ELO than one that beats a weak opponent — capturing match quality, not just outcomes.

## 4.2  Exponential Form Weighting
- Window: 5 most recent matches per team
- Weights: [0.35, 0.25, 0.20, 0.12, 0.08] — most recent match carries 3.5x the weight of oldest
- Multiple windows (3, 5, 10 games) capture both short-term hot streaks and stable long-run form
- Exponential weighting prevents a single old match from distorting form readings

## 4.3  Venue-Month Interaction
- Feature: venue_month_chase_wr — historical chase win rate at (venue, month) combination
- Motivation: the same stadium can play completely differently across months due to pitch preparation, dew, and weather patterns
- Example: Ahmedabad April chase win rate = 73% | Ahmedabad May chase win rate = 33%
- Computed via expanding historical means — only data from prior matches is ever used
A simple venue win rate misses seasonal variation entirely. The venue-month interaction captures that Ahmedabad in April is a completely different pitch environment from Ahmedabad in May — a 40-percentage-point difference in chase win rates that a naive model would never detect.

## 4.4  Player Strength Scoring
- Batting score formula: Strike Rate x 0.5 + Average x 50
- Bowling score formula: (Wickets per 6 balls x 25) - Economy Rate
- Blended across seasons: 0.6 x current season + 0.4 x previous season (handles early-season sparsity)
- Minimum thresholds: 20 balls faced (batting), 12 balls bowled (bowling) — filters noise
- Team score = mean of top 5 batsmen (batting) or top 3 bowlers (bowling) from announced XI

## 4.5  Zero Data Leakage Guarantee
- All venue statistics, ELO ratings, and form metrics computed in strict chronological order
- For every match: all stats calculated using ALL PRIOR matches only — never the current match
- Test seasons completely excluded from any training computation
- Walk-forward backtesting protocol: train on [2008, ..., year-1], test on [year]
- 11-year backtest (2015-2025) provides statistically robust accuracy estimates
Data leakage is the most common failure mode in sports prediction models. This system enforces strict chronological isolation at every level — feature computation, model training, and backtesting — making the reported accuracy figures conservative and reliable.


# 5. Live Match Bot — Automated Delivery

The match bot is a fully autonomous orchestration engine that manages the complete lifecycle of every IPL match — from pre-match predictions to final result logging and model retraining. Once scheduled, zero human intervention is required.

## 5.1  Six-Phase Match Lifecycle
- Pre-toss (90 minutes before match): Pre-match prediction computed and sent to Telegram. Match logged to Excel with venue, team, and form data.
- Toss window (10 API calls, 3-minute intervals): System polls CricAPI and Cricbuzz alternately until toss result is detected. Updated post-toss prediction sent immediately.
- First innings (~22 API calls, 4-minute intervals): Per-over live predictions from Model C. Each update includes current score, wickets, projected total, and batting-first win probability.
- Innings break (20-minute sleep): System enters sleep mode. Zero API calls consumed during the interval — preserving daily budget for the second innings.
- Second innings (~30 API calls, 3-minute intervals): Per-over live predictions from Model B. Each update includes required run rate, current win probability, and chase status.
- Match end: Result detected from score string. Final result logged to Excel. Auto-retrain triggered as subprocess. Models hot-swapped via /reload-models endpoint.

API call budget: approximately 62 calls per match — well within the 100 calls/day per key limit. The innings-break sleep and smart polling intervals are specifically designed to maximise coverage while minimising API consumption.

## 5.2  Data Source Priority
The bot uses a tiered fallback strategy for both score fetching and playing XI retrieval:

Score Fetching Priority:
- CricAPI matchScore — most reliable source; only available when match ID is confirmed from currentMatches endpoint
- Cricbuzz HTML scrape — no API key required, very reliable, used as primary fallback
- CricAPI cricScore — tertiary fallback for edge cases

Playing XI Priority:
- CricAPI matchInfo — playing11:true field, most accurate when available
- Cricbuzz mobile JSON API — reliable for announced lineups
- Cricbuzz RSC page scraping — 3-level escape regex patterns with teamSName extraction
- 60-second retry loop — if XI not yet announced, system waits and retries automatically

## 5.3  Powerplay Lock Logic
- At exactly over 6 (ball 36): pp_runs and pp_wickets are locked permanently for that innings
- Prevents late-start polls from overwriting accurate powerplay stats with stale cumulative values
- If over 6 is missed (bot starts mid-match): pp values remain None and model uses fallback estimates
The powerplay lock is a critical engineering detail. Without it, a bot that starts polling at over 8 would compute incorrect powerplay stats — corrupting the most important feature group in the second-innings model.

## 5.4  Dual API Key Architecture
- CRICAPI_KEY_AFTERNOON: dedicated key for 3:30 PM IST matches
- CRICAPI_KEY_EVENING: dedicated key for 7:30 PM IST matches
- Automatic slot assignment based on match start time — no manual key switching
- 100 calls/day per key
- Total daily budget: 200 API calls — sufficient for two full concurrent matches

## 5.5  Auto-Retrain After Every Match
- Match end detected from live score string (pattern matching for 'won by' / 'match tied')
- Triggers: python 03_train.py as a subprocess immediately after result detection
- Calls /reload-models endpoint to hot-swap the new model files without API server restart
- Retraining completes in approximately 3-5 minutes
- Models improve progressively throughout the IPL season — every match adds training data
Auto-retraining means the system is always as accurate as possible. A model trained after match 50 has seen 50 more matches than the opening-day model — and the walk-forward architecture ensures this additional data is used correctly.


# 6. Accuracy & Validation

All accuracy figures are computed on held-out test data using strict walk-forward methodology. No test-year data ever enters training — the figures below are conservative, real-world estimates.


## Backtest Methodology
- 11-year walk-forward backtest (2015-2025)
- Protocol: train on [2008, ..., year-1], test on [year] — strict forward-only information flow
- No data leakage: each test year is completely isolated from training computation
- Metrics reported: prediction accuracy, Brier score (calibration quality), and log-loss
- Brier score measures whether stated probabilities match observed outcomes — not just win/loss
- Log-loss penalises overconfident wrong predictions — ensuring the model is well-calibrated

A 63% pre-match accuracy sounds modest — but against a 50% random baseline across 900+ matches, it represents a statistically significant and practically valuable edge. Over a full 74-match IPL season, this translates to approximately 10 additional correct predictions compared to random.


# 7. Outputs & Deliverables


## 7.1  Telegram Messages
- Pre-toss update: win probabilities for both teams, key pre-match factors, Claude AI narrative analysis
- Post-toss update: revised probabilities incorporating toss alignment signal
- Per-over first innings: current score, wickets, projected total, batting-first win probability
- Per-over second innings: required run rate, chase win probability, status (on track / behind / critical)
- Final result: match winner, victory margin, confirmation of auto-retrain completion

## 7.2  Excel Tracker (ipl_predictions.xlsx)
Every match is logged to a structured Excel workbook with the following column groups:


- Color coding applied to win probability cells: Green (>=60%), Yellow (41-59%), Red (<=40%)
- Full season history maintained — every match, every over, every prediction preserved

## 7.3  Model Artifacts

- All three models are hot-reloadable via the /reload-models API endpoint without restarting the server
- Models are versioned by training date — previous versions preserved automatically


# 8. Why This System Is Different

Most cricket prediction models are either too simple (basic win rate lookups) or too opaque (black-box neural networks with no interpretability). This system combines rigorous ML methodology with cricket domain knowledge to produce predictions that are accurate, explainable, and continuously improving.

- Phase-aware architecture: Three separate models trained specifically for pre-match, first innings, and second innings stages. Most models apply one approach to all situations — this system recognises that the information available and relevant changes completely throughout a match.

- Zero data leakage: Strict chronological walk-forward methodology enforced at every level — feature computation, model training, and backtesting. Reported accuracy figures are conservative and trustworthy, not inflated by future information.

- Venue-month interaction: Captures seasonal and environmental conditions that a simple venue win rate misses entirely. The same stadium behaves differently in March vs May — and this model quantifies that difference.

- Powerplay intelligence: Phase-locking at ball 36 ensures powerplay statistics are always historically accurate, regardless of when the bot starts polling. This engineering detail protects the most important feature group in the live models.

- Calibrated probabilities: Isotonic regression calibration on the second-innings model ensures that a stated 70% win probability really means the chasing team wins 70% of the time — not just that they are more likely than not to win. Calibration is rarely implemented in sports models.

- Self-improving system: After every match, models automatically retrain and hot-reload. A model trained at the end of the IPL season has seen every match that season — it is materially more accurate than an opening-day model.

- Full-stack automation: The system handles the complete workflow — schedule detection, toss polling, live scoring, prediction delivery, result logging, and model retraining — with zero manual intervention from a scheduled start.

- 18 years of quality data: 900+ IPL matches from 2008-2026 with consistent ball-by-ball feature engineering. Enough historical depth to estimate venue-specific, team-specific, and month-specific statistics reliably.

No other publicly available IPL prediction system combines phase-aware multi-model design, isotonic probability calibration, strict data leakage prevention, and full match lifecycle automation in a single deployable package.


# 9. Technical Specifications



The system is designed for stability and low maintenance overhead. All dependencies are pinned to tested versions, the FastAPI server is stateless (models loaded in memory), and the Telegram bot recovers gracefully from API timeouts and rate limit errors through exponential backoff.

The system is deployable on any machine with Python 3.10+ and internet access. No GPU required. Typical memory footprint is under 500MB with all three models loaded. The FastAPI server can handle concurrent requests from multiple bot instances.

| File | Role | Output |
| --- | --- | --- |
| 01_parse.py | Parse Cricsheet JSON ball-by-ball data | matches.csv, deliveries.csv |
| 02_features.py | Feature engineering: ELO, form, venue stats | match_features.csv |
| 03_train.py | Train 3 ML models with hyperparameter tuning | 3 .pkl model files |
| 04_api.py | FastAPI REST server on port 8000 | Live prediction API |
| match_bot.py | Telegram bot + live match orchestration | Per-over Telegram updates |
| live_backtest.py | Walk-forward backtest 2015-2025 | Per-season accuracy reports |
| Feature Name | What It Captures |
| --- | --- |
| ELO RATINGS | ELO RATINGS |
| elo_diff | Team1 Elo minus Team2 Elo. 1500 baseline, K=24, chess-standard |
| FORM METRICS (5 windows) | FORM METRICS (5 windows) |
| form_diff | 5-game rolling win rate differential |
| form_3_diff | 3-game short-term form |
| form_10_diff | 10-game long-term form |
| form_weighted_diff | Exponential weights [0.35, 0.25, 0.20, 0.12, 0.08] |
| HEAD-TO-HEAD | HEAD-TO-HEAD |
| h2h_win_rate_team1 | Win rate in last 10 head-to-head matches |
| TOSS FEATURES | TOSS FEATURES |
| team1_won_toss | Did Team 1 win the toss? |
| toss_chose_bat | Did the toss winner choose to bat? |
| toss_venue_aligned | KEY SIGNAL: Did the toss winner make venue-optimal decision? 58% win if aligned vs 42% if misaligned |
| VENUE FEATURES | VENUE FEATURES |
| venue_bat_first_win_rate | Historical bat-first win rate at this venue |
| venue_chase_win_rate | Historical chase win rate at this venue |
| venue_avg_first_innings | Average 1st innings score (default 165 if no history) |
| team1_venue_win_rate | Team1's win rate at this specific venue |
| team2_venue_win_rate | Team2's win rate at this specific venue |
| venue_month_chase_wr | Monthly venue-chase interaction (e.g., Ahmedabad: April 73%, May 33%) |
| PLAYER STRENGTH | PLAYER STRENGTH |
| team1_bat_strength | Mean score of top-5 batsmen: SR x 0.5 + Avg x 50 |
| team2_bat_strength | Same for Team2 |
| team1_bowl_strength | Mean score of top-3 bowlers: wkts/6 x 25 minus economy |
| team2_bowl_strength | Same for Team2 |
| bat_diff | team1_bat minus team2_bat |
| bowl_diff | team1_bowl minus team2_bowl |
| CHASE METRICS | CHASE METRICS |
| team1_chase_wr | Team1's historical win rate when batting second |
| team2_chase_wr | Team2's historical win rate when batting second |
| chase_advantage_diff | Combined chase capability differential |
| SEASON CONTEXT | SEASON CONTEXT |
| match_num_in_season | Match 1-60; early season = higher uncertainty |
| is_playoff | 1 if match_num > 56 |
| is_march | March is the only bat-first dominant month (48.6%) |
| Feature Name | What It Captures |
| --- | --- |
| MATCH STATE | MATCH STATE |
| ball_num | Current ball (1-120) |
| balls_remaining | Balls left in innings |
| balls_pct | Progress through innings (0-1) |
| cum_runs | Runs scored so far |
| runs_needed | Target minus current runs |
| cum_wickets | Wickets fallen |
| wickets_left | Wickets remaining (10 minus cum_wickets) |
| wickets_pct | Wickets lost as fraction |
| crr | Current run rate (runs per 6 balls) |
| rrr | Required run rate |
| rrr_diff | crr minus rrr (positive = ahead) |
| run_rate_ratio | crr / rrr |
| MOMENTUM | MOMENTUM |
| partnership_runs | Runs since last wicket |
| partnership_balls | Balls since last wicket |
| last_3ov_runs | Runs in last 18 balls |
| last_3ov_wkts | Wickets in last 18 balls |
| boundary_pct | Cumulative boundary percentage |
| dot_ball_pct | Cumulative dot ball percentage |
| CONTEXT | CONTEXT |
| first_innings_run_rate | How fast the batting-first team scored |
| target_vs_venue_avg | Is the target above or below venue average? |
| first_innings_wickets | How many wickets did batting-first team lose? |
| POWERPLAY (locked at over 6) | POWERPLAY (locked at over 6) |
| is_pp | Currently in powerplay? (balls <= 36) |
| pp_runs | Runs scored in overs 1-6 |
| pp_wickets | Wickets in powerplay |
| pp_run_rate | Powerplay run rate (pp_runs/36 x 6) |
| pp_req_rate | Required rate from over 7 onwards ((target-pp_runs)/84 x 6) |
| pp_rate_gap | pp_run_rate minus pp_req_rate (positive = ahead at powerplay end) |
| Feature Name | What It Captures |
| --- | --- |
| SCORE STATE | SCORE STATE |
| cum_runs | Core scorecard: runs scored so far |
| cum_wickets | Core scorecard: wickets fallen |
| crr | Core scorecard: current run rate |
| balls_remaining | How far through the innings |
| balls_pct | Progress as fraction (0-1) |
| wickets_pct | Fraction of wickets lost |
| PROJECTION | PROJECTION |
| projected_score | crr x 20 overs = simple linear projection |
| venue_avg | Historical average first innings score at this venue |
| score_vs_expected | cum_runs minus (venue_avg x progress) = above/below pace |
| score_vs_expected_pct | Ratio version of score_vs_expected |
| PARTNERSHIP | PARTNERSHIP |
| partnership_runs | Active partnership: runs since last wicket |
| partnership_balls | Active partnership: balls since last wicket |
| RECENT FORM | RECENT FORM |
| last_3ov_runs | Last 18 balls: recent run scoring rate |
| last_3ov_wkts | Last 18 balls: wickets fallen recently |
| boundary_pct | Cumulative boundary percentage |
| dot_pct | Cumulative dot ball percentage |
| acceleration | Recent run rate minus early run rate |
| TEAM STRENGTH | TEAM STRENGTH |
| elo_diff | Team strength: same as pre-match ELO feature |
| form_diff | Team strength: same as pre-match form feature |
| POWERPLAY (locked at over 6) | POWERPLAY (locked at over 6) |
| is_pp | In powerplay? (over 1-6) |
| pp_runs | Powerplay runs (locked at end of over 6) |
| pp_wickets | Powerplay wickets (locked at end of over 6) |
| pp_vs_venue_avg | pp_runs / (venue_avg x 6/20): powerplay vs venue expectation |
| Over | Balls Delivered | Model Accuracy |
| --- | --- | --- |
| 3 | 18 | 68% |
| 6 | 36 | 70% |
| 10 | 60 | 72% |
| 15 | 90 | 73% |
| 20 | 120 | 75% |
| Model | Stage | Algorithm | Accuracy | Notes |
| --- | --- | --- | --- | --- |
| Pre-match | Before toss | XGBoost (Optuna) | 63% | Random baseline = 50% |
| Pre-match | After toss | XGBoost (Optuna) | 65% | Toss alignment drives uplift |
| 1st Innings | Over 3 | LightGBM | 68% | Early stage |
| 1st Innings | Over 10 | LightGBM | 72% | Mid-innings |
| 1st Innings | Over 20 | LightGBM | 75% | Full innings complete |
| 2nd Innings | Any over | LightGBM (calibrated) | 75% | Powerplay features key |
| Column Range | Content |
| --- | --- |
| A-H | Match metadata: teams, venue, toss winner, toss decision, date, season, match number |
| I-P | Pre-match prediction: win probabilities, predicted winner, confidence level, Claude AI narrative summary |
| Q-BT | Per-over win probability and score: 20 overs x 2 columns each (win%, cumulative score) |
| BE-BJ | Result: actual winner, margin, accuracy flag (correct/incorrect), margin type |
| File | Contents |
| --- | --- |
| prematch_model.pkl | XGBoost classifier + feature list + training medians for imputation |
| live_model.pkl | LightGBM classifier + scaler + feature list (2nd innings) |
| inn1_live_model.pkl | LightGBM classifier + scaler + feature list (1st innings) |
| Component | Specification |
| --- | --- |
| Language | Python 3.10+ |
| ML Frameworks | XGBoost 2.x, LightGBM 4.x, scikit-learn |
| Hyperparameter Tuning | Optuna (60 trials per model) |
| API Framework | FastAPI + Uvicorn |
| Live Data Source | CricAPI + Cricbuzz HTML scrape |
| Delivery Channel | Telegram Bot API |
| Excel Output | openpyxl |
| AI Explanations | Anthropic Claude API |
| Historical Data Source | Cricsheet JSON (ball-by-ball) |
| Historical Coverage | 2008-2026 (18 seasons, 900+ matches) |
| API Rate Limit | 100 calls/day per key (dual key = 200/day) |
| Retrain Time | Approximately 3-5 minutes after match end |