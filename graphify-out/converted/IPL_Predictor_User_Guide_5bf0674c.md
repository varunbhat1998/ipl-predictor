<!-- converted from IPL_Predictor_User_Guide.docx -->


IPL Match Prediction Bot
Subscriber User Guide & Standard Operating Procedure

v4.0  |  IPL 2026
Confidential — Subscriber Distribution Only

# Table of Contents

# Section 1: System Overview
## 1.1 What You Get
As a subscriber you receive the following capabilities for every IPL match:
- Real-time win probability updates via Telegram for every IPL match
- Pre-toss prediction (90 min before match)
- Post-toss updated prediction (after toss)
- Per-over live updates throughout 1st and 2nd innings
- Self-improving AI: models retrain after every match
- Excel tracker: full season record of all predictions and accuracy

## 1.2 Three-Model Architecture
The system uses three distinct machine learning models, each specialised for a different stage of the match:

## 1.3 System Components
### Python Files

### Model Files (models/ folder)

### Data Files (data/ folder)

# Section 2: Installation & Setup
## 2.1 Prerequisites
- Python 3.10 or higher
- Node.js 18+ (for any JS utilities)
- pip install: pandas numpy scikit-learn xgboost lightgbm optuna fastapi uvicorn openpyxl requests anthropic

## 2.2 Environment Variables
Create a .env file in the project root with the following variables:

## 2.3 First-Time Setup Steps
- Run python 01_parse.py — parses Cricsheet JSONs into matches.csv and deliveries.csv
- Run python 02_features.py — builds feature matrix (takes ~2–5 min)
- Run python 03_train.py — trains all 3 models (takes ~5–15 min with Optuna)
- Run python 04_api.py — starts FastAPI server on port 8000 (keep running in background)
- Run python match_bot.py — starts the live match bot

## 2.4 Verifying Setup
Check API health:
GET http://localhost:8000/health
Expected response:
{"status": "ok", "model_version": "4.0", "latest_season": "2026"}
Bot startup output confirms:
- Telegram: OK
- Claude: OK (if ANTHROPIC_API_KEY set)
- CricAPI afternoon/evening keys: SET

# Section 3: Standard Operating Procedure (SOP)
## 3.1 Daily Match Day SOP
### STEP 1 — Before Match Day (Evening Prior)
- Verify API server is running: curl http://localhost:8000/health
- Verify match_bot.py is not already running (check Task Manager / ps aux)
- Confirm today's match is in ipl_schedule_2026.py

### STEP 2 — Start Bot
- Start API server (if not running): python 04_api.py
- Start match bot: python match_bot.py
- Bot will print: "Found N IPL match(es) today" and begin its lifecycle

### STEP 3 — Monitor (Optional)
- Watch console for phase transitions and score updates
- Check Telegram channel for each prediction message
- Monitor ipl_predictions.xlsx for live data logging

### STEP 4 — After Match
- Bot auto-retrains: "Running auto-retrain..." will appear in console
- Retrain takes ~3–5 minutes
- Bot posts "Retrain complete" to Telegram
- Models are hot-reloaded without restarting the API

### STEP 5 — Double-Header Days
- When two matches are scheduled (afternoon + evening), the bot handles both automatically
- Afternoon key used for 3:30 PM match, evening key for 7:30 PM match
- Both predictions appear in the same Telegram channel

## 3.2 Weekly Maintenance SOP
- ipl_predictions.xlsx — check for accuracy tracking
- Confirm player databases are up to date (player_database_2026.csv)
- Archive Excel if file becomes too large (>200 rows)
- Review Telegram message history for missed predictions

## 3.3 Manual Retrain SOP (If Needed)
- Stop match_bot.py (Ctrl+C)
- Run: python 03_train.py
- Wait for "Training complete" output
- Restart match_bot.py
OR (if API is running):
POST http://localhost:8000/reload-models

## 3.4 If the Bot Crashes During a Match
- Note the current match phase from console log (pre-toss / inn1 / inn2)
- Restart match_bot.py — it will detect current match state from CricAPI
- If mid-innings: it may miss 1–2 over predictions but will resume automatically
- Check Excel: manually fill any missed over columns if needed

# Section 4: Code Reference — All Files with Key Functions
## 4.1 match_bot.py — The Orchestrator

### Key Constants
- TEAM_SHORT dict — maps team full names to abbreviations (CSK, MI, RCB, etc.)
- CRICBUZZ_SHORT_TO_FULL — reverse mapping
- _match_state dict — tracks all live match state (runs, wickets, phase, XI, powerplay)
- inn1_pp_runs / inn1_pp_wkts — powerplay lock variables (set exactly at over 6)

## 4.2 04_api.py — The Prediction Server

### Request Schema Highlights
/predict/prematch:  team1, team2, venue, toss_winner (optional), toss_decision (optional), team1_players (list), team2_players (list)
/predict/live:  batting_team, bowling_team, runs_scored, wickets_fallen, balls_bowled, target, venue + optional: partnership_runs, partnership_balls, last_3ov_runs, last_3ov_wkts, boundary_pct, dot_ball_pct, first_innings_wickets, pp_runs, pp_wickets
/predict/live_inn1:  batting_team, bowling_team, runs_scored, wickets_fallen, balls_bowled, venue, pp_runs (optional), pp_wickets (optional)

## 4.3 03_train.py — Model Training

## 4.4 02_features.py — Feature Engineering

## 4.5 01_parse.py — Data Parser
- Input: Cricsheet JSON files (one per match)
- Output: matches.csv and deliveries.csv
- matches.csv columns: match_id, date, season, venue, team1, team2, winner, toss_winner, toss_decision, etc.
- deliveries.csv columns: match_id, inning, over, ball, batsman, bowler, runs, wickets, etc.

## 4.6 live_backtest.py — Backtester
- Walk-forward: trains on all seasons BEFORE the test season
- Tests on seasons 2015–2025 one at a time
- Reports: per-season accuracy, per-over accuracy (INN1 and INN2), pp_rate_gap activation analysis
- Output: printed table and backtest_results.csv

# Section 5: Dos and Don'ts
## 5.1 DOs
Follow these practices to keep the system running reliably throughout the IPL season.
- DO keep 04_api.py running at all times — match_bot.py depends on it for all predictions
- DO use separate CricAPI keys for afternoon and evening matches — prevents hitting daily limits
- DO let the bot retrain automatically after each match — this is how the model improves during the season
- DO check Telegram messages manually if a match has unusual timing — bot may need schedule adjustment
- DO monitor the console for "[CB XI] No player entries found" — if repeated, check Cricbuzz access
- DO keep ipl_schedule_2026.py up to date — bot reads match times from here
- DO run python 03_train.py at the start of each season with fresh Cricsheet data
- DO keep player_database_2026.csv current — player strength features depend on it
- DO back up ipl_predictions.xlsx weekly — it accumulates all season accuracy data
- DO use the /status Telegram command to check bot phase mid-match
- DO set ANTHROPIC_API_KEY for richer Telegram narratives (explains WHY the model picked the winner)
- DO watch for "API calls remaining" in console — stay well under 100/day per key

## 5.2 DON'Ts
Avoid these actions — they can corrupt data, crash the bot, or produce wrong predictions.
- DON'T run two instances of match_bot.py simultaneously — causes duplicate Telegram messages and double API calls
- DON'T run python 03_train.py while a match is in progress — mid-match retrain disrupts live predictions
- DON'T edit match_features.csv or deliveries.csv manually — any edit corrupts the feature engineering pipeline
- DON'T skip python 02_features.py after updating raw data — models will train on stale features
- DON'T hardcode team names differently from TEAM_SHORT keys — the matching system is case-sensitive
- DON'T delete .pkl model files during a live match — the API will crash on next reload
- DON'T exceed 100 CricAPI calls/day per key — the API will return errors and live scores will be unavailable
- DON'T change the column order in ipl_predictions.xlsx — the Excel writer uses fixed column indices
- DON'T rename any of the 8 core Python files — they import from each other by name
- DON'T use Windows line endings (CRLF) in .env files — some environment variable parsers fail
- DON'T stop the API server mid-match — match_bot.py cannot recover prediction calls
- DON'T manually edit the .pkl model files — they are binary pickle format and will corrupt if touched

# Section 6: Telegram Commands
The bot listens for the following commands in a background thread. Send them in the Telegram group where the bot is active.


# Section 7: Troubleshooting & FAQs
## 7.1 FAQ: Pre-Match Predictions
Q: Why does the pre-toss prediction show 50/50?
A: This is correct behavior. Pre-toss, without toss information, some matchups are genuinely balanced. After toss, the prediction updates significantly — especially for venue-aligned toss decisions.
Q: Why did the prediction not change much after toss?
A: The toss-venue alignment signal is the main driver. If the toss winner made the expected decision (e.g., choosing to field at a chase-friendly venue), the model confirms it. If they did the unexpected, the model may still lean toward the better team. ELO and form dominate when teams are mismatched.
Q: The bot shows 75% for Team A but Team B won. Is the model broken?
A: No. 75% means Team A wins 3 out of 4 times in similar situations. In any individual match, the other 25% scenario can occur. Evaluate accuracy over 20+ matches.
Q: Why are player strengths sometimes labeled "team_profile_average"?
A: This means the playing XI wasn't available (not yet announced or Cricbuzz scraping failed). The model uses the team's average player database profile instead of actual XI players.

## 7.2 FAQ: Live Predictions
Q: The live win% jumped dramatically in one over. Why?
A: Large jumps are caused by wickets (a wicket in the powerplay can shift probability by 10–15%) or a high-scoring over (6s/4s push up the run rate ratio). This is correct — the model responds to actual match events.
Q: The 2nd innings prediction is stuck showing the same % for several overs.
A: If the API is unavailable, match_bot.py shows the last known prediction. Check that 04_api.py is still running.
Q: The powerplay stats were not applied. I see "pp_runs=None" in console.
A: The bot missed the exact over-6 polling window. The powerplay lock triggers only at over==6. If the first poll came at over 7 or later, powerplay stats are not available and the model uses baseline estimates.
Q: Why does the 1st innings prediction fluctuate more in early overs?
A: Early overs have higher variance — a single wicket or big over has massive impact. The model is correctly reflecting genuine uncertainty. Predictions stabilize from over 10 onwards.

## 7.3 FAQ: Playing XI
Q: The playing XI shows wrong players or shows a coach name.
A: This was a known bug (fixed in v4.0). The RSC scraping now uses [^}] bounded patterns to prevent cross-object JSON matching, and shortName/teamSName extraction ensures correct team assignment. Update to the latest version.
Q: Playing XI is blank in the Telegram message.
A: Either Cricbuzz is blocking the scrape (try VPN), the XI was not yet announced (retry in 60 seconds, which the bot does automatically), or the Cricbuzz match ID was not found. Check console for "[CB XI]" log lines.
Q: The bot says "PBKS XI (0 mapped)". What does this mean?
A: The players were fetched but none matched the model's name database. This happens for new players. Update name_map.py with the new player entries. Mapped count 0 means XI data will not be used for player strength features.

## 7.4 FAQ: API & Server
Q: API returns {"detail": "Not Found"} for /predict/prematch.
A: The API server is not running or is running on a different port. Run: python 04_api.py and check for "Uvicorn running on http://0.0.0.0:8000".
Q: API returns 500 error mid-match.
A: Usually caused by a missing feature value. Check console for KeyError. Most common cause: team name not in TEAM_SHORT dict. Add the new team name.
Q: "model_version: 3.0" but I just retrained. Why?
A: The model version in the health endpoint is hardcoded in 04_api.py. To update it, edit the VERSION variable in 04_api.py. This does not affect prediction quality.
Q: How do I run the API on a different port?
A: Edit 04_api.py, change uvicorn.run(app, host="0.0.0.0", port=8000) to your preferred port. Also update API_BASE environment variable in match_bot.py.

## 7.5 FAQ: Excel & Logging
Q: ipl_predictions.xlsx is not being updated.
A: Check that 04_api.py is running AND that the /update-excel/* endpoints are reachable. The Excel writer is part of the API server, not match_bot.py.
Q: Excel columns are misaligned after editing the file.
A: The Excel writer uses fixed column indices (A=1, B=2, etc.). If you insert or delete columns, it will write to wrong cells. Restore from backup or recreate from the standard template.
Q: Some over columns are empty (no win%).
A: The bot may have been offline during those overs, or the score update came in between polling intervals. This is normal — the bot polls every 3–4 minutes and may skip an over if it was between calls.

## 7.6 FAQ: Retraining
Q: Retrain fails with "FileNotFoundError: match_features.csv".
A: Run python 02_features.py first to generate the feature file. Then run python 03_train.py.
Q: Retrain takes more than 20 minutes.
A: This usually means Optuna is running too many trials. Reduce n_trials from 60 to 30 in 03_train.py for faster training. Optuna finds 80% of optimal performance in the first 30 trials.
Q: The model accuracy dropped after retrain.
A: This can happen when recent season data is added. The model is seeing new teams, new venues, or new batting conditions. Accuracy typically recovers within 3–5 matches as the model adapts.
Q: Can I retrain without running a full Optuna tuning?
A: Yes. In 03_train.py, set USE_OPTUNA = False (if that flag exists) or manually set best_params to a reasonable XGBoost config and skip the study.optimize() call.

## 7.7 FAQ: Data & Sources
Q: Where does the historical match data come from?
A: Cricsheet (cricsheet.org) — free, community-maintained ball-by-ball JSON files for all IPL matches from 2008 onwards. Download the IPL JSON pack and place in the /data/raw/ directory before running 01_parse.py.
Q: How often should I update the Cricsheet data?
A: At the end of each IPL season. The auto-retrain captures in-season matches from the live bot. For off-season use, download the full updated Cricsheet pack.
Q: Can I add non-IPL data (e.g., international T20s) to improve the model?
A: Not recommended without significant feature engineering changes. Venue stats, team ELO, and player databases are IPL-specific. Adding other formats would require separate normalization.
Q: The venue "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur" returns 165 as venue avg.
A: This is the global default when venue history is sparse (< 5 matches). The model uses 165 as a conservative average. Accuracy will improve as the venue builds up historical data.

# Section 8: Known Limitations
The following limitations are inherent to the current system design and should be understood when interpreting predictions:

- Weather & dew factor — not modeled. Dew heavily favors chasing teams in evening matches; the venue_month_chase_wr feature partially captures this but does not use actual weather data
- Pitch behavior — not modeled. A rank turner or green seamer affects outcomes independently of historical venue stats
- Player injuries — not real-time. If a key player is injured during the match, the model does not update player strength mid-innings
- Umpiring decisions — not modeled. DRS outcomes, no-balls, etc. affect match state but are treated the same as regular balls
- New venues — the model defaults to 165 runs average for venues with < 5 historical matches; accuracy is lower for these
- New teams — a team's first 5 matches will have thin ELO and form history; confidence intervals are wider
- Super Overs — not modeled; the bot does not track super over scenarios
- Rain interruptions — DLS-adjusted targets are consumed as-is; no DLS-specific modeling

# Section 9: Glossary
Key terms used throughout this guide and the codebase:

| Model | When | What It Predicts | Accuracy |
| --- | --- | --- | --- |
| Pre-Match | 90 min before toss | Match winner | 63% |
| 1st Innings Live | Over 1–20 | Will batting team post winning total? | 70–75% |
| 2nd Innings Live | Over 1–20 | Will chasing team win? | 75% |
| File | Purpose |
| --- | --- |
| 01_parse.py | Parse raw Cricsheet JSON data into CSVs |
| 02_features.py | Compute ELO, form, venue stats, player strength |
| 03_train.py | Train all 3 ML models (XGBoost + LightGBM) |
| 04_api.py | FastAPI server — serves live predictions on port 8000 |
| match_bot.py | Telegram bot — orchestrates full match lifecycle |
| live_backtest.py | Walk-forward backtest 2015–2025 |
| excel_writer.py | Excel logging via API endpoints |
| name_map.py | Player name normalization (CricAPI → model names) |
| ipl_schedule_2026.py | 2026 IPL schedule with match slots |
| File | Contents |
| --- | --- |
| prematch_model.pkl | XGBoost classifier + 33 feature names + train medians |
| live_model.pkl | LightGBM + CalibratedClassifierCV + 26 features + scaler |
| inn1_live_model.pkl | LightGBM + 23 features + scaler |
| File | Contents |
| --- | --- |
| match_features.csv | One row per historical match with all computed features |
| player_bat_stats.csv | Career batting stats per player per season |
| player_bowl_stats.csv | Career bowling stats per player per season |
| ipl_predictions.xlsx | Live Excel tracker (auto-updated during matches) |
| Variable | Required | Description |
| --- | --- | --- |
| TELEGRAM_BOT_TOKEN | Required | Your Telegram bot token from @BotFather |
| TELEGRAM_CHAT_ID | Required | Channel/group ID to receive predictions |
| CRICAPI_KEY | Required | Default CricAPI key (100 calls/day) |
| CRICAPI_KEY_AFTERNOON | Recommended | Dedicated key for 3:30 PM matches |
| CRICAPI_KEY_EVENING | Recommended | Dedicated key for 7:30 PM matches |
| ANTHROPIC_API_KEY | Optional | Enables Claude AI narrative explanations |
| API_BASE | Optional | Default: http://localhost:8000 |
| Function | Purpose |
| --- | --- |
| get_cricbuzz_playing11(cb_match_id, cb_slug) | Fetch playing XI from Cricbuzz using mobile API or RSC scraping |
| get_cricbuzz_score(match_id, slug, bat_first, bat_second) | Scrape live score from Cricbuzz HTML |
| ml_prematch(team1, team2, venue, toss_winner, toss_decision, team1_players, team2_players) | Call /predict/prematch API endpoint |
| ml_live_inn1(batting_team, bowling_team, runs, wickets, balls, venue, pp_runs, pp_wickets) | Call /predict/live_inn1 endpoint |
| ml_live_inn2(batting_team, bowling_team, runs, wickets, balls, target, venue, pp_runs, pp_wickets) | Call /predict/live endpoint |
| run_match(match_info, slot) | Main match lifecycle (6 phases) |
| xi_to_data_names(players) | Convert CricAPI names to model data names |
| send_telegram(text) | Send HTML-formatted message to Telegram |
| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | /health | Check API and model status |
| GET | /teams | List all known teams |
| GET | /venues | List all known venues |
| POST | /predict/prematch | Pre-match or post-toss prediction |
| POST | /predict/live | 2nd innings live prediction |
| POST | /predict/live_inn1 | 1st innings live prediction |
| POST | /reload-models | Hot-reload models from disk |
| POST | /update-excel/prematch | Log pre-match prediction to Excel |
| POST | /update-excel/live | Log over-by-over live win% to Excel |
| POST | /update-excel/result | Log final result to Excel |
| Key Section | Description |
| --- | --- |
| MODEL A (pre-match) | Lines ~25–130 — XGBoost + Optuna |
| MODEL B (2nd innings) | Lines ~200–360 — LightGBM + CalibratedClassifierCV |
| MODEL C (1st innings) | Lines ~440–600 — LightGBM standalone |
| INN2 powerplay features | Built using ball_num==36 merge |
| INN1 powerplay features | pp_runs_at6 locked at over_num==6 in per-ball loop |
| Feature lists | INN1_FEATURES (23 items), LIVE_FEATURES (26 items) |
| Computation | Details |
| --- | --- |
| ELO ratings | K=24, init=1500 — expanding chronologically |
| Team form | 3-game, 5-game, 10-game, exponential weighted — expanding |
| Head-to-head | Last 10 matches — expanding |
| Venue stats | Bat-first rate, chase rate, avg score — expanding |
| Player strength | Bat score = SR×0.5 + Avg×50, Bowl score = wkts/6×25 − eco |
| venue_month_chase_wr | (venue, month) interaction — expanding |
| is_march | Binary flag |
| Command | When to Use | What It Does |
| --- | --- | --- |
| predict | Any time during match | Sends current prediction to your chat |
| status | Any time | Shows current match phase, teams, score |
| /xi MI: Rohit, Bumrah... KKR: Iyer... | Before toss if XI not fetched | Manually set playing XI for post-toss prediction |
| Term | Definition |
| --- | --- |
| ELO Rating | A chess-derived rating system capturing team strength. Starts at 1500, updates after every match |
| Walk-forward backtest | Backtesting method where models are only trained on data prior to the test period — no future leakage |
| Toss-venue alignment | When the toss winner makes the venue-optimal decision (e.g., field first at a chase-friendly venue) |
| Powerplay | Overs 1–6 in T20 cricket. Only 2 fielders allowed outside the inner ring. High-impact phase |
| CRR | Current run rate: runs per 6 balls at current point in innings |
| RRR | Required run rate: runs per 6 balls needed to win from current point |
| pp_rate_gap | Powerplay run rate minus powerplay required rate. Positive = chasing team was ahead after 6 overs |
| Isotonic calibration | A method to ensure ML model probabilities are statistically accurate, not just ranked correctly |
| GroupKFold CV | Cross-validation where all rows from the same match (group) stay together — prevents same-match leakage |
| RSC payload | React Server Component data embedded in Cricbuzz Next.js HTML pages, containing match/player JSON |
| data_name | The standardized player name used in the training data (e.g., "V Kohli" not "Virat Kohli") |
| Expanding mean | A running average computed only from past data, never including the current or future rows |
| Hot-reload | Replacing ML models in memory without restarting the API server |