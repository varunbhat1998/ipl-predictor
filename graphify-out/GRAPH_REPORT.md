# Graph Report - .  (2026-04-24)

## Corpus Check
- 72 files · ~155,858 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 590 nodes · 784 edges · 68 communities detected
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Match Bot Core Functions|Match Bot Core Functions]]
- [[_COMMUNITY_API & Model Architecture|API & Model Architecture]]
- [[_COMMUNITY_Pre-Toss & XI Estimation|Pre-Toss & XI Estimation]]
- [[_COMMUNITY_Match Logging & Reporting|Match Logging & Reporting]]
- [[_COMMUNITY_Match Bot Pipeline|Match Bot Pipeline]]
- [[_COMMUNITY_Excel Output Layer|Excel Output Layer]]
- [[_COMMUNITY_Player Scoring API|Player Scoring API]]
- [[_COMMUNITY_Model Issues & Observations|Model Issues & Observations]]
- [[_COMMUNITY_Feature Engineering|Feature Engineering]]
- [[_COMMUNITY_Toss Replay & Validation|Toss Replay & Validation]]
- [[_COMMUNITY_Feature Engineering Pipeline|Feature Engineering Pipeline]]
- [[_COMMUNITY_Player Database Builder|Player Database Builder]]
- [[_COMMUNITY_Full Backtest Suite|Full Backtest Suite]]
- [[_COMMUNITY_2024-25 Model Retraining|2024-25 Model Retraining]]
- [[_COMMUNITY_Unified Live Model|Unified Live Model]]
- [[_COMMUNITY_Pre-Match Model Training|Pre-Match Model Training]]
- [[_COMMUNITY_X-Factor Calibration|X-Factor Calibration]]
- [[_COMMUNITY_Player Profile Backtest|Player Profile Backtest]]
- [[_COMMUNITY_5-Year Backtest|5-Year Backtest]]
- [[_COMMUNITY_Supporting Utilities 19|Supporting Utilities 19]]
- [[_COMMUNITY_Supporting Utilities 20|Supporting Utilities 20]]
- [[_COMMUNITY_Supporting Utilities 21|Supporting Utilities 21]]
- [[_COMMUNITY_Supporting Utilities 22|Supporting Utilities 22]]
- [[_COMMUNITY_Supporting Utilities 23|Supporting Utilities 23]]
- [[_COMMUNITY_Supporting Utilities 24|Supporting Utilities 24]]
- [[_COMMUNITY_Supporting Utilities 25|Supporting Utilities 25]]
- [[_COMMUNITY_Supporting Utilities 26|Supporting Utilities 26]]
- [[_COMMUNITY_Supporting Utilities 27|Supporting Utilities 27]]
- [[_COMMUNITY_Supporting Utilities 28|Supporting Utilities 28]]
- [[_COMMUNITY_Supporting Utilities 29|Supporting Utilities 29]]
- [[_COMMUNITY_Supporting Utilities 30|Supporting Utilities 30]]
- [[_COMMUNITY_Supporting Utilities 31|Supporting Utilities 31]]
- [[_COMMUNITY_Supporting Utilities 32|Supporting Utilities 32]]
- [[_COMMUNITY_Supporting Utilities 33|Supporting Utilities 33]]
- [[_COMMUNITY_Supporting Utilities 34|Supporting Utilities 34]]
- [[_COMMUNITY_Supporting Utilities 35|Supporting Utilities 35]]
- [[_COMMUNITY_Supporting Utilities 36|Supporting Utilities 36]]
- [[_COMMUNITY_Supporting Utilities 37|Supporting Utilities 37]]
- [[_COMMUNITY_Supporting Utilities 38|Supporting Utilities 38]]
- [[_COMMUNITY_Supporting Utilities 39|Supporting Utilities 39]]
- [[_COMMUNITY_Supporting Utilities 40|Supporting Utilities 40]]
- [[_COMMUNITY_Supporting Utilities 41|Supporting Utilities 41]]
- [[_COMMUNITY_Supporting Utilities 42|Supporting Utilities 42]]
- [[_COMMUNITY_Supporting Utilities 43|Supporting Utilities 43]]
- [[_COMMUNITY_Supporting Utilities 44|Supporting Utilities 44]]
- [[_COMMUNITY_Supporting Utilities 45|Supporting Utilities 45]]
- [[_COMMUNITY_Supporting Utilities 46|Supporting Utilities 46]]
- [[_COMMUNITY_Supporting Utilities 47|Supporting Utilities 47]]
- [[_COMMUNITY_Supporting Utilities 48|Supporting Utilities 48]]
- [[_COMMUNITY_Supporting Utilities 49|Supporting Utilities 49]]
- [[_COMMUNITY_Supporting Utilities 50|Supporting Utilities 50]]
- [[_COMMUNITY_Supporting Utilities 51|Supporting Utilities 51]]
- [[_COMMUNITY_Supporting Utilities 52|Supporting Utilities 52]]
- [[_COMMUNITY_Supporting Utilities 53|Supporting Utilities 53]]
- [[_COMMUNITY_Supporting Utilities 54|Supporting Utilities 54]]
- [[_COMMUNITY_Supporting Utilities 55|Supporting Utilities 55]]
- [[_COMMUNITY_Supporting Utilities 56|Supporting Utilities 56]]
- [[_COMMUNITY_Supporting Utilities 57|Supporting Utilities 57]]
- [[_COMMUNITY_Supporting Utilities 58|Supporting Utilities 58]]
- [[_COMMUNITY_Supporting Utilities 59|Supporting Utilities 59]]
- [[_COMMUNITY_Supporting Utilities 60|Supporting Utilities 60]]
- [[_COMMUNITY_Supporting Utilities 61|Supporting Utilities 61]]
- [[_COMMUNITY_Supporting Utilities 62|Supporting Utilities 62]]
- [[_COMMUNITY_Supporting Utilities 63|Supporting Utilities 63]]
- [[_COMMUNITY_Supporting Utilities 64|Supporting Utilities 64]]
- [[_COMMUNITY_Supporting Utilities 65|Supporting Utilities 65]]
- [[_COMMUNITY_Supporting Utilities 66|Supporting Utilities 66]]
- [[_COMMUNITY_Supporting Utilities 67|Supporting Utilities 67]]

## God Nodes (most connected - your core abstractions)
1. `run_match()` - 47 edges
2. `EnsemblePreMatchModel` - 35 edges
3. `MatchLogger` - 21 edges
4. `_handle_predict_command()` - 17 edges
5. `now_ist()` - 14 edges
6. `04_api.py â€” FastAPI Prediction Server` - 14 edges
7. `match_bot.py â€” Live Match Bot Orchestrator` - 14 edges
8. `send_telegram()` - 12 edges
9. `main()` - 11 edges
10. `cricapi_get()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `09_train_2024_2025.py -- Retrain all 3 models on 2024 + 2025 data only.  Key d` --uses--> `EnsemblePreMatchModel`  [INFERRED]
  09_train_2024_2025.py → model_classes.py
- `Match venue string to VENUE_COORDS key.` --uses--> `EnsemblePreMatchModel`  [INFERRED]
  10_post_toss_model.py → model_classes.py
- `Fetch weather for each match at the correct hour (afternoon or evening).` --uses--> `EnsemblePreMatchModel`  [INFERRED]
  10_post_toss_model.py → model_classes.py
- `Get playing XI for each team from deliveries.` --uses--> `EnsemblePreMatchModel`  [INFERRED]
  10_post_toss_model.py → model_classes.py
- `Compute aggregated XI batting/bowling strength for a team.` --uses--> `EnsemblePreMatchModel`  [INFERRED]
  10_post_toss_model.py → model_classes.py

## Hyperedges (group relationships)
- **Data Pipeline: Cricsheet JSON -> Parse -> Features -> Train -> API Serve** — devguide_01_parse, devguide_02_features, devguide_03_train, devguide_04_api, devguide_matches_csv, devguide_deliveries_csv, devguide_match_features_csv, arch_prematch_pkl, arch_live_model_pkl, arch_inn1_live_model_pkl, devguide_posttoss_pkl [EXTRACTED 1.00]
- **Live Match Day: Bot + API + CricAPI + Telegram + Auto-Retrain** — devguide_match_bot, devguide_04_api, devguide_06_retrain, concept_cricbuzz_scraping, concept_hot_reload, arch_telegram_output, arch_excel_tracker, arch_six_phase_lifecycle [EXTRACTED 1.00]
- **Recurring Live Bot Issues Across 2026 Season Matches** — issue_probability_stagnation, issue_model_freeze_inn1, issue_null_probabilities, issue_overconfident_prematch, issue_missing_overs, issue_xi_detection_failure, issue_false_rain_alerts, concept_posttoss_overconfidence, issue_long_toss_detection, issue_inn2_model_rigidity [INFERRED 0.82]

## Communities

### Community 0 - "Match Bot Core Functions"
Cohesion: 0.04
Nodes (106): auto_retrain(), _cb_short_to_full(), _check_rain_status(), claude_explain(), compute_weather_adjustment(), conf_label(), cricapi_get(), divider() (+98 more)

### Community 1 - "API & Model Architecture"
Cohesion: 0.05
Nodes (56): FastAPI Server (Uvicorn), Global Avg 1st Innings Score 165.4 (35 venues), GET /health Endpoint, Models Loaded at Startup (including posttoss), Accuracy & Validation (Walk-Forward 2015-2025), Auto-Retrain After Every Match, ELO Rating System (K=24, baseline 1500), inn1_live_model.pkl (1st innings LightGBM) (+48 more)

### Community 2 - "Pre-Toss & XI Estimation"
Cohesion: 0.07
Nodes (45): _blended_player_scores(), estimate_likely_xi(), estimate_toss_probabilities(), _fetch_weather_for_pretoss(), get_player_scores(), _get_team_phase_wkt_wr(), get_xi_strengths(), get_xi_strengths_extended() (+37 more)

### Community 3 - "Match Logging & Reporting"
Cohesion: 0.13
Nodes (9): _html_escape(), MatchLogger, match_logger.py - Structured match event logger for post-match analysis.  Captur, Log a model prediction with full inputs and outputs.          model_type: 'prema, Log each score poll attempt.          source: 'cricapi_matchScore', 'cricbuzz',, Build a compact summary of all predictions for Claude analysis., Build a structured prompt for Claude to analyze the match., Convert to JSON-safe types. (+1 more)

### Community 4 - "Match Bot Pipeline"
Cohesion: 0.11
Nodes (21): Data Source Priority (CricAPI + Cricbuzz Fallback), Telegram Message Outputs, Cricbuzz HTML/RSC Scraping for Live Scores and XI, Model Lock-On Point (correct winner identified mid-match), Match Bot Phase-by-Phase Flow (8 Phases), Fix 1: Match End Detection (Tied/Super Over), Fix 2: Phase Safety Deadlines (match_start + 6h), Fix 3: D/L Target Detection (4 Regex Patterns) (+13 more)

### Community 5 - "Excel Output Layer"
Cohesion: 0.14
Nodes (19): excel_status(), find_or_create_row(), LiveOverWrite, over_score_col(), over_win_pct_col(), PreMatchWrite, 04b_excel_writer.py Adds /update-excel endpoint to the FastAPI app. n8n calls th, Write pre-match info + prediction to Excel. Creates new row. (+11 more)

### Community 6 - "Player Scoring API"
Cohesion: 0.14
Nodes (12): _get_impact_bowl_score_bt(), _match_adv(), norm_venue(), predict_posttoss(), predict_prematch(), predict_unified(), backtest_2024_25.py — Comprehensive backtest of all 3 models on 2024-2025 data, Expanding-window bowl score for impact player at match time. (+4 more)

### Community 7 - "Model Issues & Observations"
Cohesion: 0.13
Nodes (18): Pattern: Post-Toss Model Overconfidence Across Matches, Weather Adjustment (+1.5% to +6.5% per match), Issue: False Rain Delay Detection, Issue: 2nd Innings Model Rigidity (CSK 48.5% at 30 off 12 balls), Issue: Toss Detection Delay (3h 52m at GT vs KKR), Issue: Missing Over Tracking (gaps in per-over predictions), Issue: Model Freeze in 1st Innings (60.9% for 10 consecutive overs), Issue: Pre-Match Overconfidence (70%+ wrong predictions) (+10 more)

### Community 8 - "Feature Engineering"
Cohesion: 0.12
Nodes (11): extract_xi(), fetch_weather_for_matches(), _fuzzy_venue_coords(), _get_impact_bowl_score(), 10_post_toss_model.py - Post-toss prediction model (toss + XI + weather known), Fetch weather for each match at the correct hour (afternoon or evening)., Get playing XI for each team from deliveries., Compute aggregated XI batting/bowling strength for a team. (+3 more)

### Community 9 - "Toss Replay & Validation"
Cohesion: 0.19
Nodes (14): call_posttoss(), check_invariants(), emoji(), load_matches(), main(), player_section(), Replay toss output for historical matches to validate:   1. Team labels match pl, Extract data_names for a team's XI from matches.csv pipe-separated list. (+6 more)

### Community 10 - "Feature Engineering Pipeline"
Cohesion: 0.15
Nodes (11): compute_bat_score(), compute_bowl_score(), get_player_bat_scores(), get_player_bowl_scores(), norm_venue(), 02_features.py  —  Build match_features.csv Fixes data leakage bugs, adds multi-, Normalize venue names — same stadium has 2-3 different names in data., Normalized batting score 0-100. SR=150 → 100, SR=0 → 0. Matches post-toss model (+3 more)

### Community 11 - "Player Database Builder"
Cohesion: 0.17
Nodes (9): bat_score(), bowl_score(), compute_bat_stats(), compute_bowl_stats(), 07_build_player_db.py Build IPL 2026 team/player database with per-venue scores,, Compute batting stats for a player, optionally filtered by venue/recency., Compute bowling stats for a player, optionally filtered by venue/recency., Composite batting score: blend of career + recent + form. (+1 more)

### Community 12 - "Full Backtest Suite"
Cohesion: 0.21
Nodes (8): cell_border(), pct_bar(), 10_backtest_full.py — Walk-forward backtest 2015-2025 For each season: train on, Return a simple bar string for readability., Write a labelled block of key-value pairs., summary_block(), write_data_row(), write_header_row()

### Community 13 - "2024-25 Model Retraining"
Cohesion: 0.22
Nodes (1): 09_train_2024_2025.py -- Retrain all 3 models on 2024 + 2025 data only.  Key d

### Community 14 - "Unified Live Model"
Cohesion: 0.22
Nodes (7): compute_momentum(), _get_phase_wkt_wr(), _phase_aware_projection(), 11_unified_live_model.py — Unified Live Match Predictor  Trains a SINGLE LightGB, Get team's historical phase-wicket win rate for completed phases.     Over 1-6:, Given sorted ball-level rows, compute running momentum state per ball., Phase-aware projected score. Must match 04_api.py exactly.

### Community 15 - "Pre-Match Model Training"
Cohesion: 0.25
Nodes (1): 03_train.py  —  Train both models + backtest on 2023/2024/2025 Pre-match: XGBoos

### Community 16 - "X-Factor Calibration"
Cohesion: 0.32
Nodes (7): bat_score_raw(), bowl_score_raw(), compute_estimated_score(), 07b_apply_x_factor.py Apply calibrated X-factor to assign scores for 82 NA playe, Batting score on 0-100+ scale., Bowling score on 0-100+ scale., Compute estimated IPL score from pre-IPL stats * X-factor.

### Community 17 - "Player Profile Backtest"
Cohesion: 0.29
Nodes (5): compute_team_score(), get_player_scores(), 09_player_profile_backtest.py Backtest player-profile-based prediction model fo, Given pipe-separated player data_names, compute team score.     bat_strength  =, Get blended (career + venue) bat and bowl scores for a player.

### Community 18 - "5-Year Backtest"
Cohesion: 0.25
Nodes (5): _bt_phase_wkt_wr(), prematch_prob(), backtest_5yr.py — 5-year historical backtest (2021-2025)  Models tested:   1. Pr, For backtest: get team_phase_wkt_wr at a given over., P(team1 wins) from pre-match model using match_features row.

### Community 19 - "Supporting Utilities 19"
Cohesion: 0.29
Nodes (7): name_map.py Maps CricAPI full player names → data_name abbreviations used in pla, Convert a CricAPI full player name to data_name format.     Falls back to the or, Convert a list of CricAPI player names to data_name format.     Handles both str, Check all NAME_MAP targets exist in player_database_2026.csv., to_data_name(), verify_map(), xi_to_data_names()

### Community 20 - "Supporting Utilities 20"
Cohesion: 0.25
Nodes (7): fake_score_inn1(), fake_score_inn2(), Test the /predict command at every match phase. Mocks CricAPI so no real API cal, Reset _match_state to defaults., Mock CricAPI response — 1st innings: SRH 95/2 (12.3 ov)., Mock CricAPI response — 2nd innings: RCB 112/3 (14.2 ov), SRH 174/6 done., reset_state()

### Community 21 - "Supporting Utilities 21"
Cohesion: 0.29
Nodes (5): matchup_adv(), 13_h2h_matchups.py — Expanding-window batter vs bowler H2H matchup matrix  For e, Mean H2H advantage of bat-first top batters vs bat-second top bowlers.     > 0.5, Advantage score from batter's perspective: 0=bowler dominates, 1=batter dominate, team_matchup_advantage()

### Community 22 - "Supporting Utilities 22"
Cohesion: 0.29
Nodes (1): Evaluate all 4 models per season (2015-2025).

### Community 23 - "Supporting Utilities 23"
Cohesion: 0.29
Nodes (1): Full match-day dry run — RCB vs SRH, March 28. Simulates every stage: pre-toss →

### Community 24 - "Supporting Utilities 24"
Cohesion: 0.29
Nodes (7): Excel Tracker (ipl_predictions.xlsx), Live Win Probability Per Over (O1-O20), IPL 2026 Match Predictions Sheet, Pre-Match Risk Flags, IPL 2026 Season Accuracy Summary, Toss-Venue Alignment Signal, openpyxl >= 3.1

### Community 25 - "Supporting Utilities 25"
Cohesion: 0.33
Nodes (1): backtest_posttoss.py — Post-toss model backtest for seasons 2021-2025 Rebuilds t

### Community 26 - "Supporting Utilities 26"
Cohesion: 0.33
Nodes (1): Preview all 5 Telegram message templates with mock RCB vs SRH data.

### Community 27 - "Supporting Utilities 27"
Cohesion: 0.5
Nodes (4): load_and_parse(), parse_one_match(), 01_parse.py  —  Parse Cricsheet IPL JSON files → matches.csv + deliveries.csv  H, Returns (match_row dict, list of delivery rows)

### Community 28 - "Supporting Utilities 28"
Cohesion: 0.4
Nodes (3): compute_innings_phase_stats(), 12_pp_wicket_analysis.py  —  Powerplay / Phase Wicket Win Rate Analysis  Builds, For each (file_id, innings), compute wickets at end of each phase.

### Community 29 - "Supporting Utilities 29"
Cohesion: 0.4
Nodes (1): Test full live pipeline against AUS vs WI Women match currently live. Simulates

### Community 30 - "Supporting Utilities 30"
Cohesion: 0.5
Nodes (1): 05_create_excel_tracker.py Creates ipl_predictions.xlsx with proper structure. R

### Community 31 - "Supporting Utilities 31"
Cohesion: 0.5
Nodes (3): 08_update_player_db.py Update player database after each match. Called automatic, Re-run the full player database builder with latest data., update_player_db()

### Community 32 - "Supporting Utilities 32"
Cohesion: 0.5
Nodes (1): Chase-perspective model: reframe prediction as "will chasing team win?" This rem

### Community 33 - "Supporting Utilities 33"
Cohesion: 0.5
Nodes (1): Live model backtest: First innings + Second innings prediction accuracy Walk-for

### Community 34 - "Supporting Utilities 34"
Cohesion: 0.67
Nodes (3): load_scores(), Snapshot test for player bat/bowl score sanity.  Run AFTER rebuilding player_dat, run()

### Community 35 - "Supporting Utilities 35"
Cohesion: 0.5
Nodes (4): 07_build_player_db.py â€” Player Database Builder, 07b_apply_x_factor.py â€” X-Factor Adjustment, 08_update_player_db.py â€” Player DB Updater, data/player_database_2026.csv â€” Player Scores with X-Factor

### Community 36 - "Supporting Utilities 36"
Cohesion: 0.67
Nodes (1): backtest_2026.py — Backtest all 3 models on 2026 IPL season matches.  Pre-match:

### Community 37 - "Supporting Utilities 37"
Cohesion: 0.67
Nodes (1): Test that the XI Analysis section in match_bot.py uses consistent team1/team2 la

### Community 38 - "Supporting Utilities 38"
Cohesion: 0.67
Nodes (3): GRAPH_REPORT: 47 Communities, 439 Nodes, 595 Edges, God Nodes: run_match() (45 edges), EnsemblePreMatchModel (27 edges), Knowledge Gaps: 139 Isolated Nodes

### Community 39 - "Supporting Utilities 39"
Cohesion: 1.0
Nodes (1): 06_retrain_after_match.py Run this after each match result is confirmed. Appen

### Community 40 - "Supporting Utilities 40"
Cohesion: 1.0
Nodes (1): 14_partnership_analysis.py — Partnership Impact Analysis  Computes per-partnersh

### Community 41 - "Supporting Utilities 41"
Cohesion: 1.0
Nodes (1): 15_impact_player_analysis.py — Impact Player Analysis (IPL 2023-2025)  The Impac

### Community 42 - "Supporting Utilities 42"
Cohesion: 1.0
Nodes (1): Check CricAPI for any live or upcoming matches right now.

### Community 43 - "Supporting Utilities 43"
Cohesion: 1.0
Nodes (1): Page through CricAPI to find IPL 2026 Match 1 (March 28 opener).

### Community 44 - "Supporting Utilities 44"
Cohesion: 1.0
Nodes (1): Find IPL 2026 Match 1 (March 28) via CricAPI series endpoint.

### Community 45 - "Supporting Utilities 45"
Cohesion: 1.0
Nodes (1): Test prediction for IPL 2026 opener: RCB vs SRH at Chinnaswamy

### Community 46 - "Supporting Utilities 46"
Cohesion: 1.0
Nodes (2): Phase-Aware Architecture Rationale, Rationale: Phase-Aware Design Avoids One-Model-Fits-All

### Community 47 - "Supporting Utilities 47"
Cohesion: 1.0
Nodes (2): Issue: Null Live Probabilities (all live model outputs null), Match Analysis: RR vs RCB (2026-04-11, RR won by 6 wkts)

### Community 48 - "Supporting Utilities 48"
Cohesion: 1.0
Nodes (2): Issue: Incomplete Data Feed (match ended prematurely), Match Analysis: RCB vs LSG (2026-04-15, RCB won by 0 wkts)

### Community 49 - "Supporting Utilities 49"
Cohesion: 1.0
Nodes (0): 

### Community 50 - "Supporting Utilities 50"
Cohesion: 1.0
Nodes (0): 

### Community 51 - "Supporting Utilities 51"
Cohesion: 1.0
Nodes (0): 

### Community 52 - "Supporting Utilities 52"
Cohesion: 1.0
Nodes (0): 

### Community 53 - "Supporting Utilities 53"
Cohesion: 1.0
Nodes (0): 

### Community 54 - "Supporting Utilities 54"
Cohesion: 1.0
Nodes (0): 

### Community 55 - "Supporting Utilities 55"
Cohesion: 1.0
Nodes (0): 

### Community 56 - "Supporting Utilities 56"
Cohesion: 1.0
Nodes (1): IPL Prediction Engine Executive Summary

### Community 57 - "Supporting Utilities 57"
Cohesion: 1.0
Nodes (1): Exponential Form Weighting (5-match window)

### Community 58 - "Supporting Utilities 58"
Cohesion: 1.0
Nodes (1): Six-Phase Match Lifecycle

### Community 59 - "Supporting Utilities 59"
Cohesion: 1.0
Nodes (1): Dual API Key Architecture (Afternoon/Evening)

### Community 60 - "Supporting Utilities 60"
Cohesion: 1.0
Nodes (1): Model Artifacts (.pkl Files)

### Community 61 - "Supporting Utilities 61"
Cohesion: 1.0
Nodes (1): User Guide: Standard Operating Procedure

### Community 62 - "Supporting Utilities 62"
Cohesion: 1.0
Nodes (1): User Guide: DOs and DON'Ts

### Community 63 - "Supporting Utilities 63"
Cohesion: 1.0
Nodes (1): numpy >= 1.24

### Community 64 - "Supporting Utilities 64"
Cohesion: 1.0
Nodes (1): scikit-learn >= 1.3

### Community 65 - "Supporting Utilities 65"
Cohesion: 1.0
Nodes (1): shap >= 0.42

### Community 66 - "Supporting Utilities 66"
Cohesion: 1.0
Nodes (1): pydantic >= 2.0

### Community 67 - "Supporting Utilities 67"
Cohesion: 1.0
Nodes (1): requests >= 2.31

## Knowledge Gaps
- **190 isolated node(s):** `01_parse.py  —  Parse Cricsheet IPL JSON files → matches.csv + deliveries.csv  H`, `Returns (match_row dict, list of delivery rows)`, `02_features.py  —  Build match_features.csv Fixes data leakage bugs, adds multi-`, `Normalize venue names — same stadium has 2-3 different names in data.`, `Normalized batting score 0-100. SR=150 → 100, SR=0 → 0. Matches post-toss model` (+185 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Supporting Utilities 39`** (2 nodes): `06_retrain_after_match.py`, `06_retrain_after_match.py Run this after each match result is confirmed. Appen`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 40`** (2 nodes): `14_partnership_analysis.py`, `14_partnership_analysis.py — Partnership Impact Analysis  Computes per-partnersh`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 41`** (2 nodes): `15_impact_player_analysis.py`, `15_impact_player_analysis.py — Impact Player Analysis (IPL 2023-2025)  The Impac`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 42`** (2 nodes): `test_cricapi_live.py`, `Check CricAPI for any live or upcoming matches right now.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 43`** (2 nodes): `test_find_ipl_opener.py`, `Page through CricAPI to find IPL 2026 Match 1 (March 28 opener).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 44`** (2 nodes): `test_find_match_id.py`, `Find IPL 2026 Match 1 (March 28) via CricAPI series endpoint.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 45`** (2 nodes): `test_match_prediction.py`, `Test prediction for IPL 2026 opener: RCB vs SRH at Chinnaswamy`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 46`** (2 nodes): `Phase-Aware Architecture Rationale`, `Rationale: Phase-Aware Design Avoids One-Model-Fits-All`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 47`** (2 nodes): `Issue: Null Live Probabilities (all live model outputs null)`, `Match Analysis: RR vs RCB (2026-04-11, RR won by 6 wkts)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 48`** (2 nodes): `Issue: Incomplete Data Feed (match ended prematurely)`, `Match Analysis: RCB vs LSG (2026-04-15, RCB won by 0 wkts)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 49`** (1 nodes): `03_live_model.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 50`** (1 nodes): `ipl_schedule_2026.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 51`** (1 nodes): `test_api.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 52`** (1 nodes): `test_telegram.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 53`** (1 nodes): `_tmp_check.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 54`** (1 nodes): `_tmp_check2.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 55`** (1 nodes): `_tmp_check3.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 56`** (1 nodes): `IPL Prediction Engine Executive Summary`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 57`** (1 nodes): `Exponential Form Weighting (5-match window)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 58`** (1 nodes): `Six-Phase Match Lifecycle`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 59`** (1 nodes): `Dual API Key Architecture (Afternoon/Evening)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 60`** (1 nodes): `Model Artifacts (.pkl Files)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 61`** (1 nodes): `User Guide: Standard Operating Procedure`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 62`** (1 nodes): `User Guide: DOs and DON'Ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 63`** (1 nodes): `numpy >= 1.24`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 64`** (1 nodes): `scikit-learn >= 1.3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 65`** (1 nodes): `shap >= 0.42`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 66`** (1 nodes): `pydantic >= 2.0`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Supporting Utilities 67`** (1 nodes): `requests >= 2.31`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EnsemblePreMatchModel` connect `Pre-Toss & XI Estimation` to `Feature Engineering`, `2024-25 Model Retraining`, `Supporting Utilities 22`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `04_api.py â€” FastAPI Prediction Server` connect `API & Model Architecture` to `Match Bot Pipeline`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `match_bot.py â€” Live Match Bot Orchestrator` connect `Match Bot Pipeline` to `API & Model Architecture`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `EnsemblePreMatchModel` (e.g. with `PreMatchRequest` and `PreTossRequest`) actually correct?**
  _`EnsemblePreMatchModel` has 30 INFERRED edges - model-reasoned connections that need verification._
- **What connects `01_parse.py  —  Parse Cricsheet IPL JSON files → matches.csv + deliveries.csv  H`, `Returns (match_row dict, list of delivery rows)`, `02_features.py  —  Build match_features.csv Fixes data leakage bugs, adds multi-` to the rest of the system?**
  _190 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Match Bot Core Functions` be split into smaller, more focused modules?**
  _Cohesion score 0.04 - nodes in this community are weakly interconnected._
- **Should `API & Model Architecture` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._