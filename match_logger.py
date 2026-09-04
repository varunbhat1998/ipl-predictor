"""
match_logger.py - Structured match event logger for post-match analysis.

Captures every prediction, score poll, phase transition, error, and decision
during a live match. After the match, sends the full log to Claude for
automated analysis with improvement suggestions.

Usage:
    from match_logger import mlog
    mlog.start("CSK", "MI", "Wankhede Stadium", "evening")
    mlog.phase("pre_toss")
    mlog.prediction("prematch", {...inputs}, {...result})
    ...
    mlog.end("CSK won by 6 wickets")
    mlog.run_analysis()   # sends to Claude + saves report
"""

import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class MatchLogger:
    def __init__(self):
        self.events = []
        self.match_meta = {}
        self._start_time = None
        self._predictions = []          # all prediction snapshots
        self._score_polls = []           # raw score data from each poll
        self._errors = []                # all errors encountered
        self._prob_trajectory = []       # probability over time for accuracy tracking
        self._phase_times = {}           # phase -> (start_ts, end_ts)
        self._current_phase = None
        self._toss_detection = {}        # how toss was detected
        self._xi_detection = {}          # how XI was obtained
        self._weather = {}
        self._final_result = None
        self._api_call_count = 0
        self._started = False
        self._ended = False
        self._analysis_run = False

    # ── Core event logging ──────────────────────────────────────────────

    def _ts(self):
        return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

    def _elapsed(self):
        if self._start_time:
            return round(time.time() - self._start_time, 1)
        return 0

    def _add(self, event_type, data):
        entry = {
            "ts": self._ts(),
            "elapsed_s": self._elapsed(),
            "type": event_type,
            "phase": self._current_phase,
            **data,
        }
        self.events.append(entry)
        return entry

    # ── Match lifecycle ─────────────────────────────────────────────────

    def start(self, team1, team2, venue, slot, match_id=None):
        self._start_time = time.time()
        self.events = []
        self._predictions = []
        self._score_polls = []
        self._errors = []
        self._prob_trajectory = []
        self._phase_times = {}
        self._current_phase = None
        self._toss_detection = {}
        self._xi_detection = {}
        self._weather = {}
        self._final_result = None
        self._api_call_count = 0
        self._started = True
        self._ended = False
        self._analysis_run = False
        self.match_meta = {
            "team1": team1, "team2": team2, "venue": venue,
            "slot": slot, "match_id": match_id,
            "date": datetime.now(IST).strftime("%Y-%m-%d"),
            "start_time": self._ts(),
        }
        self._add("match_start", {"team1": team1, "team2": team2, "venue": venue, "slot": slot})

    def end(self, result_status, winner=None, scores=None):
        if self._ended:
            return  # don't double-end
        self._ended = True
        self._final_result = {
            "status": result_status, "winner": winner, "scores": scores,
            "total_predictions": len(self._predictions),
            "total_score_polls": len(self._score_polls),
            "total_errors": len(self._errors),
            "total_api_calls": self._api_call_count,
            "duration_minutes": round(self._elapsed() / 60, 1),
        }
        self.match_meta["end_time"] = self._ts()
        self.match_meta["result"] = self._final_result
        self._add("match_end", self._final_result)
        self._save_raw_log()

    # ── Phase tracking ──────────────────────────────────────────────────

    def phase(self, phase_name, details=None):
        if self._current_phase:
            if self._current_phase in self._phase_times:
                self._phase_times[self._current_phase]["end"] = self._ts()
        self._current_phase = phase_name
        self._phase_times[phase_name] = {"start": self._ts(), "end": None}
        self._add("phase_change", {"new_phase": phase_name, "details": details or {}})

    # ── Predictions ─────────────────────────────────────────────────────

    def prediction(self, model_type, inputs, result, weather_adj=0.0):
        """Log a model prediction with full inputs and outputs.

        model_type: 'prematch', 'posttoss', 'inn1_live', 'inn2_live', 'inn1_break'
        inputs: dict of what was sent to the API
        result: dict of API response (or None on error)
        weather_adj: the adjustment applied after model output
        """
        entry = {
            "model": model_type,
            "inputs": _safe_serialize(inputs),
            "result": _safe_serialize(result),
            "weather_adj": weather_adj,
            "success": result is not None and "error" not in (result or {}),
        }
        self._predictions.append(entry)
        self._add("prediction", entry)

        # Track probability trajectory for accuracy analysis
        if result and entry["success"]:
            prob_entry = {"model": model_type, "ts": self._ts(), "weather_adj": weather_adj}
            if model_type in ("prematch", "posttoss"):
                prob_entry["team1_prob"] = result.get("team1_win_probability",
                                                       result.get("batting_first_win_probability"))
                prob_entry["predicted_winner"] = result.get("predicted_winner")
                prob_entry["confidence"] = result.get("confidence")
            elif model_type in ("inn1_live", "inn1_break"):
                # Unified model uses bat_first_win_probability; old model uses batting_team_win_probability
                prob_entry["bat_first_prob"] = result.get("bat_first_win_probability",
                                                           result.get("batting_team_win_probability"))
                prob_entry["over"] = inputs.get("over")
                prob_entry["score"] = f"{inputs.get('runs')}/{inputs.get('wickets')}"
            elif model_type == "inn2_live":
                # In Inn2 the chasing team is batting, so bat_first_win_probability = 1 - chase_prob
                _bf_prob = result.get("bat_first_win_probability",
                                      result.get("batting_team_win_probability"))
                prob_entry["bat_first_prob"] = _bf_prob
                prob_entry["chase_prob"] = (1 - _bf_prob) if _bf_prob is not None else None
                prob_entry["over"] = inputs.get("over")
                prob_entry["score"] = f"{inputs.get('runs')}/{inputs.get('wickets')}"
                prob_entry["runs_needed"] = inputs.get("runs_needed")
            self._prob_trajectory.append(prob_entry)

    # ── Score polling ───────────────────────────────────────────────────

    def score_poll(self, source, raw_data, parsed=None):
        """Log each score poll attempt.

        source: 'cricapi_matchScore', 'cricbuzz', 'cricapi_cricScore'
        raw_data: what the API returned (keep brief)
        parsed: the extracted score dict {r, w, o} or None
        """
        entry = {
            "source": source,
            "success": parsed is not None,
            "parsed": _safe_serialize(parsed),
        }
        self._score_polls.append(entry)
        self._add("score_poll", entry)

    # ── Toss detection ──────────────────────────────────────────────────

    def toss_detected(self, method, toss_winner, toss_decision, bat_first, bat_second,
                      attempts=0, details=None):
        self._toss_detection = {
            "method": method,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "bat_first": bat_first,
            "bat_second": bat_second,
            "attempts": attempts,
            "details": details or {},
        }
        self._add("toss_detected", self._toss_detection)

    def toss_attempt(self, method, success, details=None):
        self._add("toss_attempt", {"method": method, "success": success, "details": details or {}})

    # ── Playing XI ──────────────────────────────────────────────────────

    def xi_detected(self, source, team1_count, team2_count, team1_names=None, team2_names=None):
        self._xi_detection = {
            "source": source,
            "team1_count": team1_count,
            "team2_count": team2_count,
            "team1_names": team1_names or [],
            "team2_names": team2_names or [],
        }
        self._add("xi_detected", self._xi_detection)

    # ── Weather ─────────────────────────────────────────────────────────

    def weather_fetched(self, weather_data, adjustment, details=None):
        self._weather = {
            "data": _safe_serialize(weather_data),
            "adjustment": adjustment,
            "details": details or {},
        }
        self._add("weather", self._weather)

    # ── Errors ──────────────────────────────────────────────────────────

    def error(self, component, message, details=None):
        entry = {"component": component, "message": str(message), "details": details or {}}
        self._errors.append(entry)
        self._add("error", entry)

    # ── Telegram messages ───────────────────────────────────────────────

    def telegram_sent(self, message_type, success, char_count=0, chat_ids=None):
        self._add("telegram", {
            "message_type": message_type, "success": success,
            "chars": char_count, "chat_ids": chat_ids or [],
        })

    # ── API call counter ────────────────────────────────────────────────

    def api_call(self, source, endpoint, success):
        self._api_call_count += 1
        self._add("api_call", {
            "source": source, "endpoint": endpoint, "success": success,
            "total_calls": self._api_call_count,
        })

    # ── Save raw log ────────────────────────────────────────────────────

    def _save_raw_log(self):
        date_str = datetime.now(IST).strftime("%Y%m%d")
        t1 = self.match_meta.get("team1", "X")
        t2 = self.match_meta.get("team2", "Y")
        # Use first word of each team for filename
        t1s = t1.split()[-1] if t1 else "X"
        t2s = t2.split()[-1] if t2 else "Y"
        filename = f"match_log_{date_str}_{t1s}_vs_{t2s}.json"
        filepath = os.path.join(LOG_DIR, filename)
        full_log = {
            "meta": self.match_meta,
            "toss": self._toss_detection,
            "xi": self._xi_detection,
            "weather": self._weather,
            "result": self._final_result,
            "probability_trajectory": self._prob_trajectory,
            "predictions_summary": self._build_predictions_summary(),
            "errors": self._errors,
            "event_count": len(self.events),
            "events": self.events,
        }
        with open(filepath, "w") as f:
            json.dump(full_log, f, indent=2, default=str)
        print(f"  [Logger] Raw log saved: {filepath} ({len(self.events)} events)")
        return filepath

    def _build_predictions_summary(self):
        """Build a compact summary of all predictions for Claude analysis."""
        summary = []
        for p in self._predictions:
            s = {"model": p["model"], "success": p["success"]}
            if p["result"] and p["success"]:
                r = p["result"]
                if p["model"] in ("prematch", "posttoss"):
                    s["winner"] = r.get("predicted_winner")
                    s["confidence"] = r.get("confidence")
                    if "team1_win_probability" in r:
                        s["prob"] = round(r["team1_win_probability"] * 100, 1)
                    elif "batting_first_win_probability" in r:
                        s["prob_bf"] = round(r["batting_first_win_probability"] * 100, 1)
                elif p["model"] in ("inn1_live", "inn1_break"):
                    _bf = r.get("bat_first_win_probability",
                                r.get("batting_team_win_probability", 0))
                    s["bat_first_prob"] = round(_bf * 100, 1)
                    s["over"] = p["inputs"].get("over")
                    s["score"] = f"{p['inputs'].get('runs')}/{p['inputs'].get('wickets')}"
                elif p["model"] == "inn2_live":
                    _bf = r.get("bat_first_win_probability",
                                r.get("batting_team_win_probability", 0))
                    s["bat_first_prob"] = round(_bf * 100, 1)
                    s["chase_prob"]     = round((1 - _bf) * 100, 1)
                    s["over"] = p["inputs"].get("over")
                    s["score"] = f"{p['inputs'].get('runs')}/{p['inputs'].get('wickets')}"
                if p["weather_adj"]:
                    s["weather_adj"] = p["weather_adj"]
            summary.append(s)
        return summary

    # ── Post-match Claude analysis ──────────────────────────────────────

    def run_analysis(self, anthropic_key=None, send_telegram_fn=None):
        if self._analysis_run:
            return None  # don't run twice
        self._analysis_run = True
        """Send the full match log to Claude for post-match analysis.

        Returns the analysis text and saves it to logs/.
        """
        api_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("  [Logger] No ANTHROPIC_API_KEY — skipping analysis")
            return None

        # Build the analysis prompt
        prompt = self._build_analysis_prompt()
        if not prompt:
            print("  [Logger] No data to analyze")
            return None

        print(f"  [Logger] Sending {len(prompt)} chars to Claude for post-match analysis...")

        try:
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60)

            if r.status_code == 200:
                analysis = r.json()["content"][0]["text"].strip()
            else:
                print(f"  [Logger] Claude API error: {r.status_code} {r.text[:200]}")
                return None
        except Exception as e:
            print(f"  [Logger] Analysis error: {e}")
            return None

        # Save analysis
        date_str = datetime.now(IST).strftime("%Y%m%d")
        t1s = self.match_meta.get("team1", "X").split()[-1]
        t2s = self.match_meta.get("team2", "Y").split()[-1]
        analysis_file = os.path.join(LOG_DIR, f"analysis_{date_str}_{t1s}_vs_{t2s}.txt")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(analysis)
        print(f"  [Logger] Analysis saved: {analysis_file}")

        # Send summary via Telegram if callback provided
        if send_telegram_fn:
            # Extract just the first section or first 1000 chars for Telegram
            tg_text = analysis[:3500] if len(analysis) > 3500 else analysis
            tg_msg = f"📋 <b>POST-MATCH ANALYSIS</b>\n\n<pre>{_html_escape(tg_text)}</pre>"
            send_telegram_fn(tg_msg)

        return analysis

    def _build_analysis_prompt(self):
        """Build a structured prompt for Claude to analyze the match."""
        meta = self.match_meta
        if not meta:
            return None

        # Compact the events - only keep key events for the prompt
        key_events = []
        for e in self.events:
            if e["type"] in ("match_start", "match_end", "phase_change", "prediction",
                             "toss_detected", "xi_detected", "weather", "error"):
                key_events.append(e)
            elif e["type"] == "score_poll" and not e.get("success"):
                key_events.append(e)  # only log failed polls

        prob_traj = self._prob_trajectory
        pred_summary = self._build_predictions_summary()
        errors = self._errors

        data = json.dumps({
            "match": meta,
            "toss_detection": self._toss_detection,
            "xi_detection": self._xi_detection,
            "weather": self._weather,
            "result": self._final_result,
            "predictions": pred_summary,
            "probability_trajectory": prob_traj,
            "errors": errors,
            "phase_timing": self._phase_times,
        }, indent=1, default=str)

        prompt = f"""You are analyzing a live IPL cricket match prediction bot's performance.
The bot runs 7 phases: (1) Pre-toss prediction, (2) Toss detection, (3) Post-toss prediction with playing XI,
(4) 1st innings live over-by-over tracking, (5) Innings break prediction, (6) 2nd innings live tracking,
(7) Auto-retrain after match.

Below is the structured log from today's match. Analyze it and provide:

## 1. MATCH SUMMARY
- Teams, venue, toss, result
- Total predictions made, API calls used, errors encountered
- Duration per phase

## 2. PREDICTION ACCURACY TIMELINE
- How did win probabilities evolve over the match?
- At which overs did the model get it right vs wrong?
- Was the pre-toss / post-toss prediction correct?
- Did weather adjustment help or hurt?
- At what point (over number) did the model lock onto the correct winner?

## 3. ISSUES & GAPS
- Any errors or failed API calls?
- Did toss/XI detection work on first attempt?
- Were any overs missed (gaps in the tracking)?
- Did powerplay stats get locked correctly?
- Any score polling failures?

## 4. MODEL BEHAVIOR
- Was confidence calibrated? (High confidence predictions correct?)
- How much did probabilities swing per over? (Stability)
- Did the weather adjustment magnitude seem reasonable?
- Were live model inputs plausible (CRR, RRR, partnership, etc.)?

## 5. IMPROVEMENT SUGGESTIONS
- Specific, actionable improvements to the bot or models
- Priority: HIGH / MEDIUM / LOW for each

Be concise and data-driven. Reference specific over numbers and probabilities.

=== MATCH LOG ===
{data}
"""
        return prompt


def _safe_serialize(obj):
    """Convert to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 4) if abs(obj) < 1e6 else obj
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Singleton instance ──────────────────────────────────────────────────
mlog = MatchLogger()
