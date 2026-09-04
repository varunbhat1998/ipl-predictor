"""
Chase-perspective model: reframe prediction as "will chasing team win?"
This removes the team1/team2 ordering bias in recent seasons.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/match_features.csv", parse_dates=["date"])
matches_raw = pd.read_csv("data/matches.csv")
matches_raw["file_id"] = matches_raw["file_id"].astype(str)
df["file_id"] = df["file_id"].astype(str)

# Merge toss info
df = df.merge(
    matches_raw[["file_id","toss_winner","toss_decision"]].drop_duplicates(),
    on="file_id", how="left", suffixes=("","_raw")
)

df = df[df["team1_won"].notna()].copy()
df["season"] = df["season"].astype(str)

# Normalize toss winner
def norm_team(t):
    if not isinstance(t, str): return t
    mapping = {
        "Delhi Daredevils": "Delhi Capitals",
        "Deccan Chargers": "Sunrisers Hyderabad",
        "Punjab Kings": "Kings XI Punjab",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    }
    return mapping.get(t, t)

df["toss_winner_n"] = df["toss_winner"].apply(norm_team)

# Batting second = toss winner if chose field, else other team
df["bat_second"] = np.where(
    df["toss_decision"] == "field",
    df["toss_winner_n"],
    np.where(df["toss_winner_n"] == df["team1"], df["team2"], df["team1"])
)
df["bat_first"] = np.where(df["bat_second"] == df["team1"], df["team2"], df["team1"])
df["chase_won"] = (df["bat_second"] == df["winner"]).astype(int)

# ELO from chasing perspective
df["chase_elo"] = np.where(df["bat_second"]==df["team1"], df["team1_elo"], df["team2_elo"])
df["defend_elo"] = np.where(df["bat_first"]==df["team1"], df["team1_elo"], df["team2_elo"])
df["chase_elo_diff"] = df["chase_elo"] - df["defend_elo"]

# Form
df["chase_form5"] = np.where(df["bat_second"]==df["team1"], df["team1_form"], df["team2_form"])
df["defend_form5"] = np.where(df["bat_first"]==df["team1"], df["team1_form"], df["team2_form"])
df["chase_form_diff"] = df["chase_form5"] - df["defend_form5"]

df["chase_form3"] = np.where(df["bat_second"]==df["team1"], df["team1_form_3"], df["team2_form_3"])
df["defend_form3"] = np.where(df["bat_first"]==df["team1"], df["team1_form_3"], df["team2_form_3"])
df["chase_form3_diff"] = df["chase_form3"] - df["defend_form3"]

df["chase_form10"] = np.where(df["bat_second"]==df["team1"], df["team1_form_10"], df["team2_form_10"])
df["defend_form10"] = np.where(df["bat_first"]==df["team1"], df["team1_form_10"], df["team2_form_10"])
df["chase_form10_diff"] = df["chase_form10"] - df["defend_form10"]

# H2H from chasing perspective
df["chase_h2h"] = np.where(
    df["bat_second"] <= df["bat_first"],
    df["h2h_win_rate_team1"],
    1 - df["h2h_win_rate_team1"]
)

# Toss winner = chasing team?
df["chase_won_toss"] = (df["bat_second"] == df["toss_winner_n"]).astype(int)

# Venue
df["chase_venue_wr"] = np.where(df["bat_second"]==df["team1"], df["team1_venue_win_rate"], df["team2_venue_win_rate"])
df["defend_venue_wr"] = np.where(df["bat_first"]==df["team1"], df["team1_venue_win_rate"], df["team2_venue_win_rate"])

# Player strength
df["chase_bat"] = np.where(df["bat_second"]==df["team1"], df["team1_bat_strength"], df["team2_bat_strength"])
df["defend_bat"] = np.where(df["bat_first"]==df["team1"], df["team1_bat_strength"], df["team2_bat_strength"])
df["chase_bowl"] = np.where(df["bat_second"]==df["team1"], df["team1_bowl_strength"], df["team2_bowl_strength"])
df["defend_bowl"] = np.where(df["bat_first"]==df["team1"], df["team1_bowl_strength"], df["team2_bowl_strength"])
df["chase_bat_diff"] = df["chase_bat"] - df["defend_bat"]
df["chase_bowl_diff"] = df["chase_bowl"] - df["defend_bowl"]

# Chase ability
df["chase_team_chase_wr"] = np.where(df["bat_second"]==df["team1"], df["team1_chase_wr"], df["team2_chase_wr"])

CHASE_FEATURES = [
    "chase_elo_diff",
    "chase_form_diff", "chase_form3_diff", "chase_form10_diff",
    "chase_h2h",
    "chase_won_toss",
    "venue_chase_win_rate", "venue_bat_first_win_rate", "venue_avg_first_innings",
    "chase_venue_wr", "defend_venue_wr",
    "match_num_in_season", "is_playoff",
    "chase_bat_diff", "chase_bowl_diff",
    "chase_bat", "defend_bat", "chase_bowl", "defend_bowl",
    "chase_team_chase_wr",
    "venue_toss_win_rate",
    "venue_matches",
]
CHASE_FEATURES = [f for f in CHASE_FEATURES if f in df.columns]
print(f"Using {len(CHASE_FEATURES)} chase-perspective features")

# Walk-forward backtest
results = []
for test_season in ["2023", "2024", "2025"]:
    test_year = int(test_season)
    train_df = df[df["season"].astype(int) < test_year]
    test_df = df[df["season"] == test_season]

    med = train_df[CHASE_FEATURES].median()
    X_tr = train_df[CHASE_FEATURES].fillna(med)
    y_tr = train_df["chase_won"].astype(int)
    X_te = test_df[CHASE_FEATURES].fillna(med)
    y_te = test_df["chase_won"].astype(int)

    # XGBoost with Optuna
    kf = KFold(n_splits=5, shuffle=False)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 2.0),
        }
        m = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
        oof = cross_val_predict(m, X_tr, y_tr, cv=kf, method="predict_proba")[:, 1]
        return log_loss(y_tr, oof)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=80)

    xgb_model = xgb.XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, verbosity=0)
    xgb_model.fit(X_tr, y_tr)
    xgb_probs = xgb_model.predict_proba(X_te)[:, 1]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=-1)
    lgb_model.fit(X_tr, y_tr)
    lgb_probs = lgb_model.predict_proba(X_te)[:, 1]

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
    rf_model.fit(X_tr, y_tr)
    rf_probs = rf_model.predict_proba(X_te)[:, 1]

    # Logistic Regression
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_model.fit(X_tr_s, y_tr)
    lr_probs = lr_model.predict_proba(X_te_s)[:, 1]

    # Ensemble
    ens_probs = (xgb_probs + lgb_probs + rf_probs + lr_probs) / 4

    print(f"\n{test_season} (train={len(train_df)}, test={len(test_df)}):")
    for name, probs in [("XGBoost", xgb_probs), ("LightGBM", lgb_probs),
                         ("RF", rf_probs), ("LogReg", lr_probs), ("Ensemble", ens_probs)]:
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_te, preds)
        ll = log_loss(y_te, probs)
        print(f"  {name:<12}: Acc={acc:.3f}, LL={ll:.4f}")

    # Feature importance
    fi = pd.Series(xgb_model.feature_importances_, index=CHASE_FEATURES).sort_values(ascending=False)
    print(f"  Top: {', '.join(f'{k}={v:.3f}' for k,v in fi.head(6).items())}")

    for i, (_, row) in enumerate(test_df.iterrows()):
        pred_chase_wins = int(ens_probs[i] >= 0.5)
        actual_chase_won = int(row["chase_won"])
        results.append({
            "season": test_season,
            "date": str(row["date"])[:10],
            "bat_first": row["bat_first"],
            "bat_second": row["bat_second"],
            "prob_chase_wins": float(ens_probs[i]),
            "predicted_winner": row["bat_second"] if ens_probs[i] >= 0.5 else row["bat_first"],
            "actual_winner": row["winner"],
            "correct": int(pred_chase_wins == actual_chase_won),
        })

res_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("CHASE-PERSPECTIVE ENSEMBLE WALK-FORWARD:")
total = res_df["correct"].sum()
n = len(res_df)
print(f"OVERALL: {total}/{n} = {total/n*100:.1f}%")
for s in ["2023","2024","2025"]:
    sub = res_df[res_df["season"]==s]
    print(f"  {s}: {sub['correct'].sum()}/{len(sub)} = {sub['correct'].mean()*100:.1f}%")

# Confidence analysis
print("\nConfidence-based accuracy:")
res_df["confidence"] = res_df["prob_chase_wins"].apply(lambda p: abs(p - 0.5))
for label, lo, hi in [("High (>60%)", 0.10, 1.0), ("Medium (55-60%)", 0.05, 0.10), ("Low (<55%)", 0.0, 0.05)]:
    mask = (res_df["confidence"] >= lo) & (res_df["confidence"] < hi)
    sub = res_df[mask]
    if len(sub) > 0:
        print(f"  {label}: {sub['correct'].sum()}/{len(sub)} = {sub['correct'].mean()*100:.1f}%")

res_df.to_csv("data/chase_backtest.csv", index=False)
print("\nSaved data/chase_backtest.csv")
