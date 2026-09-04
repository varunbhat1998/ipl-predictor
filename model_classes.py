"""Shared model class definitions — imported by both training scripts and the API."""

import numpy as np
import pandas as pd


class EnsemblePreMatchModel:
    """Soft-voting ensemble: XGBoost + LightGBM + Logistic Regression.
    Optional isotonic calibrator applied to final ensemble probability."""

    def __init__(self, xgb_model, lgb_model, lr_model, lr_scaler, calibrator=None):
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.lr  = lr_model
        self.lr_scaler = lr_scaler
        self.calibrator = calibrator   # IsotonicRegression fitted on OOF preds

    def predict_proba(self, X):
        # LGB/XGB were trained with DataFrames — pass DataFrame to keep feature names
        # LR scaler was trained with numpy — extract .values to avoid mismatch warning
        X_arr = X.values if hasattr(X, "values") else X
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        X_s   = self.lr_scaler.transform(X_arr)
        p_lr  = self.lr.predict_proba(X_s)[:, 1]
        avg   = (p_xgb + p_lgb + p_lr) / 3
        return np.column_stack([1 - avg, avg])

    def feature_importances_(self, features):
        fi_xgb = pd.Series(self.xgb.feature_importances_, index=features)
        fi_lgb = pd.Series(self.lgb.feature_importances_, index=features)
        return ((fi_xgb / fi_xgb.sum()) + (fi_lgb / fi_lgb.sum())) / 2
