"""ML layer.

Trains and evaluates each candidate win-prediction model (Random
Forest, Logistic Regression, LightGBM, XGBoost, and a small neural
network) against the matchup features produced by backend.py.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Note: broad `except Exception` (not just ImportError) because some
# optional deps (e.g. xgboost) can fail at import time with unrelated
# errors (e.g. NameError from a pkg_resources/setuptools mismatch) that
# a narrow ImportError guard wouldn't catch, crashing the whole app.
try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def _status_label(correct):
    return "Correct" if correct else "Wrong"


@st.cache_resource
def train_evaluate_rf(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata):
    """Trains and evaluates Random Forest."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    )
    importances = importances.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    results_df = test_df_metadata.copy()  # Start with metadata columns
    results_df["predicted_winner"] = np.where(
        y_pred == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["actual_winner"] = np.where(
        y_test == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["confidence"] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df["correct"] = y_pred == y_test
    results_df["status"] = results_df["correct"].apply(_status_label)
    return model, accuracy, report, report_str, importances, results_df


@st.cache_resource
def train_evaluate_lr(X_train, y_train, X_test, y_test, test_df_metadata):
    """Trains and evaluates Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    if hasattr(scaler, "feature_names_in_"):
        feature_names = scaler.feature_names_in_
    else:
        feature_names = X_train.columns
    importances = pd.DataFrame(
        {"feature": feature_names, "importance": abs(model.coef_[0])}
    )
    importances = importances.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    results_df = test_df_metadata.copy()
    results_df["predicted_winner"] = np.where(
        y_pred == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["actual_winner"] = np.where(
        y_test == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["confidence"] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df["correct"] = y_pred == y_test
    results_df["status"] = results_df["correct"].apply(_status_label)
    return model, accuracy, report, report_str, importances, results_df


@st.cache_resource
def train_evaluate_lgbm(
    X_train, y_train, X_test, y_test, feature_cols, test_df_metadata
):
    """Trains and evaluates LightGBM."""
    if not LGBM_AVAILABLE:
        return None, 0.0, {}, "LightGBM not installed.", None, pd.DataFrame()
    model = lgb.LGBMClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    )
    importances = importances.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    results_df = test_df_metadata.copy()
    results_df["predicted_winner"] = np.where(
        y_pred == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["actual_winner"] = np.where(
        y_test == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["confidence"] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df["correct"] = y_pred == y_test
    results_df["status"] = results_df["correct"].apply(_status_label)
    return model, accuracy, report, report_str, importances, results_df


@st.cache_resource
def train_evaluate_xgb(
    X_train, y_train, X_test, y_test, feature_cols, test_df_metadata
):
    """Trains and evaluates XGBoost."""
    if not XGB_AVAILABLE:
        return None, 0.0, {}, "XGBoost not installed.", None, pd.DataFrame()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    )
    importances = importances.sort_values("importance", ascending=False).reset_index(
        drop=True
    )
    results_df = test_df_metadata.copy()
    results_df["predicted_winner"] = np.where(
        y_pred == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["actual_winner"] = np.where(
        y_test == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["confidence"] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df["correct"] = y_pred == y_test
    results_df["status"] = results_df["correct"].apply(_status_label)
    return model, accuracy, report, report_str, importances, results_df


@st.cache_resource
def train_evaluate_nn(X_train, y_train, X_test, y_test, test_df_metadata):
    """Trains and evaluates Neural Network."""
    if not TF_AVAILABLE:
        return None, 0.0, {}, "TensorFlow not installed.", None, pd.DataFrame()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def build_nn_model(input_shape):
        model = keras.Sequential(
            [
                layers.Input(shape=(input_shape,)),
                layers.Dense(64, activation="relu", name="layer1"),
                layers.Dropout(0.3),
                layers.Dense(32, activation="relu", name="layer2"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid", name="output_layer"),
            ]
        )
        return model

    nn_model = build_nn_model(X_train_scaled.shape[1])
    nn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    with st.spinner("Training Neural Network..."):
        history = nn_model.fit(
            X_train_scaled,
            y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0,
        )
    loss_nn, accuracy_nn = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    y_proba_nn_raw = nn_model.predict(X_test_scaled)
    y_pred_nn = (y_proba_nn_raw > 0.5).astype("int32").flatten()
    report = classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred_nn, zero_division=0)
    importances = None  # NN doesn't have direct feature importances
    results_df = test_df_metadata.copy()
    results_df["predicted_winner"] = np.where(
        y_pred_nn == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["actual_winner"] = np.where(
        y_test == 1, results_df["team_team"], results_df["team_opp"]
    )
    results_df["confidence"] = np.where(
        y_pred_nn == 1, y_proba_nn_raw.flatten(), 1 - y_proba_nn_raw.flatten()
    )
    results_df["correct"] = y_pred_nn == y_test
    results_df["status"] = results_df["correct"].apply(_status_label)
    return nn_model, accuracy_nn, report, report_str, importances, results_df
