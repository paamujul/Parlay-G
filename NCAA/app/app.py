# app.py
"""Frontend: the Streamlit UI for the NCAAB game prediction app.

Data loading/feature engineering lives in backend.py; model
training/evaluation lives in ml.py. This file only wires the two
together and renders the page.
"""

import time

import pandas as pd
import streamlit as st

from backend import load_and_preprocess_data, split_data
from config import (
    MODELS,
    PARLAY_DEFAULT_LEGS,
    PARLAY_MAX_LEGS,
    PARLAY_MIN_LEGS,
    PARLAY_STEP,
    ROLLING_WINDOW_SIZE,
    SPLIT_DATE,
)
from ml import (
    LGBM_AVAILABLE,
    TF_AVAILABLE,
    XGB_AVAILABLE,
    train_evaluate_lgbm,
    train_evaluate_lr,
    train_evaluate_nn,
    train_evaluate_rf,
    train_evaluate_xgb,
)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("NCAAB Game Prediction Model Analysis")

# --- Load Data ---
# This is cached, so it runs fast after the first time
matchups_final, feature_cols = load_and_preprocess_data(ROLLING_WINDOW_SIZE)

if matchups_final is None:
    st.error("Failed to load or process data. Please check the database connection.")
    st.stop()

# --- Split Data ---
train_df, test_df = split_data(matchups_final, SPLIT_DATE)

if train_df is None or test_df is None:
    st.error("Failed to split data. Check SPLIT_DATE configuration.")
    st.stop()

# Prepare data for models
X_train = train_df[feature_cols]
y_train = train_df["label"]
X_test = test_df[feature_cols]
y_test = test_df["label"]

# Keep metadata from test_df for displaying results (including date)
test_df_metadata = test_df[["date_team", "team_team", "team_opp"]].copy()
# Ensure date_team is datetime for filtering later
test_df_metadata["date_team"] = pd.to_datetime(test_df_metadata["date_team"])


# --- Model Selection Sidebar ---
st.sidebar.header("Model Training & Selection")
available_models = [MODELS["random_forest"], MODELS["logistic_regression"]]
if LGBM_AVAILABLE:
    available_models.append(MODELS["lightgbm"])
if XGB_AVAILABLE:
    available_models.append(MODELS["xgboost"])
if TF_AVAILABLE:
    available_models.append(MODELS["neural_network"])


# Use session state to remember the last selected model if button wasn't pressed
if "selected_model" not in st.session_state:
    st.session_state.selected_model = available_models[0]

selected_model = st.sidebar.selectbox(
    "Choose a model:",
    available_models,
    index=available_models.index(
        st.session_state.selected_model
    ),  # Default to last selection
)
st.session_state.selected_model = selected_model  # Update session state

run_button = st.sidebar.button(f"Train & Evaluate {selected_model}")

# --- Initialize placeholders for results ---
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# --- Main Area: Model Training & Display ---
if run_button:
    model_name = selected_model
    st.header(f"Results for: {model_name}")
    with st.spinner(f"Running {model_name}... This might take a moment."):
        start_train_time = time.time()
        model, accuracy, report, report_str, importances, results_df = (
            None,
            0.0,
            {},
            "",
            None,
            pd.DataFrame(),
        )

        # Call the appropriate training function
        if model_name == MODELS["random_forest"]:
            model, accuracy, report, report_str, importances, results_df = (
                train_evaluate_rf(
                    X_train, y_train, X_test, y_test, feature_cols, test_df_metadata
                )
            )
        elif model_name == MODELS["logistic_regression"]:
            model, accuracy, report, report_str, importances, results_df = (
                train_evaluate_lr(X_train, y_train, X_test, y_test, test_df_metadata)
            )
        elif model_name == MODELS["lightgbm"] and LGBM_AVAILABLE:
            model, accuracy, report, report_str, importances, results_df = (
                train_evaluate_lgbm(
                    X_train, y_train, X_test, y_test, feature_cols, test_df_metadata
                )
            )
        elif model_name == MODELS["xgboost"] and XGB_AVAILABLE:
            model, accuracy, report, report_str, importances, results_df = (
                train_evaluate_xgb(
                    X_train, y_train, X_test, y_test, feature_cols, test_df_metadata
                )
            )
        elif model_name == MODELS["neural_network"] and TF_AVAILABLE:
            model, accuracy, report, report_str, importances, results_df = (
                train_evaluate_nn(X_train, y_train, X_test, y_test, test_df_metadata)
            )
        else:
            st.error(
                f"Model {model_name} selection is invalid or dependencies missing."
            )

        end_train_time = time.time()
        st.session_state.model_results[model_name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report,
            "report_str": report_str,
            "importances": importances,
            "results_df": results_df,
            "train_time": end_train_time - start_train_time,
        }

# --- Display Selected Model Results (if available) ---
# Check if the *currently selected* model has results stored
if selected_model in st.session_state.model_results:
    results = st.session_state.model_results[selected_model]
    st.header(f"Results for: {selected_model}")

    # Display Metrics
    st.metric(label="Test Set Accuracy", value=f"{results['accuracy']:.4f}")
    st.write(f"Model training/evaluation time: {results['train_time']:.2f} seconds.")
    with st.expander("Classification Report"):
        st.text(results["report_str"])

    # Display Feature Importances (if applicable)
    if results["importances"] is not None:
        with st.expander("Feature Importances"):
            imp_name = "importance"
            if (
                "Coefficient Magnitude" in results["importances"].columns
            ):  # Adjust for LR
                imp_name = "Coefficient Magnitude"
                results["importances"][imp_name] = results["importances"]["importance"]
            st.dataframe(results["importances"].head(15))
            st.bar_chart(results["importances"].head(15).set_index("feature")[imp_name])

    # Display Predictions Table
    with st.expander("Prediction Results on Test Set"):
        st.write("Showing predictions with model confidence for the predicted outcome.")
        display_cols = [
            "date_team",
            "team_team",
            "team_opp",
            "predicted_winner",
            "actual_winner",
            "confidence",
            "status",
        ]
        res_df_display = results["results_df"].copy()
        res_df_display["date_team"] = res_df_display[
            "date_team"
        ].dt.date  # Format date for display
        st.dataframe(
            res_df_display[display_cols]
            .sort_values("date_team")
            .round({"confidence": 3})
        )

        # Add filters
        filter_status = st.selectbox(
            "Filter by prediction status:",
            ["All", "Correct", "Wrong"],
            key=f"filter_pred_{selected_model}",
        )
        if filter_status != "All":
            filtered_df = res_df_display[res_df_display["status"] == filter_status]
            st.dataframe(
                filtered_df[display_cols]
                .sort_values("date_team")
                .round({"confidence": 3})
            )

    # --- Parlay Builder Section ---
    st.markdown("---")  # Separator
    st.header("Parlay Builder (Based on Model Confidence)")
    st.write(
        f"Build a parlay using the predictions from the **{selected_model}** model."
    )

    # Get available dates from the test set results
    min_date = results["results_df"]["date_team"].min().date()
    max_date = results["results_df"]["date_team"].max().date()

    parlay_date = st.date_input(
        "Select Parlay Date:",
        value=min_date,  # Default to first date in test set
        min_value=min_date,
        max_value=max_date,
        key=f"parlay_date_{selected_model}",
    )

    num_legs = st.number_input(
        "Number of Parlay Legs:",
        min_value=PARLAY_MIN_LEGS,
        max_value=PARLAY_MAX_LEGS,
        value=PARLAY_DEFAULT_LEGS,
        step=PARLAY_STEP,
        key=f"parlay_legs_{selected_model}",
    )
    if st.button("Build Parlay", key=f"build_parlay_{selected_model}"):
        # Ensure parlay_date is Timestamp for comparison if needed, though date() comparison works
        target_date = pd.to_datetime(
            parlay_date
        )  # Convert selected date back for filtering
        # Filter the results dataframe for the selected date
        daily_predictions = results["results_df"][
            results["results_df"]["date_team"].dt.date == target_date.date()
        ].copy()

        if daily_predictions.empty:
            st.warning(
                f"No games found in the test set for {parlay_date.strftime('%Y-%m-%d')}."
            )
        elif len(daily_predictions) < num_legs:
            st.warning(
                f"Only {len(daily_predictions)} games predicted for {parlay_date.strftime('%Y-%m-%d')}. Cannot build a {num_legs}-leg parlay. Showing available:"
            )
            # Show the available games sorted by confidence
            parlay_legs = daily_predictions.sort_values("confidence", ascending=False)
            st.dataframe(
                parlay_legs[
                    ["team_team", "team_opp", "predicted_winner", "confidence"]
                ].round({"confidence": 3})
            )
        else:
            # Sort by confidence and select top N
            parlay_legs = daily_predictions.sort_values(
                "confidence", ascending=False
            ).head(num_legs)
            st.subheader(
                f"Top {num_legs} Confidence Parlay Legs for {parlay_date.strftime('%Y-%m-%d')}:"
            )
            st.dataframe(
                parlay_legs[
                    ["team_team", "team_opp", "predicted_winner", "confidence"]
                ].round({"confidence": 3})
            )
            overall_parlay_confidence = parlay_legs[
                "confidence"
            ].prod()  # Naive probability multiplication
            st.write(
                f"_(Note: Multiplying model confidences ({overall_parlay_confidence:.3f}) isn't a true parlay probability but indicates combined model certainty.)_"
            )

# --- Display Summary of Run Models ---
st.sidebar.header("Model Performance Summary")
summary_data = []
if st.session_state.model_results:
    for name, res in st.session_state.model_results.items():
        f1_weighted = (
            res["report"].get("weighted avg", {}).get("f1-score", 0.0)
            if isinstance(res["report"], dict)
            else 0.0
        )
        summary_data.append(
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "F1 Score (Weighted)": f1_weighted,
                "Train Time (s)": res["train_time"],
            }
        )
    summary_df = pd.DataFrame(summary_data).sort_values("Accuracy", ascending=False)
    st.sidebar.dataframe(
        summary_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "F1 Score (Weighted)": "{:.4f}",
                "Train Time (s)": "{:.2f}",
            }
        )
    )
else:
    st.sidebar.write(
        "Train a model using the button above to see results and build parlays."
    )

st.sidebar.info(
    "Refresh the page (F5) to clear cached data and all stored model results."
)
