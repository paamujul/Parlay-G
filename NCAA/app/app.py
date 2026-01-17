# app.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split # Using the same split as notebook
from sklearn.preprocessing import StandardScaler
import time
import datetime

# --- Configuration (from notebook) ---
DATA_DIRECTORY = 'ncaa_box_scores_2024_25'
SPLIT_DATE = '2025-02-28'
ROLLING_WINDOW_SIZE = 3
INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE_ELO = 100

# --- Optional Imports ---
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# --- Caching Functions ---

# @st.cache_data(ttl=3600) # Cache data loading/preprocessing for 1 hour
@st.cache_data # Use simpler caching for potentially large data
def load_and_preprocess_data(data_dir, rolling_window_size):
    """Loads data, cleans, calculates Elo, rolling avgs, and features."""
    # st.write(f"--- Step 1: Loading Data from {data_dir} ---") # Removed
    start_time = time.time()
    dataframes = []
    if not os.path.exists(data_dir):
        st.error(f"Data directory not found: {data_dir}")
        return None, None

    files_found = 0
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.csv')]
    progress_bar = st.progress(0, text="Loading files...")

    for i, file_name in enumerate(all_files):
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            if 'game_id' in df.columns and 'team' in df.columns and 'player' in df.columns:
                dataframes.append(df)
                files_found += 1
            # else: # Reduce verbosity
            #     st.warning(f"Skipping {file_name}: Missing essential columns.")
        except Exception as e:
            st.warning(f"Error reading {file_path}: {e}")
        progress_bar.progress((i + 1) / len(all_files), text=f"Loading file {i+1}/{len(all_files)}...")


    progress_bar.empty() # Remove progress bar after completion


    if not dataframes:
        st.error(f"No valid CSV files found or read from the directory '{data_dir}'.")
        return None, None

    raw_dataframe = pd.concat(dataframes, ignore_index=True)
    # st.write(f"Loaded {len(raw_dataframe)} rows from {files_found} files.") # Removed

    # --- Step 2: Basic Cleaning & Preprocessing ---
    # st.write("\n--- Step 2: Basic Cleaning & Preprocessing ---") # Removed
    team_dataframe = raw_dataframe[raw_dataframe['player'] == 'TEAM'].copy()
    team_dataframe['date'] = pd.to_datetime(team_dataframe['date'])
    team_dataframe = team_dataframe.sort_values(by=['date', 'game_id', 'team'])
    team_dataframe['result_numeric'] = (team_dataframe['result'] == 'W').astype(int)
    cols_to_drop = ['player_id', 'player', 'MIN', 'starter', 'location', 'result']
    team_dataframe = team_dataframe.drop(columns=cols_to_drop, errors='ignore')

    # Add Opponent's Actual Stats
    opp_stats_temp = team_dataframe[[
        'game_id', 'team', 'FGM', 'FGA', '3PTM', '3PTA', 'FTM', 'FTA',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS'
    ]].copy()
    opp_stats_temp = opp_stats_temp.add_prefix('opp_')
    opp_stats_temp.rename(columns={'opp_game_id': 'game_id', 'opp_team': 'opponent_team_actual'}, inplace=True)

    team_dataframe = pd.merge(
        team_dataframe, opp_stats_temp, how='left', on='game_id', suffixes=('', '_y_drop')
    )
    team_dataframe = team_dataframe[team_dataframe['opponent'] == team_dataframe['opponent_team_actual']].copy()
    team_dataframe = team_dataframe.drop(columns=[col for col in team_dataframe.columns if '_y_drop' in col or col == 'opponent_team_actual'])

    opp_allowed_cols_map = {col: col.replace('opp_', 'opp_allowed_') for col in opp_stats_temp.columns if col.startswith('opp_') and col != 'opp_game_id' and col != 'opp_team'}
    team_dataframe.rename(columns=opp_allowed_cols_map, inplace=True)
    # st.write(f"Cleaned team data shape: {team_dataframe.shape}") # Removed

    # --- Step 3: Elo Rating Calculation ---
    # st.write("\n--- Step 3: Calculating Elo Ratings ---") # Removed
    def calculate_expected_outcome(rating1, rating2):
        return 1 / (1 + 10**((rating2 - rating1) / 400))

    team_dataframe = team_dataframe.sort_values(by=['date', 'game_id', 'team']).reset_index(drop=True)
    teams = pd.concat([team_dataframe['team'], team_dataframe['opponent']]).unique()
    elo_ratings = {team: INITIAL_ELO for team in teams}
    team_dataframe['team_elo_before'] = np.nan
    team_dataframe['opp_elo_before'] = np.nan
    unique_games = team_dataframe[['date', 'game_id']].drop_duplicates().sort_values(by=['date', 'game_id'])
    rows_processed = 0
    skipped_games = 0

    # Elo Calculation Loop (remains the same logic)
    for _, game_info in unique_games.iterrows():
        game_id = game_info['game_id']
        game_rows = team_dataframe[team_dataframe['game_id'] == game_id]
        if len(game_rows) != 2: skipped_games +=1; continue
        row1, row2 = game_rows.iloc[0], game_rows.iloc[1]
        team1, team2 = row1['team'], row1['opponent']
        elo1, elo2 = elo_ratings.get(team1, INITIAL_ELO), elo_ratings.get(team2, INITIAL_ELO)
        team_dataframe.loc[row1.name, 'team_elo_before'] = elo1
        team_dataframe.loc[row1.name, 'opp_elo_before'] = elo2
        team_dataframe.loc[row2.name, 'team_elo_before'] = elo2
        team_dataframe.loc[row2.name, 'opp_elo_before'] = elo1
        home_team, away_team = (team1, team2) if row1['home'] else (team2, team1)
        elo_home = elo_ratings.get(home_team, INITIAL_ELO) + HOME_ADVANTAGE_ELO
        elo_away = elo_ratings.get(away_team, INITIAL_ELO)
        expected_home_win = calculate_expected_outcome(elo_home, elo_away)
        actual_home_win = row1['result_numeric'] if row1['home'] else row2['result_numeric']
        update_factor_home = K_FACTOR * (actual_home_win - expected_home_win)
        elo_ratings[home_team] += update_factor_home
        elo_ratings[away_team] -= update_factor_home
        rows_processed += 2

    initial_rows = len(team_dataframe)
    team_dataframe.dropna(subset=['team_elo_before', 'opp_elo_before'], inplace=True)
    # st.write(f"Dropped {initial_rows - len(team_dataframe)} rows with missing Elo ratings. Skipped {skipped_games} games.") # Removed
    # st.write("Elo calculation complete.") # Removed

    # --- Step 4: Calculate Rolling Averages ---
    # st.write("\n--- Step 4: Calculating Rolling Averages ---") # Removed
    stats_cols_base = ['FGM', 'FGA', '3PTM', '3PTA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'team_score', 'opponent_score']
    opp_allowed_cols = [col for col in team_dataframe.columns if col.startswith('opp_allowed_')]
    all_stats_cols = stats_cols_base + opp_allowed_cols
    rolling_feature_cols = []

    # Rolling Average Calculation Loop (remains the same logic)
    for col in all_stats_cols:
        new_col_name = f'{col}_roll{rolling_window_size}'
        team_dataframe[new_col_name] = team_dataframe.groupby('team')[col].transform(
            lambda x: x.shift(1).rolling(window=rolling_window_size, min_periods=rolling_window_size).mean()
        )
        rolling_feature_cols.append(new_col_name)

    # Derived rolling averages (remains the same logic)
    team_dataframe[f'FG%_roll{rolling_window_size}'] = (team_dataframe[f'FGM_roll{rolling_window_size}'] / team_dataframe[f'FGA_roll{rolling_window_size}']).fillna(0)
    team_dataframe[f'3PT%_roll{rolling_window_size}'] = (team_dataframe[f'3PTM_roll{rolling_window_size}'] / team_dataframe[f'3PTA_roll{rolling_window_size}']).fillna(0)
    team_dataframe[f'FT%_roll{rolling_window_size}'] = (team_dataframe[f'FTM_roll{rolling_window_size}'] / team_dataframe[f'FTA_roll{rolling_window_size}']).fillna(0)
    team_dataframe[f'opp_allowed_FG%_roll{rolling_window_size}'] = (team_dataframe[f'opp_allowed_FGM_roll{rolling_window_size}'] / team_dataframe[f'opp_allowed_FGA_roll{rolling_window_size}']).fillna(0)
    team_dataframe[f'opp_allowed_3PT%_roll{rolling_window_size}'] = (team_dataframe[f'opp_allowed_3PTM_roll{rolling_window_size}'] / team_dataframe[f'opp_allowed_3PTA_roll{rolling_window_size}']).fillna(0)
    derived_rolling_cols = [
        f'FG%_roll{rolling_window_size}', f'3PT%_roll{rolling_window_size}', f'FT%_roll{rolling_window_size}',
        f'opp_allowed_FG%_roll{rolling_window_size}', f'opp_allowed_3PT%_roll{rolling_window_size}'
    ]
    rolling_feature_cols.extend(derived_rolling_cols)

    initial_rows = len(team_dataframe)
    team_dataframe.dropna(subset=[f'FGM_roll{rolling_window_size}'], inplace=True)
    # st.write(f"Dropped {initial_rows - len(team_dataframe)} rows with missing rolling averages.") # Removed
    # st.write("Rolling averages calculated.") # Removed

    # --- Step 5: Create Matchup DataFrame ---
    # st.write("\n--- Step 5: Creating Matchup DataFrame ---") # Removed
    cols_for_matchup = ['game_id', 'date', 'team', 'opponent', 'home', 'result_numeric',
                        'team_elo_before', 'opp_elo_before'] + rolling_feature_cols
    matchup_data = team_dataframe[cols_for_matchup].copy()
    matchup_merged = pd.merge(
        matchup_data, matchup_data, on='game_id', suffixes=('_team', '_opp')
    )
    matchups_final = matchup_merged[matchup_merged['team_team'] != matchup_merged['team_opp']].copy()
    matchups_final = matchups_final[matchups_final['team_team'] == matchups_final['opponent_opp']].copy()
    matchups_final['game_pair_id'] = matchups_final.apply(
        lambda row: f"{row['game_id']}_{'_'.join(sorted([row['team_team'], row['team_opp']]))}", axis=1
    )
    matchups_final = matchups_final.drop_duplicates(subset='game_pair_id')
    matchups_final.drop(columns=['game_pair_id'], inplace=True)
    matchups_final['label'] = matchups_final['result_numeric_team']
    # st.write(f"Created final matchup data with shape: {matchups_final.shape}") # Removed

    # --- Step 6: Feature Engineering (Differences) ---
    # st.write("\n--- Step 6: Engineering Difference Features ---") # Removed
    feature_cols_final = []
    matchups_final['elo_diff'] = matchups_final['team_elo_before_team'] - matchups_final['team_elo_before_opp']
    feature_cols_final.append('elo_diff')

    stats_to_diff = ['PTS', 'team_score', 'opponent_score', 'FG%', '3PT%', 'FT%', 'REB', 'AST', 'TO', 'STL', 'BLK', 'PF', 'opp_allowed_FG%', 'opp_allowed_PTS']
    missing_diff_stats = []
    # Feature Engineering Loop (remains the same logic)
    for stat in stats_to_diff:
        col_team = f'{stat}_roll{rolling_window_size}_team'
        col_opp = f'{stat}_roll{rolling_window_size}_opp'
        if col_team in matchups_final.columns and col_opp in matchups_final.columns:
            diff_col_name = f'{stat}_roll_diff'
            matchups_final[diff_col_name] = matchups_final[col_team] - matchups_final[col_opp]
            feature_cols_final.append(diff_col_name)
        else: missing_diff_stats.append(stat)
    if missing_diff_stats: st.warning(f"Columns for difference not found for stats: {', '.join(missing_diff_stats)}")

    # Offense vs Defense Features (remains the same logic)
    team_fg_roll_col = f'FG%_roll{rolling_window_size}_team'
    opp_allowed_fg_roll_col_opp = f'opp_allowed_FG%_roll{rolling_window_size}_opp'
    if team_fg_roll_col in matchups_final.columns and opp_allowed_fg_roll_col_opp in matchups_final.columns:
         matchups_final['team_off_vs_opp_def_fg%'] = matchups_final[team_fg_roll_col] - matchups_final[opp_allowed_fg_roll_col_opp]
         feature_cols_final.append('team_off_vs_opp_def_fg%')
    else: st.warning("Could not create 'team_off_vs_opp_def_fg%' feature.")
    opp_fg_roll_col = f'FG%_roll{rolling_window_size}_opp'
    team_allowed_fg_roll_col = f'opp_allowed_FG%_roll{rolling_window_size}_team'
    if opp_fg_roll_col in matchups_final.columns and team_allowed_fg_roll_col in matchups_final.columns:
        matchups_final['opp_off_vs_team_def_fg%'] = matchups_final[opp_fg_roll_col] - matchups_final[team_allowed_fg_roll_col]
        feature_cols_final.append('opp_off_vs_team_def_fg%')
    else: st.warning("Could not create 'opp_off_vs_team_def_fg%' feature.")
    matchups_final['team_is_home'] = matchups_final['home_team'].astype(int)
    feature_cols_final.append('team_is_home')

    matchups_final[feature_cols_final] = matchups_final[feature_cols_final].fillna(0)
    # st.write(f"Engineered {len(feature_cols_final)} features.") # Removed
    # st.write("Final features:", feature_cols_final) # Removed

    end_time = time.time()
    st.success(f"Data Loading & Preprocessing Complete! (took {end_time - start_time:.2f} seconds)")

    # Keep only necessary columns
    id_cols = ['game_id', 'date_team', 'team_team', 'team_opp', 'home_team', 'label']
    needed_cols = id_cols + feature_cols_final
    matchups_final_reduced = matchups_final[needed_cols].copy()

    return matchups_final_reduced, feature_cols_final

def split_data(matchups_final, split_date_str):
    """Splits data based on date."""
    # st.write("\n--- Step 7: Splitting Data ---") # Removed
    split_timestamp = pd.Timestamp(split_date_str)
    date_col_to_use = 'date_team'

    if date_col_to_use not in matchups_final.columns:
        st.error(f"Date column '{date_col_to_use}' not found for splitting.")
        return None, None, None, None

    matchups_final[date_col_to_use] = pd.to_datetime(matchups_final[date_col_to_use])
    train_df = matchups_final[matchups_final[date_col_to_use] < split_timestamp].copy()
    test_df = matchups_final[matchups_final[date_col_to_use] >= split_timestamp].copy()

    # st.write(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}") # Removed

    if train_df.empty or test_df.empty:
        st.error("Train or test split resulted in empty DataFrame. Check SPLIT_DATE and data range.")
        return None, None, None, None

    return train_df, test_df

# --- Model Training and Evaluation Functions (Mostly unchanged) ---
# Use st.cache_resource for model objects
@st.cache_resource
def train_evaluate_rf(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata):
    """Trains and evaluates Random Forest."""
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10, min_samples_leaf=5, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    importances = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
    results_df = test_df_metadata.copy() # Start with metadata columns
    results_df['predicted_winner'] = np.where(y_pred == 1, results_df['team_team'], results_df['team_opp'])
    results_df['actual_winner'] = np.where(y_test == 1, results_df['team_team'], results_df['team_opp'])
    results_df['confidence'] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df['correct'] = (y_pred == y_test)
    results_df['status'] = results_df['correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
    return model, accuracy, report, report_str, importances, results_df

@st.cache_resource
def train_evaluate_lr(X_train, y_train, X_test, y_test, test_df_metadata):
    """Trains and evaluates Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    if hasattr(scaler, 'feature_names_in_'): feature_names = scaler.feature_names_in_
    else: feature_names = X_train.columns
    importances = pd.DataFrame({'feature': feature_names, 'importance': abs(model.coef_[0])})
    importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
    results_df = test_df_metadata.copy()
    results_df['predicted_winner'] = np.where(y_pred == 1, results_df['team_team'], results_df['team_opp'])
    results_df['actual_winner'] = np.where(y_test == 1, results_df['team_team'], results_df['team_opp'])
    results_df['confidence'] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df['correct'] = (y_pred == y_test)
    results_df['status'] = results_df['correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
    return model, accuracy, report, report_str, importances, results_df

@st.cache_resource
def train_evaluate_lgbm(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata):
    """Trains and evaluates LightGBM."""
    if not LGBM_AVAILABLE: return None, 0.0, {}, "LightGBM not installed.", None, pd.DataFrame()
    model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    importances = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
    results_df = test_df_metadata.copy()
    results_df['predicted_winner'] = np.where(y_pred == 1, results_df['team_team'], results_df['team_opp'])
    results_df['actual_winner'] = np.where(y_test == 1, results_df['team_team'], results_df['team_opp'])
    results_df['confidence'] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df['correct'] = (y_pred == y_test)
    results_df['status'] = results_df['correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
    return model, accuracy, report, report_str, importances, results_df

@st.cache_resource
def train_evaluate_xgb(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata):
    """Trains and evaluates XGBoost."""
    if not XGB_AVAILABLE: return None, 0.0, {}, "XGBoost not installed.", None, pd.DataFrame()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    importances = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
    results_df = test_df_metadata.copy()
    results_df['predicted_winner'] = np.where(y_pred == 1, results_df['team_team'], results_df['team_opp'])
    results_df['actual_winner'] = np.where(y_test == 1, results_df['team_team'], results_df['team_opp'])
    results_df['confidence'] = np.where(y_pred == 1, y_proba[:, 1], y_proba[:, 0])
    results_df['correct'] = (y_pred == y_test)
    results_df['status'] = results_df['correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
    return model, accuracy, report, report_str, importances, results_df

@st.cache_resource
def train_evaluate_nn(X_train, y_train, X_test, y_test, test_df_metadata):
    """Trains and evaluates Neural Network."""
    if not TF_AVAILABLE: return None, 0.0, {}, "TensorFlow not installed.", None, pd.DataFrame()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    def build_nn_model(input_shape):
        model = keras.Sequential([layers.Input(shape=(input_shape,)), layers.Dense(64, activation="relu", name="layer1"), layers.Dropout(0.3), layers.Dense(32, activation="relu", name="layer2"), layers.Dropout(0.2), layers.Dense(1, activation="sigmoid", name="output_layer")])
        return model
    nn_model = build_nn_model(X_train_scaled.shape[1])
    nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    with st.spinner("Training Neural Network..."):
         history = nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    # st.write("NN Training History (Loss):"); st.line_chart(pd.DataFrame(history.history)[['loss', 'val_loss']]) # Optional: Can uncomment if desired
    loss_nn, accuracy_nn = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
    y_proba_nn_raw = nn_model.predict(X_test_scaled)
    y_pred_nn = (y_proba_nn_raw > 0.5).astype("int32").flatten()
    report = classification_report(y_test, y_pred_nn, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred_nn, zero_division=0)
    importances = None # NN doesn't have direct feature importances
    results_df = test_df_metadata.copy()
    results_df['predicted_winner'] = np.where(y_pred_nn == 1, results_df['team_team'], results_df['team_opp'])
    results_df['actual_winner'] = np.where(y_test == 1, results_df['team_team'], results_df['team_opp'])
    results_df['confidence'] = np.where(y_pred_nn == 1, y_proba_nn_raw.flatten(), 1 - y_proba_nn_raw.flatten())
    results_df['correct'] = (y_pred_nn == y_test)
    results_df['status'] = results_df['correct'].apply(lambda x: '✅ Correct' if x else '❌ Wrong')
    return nn_model, accuracy_nn, report, report_str, importances, results_df


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("🏀 NCAAB Game Prediction Model Analysis")

# --- Load Data ---
# This is cached, so it runs fast after the first time
matchups_final, feature_cols = load_and_preprocess_data(DATA_DIRECTORY, ROLLING_WINDOW_SIZE)

if matchups_final is None:
    st.error("Failed to load or process data. Please check the data directory and file formats.")
    st.stop()

# --- Split Data ---
train_df, test_df = split_data(matchups_final, SPLIT_DATE)

if train_df is None or test_df is None:
    st.error("Failed to split data. Check SPLIT_DATE configuration.")
    st.stop()

# Prepare data for models
X_train = train_df[feature_cols]
y_train = train_df['label']
X_test = test_df[feature_cols]
y_test = test_df['label']

# Keep metadata from test_df for displaying results (including date)
test_df_metadata = test_df[['date_team', 'team_team', 'team_opp']].copy()
# Ensure date_team is datetime for filtering later
test_df_metadata['date_team'] = pd.to_datetime(test_df_metadata['date_team'])


# --- Model Selection Sidebar ---
st.sidebar.header("🤖 Model Training & Selection")
available_models = ["Random Forest", "Logistic Regression"]
if LGBM_AVAILABLE: available_models.append("LightGBM")
if XGB_AVAILABLE: available_models.append("XGBoost")
if TF_AVAILABLE: available_models.append("Neural Network (Keras)")

# Use session state to remember the last selected model if button wasn't pressed
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = available_models[0]

selected_model = st.sidebar.selectbox(
    "Choose a model:",
    available_models,
    index=available_models.index(st.session_state.selected_model) # Default to last selection
)
st.session_state.selected_model = selected_model # Update session state

run_button = st.sidebar.button(f"Train & Evaluate {selected_model}")

# --- Initialize placeholders for results ---
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# --- Main Area: Model Training & Display ---
if run_button:
    model_name = selected_model
    st.header(f"📊 Results for: {model_name}")
    with st.spinner(f"Running {model_name}... This might take a moment."):
        start_train_time = time.time()
        model, accuracy, report, report_str, importances, results_df = (None, 0.0, {}, "", None, pd.DataFrame())

        # Call the appropriate training function
        if model_name == "Random Forest": model, accuracy, report, report_str, importances, results_df = train_evaluate_rf(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata)
        elif model_name == "Logistic Regression": model, accuracy, report, report_str, importances, results_df = train_evaluate_lr(X_train, y_train, X_test, y_test, test_df_metadata)
        elif model_name == "LightGBM" and LGBM_AVAILABLE: model, accuracy, report, report_str, importances, results_df = train_evaluate_lgbm(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata)
        elif model_name == "XGBoost" and XGB_AVAILABLE: model, accuracy, report, report_str, importances, results_df = train_evaluate_xgb(X_train, y_train, X_test, y_test, feature_cols, test_df_metadata)
        elif model_name == "Neural Network (Keras)" and TF_AVAILABLE: model, accuracy, report, report_str, importances, results_df = train_evaluate_nn(X_train, y_train, X_test, y_test, test_df_metadata)
        else: st.error(f"Model {model_name} selection is invalid or dependencies missing.")

        end_train_time = time.time()
        st.session_state.model_results[model_name] = {'model': model, 'accuracy': accuracy, 'report': report, 'report_str': report_str, 'importances': importances, 'results_df': results_df, 'train_time': end_train_time - start_train_time}

# --- Display Selected Model Results (if available) ---
# Check if the *currently selected* model has results stored
if selected_model in st.session_state.model_results:
    results = st.session_state.model_results[selected_model]
    st.header(f"📊 Results for: {selected_model}")

    # Display Metrics
    st.metric(label="Test Set Accuracy", value=f"{results['accuracy']:.4f}")
    st.write(f"Model training/evaluation time: {results['train_time']:.2f} seconds.")
    with st.expander("Classification Report"):
        st.text(results['report_str'])

    # Display Feature Importances (if applicable)
    if results['importances'] is not None:
        with st.expander("Feature Importances"):
             imp_name = 'importance'
             if 'Coefficient Magnitude' in results['importances'].columns: # Adjust for LR
                 imp_name = 'Coefficient Magnitude'
                 results['importances'][imp_name] = results['importances']['importance']
             st.dataframe(results['importances'].head(15))
             st.bar_chart(results['importances'].head(15).set_index('feature')[imp_name])

    # Display Predictions Table
    with st.expander("Prediction Results on Test Set"):
        st.write("Showing predictions with model confidence for the predicted outcome.")
        display_cols = ['date_team', 'team_team', 'team_opp', 'predicted_winner', 'actual_winner', 'confidence', 'status']
        res_df_display = results['results_df'].copy()
        res_df_display['date_team'] = res_df_display['date_team'].dt.date # Format date for display
        st.dataframe(res_df_display[display_cols].sort_values('date_team').round({'confidence': 3}))

        # Add filters
        filter_status = st.selectbox("Filter by prediction status:", ["All", "✅ Correct", "❌ Wrong"], key=f"filter_pred_{selected_model}")
        if filter_status != "All":
             filtered_df = res_df_display[res_df_display['status'] == filter_status]
             st.dataframe(filtered_df[display_cols].sort_values('date_team').round({'confidence': 3}))

    # --- Parlay Builder Section ---
    st.markdown("---") # Separator
    st.header("🎰 Parlay Builder (Based on Model Confidence)")
    st.write(f"Build a parlay using the predictions from the **{selected_model}** model.")

    # Get available dates from the test set results
    min_date = results['results_df']['date_team'].min().date()
    max_date = results['results_df']['date_team'].max().date()

    parlay_date = st.date_input(
        "Select Parlay Date:",
        value=min_date, # Default to first date in test set
        min_value=min_date,
        max_value=max_date,
        key=f"parlay_date_{selected_model}"
    )

    num_legs = st.number_input(
        "Number of Parlay Legs:",
        min_value=2,
        max_value=15, # Set a reasonable max
        value=3,
        step=1,
        key=f"parlay_legs_{selected_model}"
    )

    if st.button("Build Parlay", key=f"build_parlay_{selected_model}"):
        # Ensure parlay_date is Timestamp for comparison if needed, though date() comparison works
        target_date = pd.to_datetime(parlay_date) # Convert selected date back for filtering
        # Filter the results dataframe for the selected date
        daily_predictions = results['results_df'][results['results_df']['date_team'].dt.date == target_date.date()].copy()

        if daily_predictions.empty:
            st.warning(f"No games found in the test set for {parlay_date.strftime('%Y-%m-%d')}.")
        elif len(daily_predictions) < num_legs:
            st.warning(f"Only {len(daily_predictions)} games predicted for {parlay_date.strftime('%Y-%m-%d')}. Cannot build a {num_legs}-leg parlay. Showing available:")
            # Show the available games sorted by confidence
            parlay_legs = daily_predictions.sort_values('confidence', ascending=False)
            st.dataframe(parlay_legs[['team_team', 'team_opp', 'predicted_winner', 'confidence']].round({'confidence': 3}))
        else:
            # Sort by confidence and select top N
            parlay_legs = daily_predictions.sort_values('confidence', ascending=False).head(num_legs)
            st.subheader(f"Top {num_legs} Confidence Parlay Legs for {parlay_date.strftime('%Y-%m-%d')}:")
            st.dataframe(parlay_legs[['team_team', 'team_opp', 'predicted_winner', 'confidence']].round({'confidence': 3}))
            overall_parlay_confidence = parlay_legs['confidence'].prod() # Naive probability multiplication
            st.write(f"_(Note: Multiplying model confidences ({overall_parlay_confidence:.3f}) isn't a true parlay probability but indicates combined model certainty.)_")

# --- Display Summary of Run Models ---
st.sidebar.header("📈 Model Performance Summary")
summary_data = []
if st.session_state.model_results:
    for name, res in st.session_state.model_results.items():
        f1_weighted = res['report'].get('weighted avg', {}).get('f1-score', 0.0) if isinstance(res['report'], dict) else 0.0
        summary_data.append({'Model': name, 'Accuracy': res['accuracy'], 'F1 Score (Weighted)': f1_weighted, 'Train Time (s)': res['train_time']})
    summary_df = pd.DataFrame(summary_data).sort_values('Accuracy', ascending=False)
    st.sidebar.dataframe(summary_df.style.format({'Accuracy': '{:.4f}', 'F1 Score (Weighted)': '{:.4f}', 'Train Time (s)': '{:.2f}'}))
else:
    st.sidebar.write("Train a model using the button above to see results and build parlays.")

st.sidebar.info("Refresh the page (F5) to clear cached data and all stored model results.")