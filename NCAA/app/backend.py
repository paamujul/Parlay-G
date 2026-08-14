"""Backend data layer.

Connects to the PostgreSQL database created by NCAA/db/schema.sql,
loads box-score rows, and turns them into the model-ready matchup
feature table (Elo ratings, rolling averages, engineered differences).
"""

import os
import time

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from config import HOME_ADVANTAGE_ELO, INITIAL_ELO, K_FACTOR

load_dotenv()

# --- Database configuration (override via .env or real env vars) ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ncaa_basketball")
DB_USER = os.getenv("DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# --- Feature engineering configuration ---
# INITIAL_ELO / K_FACTOR / HOME_ADVANTAGE_ELO come from config/ncaa.yaml
# via config.py (imported above) and are used directly below. SPLIT_DATE
# and ROLLING_WINDOW_SIZE live in config/ncaa.yaml too, but aren't
# referenced in this file -- app.py imports them from config directly
# and passes them in as arguments to load_and_preprocess_data() and
# split_data() below.

# Selected/aliased back to the original box-score column names so the
# feature-engineering logic below doesn't need to care about the
# database's own naming conventions (see NCAA/db/schema.sql).
BOX_SCORES_QUERY = """
    SELECT
        player_id,
        player,
        minutes_played AS "MIN",
        fgm AS "FGM",
        fga AS "FGA",
        three_ptm AS "3PTM",
        three_pta AS "3PTA",
        ftm AS "FTM",
        fta AS "FTA",
        oreb AS "OREB",
        dreb AS "DREB",
        reb AS "REB",
        ast AS "AST",
        stl AS "STL",
        blk AS "BLK",
        turnovers AS "TO",
        pf AS "PF",
        pts AS "PTS",
        team,
        opponent,
        home,
        starter,
        game_id,
        game_date AS "date",
        location,
        team_score,
        opponent_score,
        result
    FROM box_scores
"""

_engine = None


def get_engine():
    """Returns a lazily-created SQLAlchemy engine for the NCAA database."""
    global _engine
    if _engine is None:
        url = URL.create(
            "postgresql+psycopg2",
            username=DB_USER,
            password=DB_PASSWORD or None,
            host=DB_HOST,
            port=int(DB_PORT),
            database=DB_NAME,
        )
        _engine = create_engine(url)
    return _engine


def load_box_scores_from_db():
    """Loads every player/team box-score row from the database."""
    return pd.read_sql(BOX_SCORES_QUERY, get_engine())


@st.cache_data
def load_and_preprocess_data(rolling_window_size):
    """Loads data from Postgres, cleans it, and calculates Elo, rolling
    averages, and matchup features."""
    start_time = time.time()

    try:
        with st.spinner("Loading box scores from the database..."):
            raw_dataframe = load_box_scores_from_db()
    except Exception as e:
        st.error(
            f"Could not load data from database '{DB_NAME}' at {DB_HOST}:{DB_PORT}: {e}"
        )
        return None, None

    if raw_dataframe.empty:
        st.error(f"No box score data found in database '{DB_NAME}'.")
        return None, None

    # --- Step 2: Basic Cleaning & Preprocessing ---
    team_dataframe = raw_dataframe[raw_dataframe["player"] == "TEAM"].copy()
    team_dataframe["date"] = pd.to_datetime(team_dataframe["date"])
    team_dataframe = team_dataframe.sort_values(by=["date", "game_id", "team"])
    team_dataframe["result_numeric"] = (team_dataframe["result"] == "W").astype(int)
    cols_to_drop = ["player_id", "player", "MIN", "starter", "location", "result"]
    team_dataframe = team_dataframe.drop(columns=cols_to_drop, errors="ignore")

    # Add Opponent's Actual Stats
    opp_stats_temp = team_dataframe[
        [
            "game_id",
            "team",
            "FGM",
            "FGA",
            "3PTM",
            "3PTA",
            "FTM",
            "FTA",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TO",
            "PF",
            "PTS",
        ]
    ].copy()
    opp_stats_temp = opp_stats_temp.add_prefix("opp_")
    opp_stats_temp.rename(
        columns={"opp_game_id": "game_id", "opp_team": "opponent_team_actual"},
        inplace=True,
    )

    team_dataframe = pd.merge(
        team_dataframe,
        opp_stats_temp,
        how="left",
        on="game_id",
        suffixes=("", "_y_drop"),
    )
    team_dataframe = team_dataframe[
        team_dataframe["opponent"] == team_dataframe["opponent_team_actual"]
    ].copy()
    team_dataframe = team_dataframe.drop(
        columns=[
            col
            for col in team_dataframe.columns
            if "_y_drop" in col or col == "opponent_team_actual"
        ]
    )

    opp_allowed_cols_map = {
        col: col.replace("opp_", "opp_allowed_")
        for col in opp_stats_temp.columns
        if col.startswith("opp_") and col != "opp_game_id" and col != "opp_team"
    }
    team_dataframe.rename(columns=opp_allowed_cols_map, inplace=True)

    # --- Step 3: Elo Rating Calculation ---
    def calculate_expected_outcome(rating1, rating2):
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    team_dataframe = team_dataframe.sort_values(
        by=["date", "game_id", "team"]
    ).reset_index(drop=True)
    teams = pd.concat([team_dataframe["team"], team_dataframe["opponent"]]).unique()
    elo_ratings = {team: INITIAL_ELO for team in teams}
    team_dataframe["team_elo_before"] = np.nan
    team_dataframe["opp_elo_before"] = np.nan
    unique_games = (
        team_dataframe[["date", "game_id"]]
        .drop_duplicates()
        .sort_values(by=["date", "game_id"])
    )
    rows_processed = 0
    skipped_games = 0

    for _, game_info in unique_games.iterrows():
        game_id = game_info["game_id"]
        game_rows = team_dataframe[team_dataframe["game_id"] == game_id]
        if len(game_rows) != 2:
            skipped_games += 1
            continue
        row1, row2 = game_rows.iloc[0], game_rows.iloc[1]
        team1, team2 = row1["team"], row1["opponent"]
        elo1, elo2 = (
            elo_ratings.get(team1, INITIAL_ELO),
            elo_ratings.get(team2, INITIAL_ELO),
        )
        team_dataframe.loc[row1.name, "team_elo_before"] = elo1
        team_dataframe.loc[row1.name, "opp_elo_before"] = elo2
        team_dataframe.loc[row2.name, "team_elo_before"] = elo2
        team_dataframe.loc[row2.name, "opp_elo_before"] = elo1
        home_team, away_team = (team1, team2) if row1["home"] else (team2, team1)
        elo_home = elo_ratings.get(home_team, INITIAL_ELO) + HOME_ADVANTAGE_ELO
        elo_away = elo_ratings.get(away_team, INITIAL_ELO)
        expected_home_win = calculate_expected_outcome(elo_home, elo_away)
        actual_home_win = (
            row1["result_numeric"] if row1["home"] else row2["result_numeric"]
        )
        update_factor_home = K_FACTOR * (actual_home_win - expected_home_win)
        elo_ratings[home_team] += update_factor_home
        elo_ratings[away_team] -= update_factor_home
        rows_processed += 2

    team_dataframe.dropna(subset=["team_elo_before", "opp_elo_before"], inplace=True)

    # --- Step 4: Calculate Rolling Averages ---
    stats_cols_base = [
        "FGM",
        "FGA",
        "3PTM",
        "3PTA",
        "FTM",
        "FTA",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TO",
        "PF",
        "PTS",
        "team_score",
        "opponent_score",
    ]
    opp_allowed_cols = [
        col for col in team_dataframe.columns if col.startswith("opp_allowed_")
    ]
    all_stats_cols = stats_cols_base + opp_allowed_cols
    rolling_feature_cols = []

    for col in all_stats_cols:
        new_col_name = f"{col}_roll{rolling_window_size}"
        team_dataframe[new_col_name] = team_dataframe.groupby("team")[col].transform(
            lambda x: (
                x.shift(1)
                .rolling(window=rolling_window_size, min_periods=rolling_window_size)
                .mean()
            )
        )
        rolling_feature_cols.append(new_col_name)

    team_dataframe[f"FG%_roll{rolling_window_size}"] = (
        team_dataframe[f"FGM_roll{rolling_window_size}"]
        / team_dataframe[f"FGA_roll{rolling_window_size}"]
    ).fillna(0)
    team_dataframe[f"3PT%_roll{rolling_window_size}"] = (
        team_dataframe[f"3PTM_roll{rolling_window_size}"]
        / team_dataframe[f"3PTA_roll{rolling_window_size}"]
    ).fillna(0)
    team_dataframe[f"FT%_roll{rolling_window_size}"] = (
        team_dataframe[f"FTM_roll{rolling_window_size}"]
        / team_dataframe[f"FTA_roll{rolling_window_size}"]
    ).fillna(0)
    team_dataframe[f"opp_allowed_FG%_roll{rolling_window_size}"] = (
        team_dataframe[f"opp_allowed_FGM_roll{rolling_window_size}"]
        / team_dataframe[f"opp_allowed_FGA_roll{rolling_window_size}"]
    ).fillna(0)
    team_dataframe[f"opp_allowed_3PT%_roll{rolling_window_size}"] = (
        team_dataframe[f"opp_allowed_3PTM_roll{rolling_window_size}"]
        / team_dataframe[f"opp_allowed_3PTA_roll{rolling_window_size}"]
    ).fillna(0)
    derived_rolling_cols = [
        f"FG%_roll{rolling_window_size}",
        f"3PT%_roll{rolling_window_size}",
        f"FT%_roll{rolling_window_size}",
        f"opp_allowed_FG%_roll{rolling_window_size}",
        f"opp_allowed_3PT%_roll{rolling_window_size}",
    ]
    rolling_feature_cols.extend(derived_rolling_cols)

    team_dataframe.dropna(subset=[f"FGM_roll{rolling_window_size}"], inplace=True)

    # --- Step 5: Create Matchup DataFrame ---
    cols_for_matchup = [
        "game_id",
        "date",
        "team",
        "opponent",
        "home",
        "result_numeric",
        "team_elo_before",
        "opp_elo_before",
    ] + rolling_feature_cols
    matchup_data = team_dataframe[cols_for_matchup].copy()
    matchup_merged = pd.merge(
        matchup_data, matchup_data, on="game_id", suffixes=("_team", "_opp")
    )
    matchups_final = matchup_merged[
        matchup_merged["team_team"] != matchup_merged["team_opp"]
    ].copy()
    matchups_final = matchups_final[
        matchups_final["team_team"] == matchups_final["opponent_opp"]
    ].copy()
    matchups_final["game_pair_id"] = matchups_final.apply(
        lambda row: (
            f"{row['game_id']}_{'_'.join(sorted([row['team_team'], row['team_opp']]))}"
        ),
        axis=1,
    )
    matchups_final = matchups_final.drop_duplicates(subset="game_pair_id")
    matchups_final.drop(columns=["game_pair_id"], inplace=True)
    matchups_final["label"] = matchups_final["result_numeric_team"]

    # --- Step 6: Feature Engineering (Differences) ---
    feature_cols_final = []
    matchups_final["elo_diff"] = (
        matchups_final["team_elo_before_team"] - matchups_final["team_elo_before_opp"]
    )
    feature_cols_final.append("elo_diff")

    stats_to_diff = [
        "PTS",
        "team_score",
        "opponent_score",
        "FG%",
        "3PT%",
        "FT%",
        "REB",
        "AST",
        "TO",
        "STL",
        "BLK",
        "PF",
        "opp_allowed_FG%",
        "opp_allowed_PTS",
    ]
    missing_diff_stats = []
    for stat in stats_to_diff:
        col_team = f"{stat}_roll{rolling_window_size}_team"
        col_opp = f"{stat}_roll{rolling_window_size}_opp"
        if col_team in matchups_final.columns and col_opp in matchups_final.columns:
            diff_col_name = f"{stat}_roll_diff"
            matchups_final[diff_col_name] = (
                matchups_final[col_team] - matchups_final[col_opp]
            )
            feature_cols_final.append(diff_col_name)
        else:
            missing_diff_stats.append(stat)
    if missing_diff_stats:
        st.warning(
            f"Columns for difference not found for stats: {', '.join(missing_diff_stats)}"
        )

    # Offense vs Defense Features
    team_fg_roll_col = f"FG%_roll{rolling_window_size}_team"
    opp_allowed_fg_roll_col_opp = f"opp_allowed_FG%_roll{rolling_window_size}_opp"
    if (
        team_fg_roll_col in matchups_final.columns
        and opp_allowed_fg_roll_col_opp in matchups_final.columns
    ):
        matchups_final["team_off_vs_opp_def_fg%"] = (
            matchups_final[team_fg_roll_col]
            - matchups_final[opp_allowed_fg_roll_col_opp]
        )
        feature_cols_final.append("team_off_vs_opp_def_fg%")
    else:
        st.warning("Could not create 'team_off_vs_opp_def_fg%' feature.")
    opp_fg_roll_col = f"FG%_roll{rolling_window_size}_opp"
    team_allowed_fg_roll_col = f"opp_allowed_FG%_roll{rolling_window_size}_team"
    if (
        opp_fg_roll_col in matchups_final.columns
        and team_allowed_fg_roll_col in matchups_final.columns
    ):
        matchups_final["opp_off_vs_team_def_fg%"] = (
            matchups_final[opp_fg_roll_col] - matchups_final[team_allowed_fg_roll_col]
        )
        feature_cols_final.append("opp_off_vs_team_def_fg%")
    else:
        st.warning("Could not create 'opp_off_vs_team_def_fg%' feature.")
    matchups_final["team_is_home"] = matchups_final["home_team"].astype(int)
    feature_cols_final.append("team_is_home")

    matchups_final[feature_cols_final] = matchups_final[feature_cols_final].fillna(0)

    end_time = time.time()
    st.success(
        f"Data Loading & Preprocessing Complete! (took {end_time - start_time:.2f} seconds)"
    )

    # Keep only necessary columns
    id_cols = ["game_id", "date_team", "team_team", "team_opp", "home_team", "label"]
    needed_cols = id_cols + feature_cols_final
    matchups_final_reduced = matchups_final[needed_cols].copy()

    return matchups_final_reduced, feature_cols_final


def split_data(matchups_final, split_date_str):
    """Splits data based on date."""
    split_timestamp = pd.Timestamp(split_date_str)
    date_col_to_use = "date_team"

    if date_col_to_use not in matchups_final.columns:
        st.error(f"Date column '{date_col_to_use}' not found for splitting.")
        return None, None

    matchups_final[date_col_to_use] = pd.to_datetime(matchups_final[date_col_to_use])
    train_df = matchups_final[matchups_final[date_col_to_use] < split_timestamp].copy()
    test_df = matchups_final[matchups_final[date_col_to_use] >= split_timestamp].copy()

    if train_df.empty or test_df.empty:
        st.error(
            "Train or test split resulted in empty DataFrame. Check SPLIT_DATE and data range."
        )
        return None, None

    return train_df, test_df
