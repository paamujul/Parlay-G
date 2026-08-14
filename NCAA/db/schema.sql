-- NCAA basketball database schema
-- Covers the three data shapes found in NCAA/data:
--   1. NCAA/data/ncaa_box_scores_2024_25/<Team>_2024_25.csv  (364 files, one per team)
--   2. NCAA/data/ncaa_box_scores_2024_25/schedule.csv
--   3. NCAA/data/ncaa_box_scores_2024_25/scraped_ncaa_scores.csv
--
-- Run with:
--   psql -d ncaa_basketball -f NCAA/db/schema.sql

BEGIN;

-- One row per player per game, plus one "TEAM" totals row per team per game
-- (player_id / minutes_played are NULL on the TEAM row).
CREATE TABLE IF NOT EXISTS box_scores (
    id              BIGSERIAL PRIMARY KEY,
    player_id       BIGINT,             -- NULL for the per-game "TEAM" row
    player          TEXT,               -- player name, or literal 'TEAM'
    minutes_played  INTEGER,            -- source column "MIN"
    fgm             INTEGER,
    fga             INTEGER,
    three_ptm       INTEGER,            -- source column "3PTM"
    three_pta       INTEGER,            -- source column "3PTA"
    ftm             INTEGER,
    fta             INTEGER,
    oreb            INTEGER,
    dreb            INTEGER,
    reb             INTEGER,
    ast             INTEGER,
    stl             INTEGER,
    blk             INTEGER,
    turnovers       INTEGER,            -- source column "TO"
    pf              INTEGER,
    pts             INTEGER,
    team            TEXT NOT NULL,
    opponent        TEXT NOT NULL,
    home            BOOLEAN,
    starter         BOOLEAN,
    game_id         BIGINT NOT NULL,
    game_date       DATE NOT NULL,
    location        TEXT,               -- H / A / N
    team_score      INTEGER,
    opponent_score  INTEGER,
    result          TEXT                -- W / L
);

CREATE INDEX IF NOT EXISTS idx_box_scores_game_id   ON box_scores (game_id);
CREATE INDEX IF NOT EXISTS idx_box_scores_team      ON box_scores (team);
CREATE INDEX IF NOT EXISTS idx_box_scores_player_id ON box_scores (player_id);
CREATE INDEX IF NOT EXISTS idx_box_scores_date      ON box_scores (game_date);

-- Full-season schedule, one row per calendar date. game_id/home/away are
-- NULL on days with no games.
CREATE TABLE IF NOT EXISTS schedule (
    game_date  DATE,
    game_id    BIGINT,
    home_team  TEXT,
    away_team  TEXT
);

CREATE INDEX IF NOT EXISTS idx_schedule_game_id ON schedule (game_id);

-- Scraped final scores + closing spreads / predicted winners.
CREATE TABLE IF NOT EXISTS scraped_scores (
    id                     BIGSERIAL PRIMARY KEY,
    game_date              DATE,
    team_1_id              INTEGER,
    team_1_name            TEXT,
    team_1_score           INTEGER,
    team_1_spread          TEXT,       -- numeric string, or 'PK' for a pick 'em (spread of 0)
    team_2_id              INTEGER,
    team_2_name            TEXT,
    team_2_score           INTEGER,
    team_2_spread          TEXT,       -- numeric string, or 'PK' for a pick 'em (spread of 0)
    predicted_winner_id    INTEGER,
    predicted_winner_name  TEXT
);

CREATE INDEX IF NOT EXISTS idx_scraped_scores_date ON scraped_scores (game_date);

COMMIT;
