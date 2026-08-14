#!/usr/bin/env bash
# Loads NCAA/data into the box_scores / schedule / scraped_scores tables
# created by schema.sql.
#
# Usage:
#   ./NCAA/db/load_data.sh [database_name]
#
# database_name defaults to ncaa_basketball.

set -euo pipefail

DB_NAME="${1:-ncaa_basketball}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../data/ncaa_box_scores_2024_25}"

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found: $DATA_DIR" >&2
    exit 1
fi

TEAM_FILES=("$DATA_DIR"/*_2024_25.csv)

COPY_SQL="$(mktemp)"
# scraped_ncaa_scores.csv uses CRLF line endings and has a trailing blank
# line, which COPY chokes on. Strip CRs with tr (portable -- grep's \r
# handling differs between BSD/ugrep and GNU), then drop empty lines.
CLEAN_SCRAPED="$(mktemp)"
trap 'rm -f "$COPY_SQL" "$CLEAN_SCRAPED"' EXIT

tr -d '\r' < "$DATA_DIR/scraped_ncaa_scores.csv" | grep -v '^$' > "$CLEAN_SCRAPED"

# A handful of filenames (Hawai'i, St_John's, ...) contain an apostrophe,
# which needs doubling inside a single-quoted psql string literal.
sql_quote() {
    printf '%s' "$1" | sed "s/'/''/g"
}

{
    echo "\\copy schedule(game_date, game_id, home_team, away_team) FROM '$(sql_quote "$DATA_DIR/schedule.csv")' WITH (FORMAT csv, HEADER true, NULL 'NA')"

    echo "\\copy scraped_scores(game_date, team_1_id, team_1_name, team_1_score, team_1_spread, team_2_id, team_2_name, team_2_score, team_2_spread, predicted_winner_id, predicted_winner_name) FROM '$(sql_quote "$CLEAN_SCRAPED")' WITH (FORMAT csv, HEADER true, NULL 'NA')"

    for f in "${TEAM_FILES[@]}"; do
        echo "\\copy box_scores(player_id, player, minutes_played, fgm, fga, three_ptm, three_pta, ftm, fta, oreb, dreb, reb, ast, stl, blk, turnovers, pf, pts, team, opponent, home, starter, game_id, game_date, location, team_score, opponent_score, result) FROM '$(sql_quote "$f")' WITH (FORMAT csv, HEADER true, NULL 'NA')"
    done
} > "$COPY_SQL"

echo "Loading ${#TEAM_FILES[@]} team box-score files, schedule.csv, and scraped_ncaa_scores.csv into '$DB_NAME'..."
psql -d "$DB_NAME" -v ON_ERROR_STOP=1 -f "$COPY_SQL"
echo "Done."
