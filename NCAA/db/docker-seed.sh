
# Runs inside the postgres container on first init only (empty data volume),
# via /docker-entrypoint-initdb.d/. Thin wrapper that points load_data.sh at
# the mounted CSVs and lets psql pick up credentials from libpq env vars.
set -euo pipefail

export PGUSER="$POSTGRES_USER"
export PGDATABASE="$POSTGRES_DB"
export DATA_DIR=/opt/ncaa/data/ncaa_box_scores_2024_25

bash /opt/ncaa/load_data.sh "$POSTGRES_DB"