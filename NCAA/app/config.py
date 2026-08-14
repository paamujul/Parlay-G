"""Runtime configuration for the NCAA app, loaded from config/ncaa.yaml.

Resolves the YAML file's path relative to this module's own location
(mirroring the SCRIPT_DIR pattern in NCAA/db/load_data.sh) so it loads
correctly no matter what directory the app is launched from.

Every key has a hardcoded default below; config/ncaa.yaml only needs
to contain values that override a default. A missing, empty, or
partially filled-in config/ncaa.yaml is not an error -- the app just
runs with defaults for whatever wasn't specified.

DB credentials are configured separately, directly in backend.py via
python-dotenv/.env, and are intentionally NOT part of this file.

NOTE: this module's top-level code (the YAML read + merge) only runs
once per Python process, the first time something imports `config`.
Editing config/ncaa.yaml while `streamlit run` is already running
requires restarting that process -- a browser refresh (F5) alone will
NOT pick up the change.
"""

import os

import yaml

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.normpath(
    os.path.join(_APP_DIR, "..", "..", "config", "ncaa.yaml")
)

DEFAULTS = {
    "split_date": "2025-02-28",
    "rolling_window_size": 3,
    "elo": {
        "initial_rating": 1500,
        "k_factor": 20,
        "home_advantage": 100,
    },
    "models": {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "neural_network": "Neural Network (Keras)",
    },
    "parlay": {
        "min_legs": 2,
        "max_legs": 15,
        "default_legs": 3,
        "step": 1,
    },
}


def _deep_merge(base, override):
    """Returns a new dict: `base` with `override` merged on top,
    recursing into nested dicts. Values present in `override` win;
    keys missing from `override` keep their `base` value."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            loaded = yaml.safe_load(f)
    except FileNotFoundError:
        loaded = None
    if not isinstance(loaded, dict):
        # Empty file (yaml.safe_load returns None), or a YAML file
        # that isn't a mapping at the top level -- fall back cleanly.
        loaded = {}
    return _deep_merge(DEFAULTS, loaded)


_config = _load_config()

# --- Feature engineering configuration ---
SPLIT_DATE = _config["split_date"]
ROLLING_WINDOW_SIZE = _config["rolling_window_size"]
INITIAL_ELO = _config["elo"]["initial_rating"]
K_FACTOR = _config["elo"]["k_factor"]
HOME_ADVANTAGE_ELO = _config["elo"]["home_advantage"]

# --- Model registry: stable identifier -> display name ---
# Treat as read-only; mutating in place would also mutate DEFAULTS.
MODELS = _config["models"]

# --- Parlay-builder UI defaults ---
PARLAY_MIN_LEGS = _config["parlay"]["min_legs"]
PARLAY_MAX_LEGS = _config["parlay"]["max_legs"]
PARLAY_DEFAULT_LEGS = _config["parlay"]["default_legs"]
PARLAY_STEP = _config["parlay"]["step"]
