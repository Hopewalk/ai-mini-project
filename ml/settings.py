"""Paths and constants shared by the ML pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# .env must be loaded before MODEL_VERSION / LAND_PROVINCES are read below
load_dotenv(ROOT / ".env")
ML_DIR = ROOT / "ml"

DATA_DIR = ML_DIR / "data"
RAW_LAND_DIR = DATA_DIR / "raw" / "land"
RAW_OSM_DIR = DATA_DIR / "raw" / "osm"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

PARCELS_DIR = INTERIM_DIR / "parcels"  # one parquet file per province
CELLS_PATH = INTERIM_DIR / "cells.parquet"
CELLS_FEATURES_PATH = PROCESSED_DIR / "cells_features.parquet"

MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
ARTIFACTS_DIR = ML_DIR / "artifacts" / MODEL_VERSION

CKAN_PACKAGE_URL = "https://catalog.treasury.go.th/api/3/action/package_show?id=land-valuation"
OSM_GPKG_URL = "https://download.geofabrik.de/asia/thailand-latest-free.gpkg.zip"
OSM_GPKG_PATH = RAW_OSM_DIR / "thailand-latest-free.gpkg"

# กรุงเทพฯ, พระนครศรีอยุธยา, เชียงใหม่, ภูเก็ต, สงขลา
PILOT_PROVINCES = [10, 14, 50, 83, 90]


def selected_provinces():
    """Province codes from LAND_PROVINCES env var: "all", "10,14,...", or unset (= pilot)."""
    value = os.getenv("LAND_PROVINCES", "").strip().lower()
    if value == "all":
        return None
    if value:
        return [int(v) for v in value.split(",")]
    return PILOT_PROVINCES


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Province code first digit -> region (Thai province code convention).
REGIONS = {
    1: "Central",
    2: "East",
    3: "Northeast-Lower",
    4: "Northeast-Upper",
    5: "North-Upper",
    6: "North-Lower",
    7: "West",
    8: "South-Upper",
    9: "South-Lower",
}

BANGKOK_LONLAT = (100.5018, 13.7563)
