import os

from ml import settings


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/landdb")
    MODEL_VERSION = settings.MODEL_VERSION
    ARTIFACTS_DIR = settings.ARTIFACTS_DIR

    # a clicked point uses the nearest cell within this distance
    NEAREST_CELL_MAX_KM = 5
    NEIGHBOUR_LIMIT = 6

    # asking price / predicted appraisal ratio -> risk level. Appraisals sit below market prices,
    # so these are deliberately loose; tune once real transaction data is available.
    RISK_RATIO_MEDIUM = 2.0
    RISK_RATIO_HIGH = 3.0

    HISTORY_PAGE_SIZE = 20
    MAP_CELL_LIMIT = 8000
