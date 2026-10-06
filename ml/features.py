"""Shared feature definitions, preprocessing and spatial train/test split."""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml import settings

# skewed, non-negative -> log1p before scaling
LOG_FEATURES = [
    "parcel_count", "dist_main_road_km", "dist_secondary_road_km", "dist_city_km", "dist_town_km",
    "dist_rail_station_km", "dist_beach_km", "dist_bangkok_km", "road_km", "poi_count", "building_count",
]
LINEAR_FEATURES = [
    "lat", "lon", "urban_map_share",
    "landuse_urban_share", "landuse_agri_share", "landuse_forest_share",
]
CATEGORICAL_FEATURES = ["region"]
FEATURES = LOG_FEATURES + LINEAR_FEATURES + CATEGORICAL_FEATURES

TARGET = "price_median"
GROUP = "sheet50k"
GRADES = ["C", "B", "A"]


def make_preprocessor():
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    lin_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("log", log_pipe, LOG_FEATURES),
        ("lin", lin_pipe, LINEAR_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def load_cells():
    return pd.read_parquet(settings.CELLS_FEATURES_PATH)


def spatial_split(df):
    """Hold out whole 1:50,000 sheets so neighbouring cells never straddle train and test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=settings.TEST_SIZE,
                                 random_state=settings.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df[GROUP]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def group_cv():
    return GroupKFold(n_splits=settings.CV_FOLDS)


def grade_thresholds(train_prices):
    """Tertile cut points computed on the training set only."""
    q33, q67 = np.quantile(train_prices, [1 / 3, 2 / 3])
    return {"q33": float(q33), "q67": float(q67)}


def to_grade(prices, thresholds):
    return pd.cut(prices, [-np.inf, thresholds["q33"], thresholds["q67"], np.inf],
                  labels=GRADES, right=False).astype(str)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
