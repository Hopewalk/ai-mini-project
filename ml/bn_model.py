"""Bayesian Network variables and discretization, shared by training and the web app."""
import numpy as np
import pandas as pd

from ml.features import TARGET

STATES = {
    "CityProximity": ["near", "mid", "far"],
    "RoadAccess": ["near", "mid", "far"],
    "UrbanMap": ["yes", "no"],
    "Landuse": ["urban", "agri", "forest", "unmapped"],
    "PriceLevel": ["Low", "Mid", "High"],
}
# raw cell columns needed to build the evidence
RAW_COLUMNS = ["region", "dist_town_km", "dist_main_road_km", "urban_map_share",
               "landuse_urban_share", "landuse_agri_share", "landuse_forest_share"]


def discretize(df, thresholds):
    """Fixed km cut-offs (domain-defined) + price tertiles from the training set."""
    out = pd.DataFrame(index=df.index)
    out["Region"] = df["region"]
    out["CityProximity"] = pd.cut(df["dist_town_km"], [-np.inf, 5, 20, np.inf], labels=["near", "mid", "far"])
    out["RoadAccess"] = pd.cut(df["dist_main_road_km"], [-np.inf, 1, 5, np.inf], labels=["near", "mid", "far"])
    out["UrbanMap"] = np.where(df["urban_map_share"] >= 0.5, "yes", "no")
    out["Landuse"] = np.select(
        [df["landuse_urban_share"] >= 0.1, df["landuse_agri_share"] >= 0.1, df["landuse_forest_share"] >= 0.1],
        ["urban", "agri", "forest"], default="unmapped")
    if TARGET in df:
        out["PriceLevel"] = pd.cut(df[TARGET], [-np.inf, thresholds["q33"], thresholds["q67"], np.inf],
                                   labels=STATES["PriceLevel"], right=False)
    return out.astype(str)
