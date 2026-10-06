"""Location clustering model: scaled features -> KMeans, with readable cluster names."""
CLUSTER_LOG_FEATURES = [
    "price_median", "parcel_count", "building_count", "poi_count", "road_km",
    "dist_main_road_km", "dist_town_km", "dist_beach_km",
]
CLUSTER_LINEAR_FEATURES = ["urban_map_share", "landuse_urban_share", "landuse_agri_share",
                           "landuse_forest_share"]
CLUSTER_FEATURES = CLUSTER_LOG_FEATURES + CLUSTER_LINEAR_FEATURES


class LocationCluster:
    def __init__(self, pipeline, names):
        self.pipeline = pipeline  # preprocessor -> KMeans
        self.names = names

    def predict(self, X):
        return self.pipeline.predict(X[CLUSTER_FEATURES])

    def name_of(self, label):
        return self.names[int(label)]
