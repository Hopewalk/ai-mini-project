"""Two-stage clustering model: MiniBatchKMeans micro-clusters -> Ward merge, with predict()."""
import numpy as np

CLUSTER_LOG_FEATURES = [
    "price_median", "parcel_count", "building_count", "poi_count", "road_km",
    "dist_main_road_km", "dist_town_km", "dist_beach_km",
]
CLUSTER_LINEAR_FEATURES = ["urban_map_share", "landuse_urban_share", "landuse_agri_share",
                           "landuse_forest_share"]
CLUSTER_FEATURES = CLUSTER_LOG_FEATURES + CLUSTER_LINEAR_FEATURES


class TwoStageCluster:
    def __init__(self, preprocessor, kmeans, micro_to_cluster, names):
        self.preprocessor = preprocessor
        self.kmeans = kmeans
        self.micro_to_cluster = np.asarray(micro_to_cluster)
        self.names = names

    def predict(self, X):
        micro = self.kmeans.predict(self.preprocessor.transform(X[CLUSTER_FEATURES]))
        return self.micro_to_cluster[micro]

    def name_of(self, label):
        return self.names[int(label)]
