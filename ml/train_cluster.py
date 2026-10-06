"""Clustering: location profiles via MiniBatchKMeans (micro) -> Agglomerative Ward (macro).

Ward on ~10^5 rows needs O(n^2) memory and has no predict(), so Ward runs on the KMeans
centroids instead; new points go kmeans.predict -> micro id -> ward label.

Usage: python -m ml.train_cluster
"""
import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml import settings
from ml.cluster_model import CLUSTER_FEATURES, CLUSTER_LINEAR_FEATURES, CLUSTER_LOG_FEATURES, TwoStageCluster
from ml.features import load_cells, save_json
from ml.progress import step

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Thai-capable font: Tahoma (Windows/macOS) or Loma (fonts-thai-tlwg in Docker)
_installed = {f.name for f in font_manager.fontManager.ttflist}
_thai = next((f for f in ["Tahoma", "Loma", "Garuda", "Thonburi"] if f in _installed), None)
if _thai:
    plt.rcParams["font.family"] = [_thai, "DejaVu Sans"]

N_MICRO = 60
K_RANGE = range(4, 8)


def make_cluster_preprocessor():
    return ColumnTransformer([
        ("log", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]),
         CLUSTER_LOG_FEATURES),
        ("lin", StandardScaler(), CLUSTER_LINEAR_FEATURES),
    ])


def describe(profile, overall):
    """Readable Thai name from a cluster's median profile.

    Uses map-sheet scale and landuse rather than OSM building counts: buildings are unmapped in most
    rural cells, so their quantiles collapse to 0 nationwide.
    """
    price = profile["price_median"]
    q33, q67, q95 = overall["price_median"].quantile([1 / 3, 2 / 3, 0.95])
    tier = ("ราคาสูงมาก" if price >= q95 else "ราคาสูง" if price >= q67
            else "ราคากลาง" if price >= q33 else "ราคาต่ำ")
    if profile["landuse_urban_share"] >= 0.5:
        area = "ใจกลางเมือง"
    elif profile["urban_map_share"] >= 0.5:
        area = "ย่านเมือง"
    elif profile["dist_town_km"] <= overall["dist_town_km"].median():
        area = "ชานเมือง"
    else:
        area = "ชนบท"
    landuse = max(["landuse_agri_share", "landuse_forest_share"], key=lambda c: profile[c])
    extra = ""
    if profile["dist_beach_km"] <= 5:
        extra = " · ใกล้ชายหาด"
    elif area != "ย่านเมือง" and profile[landuse] >= 0.2:
        extra = " · เกษตรกรรม" if landuse == "landuse_agri_share" else " · ป่า/ภูเขา"
    return f"{area} {tier}{extra}"


def main():
    out_dir = settings.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    pre = make_cluster_preprocessor().fit(cells[CLUSTER_FEATURES])
    Z = pre.transform(cells[CLUSTER_FEATURES])

    step(f"stage 1: MiniBatchKMeans -> {N_MICRO} micro-clusters on {len(Z):,} cells × {Z.shape[1]} features")
    kmeans = MiniBatchKMeans(n_clusters=N_MICRO, n_init=10, random_state=settings.RANDOM_STATE).fit(Z)
    centroids = kmeans.cluster_centers_
    rng = np.random.default_rng(settings.RANDOM_STATE)
    sample = rng.choice(len(Z), size=min(10_000, len(Z)), replace=False)

    step(f"stage 2: Ward on centroids, k = {K_RANGE.start}..{K_RANGE.stop - 1} vs plain KMeans "
         f"(silhouette on {len(sample):,}-cell sample)")
    scores = {}
    for k in K_RANGE:
        micro_map = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(centroids)
        labels = micro_map[kmeans.labels_]
        km_labels = KMeans(n_clusters=k, n_init=10, random_state=settings.RANDOM_STATE).fit_predict(Z)
        scores[k] = {
            "two_stage_silhouette": silhouette_score(Z[sample], labels[sample]),
            "two_stage_davies_bouldin": davies_bouldin_score(Z, labels),
            "kmeans_silhouette": silhouette_score(Z[sample], km_labels[sample]),
            "kmeans_davies_bouldin": davies_bouldin_score(Z, km_labels),
        }
        s = scores[k]
        step(f"k={k}  silhouette 2-stage {s['two_stage_silhouette']:.3f} vs KMeans {s['kmeans_silhouette']:.3f}"
             f"  | Davies–Bouldin {s['two_stage_davies_bouldin']:.3f} vs {s['kmeans_davies_bouldin']:.3f}")

    best_k = max(scores, key=lambda k: scores[k]["two_stage_silhouette"])
    micro_map = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(centroids)
    cells["cluster"] = micro_map[kmeans.labels_]

    profile = cells.groupby("cluster")[CLUSTER_FEATURES].median()
    profile["n_cells"] = cells.groupby("cluster").size()
    names = {int(c): describe(profile.loc[c], cells) for c in profile.index}
    # make duplicate names unique
    seen = {}
    for c, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} ({seen[n]})"
    profile["name"] = pd.Series(names)
    step(f"best k = {best_k} (highest 2-stage silhouette)")
    for c, row in profile.iterrows():
        step(f"  cluster {c}: {row['name']:<32} {int(row['n_cells']):>7,} cells  median "
             f"{row['price_median']:>8,.0f} THB/wah  urban map {row['urban_map_share']:.0%}")

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage(centroids, method="ward"), ax=ax, color_threshold=None, no_labels=True)
    ax.set(title=f"Ward dendrogram of {N_MICRO} KMeans micro-clusters (cut at k={best_k})",
           ylabel="Ward distance")
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_dendrogram.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 9))
    for c, grp in cells.groupby("cluster"):
        ax.scatter(grp.lon, grp.lat, s=4, label=names[int(c)])
    ax.set(title="Location clusters", xlabel="lon", ylabel="lat", aspect="equal")
    ax.legend(markerscale=4, loc="upper left", bbox_to_anchor=(1.02, 1), prop={"size": 9})
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_map.png", dpi=120)
    plt.close(fig)

    model = TwoStageCluster(pre, kmeans, micro_map, names)
    joblib.dump(model, out_dir / "cluster.joblib")
    cells[["cell_id", "cluster"]].to_parquet(out_dir / "cell_clusters.parquet", index=False)
    metrics = {"n_micro": N_MICRO, "best_k": best_k, "scores": scores, "names": names,
               "profile": profile.reset_index().to_dict(orient="records")}
    save_json(metrics, out_dir / "metrics_cluster.json")
    return metrics


if __name__ == "__main__":
    main()
