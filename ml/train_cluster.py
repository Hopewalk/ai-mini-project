"""Clustering: location profiles with KMeans.

k is chosen from the elbow (inertia) and silhouette curves; a Ward dendrogram on a sample and a
2-D PCA projection are saved to help explain the clusters.

Usage: python -m ml.train_cluster
"""
import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml import settings
from ml.cluster_model import CLUSTER_FEATURES, CLUSTER_LINEAR_FEATURES, CLUSTER_LOG_FEATURES, LocationCluster
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

# k=2 only separates town from countryside, too coarse to describe a location
K_RANGE = range(3, 9)
SAMPLE_SIZE = 10_000      # silhouette is O(n^2): score on a random sample
DENDROGRAM_SAMPLE = 2_000


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


def plot_k_selection(scores, best_k, out_dir):
    ks = list(scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ks, [scores[k]["inertia"] for k in ks], marker="o")
    axes[0].set(xlabel="k", ylabel="inertia", title="Elbow method")
    axes[1].plot(ks, [scores[k]["silhouette"] for k in ks], marker="o")
    axes[1].axvline(best_k, color="red", linestyle="--", lw=1)
    axes[1].set(xlabel="k", ylabel="silhouette", title=f"Silhouette (best k = {best_k})")
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_k_selection.png", dpi=120)
    plt.close(fig)


def main():
    out_dir = settings.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    X = cells[CLUSTER_FEATURES]
    Z = make_cluster_preprocessor().fit_transform(X)
    rng = np.random.default_rng(settings.RANDOM_STATE)
    sample = rng.choice(len(Z), size=min(SAMPLE_SIZE, len(Z)), replace=False)

    step(f"KMeans on {len(Z):,} cells × {Z.shape[1]} features, k = {K_RANGE.start}..{K_RANGE.stop - 1}"
         f" (silhouette on {len(sample):,}-cell sample)")
    scores = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=settings.RANDOM_STATE).fit(Z)
        scores[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(Z[sample], km.labels_[sample])),
            "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
        }
        s = scores[k]
        step(f"k={k}  inertia {s['inertia']:,.0f}  silhouette {s['silhouette']:.3f}"
             f"  Davies–Bouldin {s['davies_bouldin']:.3f}")

    best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    pipeline = Pipeline([
        ("pre", make_cluster_preprocessor()),
        ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=settings.RANDOM_STATE)),
    ]).fit(X)
    cells["cluster"] = pipeline.predict(X)

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
    step(f"best k = {best_k} (highest silhouette)")
    for c, row in profile.iterrows():
        step(f"  cluster {c}: {row['name']:<32} {int(row['n_cells']):>7,} cells  median "
             f"{row['price_median']:>8,.0f} THB/wah  urban map {row['urban_map_share']:.0%}")

    plot_k_selection(scores, best_k, out_dir)

    # Week 8: PCA projection to 2-D to see how well the clusters separate
    Z_fit = pipeline.named_steps["pre"].transform(X)
    pca = PCA(n_components=2, random_state=settings.RANDOM_STATE).fit(Z_fit)
    P = pca.transform(Z_fit[sample])
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(names):
        m = cells["cluster"].to_numpy()[sample] == c
        ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.5, label=names[c])
    ratio = pca.explained_variance_ratio_
    ax.set(xlabel=f"PC1 ({ratio[0]:.0%} variance)", ylabel=f"PC2 ({ratio[1]:.0%} variance)",
           title="Location clusters — PCA 2-D projection")
    ax.legend(markerscale=4, prop={"size": 9})
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_pca.png", dpi=120)
    plt.close(fig)

    # Week 7: hierarchical (Ward) dendrogram on a small sample, for exploring structure only
    dsample = rng.choice(len(Z_fit), size=min(DENDROGRAM_SAMPLE, len(Z_fit)), replace=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage(Z_fit[dsample], method="ward"), ax=ax, no_labels=True, truncate_mode="lastp", p=40)
    ax.set(title=f"Ward dendrogram ({len(dsample):,}-cell sample)", ylabel="Ward distance")
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

    joblib.dump(LocationCluster(pipeline, names), out_dir / "cluster.joblib")
    cells[["cell_id", "cluster"]].to_parquet(out_dir / "cell_clusters.parquet", index=False)
    metrics = {"best_k": best_k, "scores": scores, "names": names,
               "pca_explained_variance": [float(r) for r in ratio],
               "profile": profile.reset_index().to_dict(orient="records")}
    save_json(metrics, out_dir / "metrics_cluster.json")
    step(f"PCA 2-D keeps {ratio.sum():.0%} of variance | saved cluster.joblib, cluster_k_selection.png,"
         f" cluster_pca.png, cluster_dendrogram.png, cluster_map.png -> {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
