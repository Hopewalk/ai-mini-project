import json

from flask import Blueprint, abort, current_app, render_template, send_from_directory

from app.models import ModelRun

bp = Blueprint("dashboard", __name__)

PLOTS = {
    "regression_residuals.png": "Regression — Predicted vs Actual / Residuals",
    "classifier_confusion.png": "Classification — Confusion matrix",
    "classifier_tree.png": "Classification — Decision tree",
    "cluster_k_selection.png": "Clustering — Elbow & silhouette",
    "cluster_pca.png": "Clustering — PCA 2-D projection",
    "cluster_dendrogram.png": "Clustering — Ward dendrogram (sample)",
    "cluster_map.png": "Clustering — แผนที่กลุ่มทำเล",
    "bn_dag_expert.png": "Bayesian Network — Expert DAG",
    "bn_dag_hillclimb.png": "Bayesian Network — HillClimbSearch DAG",
}


def _metrics():
    run = ModelRun.get(current_app.config["MODEL_VERSION"]) or ModelRun.latest()
    if run:
        return run
    path = current_app.config["ARTIFACTS_DIR"] / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@bp.route("/dashboard")
def index():
    available = {name: title for name, title in PLOTS.items()
                 if (current_app.config["ARTIFACTS_DIR"] / name).exists()}
    return render_template("dashboard.html", m=_metrics(), plots=available)


@bp.route("/artifacts/<name>")
def artifact(name):
    if name not in PLOTS:
        abort(404)
    return send_from_directory(current_app.config["ARTIFACTS_DIR"], name)
