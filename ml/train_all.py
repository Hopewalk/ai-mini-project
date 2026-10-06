"""Train all four models with live progress, then write metrics.json and a training log.

Usage:
    python -m ml.train_all                              # train (data already prepared)
    python -m ml.train_all --prepare                    # rebuild cells + OSM features first
    python -m ml.train_all --download --prepare         # also download raw data
    python -m ml.train_all --provinces 10,90 --version v-test --prepare
    python -m ml.train_all --only regression,bn         # retrain some models (metrics.json is merged)
    python -m ml.train_all --no-progress                # plain output (CI / redirected logs)

Output: ml/artifacts/<version>/ (models, plots, metrics.json, train.log)
"""
import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone

STEPS = {  # name -> (module, title)
    "regression": ("ml.train_regression", "Regression — ราคาประเมิน (Baseline / Ridge / KNN / Random Forest)"),
    "classification": ("ml.train_classifier", "Classification — เกรด A/B/C (Baseline / LogReg / Decision Tree / Random Forest)"),
    "clustering": ("ml.train_cluster", "Clustering — กลุ่มทำเล (KMeans)"),
    "bayesian_network": ("ml.train_bn", "Bayesian Network — Expert DAG / HillClimb"),
}
ALIASES = {"reg": "regression", "clf": "classification", "cluster": "clustering", "bn": "bayesian_network"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train all land-appraisal models with live progress.")
    parser.add_argument("--download", action="store_true", help="download raw land CSVs + OSM first")
    parser.add_argument("--prepare", action="store_true", help="rebuild cells and OSM features first")
    parser.add_argument("--provinces", help='"all", or comma-separated codes e.g. 10,90 (overrides LAND_PROVINCES)')
    parser.add_argument("--version", help="artifact version folder, e.g. v3 (overrides MODEL_VERSION)")
    parser.add_argument("--only", help="comma-separated subset: regression,classification,clustering,bn")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return parser.parse_args()


def summary_table(metrics):
    rows = []
    if "regression" in metrics:
        r = metrics["regression"]
        t = r["models"][r["best_model"]]["test"]
        rows.append(("Regression", r["best_model"], f"R²(log) {t['r2_log']:.3f}  RMSE(log) {t['rmse_log']:.3f}"
                     f"  MAE {t['mae_thb']:,.0f} THB"))
    if "classification" in metrics:
        c = metrics["classification"]
        t = c["models"][c["best_model"]].get("test_calibrated") or c["models"][c["best_model"]]["test"]
        rows.append(("Classification", c["best_model"],
                     f"acc {t['accuracy']:.3f}  F1 {t['f1_macro']:.3f}  PR-AUC {t['pr_auc_macro']:.3f}"))
    if "clustering" in metrics:
        k = metrics["clustering"]
        s = k["scores"][k["best_k"]] if k["best_k"] in k["scores"] else k["scores"][str(k["best_k"])]
        rows.append(("Clustering", f"KMeans k={k['best_k']}",
                     f"silhouette {s['silhouette']:.3f}  Davies–Bouldin {s['davies_bouldin']:.3f}"))
    if "bayesian_network" in metrics:
        b = metrics["bayesian_network"]["models"]
        best = max(b, key=lambda n: b[n]["test_log_likelihood_per_row"])
        rows.append(("Bayesian Network", best, f"log-lik/row {b[best]['test_log_likelihood_per_row']:.3f}"
                     f"  acc {b[best]['price_level_accuracy']:.3f}"))
    width = max(len(r[1]) for r in rows) if rows else 10
    lines = [f"  {task:<17} {model:<{width}}  {score}" for task, model, score in rows]
    return "\n".join(lines)


def main():
    args = parse_args()
    # env overrides must be set before ml.settings is imported (it reads them at import time)
    if args.provinces:
        os.environ["LAND_PROVINCES"] = args.provinces
    if args.version:
        os.environ["MODEL_VERSION"] = args.version
    if hasattr(sys.stdout, "reconfigure"):  # Thai labels on consoles that default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    from ml import progress, settings
    progress.ENABLED = not args.no_progress

    selected = list(STEPS)
    if args.only:
        selected = [ALIASES.get(s.strip(), s.strip()) for s in args.only.split(",")]
        unknown = [s for s in selected if s not in STEPS]
        if unknown:
            raise SystemExit(f"unknown --only step(s): {unknown}; choose from {list(STEPS)} or {list(ALIASES)}")

    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tee = progress.Tee(settings.ARTIFACTS_DIR / "train.log")
    sys.stdout = tee
    t_start = time.perf_counter()
    provinces = settings.selected_provinces()
    try:
        print(f"model version {settings.MODEL_VERSION} | provinces: "
              f"{'all' if provinces is None else ','.join(map(str, provinces))} | "
              f"artifacts -> {settings.ARTIFACTS_DIR}")

        plan = (["download"] if args.download else []) + (["prepare"] if args.prepare else []) + selected
        n = len(plan)
        i = 0
        if args.download:
            from ml import download
            i += 1
            with progress.stage("Download raw data (Treasury CSVs + OSM)", i, n):
                download.download_land()
                download.download_osm()
        if args.prepare:
            from ml import build_cells, geo_features
            i += 1
            with progress.stage("Prepare data (decode sheets → cells → OSM features)", i, n):
                build_cells.main()
                geo_features.main()

        metrics_path = settings.ARTIFACTS_DIR / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if args.only and metrics_path.exists() else {}
        for name in selected:
            module_name, title = STEPS[name]
            i += 1
            with progress.stage(title, i, n):
                metrics[name] = importlib.import_module(module_name).main()

        metrics.update({
            "version": settings.MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provinces": "all" if provinces is None else provinces,
        })
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

        total = time.perf_counter() - t_start
        print(f"\n{'═' * 72}\n  TRAINING COMPLETE — {total / 60:.1f} min  (model {settings.MODEL_VERSION})\n{'═' * 72}")
        print(summary_table({k: v for k, v in metrics.items() if k in STEPS}))
        print(f"\n  artifacts : {settings.ARTIFACTS_DIR}\n  log       : {settings.ARTIFACTS_DIR / 'train.log'}")
        print("  next      : python -m scripts.load_mongo   (then restart the web app)")
    finally:
        tee.close()


if __name__ == "__main__":
    main()
