"""Load cells (+ cluster/grade labels), model run metrics and optionally parcels into MongoDB.

Usage:
    python -m scripts.load_mongo              # cells + model_runs
    python -m scripts.load_mongo --parcels    # also parcels (millions of rows, slower)

MONGO_URI defaults to mongodb://localhost:27017/landdb
"""
import argparse
import json
import os

import pandas as pd
from pymongo import ASCENDING, GEOSPHERE, MongoClient

from ml import settings
from ml.features import TARGET, to_grade

BATCH = 50_000


def _records(df):
    return json.loads(df.to_json(orient="records", force_ascii=False))


def load_cells(db):
    cells = pd.read_parquet(settings.CELLS_FEATURES_PATH)
    clusters = pd.read_parquet(settings.ARTIFACTS_DIR / "cell_clusters.parquet")
    metrics = json.loads((settings.ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    names = metrics["clustering"]["names"]
    thresholds = metrics["classification"]["thresholds"]

    cells = cells.merge(clusters, on="cell_id", how="left")
    cells["cluster_name"] = cells["cluster"].astype(str).map(names)
    cells["grade"] = to_grade(cells[TARGET], thresholds)
    docs = _records(cells)
    for d in docs:
        d["_id"] = d["cell_id"]
        d["location"] = {"type": "Point", "coordinates": [d["lon"], d["lat"]]}

    # full replace: a new model version / province set may drop cells that existed before
    db.cells.drop()
    for start in range(0, len(docs), BATCH):
        db.cells.insert_many(docs[start:start + BATCH], ordered=False)
    db.cells.create_index([("location", GEOSPHERE)])
    db.cells.create_index([("province_code", ASCENDING)])
    print(f"cells: {db.cells.count_documents({}):,}")


def load_model_run(db):
    metrics = json.loads((settings.ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    metrics["_id"] = metrics["version"]
    db.model_runs.replace_one({"_id": metrics["_id"]}, metrics, upsert=True)
    print(f"model_runs: {metrics['_id']} ({metrics['trained_at']})")


def load_parcels(db):
    files = sorted(settings.PARCELS_DIR.glob("province_*.parquet"))
    if not files:
        raise SystemExit(f"No parcel files in {settings.PARCELS_DIR}. Run `python -m ml.build_cells` first.")
    db.parcels.drop()
    total = 0
    for path in files:  # one province at a time keeps memory flat
        parcels = pd.read_parquet(path)
        parcels = parcels[parcels["consistent"]].drop(columns=["consistent", "offset_km"])
        for start in range(0, len(parcels), BATCH):
            db.parcels.insert_many(_records(parcels.iloc[start:start + BATCH]), ordered=False)
        total += len(parcels)
        print(f"  {path.stem}: {len(parcels):,} (total {total:,})", flush=True)
    print("creating indexes...", flush=True)
    db.parcels.create_index([("province_code", ASCENDING), ("UTMMAP1", ASCENDING), ("UTMMAP2", ASCENDING),
                             ("UTMMAP3", ASCENDING), ("UTMMAP4", ASCENDING), ("LAND_NO", ASCENDING)])
    db.parcels.create_index([("cell_id", ASCENDING)])
    print(f"parcels: {db.parcels.estimated_document_count():,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parcels", action="store_true", help="also load parcels")
    parser.add_argument("--parcels-only", action="store_true", help="load parcels only (no trained models needed)")
    args = parser.parse_args()

    client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/landdb"))
    db = client.get_default_database()
    if not args.parcels_only:
        load_cells(db)
        load_model_run(db)
    if args.parcels or args.parcels_only:
        load_parcels(db)


if __name__ == "__main__":
    main()
