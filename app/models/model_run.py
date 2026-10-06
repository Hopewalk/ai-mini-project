from pymongo import DESCENDING

from app.extensions import mongo


class ModelRun:
    """Training run metrics written by scripts/load_mongo.py (collection `model_runs`)."""

    @staticmethod
    def get(version):
        return mongo.db.model_runs.find_one({"_id": version})

    @staticmethod
    def latest():
        return mongo.db.model_runs.find_one(sort=[("trained_at", DESCENDING)])
