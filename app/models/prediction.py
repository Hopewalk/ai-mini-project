from datetime import datetime, timezone

from bson import ObjectId
from pymongo import DESCENDING

from app.extensions import mongo


class Prediction:
    """A saved analysis (input + all model outputs) shown on the result and history pages."""

    @staticmethod
    def create(doc):
        doc = {**doc, "created_at": datetime.now(timezone.utc)}
        return str(mongo.db.predictions.insert_one(doc).inserted_id)

    @staticmethod
    def get(prediction_id):
        if not ObjectId.is_valid(prediction_id):
            return None
        return mongo.db.predictions.find_one({"_id": ObjectId(prediction_id)})

    @staticmethod
    def page(page, size):
        cursor = (mongo.db.predictions.find({}, {"input": 1, "location": 1, "regression.predicted": 1,
                                                 "classification.grade": 1, "cluster.name": 1,
                                                 "asking.risk": 1, "created_at": 1, "model_version": 1})
                  .sort("created_at", DESCENDING).skip((page - 1) * size).limit(size))
        return list(cursor), mongo.db.predictions.count_documents({})
