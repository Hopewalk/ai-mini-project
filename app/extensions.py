from flask_wtf import CSRFProtect
from pymongo import MongoClient


class Mongo:
    """Minimal Flask integration for PyMongo: `mongo.db` is the database named in MONGO_URI."""

    def __init__(self):
        self.client = None
        self.db = None

    def init_app(self, app):
        self.client = MongoClient(app.config["MONGO_URI"], serverSelectionTimeoutMS=3000, tz_aware=True)
        self.db = self.client.get_default_database()


mongo = Mongo()
csrf = CSRFProtect()
