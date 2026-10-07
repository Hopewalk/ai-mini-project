from datetime import timedelta, timezone

from dotenv import load_dotenv
from flask import Flask

from app import labels
from app.extensions import csrf, mongo
from app.services.ml_service import ml_service
from config import Config
from ml.provinces import PROVINCES

TH_TZ = timezone(timedelta(hours=7))


def create_app(test_config=None):
    load_dotenv()
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    mongo.init_app(app)
    csrf.init_app(app)
    ml_service.init_app(app)

    from app.views import bayes, dashboard, explore, history, lookup, main

    for module in (main, lookup, explore, bayes, dashboard, history):
        app.register_blueprint(module.bp)

    _register_template_helpers(app)
    return app


def _register_template_helpers(app):
    @app.template_filter("thb")
    def thb(value, digits=0):
        if value is None:
            return "–"
        return f"{value:,.{digits}f}"

    @app.template_filter("pct")
    def pct(value, digits=0):
        if value is None:
            return "–"
        return f"{value * 100:.{digits}f}%"

    @app.template_filter("signed_pct")
    def signed_pct(value, digits=0):
        return f"{value * 100:+.{digits}f}%"

    @app.template_filter("pp")
    def percentage_points(value, digits=1):
        """Difference of two probabilities, in percentage points."""
        return f"{value * 100:+.{digits}f} pp"

    @app.template_filter("pts")
    def points(value, digits=1):
        """Probability difference as plain-language percentage points ("+5.2 จุด")."""
        return f"{value * 100:+.{digits}f} จุด"

    @app.template_filter("th_time")
    def th_time(value):
        """Mongo returns naive UTC datetimes; show them in Thai time (UTC+7)."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(TH_TZ).strftime("%d/%m/%Y %H:%M")

    @app.context_processor
    def inject_globals():
        return {
            "labels": labels,
            "provinces": PROVINCES,
            "model_ready": ml_service.ready,
            "model_version": app.config["MODEL_VERSION"],
            "show_lookup": app.config["SHOW_PARCEL_LOOKUP"],
        }
