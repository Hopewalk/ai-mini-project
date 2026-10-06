from urllib.parse import urlencode

from flask import Blueprint, current_app, render_template, request, url_for

from app.forms import BayesQueryForm
from app.models import ModelRun
from app.services.ml_service import ml_service

bp = Blueprint("bayes", __name__)

PRESETS = [
    ("ใกล้ถนนหลัก + อยู่ในเขตเมือง → ราคาจะเป็นอย่างไร?",
     {"target": "PriceLevel", "RoadAccess": "near", "UrbanMap": "yes"}),
    ("ไกลถนนหลัก → ราคาจะเป็นอย่างไร?", {"target": "PriceLevel", "RoadAccess": "far"}),
    ("ไกลตัวเมือง → ราคาจะเป็นอย่างไร?", {"target": "PriceLevel", "CityProximity": "far"}),
    ("รู้ว่าราคาสูง → น่าจะใกล้ถนนแค่ไหน?", {"target": "RoadAccess", "PriceLevel": "High"}),
    ("รู้ว่าราคาสูง → น่าจะอยู่ในเขตเมืองไหม?", {"target": "UrbanMap", "PriceLevel": "High"}),
    ("รู้ว่าราคาสูง + รอบ ๆ เป็นย่านเมือง → ยังต้องอยู่ในเขตเมืองไหม?",
     {"target": "UrbanMap", "PriceLevel": "High", "Landuse": "urban"}),
]


@bp.route("/bayes")
def query():
    if not ml_service.ready:
        return render_template("bayes.html", form=None, error=ml_service.error)

    form = BayesQueryForm(formdata=request.args if request.args else None)
    form.set_states(ml_service.state_names)
    target = form.target.data or "PriceLevel"
    evidence = form.evidence() if request.args and form.validate() else {}
    result = ml_service.query(target, evidence)
    prior = ml_service.query(target, {})

    run = ModelRun.get(current_app.config["MODEL_VERSION"])
    structures = run["bayesian_network"]["models"] if run else None
    presets = [(label, url_for("bayes.query") + "?" + urlencode(params)) for label, params in PRESETS]
    return render_template("bayes.html", form=form, target=target, evidence=evidence, result=result,
                           prior=prior, presets=presets, structures=structures, error=None)
