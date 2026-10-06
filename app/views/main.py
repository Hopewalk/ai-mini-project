from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for

from app.forms import LocationForm
from app.models import Cell, Prediction
from app.services import geo_service
from app.services.ml_service import ml_service
from ml.provinces import PROVINCES

bp = Blueprint("main", __name__)

DISPLAY_FEATURES = ["dist_main_road_km", "dist_secondary_road_km", "dist_town_km", "dist_city_km",
                    "dist_rail_station_km", "dist_beach_km", "dist_bangkok_km", "road_km", "poi_count",
                    "building_count", "parcel_count", "urban_map_share", "landuse_urban_share",
                    "landuse_agri_share", "landuse_forest_share"]

SAMPLE_LOCATIONS = [
    ("สีลม กรุงเทพฯ", 13.7262, 100.5310),
    ("เกาะเมืองอยุธยา", 14.3560, 100.5680),
    ("นิมมานฯ เชียงใหม่", 18.7990, 98.9680),
    ("ป่าตอง ภูเก็ต", 7.8960, 98.2990),
    ("หาดใหญ่ สงขลา", 7.0086, 100.4747),
    ("ในเมืองขอนแก่น", 16.4322, 102.8236),
    ("พัทยา ชลบุรี", 12.9276, 100.8771),
]


def _location(cell, distance_km, exact):
    return {
        "cell_id": cell["_id"],
        "exact": exact,
        "distance_km": distance_km,
        "province_code": cell["province_code"],
        "province_name": PROVINCES.get(cell["province_code"], str(cell["province_code"])),
        "amphoe": cell.get("amphoe"),
        "tambon": cell.get("tambon"),
        "lat": cell["lat"],
        "lon": cell["lon"],
        "features": {f: cell.get(f) for f in DISPLAY_FEATURES},
    }


@bp.route("/", methods=["GET", "POST"])
def index():
    form = LocationForm()
    if request.method == "GET":
        form.lat.data = request.args.get("lat", type=float)
        form.lon.data = request.args.get("lon", type=float)

    if form.validate_on_submit():
        if not ml_service.ready:
            flash(ml_service.error, "danger")
        else:
            cell, distance_km, exact = geo_service.locate(form.lon.data, form.lat.data)
            if cell is None:
                flash(f"ยังไม่มีข้อมูลราคาประเมินในรัศมี {current_app.config['NEAREST_CELL_MAX_KM']} กม. "
                      "จากจุดที่เลือก — ลองเลือกในจังหวัดที่มีข้อมูล", "warning")
            else:
                analysis = ml_service.analyze(cell)
                total_wah = form.total_wah
                asking = None
                if form.asking_total.data:
                    asking = ml_service.asking_check(analysis["regression"]["predicted"],
                                                     analysis["regression"].get("actual_median"),
                                                     total_wah, form.asking_total.data)
                prediction_id = Prediction.create({
                    "model_version": current_app.config["MODEL_VERSION"],
                    "input": {"lat": form.lat.data, "lon": form.lon.data, "area_wah": total_wah or None,
                              "asking_total": form.asking_total.data},
                    "location": _location(cell, distance_km, exact),
                    **analysis,
                    "asking": asking,
                    "neighbours": geo_service.neighbours(cell),
                })
                return redirect(url_for("main.result", prediction_id=prediction_id))

    coverage = [{**c, "name": PROVINCES.get(c["province_code"], "")} for c in Cell.coverage()]
    return render_template("index.html", form=form, coverage=coverage, samples=SAMPLE_LOCATIONS)


@bp.route("/result/<prediction_id>")
def result(prediction_id):
    prediction = Prediction.get(prediction_id)
    if prediction is None:
        abort(404)
    return render_template("result.html", p=prediction)
