from flask import Blueprint, abort, current_app, jsonify, render_template, request

from app.models import Cell
from app.services.ml_service import ml_service
from ml.provinces import PROVINCES

bp = Blueprint("explore", __name__)

# clamp map queries to Thailand so $geoWithin never gets a huge polygon
THAILAND_BBOX = (97.0, 5.0, 106.0, 21.0)


@bp.route("/map")
def map_view():
    coverage = [{**c, "name": PROVINCES.get(c["province_code"], "")} for c in Cell.coverage()]
    cluster_names = ml_service.cluster.names if ml_service.ready else {}
    return render_template("map.html", coverage=coverage, cluster_names=cluster_names)


@bp.route("/api/cells")
def api_cells():
    try:
        west, south, east, north = (float(v) for v in request.args["bbox"].split(","))
    except (KeyError, ValueError):
        abort(400, "bbox=west,south,east,north required")
    west, south = max(west, THAILAND_BBOX[0]), max(south, THAILAND_BBOX[1])
    east, north = min(east, THAILAND_BBOX[2]), min(north, THAILAND_BBOX[3])
    if west >= east or south >= north:
        return jsonify(cells=[], truncated=False)

    limit = current_app.config["MAP_CELL_LIMIT"]
    docs = Cell.in_bbox(west, south, east, north, limit)
    cells = [{"id": d["_id"], "lat": d["lat"], "lon": d["lon"], "price": d["price_median"],
              "grade": d.get("grade"), "cluster": d.get("cluster"), "cluster_name": d.get("cluster_name"),
              "amphoe": d.get("amphoe"), "province": PROVINCES.get(d.get("province_code"), "")}
             for d in docs]
    return jsonify(cells=cells, truncated=len(docs) >= limit)
