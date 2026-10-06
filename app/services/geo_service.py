from flask import current_app

from app.models import Cell
from ml.utm_sheet import cell_id_for_lonlat


def locate(lon, lat):
    """Cell containing the point, else the nearest one within NEAREST_CELL_MAX_KM.

    Returns (cell, distance_km, exact) or (None, None, False) when outside the data coverage.
    """
    cell = Cell.get(cell_id_for_lonlat(lon, lat))
    if cell:
        return cell, 0.0, True
    nearest = Cell.near(lon, lat, current_app.config["NEAREST_CELL_MAX_KM"])
    if nearest:
        return nearest[0], nearest[0]["distance_km"], False
    return None, None, False


def neighbours(cell):
    docs = Cell.near(cell["lon"], cell["lat"], current_app.config["NEAREST_CELL_MAX_KM"],
                     limit=current_app.config["NEIGHBOUR_LIMIT"] + 1)
    return [{"cell_id": d["_id"], "amphoe": d.get("amphoe"), "price_median": d["price_median"],
             "grade": d.get("grade"), "cluster_name": d.get("cluster_name"),
             "distance_km": d["distance_km"], "lat": d["lat"], "lon": d["lon"]}
            for d in docs if d["_id"] != cell["_id"]]
