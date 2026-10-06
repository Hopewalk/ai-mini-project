from app.extensions import mongo

MAP_FIELDS = {"_id": 1, "lat": 1, "lon": 1, "price_median": 1, "grade": 1, "cluster": 1,
              "cluster_name": 1, "amphoe": 1, "province_code": 1}


def _point(lon, lat):
    return {"type": "Point", "coordinates": [lon, lat]}


class Cell:
    """2x2 km grid cell with appraisal stats, OSM features and model labels (collection `cells`)."""

    @staticmethod
    def get(cell_id):
        return mongo.db.cells.find_one({"_id": cell_id})

    @staticmethod
    def near(lon, lat, max_km, limit=1):
        """Cells ordered by distance, each with `distance_km`."""
        pipeline = [
            {"$geoNear": {"near": _point(lon, lat), "distanceField": "distance_m",
                          "maxDistance": max_km * 1000, "spherical": True}},
            {"$limit": limit},
        ]
        docs = list(mongo.db.cells.aggregate(pipeline))
        for d in docs:
            d["distance_km"] = d.pop("distance_m") / 1000
        return docs

    @staticmethod
    def in_bbox(west, south, east, north, limit):
        box = {"type": "Polygon", "coordinates": [[[west, south], [east, south], [east, north],
                                                   [west, north], [west, south]]]}
        return list(mongo.db.cells.find({"location": {"$geoWithin": {"$geometry": box}}}, MAP_FIELDS)
                    .limit(limit))

    @staticmethod
    def coverage():
        """[{province_code, cells, lat, lon}] — provinces that have data, with a centre point."""
        return list(mongo.db.cells.aggregate([
            {"$group": {"_id": "$province_code", "cells": {"$sum": 1},
                        "lat": {"$avg": "$lat"}, "lon": {"$avg": "$lon"}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "province_code": "$_id", "cells": 1, "lat": 1, "lon": 1}},
        ]))
