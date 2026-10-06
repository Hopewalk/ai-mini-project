"""Add OpenStreetMap-derived features to each 2x2 km cell.

Input : ml/data/interim/cells.parquet + ml/data/raw/osm/thailand-latest-free.gpkg
Output: ml/data/processed/cells_features.parquet

Usage: python -m ml.geo_features
"""
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from shapely import STRtree

from ml import settings

# All distances are measured in UTM 47N. Cells east of 102°E (zone 48) get ~0.5% scale error,
# negligible for km-level features.
METRIC_CRS = 32647
HALF_CELL_M = 1000

MAIN_ROADS = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link")
SECONDARY_ROADS = ("secondary", "secondary_link", "tertiary", "tertiary_link")
DRIVABLE_ROADS = MAIN_ROADS + SECONDARY_ROADS + ("unclassified", "residential", "service", "living_street")
LANDUSE_GROUPS = {
    "urban": ("residential", "commercial", "retail", "industrial"),
    "agri": ("farmland", "farmyard", "orchard", "meadow", "vineyard", "allotments"),
    "forest": ("forest", "scrub", "heath"),
}


def _in(values):
    return "fclass IN (" + ",".join(f"'{v}'" for v in values) + ")"


def _read(layer, where=None, bbox=None):
    gdf = pyogrio.read_dataframe(settings.OSM_GPKG_PATH, layer=layer, columns=["fclass"],
                                 where=where, bbox=bbox)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf.to_crs(METRIC_CRS)


def _nearest_km(points, targets):
    out = np.full(len(points), np.nan)
    if len(targets) == 0:
        return out
    idx, dist = STRtree(targets.geometry.values).query_nearest(points, return_distance=True)
    out[idx[0]] = dist / 1000
    return out


def _count_in(squares, geoms):
    if len(geoms) == 0:
        return np.zeros(len(squares))
    pts = shapely.point_on_surface(geoms.geometry.values)
    sq_idx = STRtree(squares).query(pts, predicate="within")[1]
    return np.bincount(sq_idx, minlength=len(squares)).astype(float)


def _overlap_sum(squares, geoms, measure):
    """Sum of `measure` (shapely.length / shapely.area) of geoms clipped to each square."""
    if len(geoms) == 0:
        return np.zeros(len(squares))
    g = shapely.make_valid(geoms.geometry.values)
    g_idx, sq_idx = STRtree(squares).query(g, predicate="intersects")
    clipped = shapely.intersection(g[g_idx], squares[sq_idx])
    return np.bincount(sq_idx, weights=measure(clipped), minlength=len(squares))


def _haversine_km(lon, lat, lon0, lat0):
    lon, lat, lon0, lat0 = map(np.radians, (lon, lat, lon0, lat0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def nationwide_distances(cells, points):
    print("reading nationwide layers (main roads, places, stations, beaches)...")
    main = _read("gis_osm_roads_free", where=_in(MAIN_ROADS))
    secondary = _read("gis_osm_roads_free", where=_in(SECONDARY_ROADS))
    places = _read("gis_osm_places_free", where=_in(("city", "town", "national_capital")))
    stations = _read("gis_osm_transport_free", where=_in(("railway_station", "railway_halt")))
    beaches = _read("gis_osm_natural_free", where="fclass = 'beach'")

    cells["dist_main_road_km"] = _nearest_km(points, main)
    cells["dist_secondary_road_km"] = _nearest_km(points, secondary)
    cells["dist_city_km"] = _nearest_km(points, places[places.fclass.isin(["city", "national_capital"])])
    cells["dist_town_km"] = _nearest_km(points, places)
    cells["dist_rail_station_km"] = _nearest_km(points, stations)
    cells["dist_beach_km"] = _nearest_km(points, beaches)
    cells["dist_bangkok_km"] = _haversine_km(cells.lon, cells.lat, *settings.BANGKOK_LONLAT)


def admin_names(cells, points):
    """District (admin_level6: อำเภอ/เขต) and subdistrict (admin_level8: ตำบล, sparse in OSM) names.
    Display only — not used as model features."""
    admin = pyogrio.read_dataframe(settings.OSM_GPKG_PATH, layer="gis_osm_adminareas_a_free",
                                   columns=["fclass", "name"],
                                   where="fclass IN ('admin_level6', 'admin_level8')").to_crs(METRIC_CRS)
    for fclass, col in [("admin_level6", "amphoe"), ("admin_level8", "tambon")]:
        polys = admin[admin.fclass == fclass].reset_index(drop=True)
        pt_idx, poly_idx = STRtree(shapely.make_valid(polys.geometry.values)).query(points, predicate="within")
        names = pd.Series(polys.name.values[poly_idx], index=pt_idx)
        cells[col] = names[~names.index.duplicated()].reindex(range(len(cells))).values


def local_densities(cells, squares):
    for col in ["road_km", "poi_count", "building_count",
                *[f"landuse_{g}_share" for g in LANDUSE_GROUPS]]:
        cells[col] = 0.0
    cell_area = (2 * HALF_CELL_M) ** 2

    for code, idx in cells.groupby("province_code").groups.items():
        t0 = time.time()
        sub = cells.loc[idx]
        bbox = (sub.lon.min() - 0.05, sub.lat.min() - 0.05, sub.lon.max() + 0.05, sub.lat.max() + 0.05)
        sq = squares[cells.index.get_indexer(idx)]

        roads = _read("gis_osm_roads_free", where=_in(DRIVABLE_ROADS), bbox=bbox)
        cells.loc[idx, "road_km"] = _overlap_sum(sq, roads, shapely.length) / 1000
        cells.loc[idx, "poi_count"] = _count_in(sq, _read("gis_osm_pois_free", bbox=bbox))
        cells.loc[idx, "building_count"] = _count_in(sq, _read("gis_osm_buildings_a_free", bbox=bbox))

        landuse = _read("gis_osm_landuse_a_free", bbox=bbox)
        for group, classes in LANDUSE_GROUPS.items():
            area = _overlap_sum(sq, landuse[landuse.fclass.isin(classes)], shapely.area)
            cells.loc[idx, f"landuse_{group}_share"] = np.clip(area / cell_area, 0, 1)
        print(f"  province {code}: {len(sub)} cells, {time.time() - t0:.0f}s")


def main():
    cells = pd.read_parquet(settings.CELLS_PATH)
    pts = gpd.GeoSeries(gpd.points_from_xy(cells.lon, cells.lat), crs=4326).to_crs(METRIC_CRS).values
    x, y = shapely.get_x(pts), shapely.get_y(pts)
    squares = shapely.box(x - HALF_CELL_M, y - HALF_CELL_M, x + HALF_CELL_M, y + HALF_CELL_M)

    nationwide_distances(cells, pts)
    local_densities(cells, squares)
    admin_names(cells, pts)

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(settings.CELLS_FEATURES_PATH, index=False)
    print(f"\nsaved {len(cells):,} cells -> {settings.CELLS_FEATURES_PATH}")
    print(cells.describe().T[["mean", "50%", "max"]].round(2).to_string())
    print("\nmissing %:", (cells.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict())
    print(cells[["province_code", "amphoe", "tambon"]].dropna(subset=["amphoe"]).drop_duplicates("amphoe")
          .head(5).to_string(index=False))


if __name__ == "__main__":
    main()
