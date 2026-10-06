"""Decode Thai cadastral UTM map sheet numbers (ระวาง) into 2x2 km grid cells and lat/lon.

Sheet system (verified against hand-checked rows, see tests/test_utm_sheet.py):
  * UTMMAP1 = 1:50,000 sheet "CCRR": column CC -> lon 75 + 0.5*CC, row RR -> lat -4.5 + 0.5*RR
    (30' block, SW corner).
  * UTMMAP2 = quadrant of that block: 1=I (NE), 2=II (SE), 3=III (SW), 4=IV (NW), each 15'x15'.
  * UTMMAP3 = 1:4000 sheet "EENN": tens/units of UTM easting and northing in km (SW corner of a
    2x2 km square). The hundreds/thousands are recovered from the 1:50,000 quadrant's extent.
  * UTM zone 47 west of 102°E, zone 48 east of it.
"""
import numpy as np
import pandas as pd
from pyproj import Transformer

CELL_KM = 2
# Some rows (mostly in the South) sit one quadrant outside the 1:50,000 sheet they are labelled
# with while the decoded position is still plausible; accept up to one quadrant (~27 km) of offset.
MAX_OFFSET_KM = 30

# quadrant -> (lon offset, lat offset) of its SW corner inside the 30' block
_QUADRANT_OFFSET = {1: (0.25, 0.25), 2: (0.25, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.25)}

_to_utm = {z: Transformer.from_crs(4326, 32600 + z, always_xy=True) for z in (47, 48)}
_to_wgs = {z: Transformer.from_crs(32600 + z, 4326, always_xy=True) for z in (47, 48)}


def quadrant_bounds(utmmap1, utmmap2):
    """(lon_w, lat_s, lon_e, lat_n) of a 1:50,000 quadrant."""
    col, row = divmod(int(utmmap1), 100)
    dlon, dlat = _QUADRANT_OFFSET[int(utmmap2)]
    lon_w = 75 + 0.5 * col + dlon
    lat_s = -4.5 + 0.5 * row + dlat
    return lon_w, lat_s, lon_w + 0.25, lat_s + 0.25


def zone_for_lon(lon):
    return np.where(np.asarray(lon) < 102.0, 47, 48)


def cell_centre_lonlat(zone, e_km, n_km):
    """WGS84 (lon, lat) arrays of 2 km cell centres given UTM zone and SW-corner km."""
    zone, e_km, n_km = (np.asarray(a) for a in (zone, e_km, n_km))
    lon, lat = np.full(len(zone), np.nan), np.full(len(zone), np.nan)
    for z in (47, 48):
        m = zone == z
        if m.any():
            lon[m], lat[m] = _to_wgs[z].transform((e_km[m] + CELL_KM / 2) * 1000, (n_km[m] + CELL_KM / 2) * 1000)
    return lon, lat


def _quadrant_utm_extent(utmmap1, utmmap2):
    lon_w, lat_s, lon_e, lat_n = quadrant_bounds(utmmap1, utmmap2)
    zone = int(zone_for_lon((lon_w + lon_e) / 2))
    xs, ys = _to_utm[zone].transform([lon_w, lon_e, lon_w, lon_e], [lat_s, lat_s, lat_n, lat_n])
    return zone, min(xs) / 1000, max(xs) / 1000, min(ys) / 1000, max(ys) / 1000


def decode(df):
    """Add zone, e_km, n_km (SW corner of 2 km cell), cell_id, lat, lon, sheet50k, consistent.

    `df` needs integer columns UTMMAP1, UTMMAP2, UTMMAP3.
    """
    out = df.copy()
    quads = out[["UTMMAP1", "UTMMAP2"]].drop_duplicates()
    valid_quad = quads["UTMMAP2"].isin(_QUADRANT_OFFSET.keys())
    ext = pd.DataFrame(
        [_quadrant_utm_extent(a, b) for a, b in quads[valid_quad].itertuples(index=False)],
        columns=["zone", "e_min", "e_max", "n_min", "n_max"],
        index=quads[valid_quad].index,
    )
    ext = pd.concat([quads[valid_quad], ext], axis=1)
    out = out.merge(ext, on=["UTMMAP1", "UTMMAP2"], how="left")

    ee, nn = np.divmod(out["UTMMAP3"].to_numpy(), 100)
    e_mid = (out["e_min"] + out["e_max"]).to_numpy() / 2
    n_mid = (out["n_min"] + out["n_max"]).to_numpy() / 2
    # pick the hundreds digit that puts the cell centre closest to the quadrant centre
    e_km = ee + 100 * np.round((e_mid - CELL_KM / 2 - ee) / 100)
    n_km = nn + 100 * np.round((n_mid - CELL_KM / 2 - nn) / 100)

    out["e_km"] = e_km
    out["n_km"] = n_km
    # distance (km) from the cell to the labelled quadrant's extent; 0 = inside
    dx = np.maximum.reduce([out["e_min"] - (e_km + CELL_KM), e_km - out["e_max"], np.zeros(len(out))])
    dy = np.maximum.reduce([out["n_min"] - (n_km + CELL_KM), n_km - out["n_max"], np.zeros(len(out))])
    out["offset_km"] = np.hypot(dx, dy)
    # UTMMAP3 == 0 marks rows without a 1:4000 sheet (e.g. UTMSCALE 5000) -> cannot be located
    out["consistent"] = (out["UTMMAP3"] > 0) & (out["offset_km"] <= MAX_OFFSET_KM)
    out = out.drop(columns=["e_min", "e_max", "n_min", "n_max"])

    ok = out["zone"].notna() & out["e_km"].notna()
    out["zone"] = out["zone"].astype("Int64")
    out.loc[ok, "e_km"] = out.loc[ok, "e_km"].astype(int)
    out.loc[ok, "n_km"] = out.loc[ok, "n_km"].astype(int)
    out["cell_id"] = pd.NA
    out.loc[ok, "cell_id"] = (
        out.loc[ok, "zone"].astype(str) + "_"
        + out.loc[ok, "e_km"].astype(int).astype(str) + "_"
        + out.loc[ok, "n_km"].astype(int).astype(str)
    )
    out["lat"] = np.nan
    out["lon"] = np.nan
    lon, lat = cell_centre_lonlat(out.loc[ok, "zone"].to_numpy(int), out.loc[ok, "e_km"].to_numpy(float),
                                  out.loc[ok, "n_km"].to_numpy(float))
    out.loc[ok, "lon"] = lon
    out.loc[ok, "lat"] = lat
    out["sheet50k"] = out["UTMMAP1"].astype(str) + "_" + out["UTMMAP2"].astype(str)
    return out


def cell_id_for_lonlat(lon, lat):
    """Cell id of an arbitrary WGS84 point (used by the web app)."""
    zone = int(zone_for_lon(lon))
    x, y = _to_utm[zone].transform(lon, lat)
    e_km = int(x // (CELL_KM * 1000)) * CELL_KM
    n_km = int(y // (CELL_KM * 1000)) * CELL_KM
    return f"{zone}_{e_km}_{n_km}"
