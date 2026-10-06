"""Raw land CSVs -> parcels/<province>.parquet (decoded) and cells.parquet (2x2 km aggregates).

Provinces are decoded one at a time; only a slim numeric table (~19 bytes/parcel) is kept in
memory for the nationwide aggregation, so all 77 provinces (~tens of millions of parcels) fit in RAM.

Usage: python -m ml.build_cells
"""
import re

import numpy as np
import pandas as pd

from ml import settings
from ml.utm_sheet import cell_centre_lonlat, decode

MIN_PARCELS_PER_CELL = 3
# A garbage 1:50,000 sheet number can still pass the quadrant-offset check (the cell sits next to its
# own wrong sheet). Reject parcels outside Thailand or far from their province's median location;
# the longest province (Kanchanaburi) has 99.9% of parcels within ~195 km of its median.
THAILAND_BBOX = (97.3, 5.5, 105.7, 20.5)
MAX_KM_FROM_PROVINCE_CENTRE = 250
PARCEL_COLUMNS = ["province_code", "UTMMAP1", "UTMMAP2", "UTMMAP3", "UTMMAP4", "UTMSCALE", "LAND_NO",
                  "EVAPRICE", "zone", "e_km", "n_km", "cell_id", "lat", "lon", "sheet50k", "offset_km",
                  "consistent"]


def _province_code(path):
    return int(re.match(r"land_(\d+)_", path.name).group(1))


def _read_province(path):
    df = pd.read_csv(path, dtype={"UTMMAP4": str, "LAND_NO": str}, encoding="utf-8-sig")
    for col in ["UTMMAP1", "UTMMAP2", "UTMMAP3", "UTMSCALE", "EVAPRICE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    raw = len(df)
    df = df.dropna(subset=["UTMMAP1", "UTMMAP2", "UTMMAP3", "EVAPRICE"])
    df = df[df["EVAPRICE"] > 0]
    df = df.astype({"UTMMAP1": int, "UTMMAP2": int, "UTMMAP3": int})
    df["province_code"] = _province_code(path)
    return df, raw


def _cell_key(zone, e_km, n_km):
    return zone.astype(np.int64) * 10**8 + e_km.astype(np.int64) * 10**4 + n_km.astype(np.int64)


def process_province(path):
    """Decode one province, write its parcels file, return (slim aggregation rows, report row)."""
    df, raw = _read_province(path)
    df = decode(df)
    code = _province_code(path)
    implausible = _mark_implausible(df)
    df[PARCEL_COLUMNS].to_parquet(settings.PARCELS_DIR / f"province_{code}.parquet", index=False)

    ok = df[df["consistent"]]
    slim = pd.DataFrame({
        "key": _cell_key(ok["zone"].to_numpy(int), ok["e_km"].to_numpy(), ok["n_km"].to_numpy()),
        "price": ok["EVAPRICE"].to_numpy(np.float32),
        "urban": (ok["UTMSCALE"] <= 1000).to_numpy(),
        "province": np.full(len(ok), code, dtype=np.int16),
        "sheet": (ok["UTMMAP1"] * 10 + ok["UTMMAP2"]).to_numpy(np.int32),
    })
    report = {
        "province_code": code,
        "rows_raw": raw,
        "rows_valid": len(df),
        "inside_quadrant_pct": round(100 * (df["offset_km"] == 0).mean(), 2),
        "no_sheet_pct": round(100 * (df["UTMMAP3"] == 0).mean(), 2),
        "implausible": implausible,
        "consistent_pct": round(100 * df["consistent"].mean(), 2),
        "zone48_pct": round(100 * (df["zone"] == 48).mean(), 2),
    }
    return slim, report


def _mark_implausible(df):
    """Set consistent=False for parcels outside Thailand or far from the province centre; return count."""
    ok = df["consistent"]
    if not ok.any():
        return 0
    lat0, lon0 = df.loc[ok, "lat"].median(), df.loc[ok, "lon"].median()
    dist_km = 111.2 * np.hypot(df["lat"] - lat0, (df["lon"] - lon0) * np.cos(np.radians(lat0)))
    west, south, east, north = THAILAND_BBOX
    plausible = (df["lon"].between(west, east) & df["lat"].between(south, north)
                 & (dist_km <= MAX_KM_FROM_PROVINCE_CENTRE))
    bad = ok & ~plausible
    df.loc[bad, "consistent"] = False
    return int(bad.sum())


def _mode_by_key(slim, column):
    counts = slim.groupby(["key", column]).size().reset_index(name="n")
    top = counts.sort_values(["key", "n"], ascending=[True, False]).drop_duplicates("key")
    return top.set_index("key")[column]


def aggregate_cells(slim):
    g = slim.groupby("key")["price"]
    cells = pd.DataFrame({
        "parcel_count": g.size(),
        "price_median": g.median().astype(float),
        "price_p25": g.quantile(0.25).astype(float),
        "price_p75": g.quantile(0.75).astype(float),
        "price_max": g.max().astype(float),
        "urban_map_share": slim.groupby("key")["urban"].mean(),
        "province_code": _mode_by_key(slim, "province").astype(int),
        "sheet": _mode_by_key(slim, "sheet"),
    })
    cells = cells[cells["parcel_count"] >= MIN_PARCELS_PER_CELL].reset_index()

    keys = cells["key"].to_numpy()
    cells["zone"] = keys // 10**8
    cells["e_km"] = (keys // 10**4) % 10**4
    cells["n_km"] = keys % 10**4
    cells["cell_id"] = (cells["zone"].astype(str) + "_" + cells["e_km"].astype(str) + "_"
                        + cells["n_km"].astype(str))
    cells["lon"], cells["lat"] = cell_centre_lonlat(cells["zone"], cells["e_km"], cells["n_km"])
    cells["sheet50k"] = (cells["sheet"] // 10).astype(str) + "_" + (cells["sheet"] % 10).astype(str)
    cells["region"] = (cells["province_code"] // 10).map(settings.REGIONS)
    return cells.drop(columns=["key", "sheet"])[
        ["cell_id", "zone", "e_km", "n_km", "lat", "lon", "province_code", "sheet50k", "parcel_count",
         "price_median", "price_p25", "price_p75", "price_max", "urban_map_share", "region"]]


def main():
    settings.PARCELS_DIR.mkdir(parents=True, exist_ok=True)
    for old in settings.PARCELS_DIR.glob("*.parquet"):
        old.unlink()

    wanted = settings.selected_provinces()
    paths = [p for p in sorted(settings.RAW_LAND_DIR.glob("land_*.csv"))
             if wanted is None or _province_code(p) in wanted]
    if not paths:
        raise SystemExit(f"No land CSVs found in {settings.RAW_LAND_DIR}. Run `python -m ml.download` first.")

    slims, reports = [], []
    for path in paths:
        slim, report = process_province(path)
        slims.append(slim)
        reports.append(report)
        print(f"  {path.name}: {report['rows_raw']:,} rows, consistent {report['consistent_pct']}%", flush=True)
    report = pd.DataFrame(reports)
    report.to_csv(settings.INTERIM_DIR / "decode_report.csv", index=False)

    slim = pd.concat(slims, ignore_index=True)
    del slims
    cells = aggregate_cells(slim)
    cells.to_parquet(settings.CELLS_PATH, index=False)

    total_raw, total_ok = report["rows_raw"].sum(), len(slim)
    print(f"\nprovinces: {len(report)}  parcels: {total_raw:,}  located: {total_ok:,} "
          f"({100 * total_ok / total_raw:.2f}%)  -> cells (>= {MIN_PARCELS_PER_CELL} parcels): {len(cells):,}")
    print(f"rejected as implausible (outside Thailand / >{MAX_KM_FROM_PROVINCE_CENTRE} km from province): "
          f"{report['implausible'].sum():,}")
    print(f"cells lon {cells.lon.min():.2f}–{cells.lon.max():.2f}, lat {cells.lat.min():.2f}–{cells.lat.max():.2f}")
    print("lowest consistency:\n" + report.nsmallest(8, "consistent_pct").to_string(index=False))
    print(f"price_median range: {cells.price_median.min():,.0f} - {cells.price_median.max():,.0f}")


if __name__ == "__main__":
    main()
