"""Download raw data: Treasury land appraisal CSVs (CKAN) and OSM Thailand GeoPackage.

Usage:
    python -m ml.download                 # pilot provinces + OSM
    LAND_PROVINCES=all python -m ml.download
    python -m ml.download --skip-osm
"""
import argparse
import re
import zipfile

import requests
from tqdm import tqdm

from ml import settings


def _stream(url, dest, expected_size=None):
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"skip (exists): {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def land_resources():
    """Return [(province_code, filename, url, size)] from the CKAN package."""
    resp = requests.get(settings.CKAN_PACKAGE_URL, timeout=60)
    resp.raise_for_status()
    out = []
    for res in resp.json()["result"]["resources"]:
        filename = res["url"].rsplit("/", 1)[-1]
        m = re.match(r"land_(\d+)_", filename)
        if m:
            out.append((int(m.group(1)), filename, res["url"], res.get("size")))
    return sorted(out)


def download_land():
    wanted = settings.selected_provinces()
    for code, filename, url, size in land_resources():
        if wanted is None or code in wanted:
            _stream(url, settings.RAW_LAND_DIR / filename, size)


def download_osm():
    if settings.OSM_GPKG_PATH.exists():
        print(f"skip (exists): {settings.OSM_GPKG_PATH.name}")
        return
    zip_path = _stream(settings.OSM_GPKG_URL, settings.RAW_OSM_DIR / "thailand-latest-free.gpkg.zip")
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith(".gpkg"))
        print(f"extracting {member}")
        with zf.open(member) as src, open(settings.OSM_GPKG_PATH, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    zip_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-osm", action="store_true")
    args = parser.parse_args()
    download_land()
    if not args.skip_osm:
        download_osm()


if __name__ == "__main__":
    main()
