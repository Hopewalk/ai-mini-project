"""End-to-end view tests against a real MongoDB test database ($geoNear needs a real server).

Skipped when MongoDB (MONGO_TEST_URI, default mongodb://localhost:27017/landdb_test) or the trained
artifacts are not available.
"""
import os
import re

import pandas as pd
import pytest
from pymongo import GEOSPHERE, MongoClient
from pymongo.errors import PyMongoError

from ml import settings

TEST_URI = os.getenv("MONGO_TEST_URI", "mongodb://localhost:27017/landdb_test")
HAT_YAI = {"lat": 7.0086, "lon": 100.4747}


def _mongo_available():
    try:
        MongoClient(TEST_URI, serverSelectionTimeoutMS=1000).admin.command("ping")
        return True
    except PyMongoError:
        return False


pytestmark = pytest.mark.skipif(
    not (_mongo_available() and (settings.ARTIFACTS_DIR / "metrics.json").exists()
         and settings.CELLS_FEATURES_PATH.exists()),
    reason="needs MongoDB, trained artifacts and cells_features.parquet",
)


@pytest.fixture(scope="module")
def app():
    from app import create_app
    from ml.features import TARGET, to_grade
    import json

    db = MongoClient(TEST_URI).get_default_database()
    db.client.drop_database(db.name)

    cells = pd.read_parquet(settings.CELLS_FEATURES_PATH)
    near = cells[(cells.lat - HAT_YAI["lat"]).abs().lt(0.05) & (cells.lon - HAT_YAI["lon"]).abs().lt(0.05)]
    metrics = json.loads((settings.ARTIFACTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    docs = json.loads(near.to_json(orient="records"))
    grades = to_grade(near[TARGET], metrics["classification"]["thresholds"]).tolist()
    for d, g in zip(docs, grades):
        d.update(_id=d["cell_id"], grade=g, cluster=0, cluster_name="test",
                 location={"type": "Point", "coordinates": [d["lon"], d["lat"]]})
    db.cells.insert_many(docs)
    db.cells.create_index([("location", GEOSPHERE)])
    db.parcels.insert_one({"province_code": 90, "UTMMAP1": 5023, "UTMMAP2": 1, "UTMMAP3": 6272, "UTMMAP4": "13",
                           "UTMSCALE": 1000, "LAND_NO": "999", "EVAPRICE": 12345.0, "cell_id": docs[0]["_id"],
                           "lat": docs[0]["lat"], "lon": docs[0]["lon"]})

    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False, "MONGO_URI": TEST_URI})
    yield app
    db.client.drop_database(db.name)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.mark.parametrize("url", ["/", "/lookup", "/map", "/bayes", "/dashboard", "/history"])
def test_pages_render(client, url):
    assert client.get(url).status_code == 200


def test_analyze_location_end_to_end(client):
    resp = client.post("/", data={**HAT_YAI, "area_ngan": 1, "asking_total": 5_000_000})
    assert resp.status_code == 302
    page = client.get(resp.headers["Location"]).get_data(as_text=True)
    assert "บาท/ตร.ว." in page
    assert re.search(r"grade grade-[ABC] lg", page)
    assert re.search(r"risk risk-(Low|Medium|High)", page)
    assert "What-if" in page
    # shows up in history
    assert resp.headers["Location"].rsplit("/", 1)[-1] in client.get("/history").get_data(as_text=True)


def test_point_outside_coverage_is_rejected(client):
    resp = client.post("/", data={"lat": 10.0, "lon": 99.0})
    assert resp.status_code == 200
    assert "ยังไม่มีข้อมูล" in resp.get_data(as_text=True)


def test_asking_price_requires_area(client):
    resp = client.post("/", data={**HAT_YAI, "asking_total": 1_000_000})
    assert "ระบุเนื้อที่ด้วย" in resp.get_data(as_text=True)


def test_parcel_lookup(client):
    found = client.post("/lookup", data={"province": 90, "utmmap1": 5023, "utmmap2": 1, "utmmap3": 6272,
                                         "utmmap4": "13", "land_no": "999"})
    assert "12,345" in found.get_data(as_text=True)
    missing = client.post("/lookup", data={"province": 90, "utmmap1": 5023, "utmmap2": 1, "utmmap3": 6272,
                                           "utmmap4": "13", "land_no": "1"})
    assert "ไม่พบแปลงที่ดิน" in missing.get_data(as_text=True)


def test_api_cells_bbox(client):
    data = client.get("/api/cells?bbox=100.40,6.95,100.55,7.06").get_json()
    assert data["cells"] and {"id", "lat", "lon", "price", "grade"} <= data["cells"][0].keys()
    assert client.get("/api/cells").status_code == 400


def test_bayes_query_applies_evidence(client):
    """Evidence must be applied even when other evidence fields are absent from the query string."""
    page = client.get("/bayes?target=PriceLevel&RoadAccess=far").get_data(as_text=True)
    assert "Not a valid choice" not in page
    assert "ระยะถึงถนนสายหลัก=ไกล" in page


@pytest.mark.skipif(not (settings.ROOT / "app" / "static" / "vendor").exists(),
                    reason="frontend vendor assets not built (run `pnpm install`)")
def test_vendor_assets_are_served_locally(client):
    page = client.get("/").get_data(as_text=True)
    assert "cdn.jsdelivr.net" not in page and "unpkg.com" not in page
    for path in ["vendor/bootstrap/bootstrap.min.css", "vendor/leaflet/leaflet.js",
                 "vendor/ibm-plex-sans-thai/400.css"]:
        assert client.get(f"/static/{path}").status_code == 200


def test_artifact_route_only_serves_known_plots(client):
    assert client.get("/artifacts/cluster_map.png").status_code == 200
    assert client.get("/artifacts/metrics.json").status_code == 404
