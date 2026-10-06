"""Thai display labels for model variables and states."""

BN_VARIABLES = {
    "Region": "ภูมิภาค",
    "CityProximity": "ระยะถึงตัวเมือง",
    "RoadAccess": "ระยะถึงถนนสายหลัก",
    "UrbanMap": "ระวางเขตเมือง (1:1000)",
    "Landuse": "การใช้ที่ดิน (OSM)",
    "PriceLevel": "ระดับราคา",
}

BN_STATES = {
    "CityProximity": {"near": "ใกล้ (<5 กม.)", "mid": "ปานกลาง (5–20 กม.)", "far": "ไกล (>20 กม.)"},
    "RoadAccess": {"near": "ใกล้ (<1 กม.)", "mid": "ปานกลาง (1–5 กม.)", "far": "ไกล (>5 กม.)"},
    "UrbanMap": {"yes": "ใช่", "no": "ไม่ใช่"},
    "Landuse": {"urban": "เมือง", "agri": "เกษตรกรรม", "forest": "ป่า/พื้นที่ธรรมชาติ", "unmapped": "ไม่มีข้อมูล"},
    "PriceLevel": {"Low": "ต่ำ (C)", "Mid": "กลาง (B)", "High": "สูง (A)"},
    "Region": {
        "Central": "ภาคกลาง", "East": "ภาคตะวันออก", "Northeast-Lower": "อีสานล่าง",
        "Northeast-Upper": "อีสานบน", "North-Upper": "ภาคเหนือตอนบน", "North-Lower": "ภาคเหนือตอนล่าง",
        "West": "ภาคตะวันตก", "South-Upper": "ภาคใต้ตอนบน", "South-Lower": "ภาคใต้ตอนล่าง",
    },
}

GRADES = {"A": "เกรด A — ราคาสูง", "B": "เกรด B — ราคากลาง", "C": "เกรด C — ราคาต่ำ"}

RISK = {"Low": "ต่ำ", "Medium": "ปานกลาง", "High": "สูง"}

MODEL_NAMES = {
    "hist_gbr": "HistGradientBoosting",
    "poly_ridge": "Polynomial + Ridge",
    "knn_latlon": "KNN (lat/lon)",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "expert": "Expert DAG",
    "hill_climb_bic": "HillClimbSearch (BIC)",
}

FEATURE_LABELS = {
    "dist_main_road_km": "ระยะถึงถนนสายหลัก (กม.)",
    "dist_secondary_road_km": "ระยะถึงถนนสายรอง (กม.)",
    "dist_town_km": "ระยะถึงตัวเมือง/อำเภอ (กม.)",
    "dist_city_km": "ระยะถึงเมืองใหญ่ (กม.)",
    "dist_rail_station_km": "ระยะถึงสถานีรถไฟ/รถไฟฟ้า (กม.)",
    "dist_beach_km": "ระยะถึงชายหาด (กม.)",
    "dist_bangkok_km": "ระยะถึงกรุงเทพฯ (กม.)",
    "road_km": "ความยาวถนนในช่อง (กม.)",
    "poi_count": "จำนวนสถานที่สำคัญ (POI)",
    "building_count": "จำนวนอาคาร (OSM)",
    "parcel_count": "จำนวนแปลงที่ดิน",
    "urban_map_share": "สัดส่วนระวางเขตเมือง",
    "landuse_urban_share": "สัดส่วนพื้นที่เมือง",
    "landuse_agri_share": "สัดส่วนพื้นที่เกษตร",
    "landuse_forest_share": "สัดส่วนพื้นที่ป่า",
    "region": "ภูมิภาค",
    "lat": "ละติจูด",
    "lon": "ลองจิจูด",
}


def bn_state(variable, state):
    return BN_STATES.get(variable, {}).get(state, state)
