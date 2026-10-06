"""Thai display labels for model variables and states — worded for non-technical users."""

BN_VARIABLES = {
    "Region": "ภูมิภาค",
    "CityProximity": "ระยะห่างจากตัวเมือง",
    "RoadAccess": "ระยะถึงถนนสายหลัก",
    "UrbanMap": "อยู่ในเขตชุมชนเมือง",
    "Landuse": "ลักษณะพื้นที่โดยรอบ",
    "PriceLevel": "ระดับราคาที่ดิน",
}

BN_STATES = {
    "CityProximity": {"near": "ใกล้ (<5 กม.)", "mid": "ปานกลาง (5–20 กม.)", "far": "ไกล (>20 กม.)"},
    "RoadAccess": {"near": "ใกล้ (<1 กม.)", "mid": "ปานกลาง (1–5 กม.)", "far": "ไกล (>5 กม.)"},
    "UrbanMap": {"yes": "ใช่", "no": "ไม่ใช่"},
    "Landuse": {"urban": "ย่านเมือง/ชุมชน", "agri": "ไร่นา/เกษตรกรรม", "forest": "ป่า/พื้นที่ธรรมชาติ",
                "unmapped": "ไม่มีข้อมูล"},
    "PriceLevel": {"Low": "ราคาต่ำ (C)", "Mid": "ราคากลาง (B)", "High": "ราคาสูง (A)"},
    "Region": {
        "Central": "ภาคกลาง", "East": "ภาคตะวันออก", "Northeast-Lower": "อีสานล่าง",
        "Northeast-Upper": "อีสานบน", "North-Upper": "ภาคเหนือตอนบน", "North-Lower": "ภาคเหนือตอนล่าง",
        "West": "ภาคตะวันตก", "South-Upper": "ภาคใต้ตอนบน", "South-Lower": "ภาคใต้ตอนล่าง",
    },
}

GRADES = {"A": "ราคาสูง", "B": "ราคากลาง", "C": "ราคาต่ำ"}
GRADE_HINTS = {
    "A": "แพงกว่าพื้นที่ส่วนใหญ่ — อยู่ในกลุ่ม 1 ใน 3 ที่ราคาสูงที่สุด",
    "B": "ราคาระดับกลาง เมื่อเทียบกับพื้นที่อื่น",
    "C": "ถูกกว่าพื้นที่ส่วนใหญ่ — อยู่ในกลุ่ม 1 ใน 3 ที่ราคาต่ำที่สุด",
}
GRADE_SHORT = {"A": "ราคาสูง (A)", "B": "ราคากลาง (B)", "C": "ราคาต่ำ (C)"}

RISK = {"Low": "ปกติ", "Medium": "ค่อนข้างแพง", "High": "แพงมาก"}
RISK_VERDICT = {
    "Low": "ราคาที่เสนออยู่ในเกณฑ์ปกติ",
    "Medium": "ราคาที่เสนอค่อนข้างแพง — ควรต่อรองหรือหาข้อมูลเพิ่ม",
    "High": "ราคาที่เสนอแพงกว่าราคาประเมินมาก — ควรระวัง",
}

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
    "road_km": "ความยาวถนนรวมในพื้นที่ (กม.)",
    "poi_count": "จำนวนร้านค้า/สถานที่สำคัญ",
    "building_count": "จำนวนอาคาร",
    "parcel_count": "จำนวนแปลงที่ดิน",
    "urban_map_share": "สัดส่วนที่ดินในเขตชุมชนเมือง",
    "landuse_urban_share": "สัดส่วนพื้นที่ย่านเมือง",
    "landuse_agri_share": "สัดส่วนพื้นที่เกษตร",
    "landuse_forest_share": "สัดส่วนพื้นที่ป่า",
    "region": "ภูมิภาค",
    "lat": "ตำแหน่งเหนือ–ใต้ (ละติจูด)",
    "lon": "ตำแหน่งตะวันออก–ตะวันตก (ลองจิจูด)",
}


def bn_state(variable, state):
    return BN_STATES.get(variable, {}).get(state, state)
