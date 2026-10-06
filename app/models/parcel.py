from app.extensions import mongo


class Parcel:
    """Single Treasury appraisal record (collection `parcels`)."""

    @staticmethod
    def find(province_code, utmmap1, utmmap2, utmmap3, utmmap4, land_no):
        return mongo.db.parcels.find_one({
            "province_code": province_code, "UTMMAP1": utmmap1, "UTMMAP2": utmmap2,
            "UTMMAP3": utmmap3, "UTMMAP4": utmmap4, "LAND_NO": land_no,
        }, {"_id": 0})

    @staticmethod
    def province_codes():
        return sorted(mongo.db.parcels.distinct("province_code"))
