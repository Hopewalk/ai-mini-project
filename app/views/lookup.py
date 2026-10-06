from flask import Blueprint, flash, render_template

from app.forms import ParcelLookupForm
from app.models import Cell, Parcel
from app.services.ml_service import ml_service
from ml.provinces import PROVINCES

bp = Blueprint("lookup", __name__)


@bp.route("/lookup", methods=["GET", "POST"])
def parcel():
    form = ParcelLookupForm()
    form.province.choices = [(c, PROVINCES.get(c, str(c))) for c in Parcel.province_codes()]
    result = None

    if form.validate_on_submit():
        found = Parcel.find(form.province.data, form.utmmap1.data, form.utmmap2.data, form.utmmap3.data,
                            form.utmmap4.data, form.land_no.data.strip())
        if found is None:
            flash("ไม่พบแปลงที่ดินตามเลขระวาง/เลขที่ดินที่ระบุ", "warning")
        else:
            cell = Cell.get(found["cell_id"])
            prediction = ml_service.predict_price(cell) if cell and ml_service.ready else None
            result = {"parcel": found, "cell": cell, "prediction": prediction}

    return render_template("lookup.html", form=form, result=result)
