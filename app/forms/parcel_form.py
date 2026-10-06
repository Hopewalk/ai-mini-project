from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange, Regexp

QUADRANTS = [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]


class ParcelLookupForm(FlaskForm):
    province = SelectField("จังหวัด", coerce=int, validators=[DataRequired()])
    utmmap1 = IntegerField("เลขระวางชุดแรก", validators=[InputRequired("กรอกเลขระวาง"),
                                                         NumberRange(1000, 9999, "เลข 4 หลัก เช่น 5137")])
    utmmap2 = SelectField("เลขโรมัน", coerce=int, choices=QUADRANTS)
    utmmap3 = IntegerField("เลขระวางชุดที่สอง", validators=[InputRequired("กรอกเลขระวาง"),
                                                       NumberRange(1, 9999, "เลข 4 หลัก เช่น 6888")])
    utmmap4 = StringField("เลขหลังขีด", default="00",
                          validators=[Regexp(r"^\d{2}$", message="เลข 2 หลัก (ถ้าไม่มีเลขหลังขีด ใส่ 00)")])
    land_no = StringField("เลขที่ดิน", validators=[DataRequired("กรอกเลขที่ดิน")])
    submit = SubmitField("ค้นหาราคา")
