from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange, Regexp

QUADRANTS = [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]


class ParcelLookupForm(FlaskForm):
    province = SelectField("จังหวัด", coerce=int, validators=[DataRequired()])
    utmmap1 = IntegerField("ระวาง 1:50,000", validators=[InputRequired("กรอกเลขระวาง"),
                                                         NumberRange(1000, 9999, "เลข 4 หลัก เช่น 5137")])
    utmmap2 = SelectField("แผ่น", coerce=int, choices=QUADRANTS)
    utmmap3 = IntegerField("ระวาง 1:4000", validators=[InputRequired("กรอกเลขระวาง"),
                                                       NumberRange(1, 9999, "เลข 4 หลัก เช่น 6888")])
    utmmap4 = StringField("แผ่นย่อย", default="00",
                          validators=[Regexp(r"^\d{2}$", message="เลข 2 หลัก (00 = ระวาง 1:4000)")])
    land_no = StringField("เลขที่ดิน", validators=[DataRequired("กรอกเลขที่ดิน")])
    submit = SubmitField("ค้นหา")
