from flask_wtf import FlaskForm
from wtforms import DecimalField, FloatField, IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional, ValidationError

SQ_WAH_PER_RAI = 400
SQ_WAH_PER_NGAN = 100


class LocationForm(FlaskForm):
    lat = FloatField("ละติจูด", validators=[DataRequired("กรุณาเลือกตำแหน่งบนแผนที่"),
                                             NumberRange(5.5, 20.5, "ละติจูดต้องอยู่ในประเทศไทย (5.5–20.5)")])
    lon = FloatField("ลองจิจูด", validators=[DataRequired("กรุณาเลือกตำแหน่งบนแผนที่"),
                                              NumberRange(97.3, 105.7, "ลองจิจูดต้องอยู่ในประเทศไทย (97.3–105.7)")])
    area_rai = IntegerField("ไร่", validators=[Optional(), NumberRange(0, 100_000)])
    area_ngan = IntegerField("งาน", validators=[Optional(), NumberRange(0, 3, "งานต้องอยู่ระหว่าง 0–3")])
    area_wah = DecimalField("ตร.ว.", places=1, validators=[Optional(), NumberRange(0, 99.9, "ตร.ว. ต้องอยู่ระหว่าง 0–99.9")])
    asking_total = FloatField("ราคาเสนอขายรวม (บาท)", validators=[Optional(), NumberRange(min=1)])
    submit = SubmitField("วิเคราะห์ทำเล")

    @property
    def total_wah(self):
        return ((self.area_rai.data or 0) * SQ_WAH_PER_RAI + (self.area_ngan.data or 0) * SQ_WAH_PER_NGAN
                + float(self.area_wah.data or 0))

    def validate_asking_total(self, field):
        if field.data and self.total_wah <= 0:
            raise ValidationError("ระบุเนื้อที่ด้วย เพื่อคำนวณราคาต่อตารางวา")
