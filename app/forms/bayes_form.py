from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField

from app.labels import BN_VARIABLES, bn_state

ANY = ""


class BayesQueryForm(FlaskForm):
    """GET form: target variable + optional evidence for each BN node (choices set from the model)."""

    class Meta:
        csrf = False

    target = SelectField("อยากรู้ความน่าจะเป็นของ", choices=list(BN_VARIABLES.items()), default="PriceLevel")
    # default=ANY: fields missing from the query string must still be a valid choice
    Region = SelectField(BN_VARIABLES["Region"], default=ANY)
    CityProximity = SelectField(BN_VARIABLES["CityProximity"], default=ANY)
    RoadAccess = SelectField(BN_VARIABLES["RoadAccess"], default=ANY)
    UrbanMap = SelectField(BN_VARIABLES["UrbanMap"], default=ANY)
    Landuse = SelectField(BN_VARIABLES["Landuse"], default=ANY)
    PriceLevel = SelectField(BN_VARIABLES["PriceLevel"], default=ANY)
    submit = SubmitField("คำนวณ")

    def set_states(self, state_names):
        for var in BN_VARIABLES:
            getattr(self, var).choices = [(ANY, "(ไม่ระบุ)")] + [(s, bn_state(var, s)) for s in state_names[var]]

    def evidence_fields(self):
        return [getattr(self, var) for var in BN_VARIABLES]

    def evidence(self):
        return {var: getattr(self, var).data for var in BN_VARIABLES
                if getattr(self, var).data not in (None, ANY) and var != self.target.data}
