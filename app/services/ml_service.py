"""Loads the trained artifacts once and runs all four analyses for a cell."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from ml.bn_model import RAW_COLUMNS, STATES, discretize
from ml.cluster_model import CLUSTER_FEATURES
from ml.features import FEATURES

WHAT_IF_VARIABLES = ["RoadAccess", "CityProximity", "UrbanMap", "Landuse"]
ARTIFACT_FILES = ["regressor.joblib", "classifier.joblib", "cluster.joblib", "bn.joblib"]


def _row(cell, columns):
    return pd.DataFrame([{c: cell.get(c) for c in columns}])


class MLService:
    def __init__(self):
        self.ready = False
        self.error = None

    def init_app(self, app):
        artifacts = Path(app.config["ARTIFACTS_DIR"])
        missing = [f for f in ARTIFACT_FILES if not (artifacts / f).exists()]
        if missing:
            self.ready = False
            self.error = f"ไม่พบไฟล์โมเดล {', '.join(missing)} ใน {artifacts} — รัน `python -m ml.train_all` ก่อน"
            app.logger.warning(self.error)
            return

        reg = joblib.load(artifacts / "regressor.joblib")
        self.regressor, self.regressor_name = reg["model"], reg["name"]
        self.log_interval = reg["log_residual_interval"]

        clf = joblib.load(artifacts / "classifier.joblib")
        self.classifier, self.classifier_name = clf["model"], clf["name"]
        self.grade_thresholds = clf["thresholds"]

        self.cluster = joblib.load(artifacts / "cluster.joblib")

        bn = joblib.load(artifacts / "bn.joblib")
        self.bn, self.bn_thresholds, self.state_names = bn["model"], bn["thresholds"], bn["state_names"]
        self.bn_infer = VariableElimination(self.bn)

        self.risk_medium = app.config["RISK_RATIO_MEDIUM"]
        self.risk_high = app.config["RISK_RATIO_HIGH"]
        self.ready = True
        self.error = None

    # ---- individual models -------------------------------------------------------------
    def predict_price(self, cell):
        pred = float(self.regressor.predict(_row(cell, FEATURES))[0])
        log_pred = np.log1p(pred)
        return {
            "model": self.regressor_name,
            "predicted": pred,
            "low": float(np.expm1(log_pred + self.log_interval["q05"])),
            "high": float(np.expm1(log_pred + self.log_interval["q95"])),
        }

    def predict_grade(self, cell):
        proba = self.classifier.predict_proba(_row(cell, FEATURES))[0]
        classes = [str(c) for c in self.classifier.classes_]
        return {
            "model": self.classifier_name,
            "grade": classes[int(np.argmax(proba))],
            "proba": {c: float(p) for c, p in zip(classes, proba)},
        }

    def predict_cluster(self, cell):
        label = int(self.cluster.predict(_row(cell, CLUSTER_FEATURES))[0])
        return {"id": label, "name": self.cluster.name_of(label)}

    # ---- Bayesian network ----------------------------------------------------------------
    def evidence_for(self, cell):
        ev = discretize(_row(cell, RAW_COLUMNS), self.bn_thresholds).iloc[0].to_dict()
        # drop states the network never saw (e.g. a region outside the training provinces)
        return {k: v for k, v in ev.items() if v in self.state_names.get(k, [])}

    def query(self, variable, evidence):
        evidence = {k: v for k, v in evidence.items() if k != variable}
        q = self.bn_infer.query([variable], evidence=evidence, show_progress=False)
        return {s: float(p) for s, p in zip(q.state_names[variable], q.values)}

    def explain(self, cell):
        evidence = self.evidence_for(cell)
        posterior = self.query("PriceLevel", evidence)
        base_high = posterior["High"]

        what_if = []
        for var in WHAT_IF_VARIABLES:
            if var not in evidence:
                continue
            for state in STATES[var]:
                if state != evidence[var]:
                    p_high = self.query("PriceLevel", {**evidence, var: state})["High"]
                    what_if.append({"variable": var, "current": evidence[var], "state": state,
                                    "p_high": p_high, "delta": p_high - base_high})

        # driver = the factor whose knowledge moves P(High) the most versus not knowing it
        impacts = []
        for var in evidence:
            without = {k: v for k, v in evidence.items() if k != var}
            impacts.append({"variable": var, "state": evidence[var],
                            "delta": base_high - self.query("PriceLevel", without)["High"]})
        impacts.sort(key=lambda d: -abs(d["delta"]))
        # what-if variables that cannot move P(High) given the rest of the evidence (d-separated)
        no_effect = [var for var in WHAT_IF_VARIABLES
                     if any(w["variable"] == var for w in what_if)
                     and all(abs(w["delta"]) < 5e-4 for w in what_if if w["variable"] == var)]
        return {"evidence": evidence, "posterior": posterior, "what_if": what_if, "impacts": impacts,
                "no_effect": no_effect}

    # ---- full analysis -------------------------------------------------------------------
    def analyze(self, cell):
        regression = self.predict_price(cell)
        actual = cell.get("price_median")
        if actual:
            regression.update({
                "actual_median": float(actual),
                "actual_p25": float(cell["price_p25"]),
                "actual_p75": float(cell["price_p75"]),
                "diff_pct": (actual - regression["predicted"]) / regression["predicted"],
            })
        classification = self.predict_grade(cell)
        classification["actual_grade"] = cell.get("grade")
        return {
            "regression": regression,
            "classification": classification,
            "cluster": self.predict_cluster(cell),
            "bayes": self.explain(cell),
        }

    def asking_check(self, predicted, actual, area_wah, asking_total):
        per_wah = asking_total / area_wah
        ratio = per_wah / predicted
        risk = "High" if ratio >= self.risk_high else "Medium" if ratio >= self.risk_medium else "Low"
        return {
            "asking_total": asking_total,
            "asking_per_wah": per_wah,
            "ratio_to_predicted": ratio,
            "ratio_to_actual": per_wah / actual if actual else None,
            "appraisal_total_predicted": predicted * area_wah,
            "risk": risk,
            "thresholds": {"medium": self.risk_medium, "high": self.risk_high},
        }


ml_service = MLService()
