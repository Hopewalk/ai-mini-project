"""Classification: Grade A/B/C (price tertiles from the training set).

Random Forest vs Logistic Regression baseline, spatial CV, isotonic calibration.

Usage: python -m ml.train_classifier
"""
import joblib
import matplotlib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, f1_score)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from ml import settings
from ml.features import (FEATURES, GRADES, GROUP, TARGET, grade_thresholds, group_cv, load_cells,
                         make_preprocessor, save_json, spatial_split, to_grade)
from ml.progress import fit_progress, fmt_params, step

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def candidates():
    rf = Pipeline([("pre", make_preprocessor()),
                   ("model", RandomForestClassifier(n_estimators=200, random_state=settings.RANDOM_STATE,
                                                    n_jobs=-1))])
    logreg = Pipeline([("pre", make_preprocessor()), ("model", LogisticRegression(max_iter=3000))])
    return {
        # bounded trees: fully grown forests on ~60k cells are several GB and too big to serve
        "random_forest": (rf, {
            "model__max_depth": [20, 30],
            "model__min_samples_leaf": [3, 10],
            "model__class_weight": [None, "balanced"],
        }),
        "logistic_regression": (logreg, {"model__C": np.logspace(-2, 2, 5)}),
    }


def evaluate(y_true, proba, classes):
    pred = classes[proba.argmax(axis=1)]
    y_bin = label_binarize(y_true, classes=classes)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro"),
        "pr_auc_macro": average_precision_score(y_bin, proba, average="macro"),
    }


def main():
    out_dir = settings.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = spatial_split(load_cells())
    thresholds = grade_thresholds(train[TARGET])
    X_tr, y_tr, g_tr = train[FEATURES], to_grade(train[TARGET], thresholds), train[GROUP]
    X_te, y_te = test[FEATURES], to_grade(test[TARGET], thresholds)
    step(f"grade thresholds (THB/sq.wah): C < {thresholds['q33']:,.0f} ≤ B < {thresholds['q67']:,.0f} ≤ A")
    step(f"train grades {y_tr.value_counts().to_dict()} | test {y_te.value_counts().to_dict()}")

    results, fitted = {}, {}
    for name, (est, grid) in candidates().items():
        search = GridSearchCV(est, grid, scoring="f1_macro", cv=group_cv(), n_jobs=-1)
        with fit_progress(len(ParameterGrid(grid)) * settings.CV_FOLDS, name):
            search.fit(X_tr, y_tr, groups=g_tr)
        fitted[name] = search.best_estimator_
        model = search.best_estimator_
        test_metrics = evaluate(y_te, model.predict_proba(X_te), model.classes_)
        results[name] = {
            "cv_f1_macro": search.best_score_,
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
            "test": test_metrics,
        }
        step(f"{name:20s} CV F1 {search.best_score_:.3f} | test acc {test_metrics['accuracy']:.3f}"
             f"  F1 {test_metrics['f1_macro']:.3f}  PR-AUC {test_metrics['pr_auc_macro']:.3f}"
             f"  | {fmt_params(results[name]['best_params'])}")

    best_name = max(results, key=lambda n: results[n]["cv_f1_macro"])
    step(f"best (highest CV F1): {best_name}")
    # calibrate probabilities with the same spatial folds; ensemble=False keeps a single forest
    # (refit on all training data) calibrated on out-of-fold predictions
    folds = list(group_cv().split(X_tr, y_tr, groups=g_tr))
    calibrated = CalibratedClassifierCV(fitted[best_name], method="isotonic", cv=folds, ensemble=False)
    with fit_progress(settings.CV_FOLDS, "isotonic calibration"):
        calibrated.fit(X_tr, y_tr)
    proba = calibrated.predict_proba(X_te)
    results[best_name]["test_calibrated"] = evaluate(y_te, proba, calibrated.classes_)
    pred = calibrated.classes_[proba.argmax(axis=1)]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, labels=GRADES[::-1], ax=ax, colorbar=False)
    ax.set_title(f"Grade — {best_name} (calibrated)")
    fig.tight_layout()
    fig.savefig(out_dir / "classifier_confusion.png", dpi=120)
    plt.close(fig)

    # few workers: each one receives a pickled copy of the model
    with fit_progress(len(FEATURES), "permutation importance"):
        imp = permutation_importance(calibrated, X_te, y_te, scoring="f1_macro", n_repeats=5,
                                     random_state=settings.RANDOM_STATE, n_jobs=2)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4)), key=lambda kv: -kv[1]))

    joblib.dump({"model": calibrated, "name": best_name, "thresholds": thresholds},
                out_dir / "classifier.joblib", compress=3)
    metrics = {"best_model": best_name, "thresholds": thresholds, "models": results,
               "report": classification_report(y_te, pred, output_dict=True),
               "permutation_importance": importance}
    save_json(metrics, out_dir / "metrics_classifier.json")
    cal = results[best_name]["test_calibrated"]
    step(f"calibrated {best_name}: test acc {cal['accuracy']:.3f}  F1 {cal['f1_macro']:.3f}"
         f"  PR-AUC {cal['pr_auc_macro']:.3f} | top features: {', '.join(list(importance)[:5])}")
    step(f"saved classifier.joblib, classifier_confusion.png -> {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
