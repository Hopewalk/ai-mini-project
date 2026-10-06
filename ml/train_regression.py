"""Regression: predict appraisal price (THB / sq.wah) per cell.

Compares a mean baseline, Linear/Polynomial Ridge, a lat/lon KNN and a Random Forest with spatial
GroupKFold CV, then saves the best model.

Usage: python -m ml.train_regression
"""
import joblib
import matplotlib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ml import settings
from ml.features import FEATURES, GROUP, TARGET, group_cv, load_cells, make_preprocessor, save_json, spatial_split
from ml.progress import fit_progress, fmt_params, step

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def log_rmse(y_true, y_pred):
    return root_mean_squared_error(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))


LOG_RMSE_SCORER = make_scorer(log_rmse, greater_is_better=False)


def _ttr(pipe):
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def candidates():
    # baseline: always predict the mean (of log price) — every model must beat this
    dummy = _ttr(Pipeline([("pre", make_preprocessor()), ("model", DummyRegressor(strategy="mean"))]))
    ridge = _ttr(Pipeline([
        ("pre", make_preprocessor()),
        ("poly", PolynomialFeatures(include_bias=False)),
        ("model", Ridge()),
    ]))
    forest = _ttr(Pipeline([
        ("pre", make_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=200, random_state=settings.RANDOM_STATE)),
    ]))
    knn = _ttr(Pipeline([
        ("pre", ColumnTransformer([("xy", "passthrough", ["lat", "lon"])])),
        ("model", KNeighborsRegressor()),
    ]))
    return {
        "dummy_mean": (dummy, {}),
        "poly_ridge": (ridge, {
            "regressor__poly__degree": [1, 2],
            "regressor__model__alpha": np.logspace(-2, 3, 6),
        }),
        # bounded trees keep the saved forest small enough to serve
        "random_forest": (forest, {
            "regressor__model__max_depth": [15, 25],
            "regressor__model__min_samples_leaf": [5, 20],
            "regressor__model__max_features": [0.5, 1.0],
        }),
        "knn_latlon": (knn, {
            "regressor__model__n_neighbors": [3, 5, 10, 20],
            "regressor__model__weights": ["uniform", "distance"],
        }),
    }


def evaluate(y_true, y_pred):
    return {
        "rmse_log": log_rmse(y_true, y_pred),
        "r2_log": r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None))),
        "mae_thb": mean_absolute_error(y_true, y_pred),
        "median_ape_pct": float(np.median(np.abs(y_pred - y_true) / y_true) * 100),
    }


def plot_diagnostics(y_true, y_pred, name, out_dir):
    lt, lp = np.log10(y_true), np.log10(np.clip(y_pred, 1, None))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(lt, lp, s=6, alpha=0.4)
    lims = [min(lt.min(), lp.min()), max(lt.max(), lp.max())]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set(xlabel="actual log10(THB/sq.wah)", ylabel="predicted", title=f"{name}: predicted vs actual")
    axes[1].scatter(lp, lt - lp, s=6, alpha=0.4)
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set(xlabel="predicted log10", ylabel="residual (log10)", title="Residuals")
    fig.tight_layout()
    fig.savefig(out_dir / "regression_residuals.png", dpi=120)
    plt.close(fig)


def main():
    out_dir = settings.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = spatial_split(load_cells())
    X_tr, y_tr, g_tr = train[FEATURES], train[TARGET], train[GROUP]
    X_te, y_te = test[FEATURES], test[TARGET]
    step(f"train {len(train):,} cells / test {len(test):,} cells (spatial split by 1:50,000 sheet)")

    results, fitted = {}, {}
    for name, (est, grid) in candidates().items():
        search = GridSearchCV(est, grid, scoring=LOG_RMSE_SCORER, cv=group_cv(), n_jobs=-1)
        with fit_progress(len(ParameterGrid(grid)) * settings.CV_FOLDS, name):
            search.fit(X_tr, y_tr, groups=g_tr)
        fitted[name] = search.best_estimator_
        test_metrics = evaluate(y_te.to_numpy(), search.predict(X_te))
        results[name] = {
            "cv_rmse_log": -search.best_score_,
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
            "test": test_metrics,
        }
        step(f"{name:11s} CV RMSE(log) {-search.best_score_:.3f} | test RMSE(log) {test_metrics['rmse_log']:.3f}"
             f"  R² {test_metrics['r2_log']:.3f}  MAE {test_metrics['mae_thb']:,.0f} THB"
             f"  | {fmt_params(results[name]['best_params'])}")

    best_name = min(results, key=lambda n: results[n]["cv_rmse_log"])
    best = fitted[best_name]
    step(f"best (lowest CV RMSE): {best_name}")

    # prediction interval from out-of-fold residuals (log space) on the training set
    with fit_progress(settings.CV_FOLDS, "out-of-fold residuals"):
        oof = cross_val_predict(best, X_tr, y_tr, groups=g_tr, cv=group_cv(), n_jobs=-1)
    resid = np.log1p(y_tr) - np.log1p(np.clip(oof, 0, None))
    interval = {"q05": float(np.quantile(resid, 0.05)), "q95": float(np.quantile(resid, 0.95))}

    with fit_progress(len(FEATURES), "permutation importance"):
        # few workers: each one receives a pickled copy of the model
        imp = permutation_importance(best, X_te, y_te, scoring=LOG_RMSE_SCORER, n_repeats=5,
                                     random_state=settings.RANDOM_STATE, n_jobs=2)
    importance = dict(sorted(zip(FEATURES, imp.importances_mean.round(4)), key=lambda kv: -kv[1]))

    plot_diagnostics(y_te.to_numpy(), best.predict(X_te), best_name, out_dir)
    joblib.dump({"model": best, "name": best_name, "log_residual_interval": interval},
                out_dir / "regressor.joblib", compress=3)
    metrics = {"best_model": best_name, "models": results, "log_residual_interval": interval,
               "permutation_importance": importance, "n_train": len(train), "n_test": len(test)}
    save_json(metrics, out_dir / "metrics_regression.json")
    step(f"90% interval ×{np.exp(interval['q05']):.2f} – ×{np.exp(interval['q95']):.2f} of prediction"
         f" | top features: {', '.join(list(importance)[:5])}")
    step(f"saved regressor.joblib, regression_residuals.png -> {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
