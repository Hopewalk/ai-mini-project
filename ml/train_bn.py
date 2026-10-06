"""Discrete Bayesian Network over location factors -> PriceLevel (pgmpy).

Expert DAG vs HillClimbSearch(BIC); parameters with BDeu prior; inference by VariableElimination.

Usage: python -m ml.train_bn
"""
import joblib
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import accuracy_score, f1_score

from ml import progress, settings
from ml.bn_model import STATES, discretize
from ml.progress import step
from ml.features import TARGET, grade_thresholds, load_cells, save_json, spatial_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERT_EDGES = [
    ("CityProximity", "RoadAccess"),
    ("CityProximity", "UrbanMap"),
    ("CityProximity", "Landuse"),
    ("RoadAccess", "PriceLevel"),
    ("UrbanMap", "PriceLevel"),
    ("Landuse", "PriceLevel"),
    ("Region", "PriceLevel"),
]
def fit_params(edges, data, state_names):
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(state_names)
    est = BayesianEstimator(model, data, state_names=state_names)
    model.add_cpds(*est.get_parameters(prior_type="BDeu", equivalent_sample_size=10))
    model.check_model()
    return model


def log_likelihood(model, data):
    total = 0.0
    for cpd in model.get_cpds():
        idx = tuple(data[v].map(cpd.name_to_no[v]).to_numpy() for v in cpd.variables)
        total += np.log(cpd.values[idx]).sum()
    return total / len(data)


def predict_price_level(model, data):
    infer = VariableElimination(model)
    evidence_cols = [c for c in data.columns if c != "PriceLevel"]
    preds = pd.Series(index=data.index, dtype=object)
    for key, grp in data.groupby(evidence_cols):
        q = infer.query(["PriceLevel"], evidence=dict(zip(evidence_cols, key)), show_progress=False)
        preds[grp.index] = q.state_names["PriceLevel"][int(np.argmax(q.values))]
    return preds


def query(infer, variable, evidence=None):
    q = infer.query([variable], evidence=evidence or {}, show_progress=False)
    return {s: round(float(p), 4) for s, p in zip(q.state_names[variable], q.values)}


def example_queries(model):
    infer = VariableElimination(model)
    return {
        "P(PriceLevel)": query(infer, "PriceLevel"),
        "P(PriceLevel | RoadAccess=near, UrbanMap=yes)":
            query(infer, "PriceLevel", {"RoadAccess": "near", "UrbanMap": "yes"}),
        "P(PriceLevel | RoadAccess=far)": query(infer, "PriceLevel", {"RoadAccess": "far"}),
        "P(PriceLevel | CityProximity=near)": query(infer, "PriceLevel", {"CityProximity": "near"}),
        "P(PriceLevel | CityProximity=far)": query(infer, "PriceLevel", {"CityProximity": "far"}),
        # diagnostic
        "P(RoadAccess | PriceLevel=High)": query(infer, "RoadAccess", {"PriceLevel": "High"}),
        "P(UrbanMap | PriceLevel=High)": query(infer, "UrbanMap", {"PriceLevel": "High"}),
        # explaining away: knowing Landuse=urban changes how much UrbanMap is needed to explain High
        "P(UrbanMap | PriceLevel=High, Landuse=urban)":
            query(infer, "UrbanMap", {"PriceLevel": "High", "Landuse": "urban"}),
    }


def plot_dag(model, path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    g = nx.DiGraph(model.edges())
    pos = nx.spring_layout(g, seed=3, k=1.5)
    nx.draw_networkx(g, pos, ax=ax, node_color="#cfe3ff", node_size=2600, font_size=9,
                     arrowsize=18, edgecolors="#3b6fb6")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    out_dir = settings.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    train, test = spatial_split(cells)
    thresholds = grade_thresholds(train[TARGET])
    d_train, d_test = discretize(train, thresholds), discretize(test, thresholds)
    state_names = {**STATES, "Region": sorted(cells["region"].unique())}

    step(f"discretized {len(d_train):,} train / {len(d_test):,} test cells into {len(d_train.columns)} nodes")
    step("fitting expert DAG parameters (BDeu prior)")
    expert = fit_params(EXPERT_EDGES, d_train, state_names)
    step("structure learning: HillClimbSearch + BIC")
    dag = HillClimbSearch(d_train).estimate(scoring_method="bic-d", max_indegree=4,
                                            show_progress=progress.ENABLED)
    learned = fit_params(list(dag.edges()), d_train, state_names)
    step(f"learned DAG: {', '.join(f'{a}→{b}' for a, b in dag.edges())}")

    results = {}
    for name, model in [("expert", expert), ("hill_climb_bic", learned)]:
        # inference works whether PriceLevel is a child or a parent in the learned DAG
        pred = predict_price_level(model, d_test)
        results[name] = {
            "edges": [list(e) for e in model.edges()],
            "test_log_likelihood_per_row": log_likelihood(model, d_test),
            "price_level_accuracy": accuracy_score(d_test["PriceLevel"], pred),
            "price_level_f1_macro": f1_score(d_test["PriceLevel"], pred, average="macro"),
        }
        r = results[name]
        step(f"{name:15s} test log-lik/row {r['test_log_likelihood_per_row']:.3f} | PriceLevel acc "
             f"{r['price_level_accuracy']:.3f}  F1 {r['price_level_f1_macro']:.3f}")

    plot_dag(expert, out_dir / "bn_dag_expert.png", "Expert DAG")
    plot_dag(learned, out_dir / "bn_dag_hillclimb.png", "HillClimbSearch (BIC) DAG")
    queries = example_queries(expert)
    for k, v in queries.items():
        step(f"{k:48s} " + "  ".join(f"{s} {p:.1%}" for s, p in v.items()))

    joblib.dump({"model": expert, "thresholds": thresholds, "state_names": state_names},
                out_dir / "bn.joblib")
    metrics = {"models": results, "queries": queries, "thresholds": thresholds,
               "train_distribution": {c: d_train[c].value_counts().to_dict() for c in d_train}}
    save_json(metrics, out_dir / "metrics_bn.json")
    step(f"saved bn.joblib, bn_dag_expert.png, bn_dag_hillclimb.png -> {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
