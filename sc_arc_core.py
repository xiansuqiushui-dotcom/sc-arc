"""SC-ARC from processed diagnostic features. No query outcomes enter this module.

The historical training labels and exactly five requested capacity measurements
are permitted inputs. Dataset seeds reproduce the archived historical ensembles.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

METHODS = ("GLOBAL", "AFFINE", "TARGET", "STACK")
ROW_ID = ("source", "cell_id", "cycle_number", "raw_cycle_index", "file_name")


def canonicalize(frame):
    if frame.duplicated(list(ROW_ID)).any():
        raise ValueError("Ambiguous record identity")
    return frame.sort_values(list(ROW_ID), kind="mergesort").reset_index(drop=True)


def snapshot(frame, phase):
    """Retrospective record-index snapshot; not an online ageing-stage estimator."""
    rows = [c.sort_values(["cycle_number", "raw_cycle_index", "file_name"],
                         kind="mergesort").iloc[int(round(phase * (len(c)-1)))]
            for _, c in frame.groupby("cell_id", sort=True)]
    return pd.DataFrame(rows).sort_values("cell_id", kind="mergesort").reset_index(drop=True)


def stable_seed(*parts):
    payload = "|".join(["20260814", *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31-1)


def forest(seed, leaf=3):
    return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=leaf,
                              max_features=.8, n_jobs=1, random_state=seed)


def simplex(x, y):
    weights, _ = nnls(x, y)
    return weights / weights.sum() if weights.sum() > 1e-12 else np.full(x.shape[1], 1/x.shape[1])


def kcenter(cells, representation, count=5):
    if len(cells) <= count:
        raise ValueError("A task must have more candidates than the measurement budget")
    distance = cdist(representation, representation)
    chosen = [min(range(len(cells)), key=lambda i: (distance[i].mean(), str(cells[i])))]
    while len(chosen) < count:
        nearest = distance[:, chosen].min(axis=1)
        chosen.append(min((i for i in range(len(cells)) if i not in chosen),
                          key=lambda i: (-nearest[i], str(cells[i]))))
    return np.asarray(chosen, dtype=int)


def knn_weights(distance, count=3):
    order = np.argsort(distance, axis=1, kind="mergesort")[:, :count]
    weights = np.zeros_like(distance)
    values = 1 / (np.take_along_axis(distance, order, axis=1) + 1e-6)
    values /= values.sum(axis=1, keepdims=True)
    np.put_along_axis(weights, order, values, axis=1)
    return weights


def local_instability(residuals, label_weights, query_weights, floor=.02):
    errors = np.abs(residuals - label_weights @ residuals)
    return floor + query_weights @ errors


@dataclass
class PreparedTask:
    build_dataset: str
    phase: float
    cells: np.ndarray
    x: np.ndarray
    raw_shap: np.ndarray
    z: np.ndarray
    global_prediction: np.ndarray
    expert_prediction: np.ndarray
    selection_order: np.ndarray

    @property
    def requested_cells(self):
        return [str(self.cells[i]) for i in self.selection_order]


class HistoricalModels:
    def __init__(self, history, feature_columns, build_dataset):
        import shap
        self.features = list(feature_columns)
        if "soh" in self.features:
            raise ValueError("SOH cannot be a diagnostic feature")
        self.build_dataset = build_dataset
        history = canonicalize(history)
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        xh = self.imputer.fit_transform(history[self.features])
        yh = history.soh.to_numpy(float)
        self.global_model = forest(stable_seed("global", build_dataset)).fit(xh, yh)
        self.experts = []
        for source in sorted(history.source.unique()):
            mask = history.source.to_numpy() == source
            self.experts.append(forest(stable_seed("expert", build_dataset, source)).fit(xh[mask], yh[mask]))
        self.explainer = shap.TreeExplainer(self.global_model)

    def prepare(self, features, phase):
        if "soh" in features.columns or "truth" in features.columns:
            raise ValueError("Remove query outcomes before preparing a task")
        features = features.sort_values("cell_id", kind="mergesort").reset_index(drop=True)
        if features.cell_id.duplicated().any():
            raise ValueError("Each task needs one record per cell")
        cells = features.cell_id.astype(str).to_numpy()
        x = self.imputer.transform(features[self.features])
        raw = np.asarray(self.explainer.shap_values(x), dtype=float)
        z = raw / (np.abs(raw).sum(axis=1, keepdims=True) + 1e-12)
        return PreparedTask(self.build_dataset, phase, cells, x, raw, z,
                            self.global_model.predict(x),
                            np.column_stack([e.predict(x) for e in self.experts]),
                            kcenter(cells, raw))


def fit_measured(task, measured_labels):
    """Accept exactly the five requested labels, keyed by cell ID.

    Type selection uses all five LOO errors, as in the final manuscript.
    These are not selection-aware nested residuals.
    """
    if set(measured_labels) != set(task.requested_cells):
        raise ValueError("Supply exactly the five requested cell labels, no others")
    ix = np.sort(task.selection_order)
    yy = np.asarray([measured_labels[str(task.cells[i])] for i in ix], dtype=float)
    if not np.isfinite(yy).all():
        raise ValueError("Measured labels must be finite")
    x, g, ex = task.x, task.global_prediction, task.expert_prediction
    oof = np.zeros((5, 4))
    oof[:, 0] = g[ix]
    for l, pos in enumerate(ix):
        keep = np.arange(5) != l
        oof[l, 1] = LinearRegression(positive=True).fit(g[ix[keep], None], yy[keep]).predict(g[pos:pos+1, None])[0]
        seed = stable_seed("target-loo", task.build_dataset, f"{task.phase:.2f}", task.cells[pos])
        oof[l, 2] = forest(seed, 1).fit(x[ix[keep]], yy[keep]).predict(x[pos:pos+1])[0]
        oof[l, 3] = ex[pos] @ simplex(ex[ix[keep]], yy[keep])
    final = np.zeros((len(g), 4))
    final[:, 0] = g
    final[:, 1] = LinearRegression(positive=True).fit(g[ix, None], yy).predict(g[:, None])
    seed = stable_seed("target-final", task.build_dataset, f"{task.phase:.2f}")
    final[:, 2] = forest(seed, 1).fit(x[ix], yy).predict(x)
    final[:, 3] = ex @ simplex(ex[ix], yy)
    worst = np.abs(yy[:, None]-oof).max(axis=0)
    model_weights = (worst <= worst.min()+1e-12).astype(float)
    model_weights /= model_weights.sum()
    return dict(selected=ix, prediction=final @ model_weights,
                residuals=yy-oof @ model_weights, model_weights=model_weights,
                worst_loo_errors=worst, oof=oof, final=final)


def score_queries(task, fit, thresholds=(.7, .8, .9), floor=.02):
    ix = fit["selected"]
    query = np.setdiff1d(np.arange(len(task.cells)), ix)
    dl = cdist(task.z[ix], task.z[ix])
    np.fill_diagonal(dl, np.inf)
    wl = knn_weights(dl)
    wq = knn_weights(cdist(task.z[query], task.z[ix]))
    r = fit["residuals"]
    a = local_instability(r, wl, wq, floor)
    prediction = fit["prediction"][query]
    frames = []
    for tau in thresholds:
        sign = np.sign(prediction-tau)
        distance = np.abs(prediction-tau)
        margins = distance[:, None] + sign[:, None]*r
        support = (wq*np.maximum(margins, 0)).sum(axis=1)
        opposition = (wq*np.maximum(-margins, 0)).sum(axis=1)
        rho = support/(support+opposition+floor)
        frames.append(pd.DataFrame(dict(cell_id=task.cells[query], threshold=tau,
            baseline_prediction=prediction, local_scale=a, support=support,
            opposition=opposition, direction_support=rho,
            action_score=distance/a*rho, predicted_sign=sign, eligible=sign != 0)))
    return pd.concat(frames, ignore_index=True)


def decide(scores, cutoff):
    """Apply an already-determined cutoff; query outcomes are neither needed nor accepted."""
    if not np.isscalar(cutoff) or np.isnan(cutoff) or cutoff < 0:
        raise ValueError("Cutoff must be non-negative, including positive infinity")
    output = scores.copy()
    output["cutoff"] = cutoff
    output["issued"] = output.eligible & (output.action_score > cutoff)
    return output
