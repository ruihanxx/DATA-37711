#!/usr/bin/env python3
"""Problem 5 ablation: original features, without centering or scaling.

Preserves the standardized results and uses the same outer holdout, clipped
losses, epoch candidates, SGD and training-only CV selection rules. Learning
rates are retuned for the larger raw feature magnitudes. Also isolates the
effect of probability clipping using the fixed standardized configurations.
"""
import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import time
import warnings

import run_experiment as experiment
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = experiment.ROOT / "no_scaling"
RAW_LRS = [1e-5, 1e-4, 1e-3]


def tune_raw(name, kind, X, y, loss_fn):
    cv = (StratifiedKFold(3, shuffle=True, random_state=experiment.SEED)
          if kind == "classification" else KFold(3, shuffle=True, random_state=experiment.SEED))
    # These are the ORIGINAL feature arrays; no transformation is fitted here.
    folds = [(X[t], y[t], X[v], y[v]) for t, v in cv.split(X, y)]
    batches = [64, 128] if kind == "classification" else [16, 32]
    candidates, fold_rows = [], []
    configurations = list(product(RAW_LRS, batches, experiment.MOMENTA))
    for number, (lr, batch, momentum) in enumerate(configurations, 1):
        config = {"epochs": max(experiment.SEARCH_EPOCHS), "batch_size": batch,
            "initial_lr": lr, "momentum": momentum, "decay_epochs": experiment.DECAY_EPOCHS}
        scores = []
        for fold, (tx, ty, vx, vy) in enumerate(folds, 1):
            _, _, _, checkpoints = experiment.fit_sgd(name, tx, ty, loss_fn, config,
                initial_bias=float(ty.mean()) if kind == "regression" else 0.,
                validation=(vx, vy), kind=kind, seed=experiment.SEED + fold, verbose=False)
            scores.extend(checkpoints)
            fold_rows.extend({"model": name, "fold": fold, "initial_lr": lr,
                "batch_size": batch, "momentum": momentum, **row} for row in checkpoints)
        for epoch in experiment.SEARCH_EPOCHS:
            rows = [r for r in scores if r["epoch"] == epoch]
            weights = [r["n_validation"] for r in rows]
            candidates.append({"model": name, **config, "epochs": epoch,
                "mean_validation_loss": float(np.average([r["validation_loss"] for r in rows], weights=weights)),
                "mean_validation_accuracy": float(np.average([r["validation_accuracy"] for r in rows], weights=weights)) if kind == "classification" else "",
                "mean_validation_mae": float(np.average([r["validation_mae"] for r in rows], weights=weights)) if kind == "regression" else ""})
        print(f"  Raw CV {name}: {number}/{len(configurations)} configurations", flush=True)
    def key(row):
        secondary = (row["mean_validation_loss"], row["epochs"], -row["batch_size"])
        return ((-row["mean_validation_accuracy"],) if kind == "classification"
                else (row["mean_validation_mae"],)) + secondary
    best = min(candidates, key=key)
    config = {k: best[k] for k in ["epochs", "batch_size", "initial_lr", "momentum", "decay_epochs"]}
    print(f"Selected raw {name}: {config}", flush=True)
    return config, best, candidates, fold_rows


def compare_clipping(X, y, train, test, standardized):
    """Diagnostic only: keep data and SGD settings fixed, vary BCE evaluation."""
    scaler = StandardScaler().fit(X[train])
    tx, vx = scaler.transform(X[train]), scaler.transform(X[test])
    output = {}
    for name, link, stable in [
        ("PyTorch logistic (SGD)", "logistic",
            lambda z, target: -(target * F.logsigmoid(z) + (1-target) * F.logsigmoid(-z)).mean()),
        ("PyTorch probit (SGD)", "probit",
            lambda z, target: -(target * torch.special.log_ndtr(z) + (1-target) * torch.special.log_ndtr(-z)).mean())]:
        params = standardized["models"][name]
        model, _, _, _ = experiment.fit_sgd(name, tx, y[train], stable, params["training_config"], verbose=False)
        stable_score = experiment.linear_score(model, vx)
        clipped_score = vx @ np.array(params["weights_standardized_X"]) + params["bias_standardized_X"]
        stable_label, clipped_label = stable_score >= 0, clipped_score >= 0
        weights = model.weight.detach().numpy().ravel()
        output[name] = {"settings": params["training_config"],
            "stable_bce_test_correct": int(np.sum(stable_label == y[test])),
            "clipped_bce_test_correct": int(np.sum(clipped_label == y[test])),
            "different_test_labels": int(np.sum(stable_label != clipped_label)),
            "max_weight_difference": float(np.max(np.abs(weights - np.array(params["weights_standardized_X"]))))}
    return output


def run(retune):
    started = time.perf_counter()
    ROOT.mkdir(exist_ok=True)
    torch.manual_seed(experiment.SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    breast = load_breast_cancer()
    ale = np.genfromtxt(experiment.ROOT / "data/mcs_ds_edited_iter_shuffled.csv", delimiter=",", skip_header=1)
    datasets = {"classification": (breast.data, breast.target), "regression": (ale[:, :4], ale[:, 4])}
    standardized = json.loads((experiment.ROOT / "results/summary.json").read_text())
    splits, split_rows, condition = {}, [], {}
    for kind, (X, y) in datasets.items():
        train, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=experiment.SEED,
            stratify=y if kind == "classification" else None)
        splits[kind] = {"train": train, "test": test}
        split_rows.extend({"dataset": kind, "subset": subset, "data_row": int(i+1)}
            for subset, ids in splits[kind].items() for i in ids)
        # Scaling here is only for the condition-number DIAGNOSTIC, not training.
        standardized_design = StandardScaler().fit_transform(X[train])
        condition[kind] = {"raw_design_condition": float(np.linalg.cond(np.c_[X[train], np.ones(len(train))])),
            "standardized_design_condition": float(np.linalg.cond(np.c_[standardized_design, np.ones(len(train))]))}
    with (experiment.ROOT / "results/split.csv").open(newline="") as f:
        prior_split = list(csv.DictReader(f))
    assert {(r["dataset"], r["subset"], int(r["data_row"])) for r in prior_split} == {
        (r["dataset"], r["subset"], r["data_row"]) for r in split_rows}
    experiment.write_csv(ROOT / "split.csv", split_rows)
    identity = {"seed": experiment.SEED, "preprocessing": "none", "epochs": experiment.SEARCH_EPOCHS,
        "learning_rates": RAW_LRS, "momentum": experiment.MOMENTA, "decay_epochs": experiment.DECAY_EPOCHS,
        "clip_eps": experiment.CLIP_EPS, "gradient_clip_norm": experiment.GRAD_CLIP_NORM,
        "classification_sha256": hashlib.sha256(breast.data.tobytes()+breast.target.tobytes()).hexdigest(),
        "regression_sha256": hashlib.sha256(ale.tobytes()).hexdigest()}
    cached = None
    cache_path = ROOT / "selected_configs.json"
    if not retune and cache_path.exists() and (ROOT / "cv_candidates.csv").exists() and (ROOT / "cv_folds.csv").exists():
        saved = json.loads(cache_path.read_text())
        if saved["identity"] == identity:
            cached = saved["selected"]
    models, parameters, selected, candidates, folds, history = {}, {}, {}, [], [], []
    specs = [
        ("PyTorch logistic (SGD)", "classification", experiment.logistic_bce, "logistic"),
        ("PyTorch probit (SGD)", "classification", experiment.probit_bce, "probit"),
        ("PyTorch linear (MSE, SGD)", "regression", experiment.mse_loss, None),
        ("PyTorch linear (t5 NLL, SGD)", "regression", experiment.student_t_nll, None)]
    for name, kind, loss, link in specs:
        X, y = datasets[kind]
        train = splits[kind]["train"]
        if cached is None:
            config, best, rows, fold_rows = tune_raw(name, kind, X[train], y[train], loss)
            candidates.extend(rows); folds.extend(fold_rows)
        else:
            best = cached[name]
            config = {k: best[k] for k in ["epochs", "batch_size", "initial_lr", "momentum", "decay_epochs"]}
        selected[name] = best
        model, h, gradient, _ = experiment.fit_sgd(name, X[train], y[train], loss, config,
            initial_bias=float(y[train].mean()) if kind == "regression" else 0.)
        history.extend(h)
        models[name] = (model, kind, link)
        parameters[name] = {"weights_original_X": model.weight.detach().numpy().ravel().tolist(),
            "bias_original_X": float(model.bias.detach()), "training_config": config,
            "training_gradient_norm": gradient}
    baseline_cl = LogisticRegression(C=1., penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-10)
    baseline_cl.fit(breast.data[splits["classification"]["train"]], breast.target[splits["classification"]["train"]])
    baseline_reg = LinearRegression().fit(ale[splits["regression"]["train"], :4], ale[splits["regression"]["train"], 4])
    models["sklearn logistic (L2)"] = (baseline_cl, "classification", "baseline")
    models["sklearn linear"] = (baseline_reg, "regression", "baseline")
    order = ["sklearn logistic (L2)", "PyTorch logistic (SGD)", "PyTorch probit (SGD)",
        "sklearn linear", "PyTorch linear (MSE, SGD)", "PyTorch linear (t5 NLL, SGD)"]
    class_metrics, reg_metrics, predictions, comparison = [], [], [], []
    for name in order:
        model, kind, link = models[name]
        for subset, ids in splits[kind].items():
            X, y = datasets[kind][0][ids], datasets[kind][1][ids]
            if kind == "classification":
                probability = model.predict_proba(X)[:, 1] if link == "baseline" else experiment.custom_probabilities(model, X, link)
                prediction = (probability >= .5).astype(int)
                correct = int(np.sum(prediction == y))
                metric = {"model": name, "subset": subset, "n": len(y), "correct": correct, "accuracy": correct/len(y)}
                class_metrics.append(metric)
            else:
                prediction = model.predict(X) if link == "baseline" else experiment.linear_score(model, X)
                probability = [""] * len(y)
                metric = {"model": name, "subset": subset, "n": len(y),
                    "mae": float(np.mean(np.abs(prediction-y))), "mse": float(np.mean((prediction-y)**2))}
                reg_metrics.append(metric)
            assert np.isfinite(prediction).all()
            predictions.extend({"dataset": kind, "model": name, "subset": subset, "data_row": int(i+1),
                "observed": float(yi), "prediction": float(pi), "probability_y1": float(pr) if kind == "classification" else ""}
                for i, yi, pi, pr in zip(ids, y, prediction, probability))
            if subset == "test":
                field = "accuracy" if kind == "classification" else "mae"
                previous = next(r for r in standardized[kind] if r["model"] == name and r["subset"] == "test")
                comparison.append({"dataset": kind, "model": name, "metric": field,
                    "standardized": previous[field], "raw_features": metric[field],
                    "raw_minus_standardized": metric[field]-previous[field]})
                print("RAW TEST", metric, flush=True)
    clip_control = compare_clipping(breast.data, breast.target,
        splits["classification"]["train"], splits["classification"]["test"], standardized)
    for name, rows in [("classification_metrics.csv", class_metrics), ("regression_metrics.csv", reg_metrics),
            ("predictions.csv", predictions), ("training_history.csv", history), ("comparison.csv", comparison)]:
        experiment.write_csv(ROOT / name, rows)
    if candidates:
        experiment.write_csv(ROOT / "cv_candidates.csv", candidates)
        experiment.write_csv(ROOT / "cv_folds.csv", folds)
    cache_path.write_text(json.dumps({"identity": identity, "selected": selected}, indent=2)+"\n")
    summary = {"preprocessing": "none: raw X and raw y; no centering, standardization or normalization",
        "same_outer_holdout": True, "identity": identity, "parameters": parameters,
        "classification": class_metrics, "regression": reg_metrics, "comparison": comparison,
        "clip_control_standardized_fixed_settings": clip_control, "training_design_condition": condition,
        "baseline_logistic_iterations": baseline_cl.n_iter_.tolist(),
        "tuning": {"folds": 3, "selection": "same training-only criteria as standardized experiment", "selected": selected},
        "elapsed_seconds": round(time.perf_counter()-started, 3)}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print("CLIP CONTROL", json.dumps(clip_control), flush=True)
    print(f"Done: {ROOT}; elapsed {summary['elapsed_seconds']} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        run(args.retune)
    records = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
    summary_path = ROOT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["warnings"] = records
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Warnings: {len(records)}", flush=True)
