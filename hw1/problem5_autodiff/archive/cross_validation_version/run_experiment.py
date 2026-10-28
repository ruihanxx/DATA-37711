#!/usr/bin/env python3
"""Problem 5: four custom PyTorch losses trained with mini-batch SGD.

Settings are selected by training-only 3-fold CV before test evaluation. Only X is scaled:
Student-t residuals retain the original ALE units and fixed scale = 1.
This script generates results, training curves, and a LaTeX results fragment;
it never edits the parent hw1.tex itself.
"""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import warnings
from itertools import product

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
(ROOT / ".cache").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from threadpoolctl import threadpool_limits

SEED = 42
DTYPE = torch.float64
NU = 5.0
FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]
CLIP_EPS = 1e-6
GRAD_CLIP_NORM = 5.0
SEARCH_EPOCHS = [50, 100, 200]
LEARNING_RATES = [0.03, 0.1, 0.3]
MOMENTA = [0.0, 0.9]
DECAY_EPOCHS = 100


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clipped_bce(probability, y):
    probability = torch.clamp(probability, min=CLIP_EPS, max=1 - CLIP_EPS)
    return -(y * torch.log(probability) + (1 - y) * torch.log1p(-probability)).mean()


def logistic_bce(score, y):
    return clipped_bce(torch.sigmoid(score), y)


def probit_bce(score, y):
    return clipped_bce(torch.distributions.Normal(0., 1.).cdf(score), y)


def mse_loss(prediction, y):
    return ((y - prediction) ** 2).mean()


def student_t_nll(prediction, y):
    distribution = torch.distributions.StudentT(
        df=torch.tensor(NU, dtype=DTYPE), loc=torch.tensor(0., dtype=DTYPE),
        scale=torch.tensor(1., dtype=DTYPE))
    return -distribution.log_prob(y - prediction).mean()


def fit_sgd(name, X, y, loss_fn, config, initial_bias=0., validation=None,
            kind=None, seed=SEED, verbose=True):
    model = nn.Linear(X.shape[1], 1, dtype=DTYPE)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(initial_bias)
    tx, ty = torch.tensor(X, dtype=DTYPE), torch.tensor(y, dtype=DTYPE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["initial_lr"],
                               momentum=config["momentum"], weight_decay=0.)
    sampler = torch.Generator().manual_seed(seed)
    history, checkpoints = [], []
    if validation is not None:
        vx, vy = (torch.tensor(value, dtype=DTYPE) for value in validation)
    def record(epoch, lr):
        with torch.no_grad():
            score = model(tx).squeeze(-1)
            history.append({"model": name, "epoch": epoch, "learning_rate": lr,
                "training_loss": float(loss_fn(score, ty))})
    record(0, config["initial_lr"])
    for epoch in range(config["epochs"]):
        lr = config["initial_lr"] / (1 + epoch / config["decay_epochs"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        permutation = torch.randperm(len(ty), generator=sampler)
        for batch in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tx[batch]).squeeze(-1), ty[batch])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in {name}")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError(f"Non-finite gradient in {name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM,
                                           error_if_nonfinite=True)
            optimizer.step()
        record(epoch + 1, lr)
        if validation is not None and epoch + 1 in SEARCH_EPOCHS:
            with torch.no_grad():
                score = model(vx).squeeze(-1)
                checkpoints.append({"epoch": epoch + 1, "n_validation": len(vy),
                    "validation_loss": float(loss_fn(score, vy)),
                    "validation_accuracy": float(((score >= 0) == (vy >= .5)).double().mean()) if kind == "classification" else "",
                    "validation_mae": float(torch.abs(score - vy).mean()) if kind == "regression" else ""})
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  {name}: epoch {epoch + 1}, loss={history[-1]['training_loss']:.6f}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(tx).squeeze(-1), ty).backward()
    grad_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())))
    return model, history, grad_norm, checkpoints


def tune_config(name, kind, X, y, loss_fn):
    """Use raw outer-training X only; fit every scaler on its CV training fold."""
    splitter = (StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                if kind == "classification" else KFold(n_splits=3, shuffle=True, random_state=SEED))
    folds = []
    for train, valid in splitter.split(X, y):
        scaler = StandardScaler().fit(X[train])
        folds.append((scaler.transform(X[train]), y[train], scaler.transform(X[valid]), y[valid]))
    batch_sizes = [64, 128] if kind == "classification" else [16, 32]
    candidates, fold_rows = [], []
    configurations = list(product(LEARNING_RATES, batch_sizes, MOMENTA))
    for number, (lr, batch, momentum) in enumerate(configurations, 1):
        config = {"epochs": max(SEARCH_EPOCHS), "batch_size": batch,
                  "initial_lr": lr, "momentum": momentum, "decay_epochs": DECAY_EPOCHS}
        checkpoints = []
        for fold, (tx, ty, vx, vy) in enumerate(folds, 1):
            _, _, _, scores = fit_sgd(name, tx, ty, loss_fn, config,
                initial_bias=float(ty.mean()) if kind == "regression" else 0.,
                validation=(vx, vy), kind=kind, seed=SEED + fold, verbose=False)
            checkpoints.extend(scores)
            fold_rows.extend({"model": name, "fold": fold, "initial_lr": lr,
                "batch_size": batch, "momentum": momentum, **score} for score in scores)
        for epoch in SEARCH_EPOCHS:
            selected = [r for r in checkpoints if r["epoch"] == epoch]
            weights = [r["n_validation"] for r in selected]
            candidates.append({"model": name, "epochs": epoch, "initial_lr": lr,
                "batch_size": batch, "momentum": momentum, "decay_epochs": DECAY_EPOCHS,
                "mean_validation_loss": float(np.average([r["validation_loss"] for r in selected], weights=weights)),
                "mean_validation_accuracy": float(np.average([r["validation_accuracy"] for r in selected], weights=weights)) if kind == "classification" else "",
                "mean_validation_mae": float(np.average([r["validation_mae"] for r in selected], weights=weights)) if kind == "regression" else ""})
        print(f"  CV {name}: {number}/{len(configurations)} optimizer configurations", flush=True)
    if kind == "classification":
        best = min(candidates, key=lambda r: (-r["mean_validation_accuracy"],
            r["mean_validation_loss"], r["epochs"], -r["batch_size"]))
    else:
        best = min(candidates, key=lambda r: (r["mean_validation_mae"],
            r["mean_validation_loss"], r["epochs"], -r["batch_size"]))
    best_config = {key: best[key] for key in ["epochs", "batch_size", "initial_lr", "momentum", "decay_epochs"]}
    print(f"Selected {name}: {best_config}", flush=True)
    return best_config, candidates, fold_rows, best


def linear_score(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=DTYPE)).squeeze(-1).numpy()


def custom_probabilities(model, X, link):
    with torch.no_grad():
        score = model(torch.tensor(X, dtype=DTYPE)).squeeze(-1)
        if link == "logistic":
            probability = torch.sigmoid(score)
        else:
            probability = torch.distributions.Normal(0., 1.).cdf(score)
        return torch.clamp(probability, min=CLIP_EPS, max=1 - CLIP_EPS).numpy()


def check_losses():
    # Verify the custom values against independent SciPy/NumPy expressions and
    # finite-difference-check autograd for all four losses before training.
    score = torch.tensor([-.8, .4, 1.1, -1.5], dtype=DTYPE, requires_grad=True)
    label = torch.tensor([0., 1., 0., 1.], dtype=DTYPE)
    target = torch.tensor([.2, -.3, 1.8, -.6], dtype=DTYPE)
    z, y = score.detach().numpy(), label.numpy()
    sigmoid = np.clip(1 / (1 + np.exp(-z)), CLIP_EPS, 1 - CLIP_EPS)
    cdf = np.clip(stats.norm.cdf(z), CLIP_EPS, 1 - CLIP_EPS)
    references = [-(y * np.log(sigmoid) + (1-y) * np.log1p(-sigmoid)).mean(),
        -(y * np.log(cdf) + (1-y) * np.log1p(-cdf)).mean(),
        np.mean((target.numpy() - z) ** 2),
        -stats.t.logpdf(target.numpy() - z, df=NU, loc=0, scale=1).mean()]
    checks = {}
    for (name, loss_fn, response), expected in zip([
            ("logistic_bce", logistic_bce, label), ("probit_bce", probit_bce, label),
            ("mse", mse_loss, target), ("student_t_nll", student_t_nll, target)], references):
        actual = float(loss_fn(score, response))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert torch.autograd.gradcheck(lambda value: loss_fn(value, response), (score,), eps=1e-6, atol=1e-5)
        checks[name] = {"reference_value": float(expected), "autodiff_value": actual, "gradcheck": "passed"}
    for name, loss_fn in [("logistic", logistic_bce), ("probit", probit_bce)]:
        extreme = torch.tensor([-100., 100.], dtype=DTYPE, requires_grad=True)
        loss = loss_fn(extreme, torch.tensor([1., 0.], dtype=DTYPE))
        gradient, = torch.autograd.grad(loss, extreme)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        checks[name + "_clip_extremes"] = {"loss": float(loss), "finite_loss_and_gradient": True}
    return checks


def parameters(model, scaler):
    standardized_w = model.weight.detach().numpy().ravel()
    standardized_b = float(model.bias.detach().item())
    original_w = standardized_w / scaler.scale_
    return {"weights_standardized_X": standardized_w.tolist(), "bias_standardized_X": standardized_b,
        "weights_original_X": original_w.tolist(),
        "bias_original_X": float(standardized_b - original_w @ scaler.mean_)}


def make_figures(histories):
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 3.3), layout="constrained")
    for name, color in [("PyTorch logistic (SGD)", "#247ba0"), ("PyTorch probit (SGD)", "#ce5e79")]:
        h = histories[name]
        ax.plot([r["epoch"] for r in h], [r["training_loss"] for r in h], color=color, label=name)
    ax.set(xlabel="Epoch", ylabel="Training binary cross-entropy")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(ROOT / "figures" / "classification_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, name, color, ylabel in zip(axes,
            ["PyTorch linear (MSE, SGD)", "PyTorch linear (t5 NLL, SGD)"],
            ["#247ba0", "#ce5e79"], ["Training MSE", "Training Student-t NLL"]):
        h = histories[name]
        ax.plot([r["epoch"] for r in h], [r["training_loss"] for r in h], color=color)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.2)
    fig.savefig(ROOT / "figures" / "regression_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_tex(class_rows, reg_rows, configs):
    ctest = [r for r in class_rows if r["subset"] == "test"]
    rtest = [r for r in reg_rows if r["subset"] == "test"]
    cbase, rbase = ctest[0]["accuracy"], rtest[0]["mae"]
    classtex = "\n".join(f"    {r['model']} & {r['correct']}/{r['n']} & {r['accuracy']:.4f} & {r['accuracy']-cbase:+.4f} \\\\" for r in ctest)
    regtex = "\n".join(f"    {r['model'].replace('t5', '$t_5$')} & {r['mae']:.6f} & {r['mae']-rbase:+.6f} \\\\" for r in rtest)
    def config_sentence(name):
        c = configs[name]
        return (f"SGD: {c['epochs']} epochs, batch size {c['batch_size']}, "
                f"initial learning rate {c['initial_lr']:g}, momentum {c['momentum']:g}.")
    logistic, probit = ctest[1], ctest[2]
    mse, student = rtest[1], rtest[2]
    def accuracy_comparison(row):
        delta = 100 * (row['accuracy'] - cbase)
        if abs(delta) < 1e-12:
            return "matches the sklearn baseline"
        return f"is {abs(delta):.2f} percentage points {'higher' if delta > 0 else 'lower'} than the sklearn baseline"
    def mae_comparison(row):
        delta = row['mae'] - rbase
        if abs(delta) < .0000005:
            return "matches the sklearn baseline to six decimal places"
        return f"is {abs(delta):.6f} m {'lower' if delta < 0 else 'higher'} than the sklearn baseline"
    fragment = r"""\clearpage
\section{problem 5}
Both datasets use an 80/20 split with seed 42. Features are standardized using training data only. Hyperparameters are selected by three-fold cross-validation within the training set, with 50, 100 or 200 epochs. Classification probabilities are clipped to $[10^{-6},1-10^{-6}]$ before computing BCE; gradient norms are clipped at 5.

\paragraph{Classification ($n_{\mathrm{train}}=455$, $n_{\mathrm{test}}=114$).}
\begin{center}
  \begin{tabular}{lrrr}
    \hline
    Model & Correct & Test accuracy & $\Delta$ accuracy \\
    \hline
""" + classtex + r"""
    \hline
  \end{tabular}

\end{center}
""" + "\n\\paragraph{Classification (a).} " + config_sentence(logistic['model']) + f" The sigmoid model has test accuracy {100*logistic['accuracy']:.2f}\\%, which {accuracy_comparison(logistic)}. The PyTorch model minimizes BCE without a penalty; the sklearn baseline uses L2 regularization.\n" + "\n\\paragraph{Classification (b).} " + config_sentence(probit['model']) + f" Replacing sigmoid with the standard normal CDF gives test accuracy {100*probit['accuracy']:.2f}\\%, which {accuracy_comparison(probit)}. Both implementations use the same clipped BCE.\n" + r"""
\begin{center}
  \includegraphics[width=0.9\textwidth]{problem5_autodiff/figures/classification_loss.png}
\end{center}

\clearpage
\paragraph{Regression ($n_{\mathrm{train}}=85$, $n_{\mathrm{test}}=22$).}
\begin{center}
  \begin{tabular}{lrr}
    \hline
    Model & Test MAE (m) & $\Delta$ MAE (m) \\
    \hline
""" + regtex + r"""
    \hline
  \end{tabular}

\end{center}
""" + "\n\\paragraph{Regression (a).} " + config_sentence(mse['model']) + f" The linear model trained with MSE has test MAE {mse['mae']:.6f} m, which {mae_comparison(mse)}. A finite number of stochastic updates can produce a small difference from the least-squares baseline.\n" + "\n\\paragraph{Regression (b).} " + config_sentence(student['model']) + f" Minimizing the Student-$t$ negative log-likelihood with $\\nu=5$ and fixed scale 1 gives test MAE {student['mae']:.6f} m, which {mae_comparison(student)}. ALE remains in its original units.\n" + r"""
\begin{center}
  \includegraphics[width=\textwidth]{problem5_autodiff/figures/regression_loss.png}
\end{center}
"""
    (ROOT / "results" / "problem5.tex").write_text(fragment, encoding="utf-8")


def run(retune=False):
    started = time.perf_counter()
    for folder in ["data", "results", "figures"]:
        (ROOT / folder).mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = check_losses()
    original = ROOT.parent / "problem4_ale" / "data" / "mcs_ds_edited_iter_shuffled.csv"
    ale = ROOT / "data" / original.name
    if not ale.exists():
        shutil.copyfile(original, ale)
    values = np.genfromtxt(ale, delimiter=",", skip_header=1)
    assert values.shape == (107, 6) and np.isfinite(values).all()
    breast = load_breast_cancer()
    np.savez_compressed(ROOT / "data" / "breast_cancer.npz", X=breast.data, y=breast.target,
        feature_names=breast.feature_names, target_names=breast.target_names)
    datasets = {"classification": (breast.data, breast.target), "regression": (values[:, :4], values[:, 4])}
    identity = {"seed": SEED, "clip_eps": CLIP_EPS, "gradient_clip_norm": GRAD_CLIP_NORM,
        "ale_csv_sha256": hashlib.sha256(ale.read_bytes()).hexdigest(),
        "breast_cancer_arrays_sha256": hashlib.sha256(breast.data.tobytes()+breast.target.tobytes()).hexdigest(),
        "search_epochs": SEARCH_EPOCHS, "learning_rates": LEARNING_RATES,
        "momenta": MOMENTA, "decay_epochs": DECAY_EPOCHS}
    cache_path = ROOT / "results/selected_configs.json"
    cached = None
    if not retune and cache_path.exists() and all((ROOT / "results" / name).exists()
            for name in ["cv_candidates.csv", "cv_folds.csv"]):
        saved = json.loads(cache_path.read_text())
        if saved["identity"] == identity:
            cached = saved["selected"]
    splits, transformed, scalers, split_rows = {}, {}, {}, []
    for kind, (X, y) in datasets.items():
        train, test = train_test_split(np.arange(len(y)), test_size=.2, random_state=SEED,
            stratify=y if kind == "classification" else None)
        assert not set(train).intersection(test)
        assert sorted(np.r_[train, test].tolist()) == list(range(len(y)))
        scaler = StandardScaler().fit(X[train])
        splits[kind] = {"train": train, "test": test}
        transformed[kind] = {"train": scaler.transform(X[train]), "test": scaler.transform(X[test])}
        scalers[kind] = scaler
        for subset, ids in splits[kind].items():
            split_rows.extend({"dataset": kind, "subset": subset, "data_row": int(i+1)} for i in ids)
    # Ensure the ALE holdout is exactly the same as in Problem 4.
    with (ROOT.parent / "problem4_ale/results/split.csv").open(newline="", encoding="utf-8") as file:
        q4split = list(csv.DictReader(file))
    assert set(splits["regression"]["test"]+1) == {int(r["data_row"]) for r in q4split if r["subset"]=="test"}
    write_csv(ROOT / "results/split.csv", split_rows)
    baseline_cl = LogisticRegression(C=1., penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-10)
    baseline_reg = LinearRegression()
    baseline_cl.fit(transformed["classification"]["train"], breast.target[splits["classification"]["train"]])
    baseline_reg.fit(transformed["regression"]["train"], values[splits["regression"]["train"], 4])
    histories, models, model_params, configs, cv_candidates, cv_folds, cv_selection = {}, {}, {}, {}, [], [], {}
    for name, kind, loss_fn, link in [
        ("PyTorch logistic (SGD)", "classification", logistic_bce, "logistic"),
        ("PyTorch probit (SGD)", "classification", probit_bce, "probit"),
        ("PyTorch linear (MSE, SGD)", "regression", mse_loss, None),
        ("PyTorch linear (t5 NLL, SGD)", "regression", student_t_nll, None)]:
        train_ids = splits[kind]["train"]
        if cached is None:
            print(f"Tuning {name} on training-only CV", flush=True)
            config, candidates, folds, best = tune_config(name, kind,
                datasets[kind][0][train_ids], datasets[kind][1][train_ids], loss_fn)
        else:
            print(f"Reusing training-CV settings for {name}", flush=True)
            best = cached[name]
            config = {key: best[key] for key in ["epochs", "batch_size", "initial_lr", "momentum", "decay_epochs"]}
            candidates, folds = [], []
        configs[name], cv_selection[name] = config, best
        cv_candidates.extend(candidates)
        cv_folds.extend(folds)
        print(f"Training {name}", flush=True)
        train_y = datasets[kind][1][splits[kind]["train"]]
        fit_started = time.perf_counter()
        model, history, grad_norm, _ = fit_sgd(name, transformed[kind]["train"], train_y, loss_fn, config,
            initial_bias=float(train_y.mean()) if kind == "regression" else 0.)
        models[name] = (model, kind, link)
        histories[name] = history
        model_params[name] = {**parameters(model, scalers[kind]), "training_gradient_norm": grad_norm,
            "fit_seconds": round(time.perf_counter() - fit_started, 4),
            "training_config": config, "initialization": "zero weights; regression bias = training y mean, classification bias = 0"}
    class_rows, reg_rows, prediction_rows, reg_predictions = [], [], [], {}
    for name, baseline, kind in [("sklearn logistic (L2)", baseline_cl, "classification"),
            ("sklearn linear", baseline_reg, "regression")]:
        models[name] = (baseline, kind, "baseline")
    # Baseline first, followed by parts (a) and (b), in every saved result table.
    order = ["sklearn logistic (L2)", "PyTorch logistic (SGD)", "PyTorch probit (SGD)",
             "sklearn linear", "PyTorch linear (MSE, SGD)", "PyTorch linear (t5 NLL, SGD)"]
    for name in order:
        model, kind, link = models[name]
        for subset, ids in splits[kind].items():
            X, y = transformed[kind][subset], datasets[kind][1][ids]
            if kind == "classification":
                probability = model.predict_proba(X)[:, 1] if link == "baseline" else custom_probabilities(model, X, link)
                pred = (probability >= .5).astype(int)
                assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
                class_rows.append({"model": name, "subset": subset, "n": len(y),
                    "correct": int(np.sum(pred == y)), "accuracy": float(accuracy_score(y, pred))})
                prediction_rows.extend({"dataset": kind, "model": name, "subset": subset,
                    "data_row": int(i+1), "observed": float(yi), "prediction": float(pi), "probability_y1": float(pr)}
                    for i, yi, pi, pr in zip(ids, y, pred, probability))
            else:
                pred = model.predict(X) if link == "baseline" else linear_score(model, X)
                assert np.isfinite(pred).all()
                reg_rows.append({"model": name, "subset": subset, "n": len(y),
                    "mae": float(mean_absolute_error(y, pred)), "mse": float(mean_squared_error(y, pred))})
                reg_predictions[name] = pred if subset == "test" else reg_predictions.get(name)
                prediction_rows.extend({"dataset": kind, "model": name, "subset": subset,
                    "data_row": int(i+1), "observed": float(yi), "prediction": float(pi), "probability_y1": ""}
                    for i, yi, pi in zip(ids, y, pred))
    write_csv(ROOT / "results/classification_metrics.csv", class_rows)
    write_csv(ROOT / "results/regression_metrics.csv", reg_rows)
    write_csv(ROOT / "results/predictions.csv", prediction_rows)
    write_csv(ROOT / "results/training_history.csv", [r for history in histories.values() for r in history])
    if cv_candidates:
        write_csv(ROOT / "results/cv_candidates.csv", cv_candidates)
        write_csv(ROOT / "results/cv_folds.csv", cv_folds)
    cache_path.write_text(json.dumps({"identity": identity, "selected": cv_selection}, indent=2) + "\n")
    make_figures(histories)
    summary = {"seed": SEED, "test_fraction": .2, "classification_stratified": True,
        "scaling": "X only, fitted on training data; regression y remains in meters",
        "classification_threshold": .5, "baseline_logistic": baseline_cl.get_params(),
        "custom_model_penalty": "none", "student_t": {"df": NU, "loc": 0., "scale": 1.},
        "optimizer": "torch.optim.SGD, shuffled mini-batches, selected momentum, weight_decay=0",
        "probability_clip": {"min": CLIP_EPS, "max": 1 - CLIP_EPS},
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "tuning": {"folds": 3, "epochs": SEARCH_EPOCHS, "initial_lr": LEARNING_RATES,
            "reused_saved_selection": cached is not None,
            "momentum": MOMENTA, "classification_batch_sizes": [64, 128], "regression_batch_sizes": [16, 32],
            "criterion_classification": "maximum pooled CV accuracy, then minimum pooled clipped BCE",
            "criterion_regression": "minimum pooled CV MAE, then minimum pooled training-objective validation loss",
            "fold_scaling": "scaler fitted on each CV training fold only", "selected": cv_selection},
        "learning_rate": "initial_lr / (1 + (epoch - 1) / decay_epochs)",
        "test_used_for_training_or_selection": False,
        "ale_csv_sha256": hashlib.sha256(ale.read_bytes()).hexdigest(),
        "breast_cancer_arrays_sha256": hashlib.sha256(breast.data.tobytes()+breast.target.tobytes()).hexdigest(),
        "features_regression": FEATURES, "features_classification": breast.feature_names.tolist(),
        "classification_label_names": breast.target_names.tolist(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "torch": torch.__version__},
        "models": model_params, "loss_checks": checks, "classification": class_rows, "regression": reg_rows,
        "elapsed_seconds": round(time.perf_counter()-started, 3)}
    (ROOT / "results/summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    write_tex(class_rows, reg_rows, configs)
    for row in class_rows + reg_rows:
        if row["subset"] == "test": print(row, flush=True)
    print(f"Done in {summary['elapsed_seconds']} seconds: {ROOT / 'results/problem5.tex'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune", action="store_true", help="Repeat training-only CV instead of reusing saved selected settings.")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            run(retune=args.retune)
        records = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
        (ROOT / "results/warnings.json").write_text(json.dumps(records, indent=2)+"\n")
        summary_path = ROOT / "results/summary.json"
        summary = json.loads(summary_path.read_text())
        summary["warnings"] = records
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        print(f"Warnings: {len(captured)}", flush=True)
