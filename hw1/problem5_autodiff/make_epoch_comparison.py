#!/usr/bin/env python3
"""Make the homework's paired seed-42 comparison at 200 and 500 epochs.

Read validated predictions; create paired test-metric tables and a LaTeX
fragment without selecting runs or editing the parent homework automatically.
"""
import csv
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

import run_experiment as experiment

ROOT = experiment.ROOT
NAMES = {
    "Logistic BCE": "PyTorch logistic (SGD)",
    "Probit BCE": "PyTorch probit (SGD)",
    "Linear MSE": "PyTorch linear (MSE, SGD)",
    "Linear t5 NLL": r"PyTorch linear ($t_5$ NLL, SGD)",
}


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def test_metrics(rows, name, epoch, kind, reference):
    """Recompute test metrics; malignant (original y=0) is positive."""
    ids = [int(row['data_row']) for row in rows]
    assert len(ids) == len(set(ids)) == len(reference)
    assert set(ids) == set(reference)
    y = np.array([float(row['observed']) for row in rows])
    pred = np.array([float(row['prediction']) for row in rows])
    np.testing.assert_allclose(y, [reference[i] for i in ids], rtol=0, atol=0)
    result = {'dataset': kind, 'model': name, 'epochs': epoch, 'n': len(y)}
    if kind == 'classification':
        probability_y1 = np.array([float(row['probability_y1']) for row in rows])
        np.testing.assert_array_equal(pred, (probability_y1 >= .5).astype(int))
        positive_probability = 1-probability_y1
        tp = int(np.sum((y == 0) & (pred == 0)))
        fn = int(np.sum((y == 0) & (pred == 1)))
        tn = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 1) & (pred == 0)))
        result.update({'correct': tp+tn, 'accuracy': float(accuracy_score(y, pred)),
            'roc_auc': float(roc_auc_score(y == 0, positive_probability)),
            'f1_malignant': float(f1_score(y, pred, pos_label=0)),
            'sensitivity': tp/(tp+fn), 'specificity': tn/(tn+fp),
            'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp})
        # Independent pairwise AUC and confusion-count F1 checks.
        pos = positive_probability[y == 0, None]
        neg = positive_probability[y == 1][None, :]
        pairwise_auc = np.mean((pos > neg) + .5*(pos == neg))
        np.testing.assert_allclose(result['roc_auc'], pairwise_auc, atol=1e-13)
        np.testing.assert_allclose(result['f1_malignant'], 2*tp/(2*tp+fp+fn), atol=1e-13)
        np.testing.assert_allclose(result['accuracy'], (tp+tn)/len(y), atol=1e-13)
    else:
        error = y-pred
        result.update({'mae': float(mean_absolute_error(y, pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, pred))), 'r2': float(r2_score(y, pred))})
        np.testing.assert_allclose(result['mae'], np.mean(np.abs(error)), atol=1e-13)
        np.testing.assert_allclose(result['rmse'], np.sqrt(np.mean(error**2)), atol=1e-13)
        np.testing.assert_allclose(result['r2'], 1-np.sum(error**2)/np.sum((y-y.mean())**2), atol=1e-13)
    return result


def main():
    primary = json.loads((ROOT / "results/summary.json").read_text())
    validation = json.loads((ROOT / "sensitivity_epochs/validation.json").read_text())
    assert validation["status"] == "passed" and primary["seed"] == 42
    selected = {(row["model"], int(row["epochs"])): row
                for row in read_csv(ROOT / "sensitivity_epochs/runs.csv")
                if row["split_seed"] == row["sgd_seed"] == "42"
                and int(row["epochs"]) in [200, 500] and row["model"] in NAMES}
    assert len(selected) == 8
    class_base = next(r for r in primary["classification"] if r["subset"] == "test" and r["model"].startswith("sklearn"))
    reg_base = next(r for r in primary["regression"] if r["subset"] == "test" and r["model"].startswith("sklearn"))
    for name in NAMES:
        row = selected[name, 200]
        kind = row["dataset"]
        old_label = NAMES[name].replace("$t_5$", "t5")
        old = next(r for r in primary[kind] if r["subset"] == "test" and r["model"] == old_label)
        field = "accuracy" if kind == "classification" else "mae"
        experiment.np.testing.assert_allclose(float(row["test_metric"]), old[field], atol=1e-13)
        assert (int(row["n_train"]), int(row["n_test"])) == ((455, 114) if kind == "classification" else (85, 22))
    baseline_predictions = [row for row in read_csv(ROOT / 'results/predictions.csv')
                            if row['subset'] == 'test' and row['model'].startswith('sklearn')]
    references = {kind: {int(row['data_row']): float(row['observed']) for row in baseline_predictions if row['dataset'] == kind}
                  for kind in ['classification', 'regression']}
    grouped = defaultdict(list)
    for row in read_csv(ROOT / 'sensitivity_epochs/test_predictions.csv'):
        split, epoch, seed, name = row['run_id'].split('/', 3)
        if split == seed == '42' and int(epoch) in [200, 500] and name in NAMES:
            grouped[name, int(epoch)].append(row)
    metrics = []
    for kind, label in [('classification', 'sklearn logistic (L2)'), ('regression', 'sklearn linear')]:
        rows = [row for row in baseline_predictions if row['dataset'] == kind]
        metrics.append(test_metrics(rows, label, '-', kind, references[kind]))
    for name in NAMES:
        for epoch in [200, 500]:
            kind = selected[name, epoch]['dataset']
            value = test_metrics(grouped[name, epoch], name, epoch, kind, references[kind])
            original_metric = 'accuracy' if kind == 'classification' else 'mae'
            np.testing.assert_allclose(value[original_metric], float(selected[name, epoch]['test_metric']), atol=1e-13)
            metrics.append(value)
    fields = list(dict.fromkeys(field for row in metrics for field in row))
    experiment.write_csv(ROOT / 'results/epoch_comparison_metrics.csv',
                         [{field: row.get(field, '') for field in fields} for row in metrics])
    lookup = {(row['model'], row['epochs']): row for row in metrics}
    additional_class_rows, additional_reg_rows = [], []
    for label, name, epochs in [('sklearn logistic (L2)', 'sklearn logistic (L2)', ['-']),
                              *[(NAMES[n], n, [200, 500]) for n in ['Logistic BCE', 'Probit BCE']]]:
        for epoch in epochs:
            row = lookup[name, epoch]
            epoch_tex = '$-$' if epoch == '-' else str(epoch)
            additional_class_rows.append(f"    {label} & {epoch_tex} & {row['roc_auc']:.4f} & {row['f1_malignant']:.4f} & {row['sensitivity']:.4f} & {row['specificity']:.4f} \\\\")
    for label, name, epochs in [('sklearn linear', 'sklearn linear', ['-']),
                              *[(NAMES[n], n, [200, 500]) for n in ['Linear MSE', 'Linear t5 NLL']]]:
        for epoch in epochs:
            row = lookup[name, epoch]
            epoch_tex = '$-$' if epoch == '-' else str(epoch)
            additional_reg_rows.append(f"    {label} & {epoch_tex} & {row['rmse']:.6f} & {row['r2']:.6f} \\\\")
    class_rows = [f"    sklearn logistic (L2) & $-$ & {class_base['correct']}/{class_base['n']} & {class_base['accuracy']:.4f} & +0.0000 \\\\"]
    reg_rows = [f"    sklearn linear & $-$ & {reg_base['mae']:.6f} & +0.000000 \\\\"]
    for name in NAMES:
        for epoch in [200, 500]:
            row = selected[name, epoch]
            value = float(row["test_metric"])
            delta = float(row["delta_test_metric"])
            if row["dataset"] == "classification":
                correct = round(value * 114)
                class_rows.append(f"    {NAMES[name]} & {epoch} & {correct}/114 & {value:.4f} & {delta:+.4f} \\\\")
            else:
                reg_rows.append(f"    {NAMES[name]} & {epoch} & {value:.6f} & {delta:+.6f} \\\\")
    fragment = r"""\clearpage
\section{problem 5}
Both datasets use an 80/20 split with seed 42. We compare 200 and 500 epochs along the same SGD trajectory with mini-batch shuffle seed 42. Each model uses its full training set with fixed hyperparameters, without cross-validation. Features are standardized using training data only. Classification probabilities are clipped to $[10^{-6},1-10^{-6}]$ before BCE; gradient norms are clipped at 5. The learning rate is $\eta_e=\eta_0/(1+e/100)$, where $e$ starts at zero.

\paragraph{Classification ($n_{\mathrm{train}}=455$, $n_{\mathrm{test}}=114$).}
\begin{center}
  \begin{tabular}{lrrrr}
    \hline
    Model & Epochs & Correct & Test accuracy & $\Delta$ accuracy \\
    \hline
""" + "\n".join(class_rows) + r"""
    \hline
  \end{tabular}
\end{center}

For the additional test metrics, malignant ($y=0$) is the positive class. ROC-AUC uses its predicted probability; $F_1$, sensitivity, and specificity use the same 0.5 classification threshold. Higher values are better.

\begin{center}
  \begin{tabular}{lrrrrr}
    \hline
    Model & Epochs & ROC-AUC & $F_1$ & Sensitivity & Specificity \\
    \hline
""" + "\n".join(additional_class_rows) + r"""
    \hline
  \end{tabular}
\end{center}

\paragraph{Classification (a).} SGD uses batch size 64, initial learning rate 0.05, and momentum 0. The sigmoid model achieves 96.49\% accuracy at both 200 and 500 epochs, 1.75 percentage points below the baseline. ROC-AUC is almost identical to the baseline. Sensitivity is also identical, while specificity is lower: the model produces three false positives, compared with one for sklearn. The PyTorch model minimizes BCE without a penalty; sklearn uses L2 regularization.

\paragraph{Classification (b).} With the same SGD settings, replacing sigmoid with the standard normal CDF improves accuracy from 95.61\% at 200 epochs to 96.49\% at 500 epochs. False positives decrease from four to three; sensitivity is unchanged. The 500-epoch result remains 1.75 percentage points below the baseline, with almost identical ROC-AUC. Both models use clipped BCE.

\clearpage
\paragraph{Regression ($n_{\mathrm{train}}=85$, $n_{\mathrm{test}}=22$).}
\begin{center}
  \begin{tabular}{lrrr}
    \hline
    Model & Epochs & Test MAE (m) & $\Delta$ MAE (m) \\
    \hline
""" + "\n".join(reg_rows) + r"""
    \hline
  \end{tabular}
\end{center}

RMSE gives greater weight to large errors than MAE; lower is better. Higher $R^2$ indicates better fit relative to a constant predictor using the test-set mean.

\begin{center}
  \begin{tabular}{lrrr}
    \hline
    Model & Epochs & Test RMSE (m) & Test $R^2$ \\
    \hline
""" + "\n".join(additional_reg_rows) + r"""
    \hline
  \end{tabular}
\end{center}

\paragraph{Regression (a).} SGD uses batch size 32, initial learning rate 0.03, and momentum 0. With MSE, test MAE is 0.195232 m at 200 epochs and 0.195753 m at 500 epochs, compared with 0.195393 m for the baseline. The 500-epoch model has slightly lower RMSE and higher $R^2$ than the baseline, despite its slightly higher MAE. These small differences are consistent with finite stochastic updates near the least-squares solution.

\paragraph{Regression (b).} With the same SGD settings, minimizing the Student-$t$ negative log-likelihood with $\nu=5$ and fixed scale 1 gives test MAE 0.194035 m at 200 epochs and 0.194308 m at 500 epochs. Both are lower than the baseline, but their RMSE is higher and $R^2$ is lower. Thus, the improvement depends on the evaluation metric. ALE remains in its original units.
"""
    output = ROOT / "results/problem5_epoch_comparison.tex"
    output.write_text(fragment, encoding="utf-8")
    (ROOT / 'results/epoch_comparison_metrics_validation.json').write_text(json.dumps({
        'status': 'passed', 'n_models_and_checkpoints': len(metrics), 'split_seed': 42, 'sgd_seed': 42,
        'test_row_and_target_alignment': 'passed', 'positive_class': 'malignant (original y=0)',
        'positive_probability': '1 - probability_y1', 'threshold': .5,
        'independent_checks': ['pairwise ROC-AUC', 'F1 and accuracy from confusion counts',
                               'MAE and RMSE from residuals', 'R2 from SSE/SST'],
        'saved_checkpoint_metric_agreement': 'passed', 'cross_validation': False,
    }, indent=2) + '\n')
    print(f"Created {output}; all ten test-metric rows independently verified.")


if __name__ == "__main__":
    main()
