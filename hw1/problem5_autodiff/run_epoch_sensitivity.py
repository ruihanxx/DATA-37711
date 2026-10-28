#!/usr/bin/env python3
"""Compare 200/500/1000 epochs on identical direct-training trajectories."""
import csv
import hashlib
import json
import time
import warnings

import numpy as np
from scipy import special, stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from threadpoolctl import threadpool_limits

import run_experiment as experiment
import run_sensitivity as sensitivity

ROOT = experiment.ROOT / "sensitivity_epochs"
CHECKPOINTS = [200, 500, 1000]
SPLIT_SEEDS = sensitivity.SPLIT_SEEDS
SGD_SEEDS = sensitivity.SGD_SEEDS


def fit_checkpoints(X, y, loss_fn, config, seed, matched_l2=False, regression=False):
    """Use the existing optimizer semantics; checkpoints never change updates."""
    model = nn.Linear(X.shape[1], 1, dtype=experiment.DTYPE)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(float(y.mean()) if regression else 0.)
    tx, ty = torch.tensor(X, dtype=experiment.DTYPE), torch.tensor(y, dtype=experiment.DTYPE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["initial_lr"],
                                momentum=config["momentum"], weight_decay=0.)
    sampler = torch.Generator().manual_seed(seed)

    def objective(ids=None):
        bx, by = (tx, ty) if ids is None else (tx[ids], ty[ids])
        result = loss_fn(model(bx).squeeze(-1), by)
        if matched_l2:
            result = result + model.weight.square().sum() / (2 * len(y))
        return result

    history, snapshots = [], {}
    with torch.no_grad():
        history.append({"epoch": 0, "learning_rate": config["initial_lr"],
                        "training_objective": float(objective())})
    for epoch in range(max(CHECKPOINTS)):
        lr = config["initial_lr"] / (1 + epoch / config["decay_epochs"])
        optimizer.param_groups[0]["lr"] = lr
        for ids in torch.randperm(len(y), generator=sampler).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(ids)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), experiment.GRAD_CLIP_NORM,
                                    error_if_nonfinite=True)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                history.append({"epoch": epoch + 1, "learning_rate": lr,
                                "training_objective": float(objective())})
        if epoch + 1 in CHECKPOINTS:
            optimizer.zero_grad(set_to_none=True)
            full_loss = objective()
            full_loss.backward()
            gradient = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
            snapshots[epoch + 1] = {
                "weights": model.weight.detach().numpy().ravel().copy(),
                "bias": float(model.bias.detach().item()),
                "training_objective": float(full_loss.detach()),
                "training_gradient_norm": gradient,
            }
    return snapshots, history


def summarize(rows):
    aggregated = []
    for epoch in CHECKPOINTS:
        for name, kind, _, _ in sensitivity.SPECS:
            group = [r for r in rows if r["epochs"] == epoch and r["model"] == name]
            original = { (r["split_seed"], r["sgd_seed"]): r for r in rows
                        if r["epochs"] == 200 and r["model"] == name }
            changes = [r["test_metric"] - original[r["split_seed"], r["sgd_seed"]]["test_metric"] for r in group]
            values = np.array([r["test_metric"] for r in group])
            deltas = np.array([r["delta_test_metric"] for r in group])
            gaps = [r["training_objective_gap"] for r in group if r["training_objective_gap"] is not None]
            aggregated.append({"epochs": epoch, "model": name, "dataset": kind,
                "n_paired_runs": len(group), "metric": group[0]["metric"],
                "mean_test_metric": float(values.mean()), "sd_test_metric": float(values.std(ddof=1)),
                "mean_baseline_metric": float(np.mean([r["baseline_test_metric"] for r in group])),
                "mean_delta": float(deltas.mean()), "min_delta": float(deltas.min()), "max_delta": float(deltas.max()),
                "mean_change_from_200": float(np.mean(changes)),
                "mean_training_objective": float(np.mean([r["training_objective"] for r in group])),
                "mean_training_objective_gap": float(np.mean(gaps)) if gaps else None,
                "mean_training_gradient_norm": float(np.mean([r["training_gradient_norm"] for r in group]))})
    return aggregated


def make_figure(aggregated):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7), layout="constrained")
    colors = ["#247ba0", "#ce5e79", "#388e68"]
    for ax, kind, names, scale in [(axes[0], "classification", [s[0] for s in sensitivity.SPECS[:3]], 100.),
                                  (axes[1], "regression", [s[0] for s in sensitivity.SPECS[3:]], 1.)]:
        for name, color in zip(names, colors):
            group = [r for r in aggregated if r["model"] == name]
            ax.plot(CHECKPOINTS, [scale*r["mean_test_metric"] for r in group], "o-", color=color, label=name)
        baseline = next(r["mean_baseline_metric"] for r in aggregated if r["dataset"] == kind)
        ax.axhline(scale*baseline, linestyle="--", color="#666", label="sklearn baseline")
        ax.set(xlabel="Epochs", ylabel="Mean accuracy (%)" if kind == "classification" else "Mean test MAE (m)")
        ax.set_xticks(CHECKPOINTS)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.19))
    fig.suptitle("Longer training: 15 paired runs, identical data and SGD trajectories")
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


def write_report(aggregated):
    import html
    def metric(r, field):
        return f"{100*r[field]:.2f}%" if r["dataset"] == "classification" else f"{r[field]:.6f}"
    body = []
    for r in aggregated:
        delta = f"{100*r['mean_delta']:+.2f} pp" if r['dataset'] == 'classification' else f"{r['mean_delta']:+.6f}"
        gap = "—" if r['mean_training_objective_gap'] is None else f"{r['mean_training_objective_gap']:.3e}"
        body.append(f"<tr><td>{r['epochs']}</td><td>{html.escape(r['model'])}</td><td>{metric(r,'mean_baseline_metric')}</td><td>{metric(r,'mean_test_metric')}</td><td>{delta}</td><td>{r['mean_training_objective']:.6f}</td><td>{gap}</td></tr>")
    findings = []
    for name, kind, _, _ in sensitivity.SPECS:
        first = next(r for r in aggregated if r['epochs'] == 200 and r['model'] == name)
        last = next(r for r in aggregated if r['epochs'] == 1000 and r['model'] == name)
        change = f"{100*last['mean_change_from_200']:+.2f} 个百分点" if kind == 'classification' else f"{last['mean_change_from_200']:+.6f} m"
        findings.append(f"{name}：200→1000 轮，平均训练目标从 {first['mean_training_objective']:.6f} 变为 {last['mean_training_objective']:.6f}；平均测试指标从 {metric(first,'mean_test_metric')} 变为 {metric(last,'mean_test_metric')}，变化 {change}。")
    findings.append("分类测试集每多错一条相差约 0.88 个百分点。训练损失与测试指标需要分开判断：训练更充分并不保证测试准确率更高或 MAE 更低；不带 L2 的 logistic 与默认 sklearn baseline 的训练目标也不同。MSE 与 OLS 的目标相同，应结合训练目标差判断是否已充分优化。")
    findings.append("学习率衰减保持不变，1000 轮的末轮学习率约为初始值的 1/11。该对照衡量的是增加训练轮数的效果，不同时改变学习率或按测试集寻找最佳停止点。")
    (ROOT / 'findings.txt').write_text('\n\n'.join(findings) + '\n', encoding='utf-8')
    findings_html = ''.join(f'<p>{html.escape(line)}</p>' for line in findings)
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Problem 5：增加 epochs</title>
<style>body{{font-family:system-ui,-apple-system,'PingFang SC',sans-serif;max-width:1120px;margin:32px auto;padding:0 20px;color:#243047;line-height:1.75}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #dde3eb}}th{{background:#eef3f8}}img{{width:100%}}.scroll{{overflow-x:auto}}</style>
<h1>Problem 5：200、500、1000 epochs 的直接训练对照</h1>
<p>固定数据划分种子 {SPLIT_SEEDS}，mini-batch 打乱种子 {SGD_SEEDS}，每个模型每个训练轮数有 15 次成对比较。每条轨迹训练到 1000 轮，并保存第 200、500、1000 轮结果，共 75 条训练轨迹和 225 个检查点。不同检查点共享相同的前期更新，不做交叉验证，不按测试分数选择 epoch。</p>
<p>只增加训练轮数。分类 batch=64、初始 lr=0.05；回归 batch=32、初始 lr=0.03；momentum=0。保持原来的 lr0/(1+epoch_index/100) 衰减、训练集标准化、float64、概率 clip=[1e-6,1-1e-6]、梯度 norm clip=5。分类使用全部 455 条训练数据、114 条测试数据；回归使用 85/22 条。</p>
<p>sklearn logistic baseline 为 L2、C=1；回归 baseline 为 OLS。除作业要求的四个模型外，保留匹配 sklearn 正则强度的 L2 logistic 诊断模型。训练目标差分别参照无正则 sklearn logistic、匹配 L2 sklearn logistic、OLS、SciPy t5 BFGS；probit 未设数值最优参考。无正则 logistic 参考是数值拟合结果，并不保证可分数据存在有限最优参数。</p>
<img src="comparison.png" alt="不同训练轮数的平均测试准确率和 MAE">
<div class="scroll"><table><tr><th>Epochs</th><th>模型</th><th>Baseline 均值</th><th>PyTorch 均值</th><th>平均差值</th><th>平均训练目标</th><th>平均目标差</th></tr>{''.join(body)}</table></div>
<p>准确率越高越好、MAE 越低越好。平均值来自 15 次配对运行；同一划分的三个种子共享测试集，划分之间也有样本重叠，因此没有把这些运行当作独立样本进行显著性检验。</p>
<h2>实测结论</h2><div id="findings">{findings_html}</div>
<h2>可复核文件</h2><p>runs.csv 保存逐次指标、训练目标与梯度；aggregate.csv 为汇总；parameters.json、test_predictions.csv、split.csv 可重算预测和指标；training_history.csv 保存每 10 轮训练目标；validation.json 保存同轨迹第 200 轮复现检查及独立指标核验。</p>
</html>"""
    (ROOT / "report.html").write_text(document, encoding="utf-8")


def run():
    ROOT.mkdir(exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old_summary = json.loads((sensitivity.ROOT / "summary.json").read_text())
    old_parameters = json.loads((sensitivity.ROOT / "parameters.json").read_text())
    with (sensitivity.ROOT / "runs.csv").open() as old_file:
        old_rows = {r["run_id"]: r for r in csv.DictReader(old_file)}
    checks = experiment.check_losses()
    breast = load_breast_cancer()
    ale = np.genfromtxt(experiment.ROOT / "data/mcs_ds_edited_iter_shuffled.csv", delimiter=",", skip_header=1)
    breast_hash = hashlib.sha256(breast.data.tobytes() + breast.target.tobytes()).hexdigest()
    ale_hash = hashlib.sha256(ale.tobytes()).hexdigest()
    assert breast_hash == old_summary["breast_cancer_sha256"] and ale_hash == old_summary["ale_sha256"]
    datasets = {"classification": (breast.data, breast.target), "regression": (ale[:, :4], ale[:, 4])}
    references = old_summary["baseline_runs"]
    rows, predictions, splits, histories, saved_parameters = [], [], [], [], {}
    replicated = 0
    for split_seed in SPLIT_SEEDS:
        print(f"Split seed {split_seed}: training identical trajectories to 1000 epochs", flush=True)
        data = {}
        for kind, (X, y) in datasets.items():
            tr, te = train_test_split(np.arange(len(y)), test_size=.2, random_state=split_seed,
                                     stratify=y if kind == "classification" else None)
            scaler = StandardScaler().fit(X[tr])
            tx, vx = scaler.transform(X[tr]), scaler.transform(X[te])
            data[kind] = (tx, y[tr], vx, y[te], tr, te, scaler)
            splits.extend({"split_seed": split_seed, "dataset": kind, "subset": subset, "data_row": int(i+1)}
                          for subset, ids in [("train", tr), ("test", te)] for i in ids)
        # Validate reused baseline metrics using fresh fits on this exact split.
        tx, ty, vx, vy, _, _, _ = data["classification"]
        base = LogisticRegression(C=1., penalty="l2", max_iter=10000, tol=1e-10).fit(tx, ty)
        ref = next(r for r in references if r["split_seed"] == split_seed and r["model"] == "sklearn logistic L2")
        np.testing.assert_allclose(np.mean(base.predict(vx) == vy), ref["test_metric"], atol=1e-15)
        tx, ty, vx, vy, _, _, _ = data["regression"]
        ols = LinearRegression().fit(tx, ty)
        ref = next(r for r in references if r["split_seed"] == split_seed and r["model"] == "sklearn linear")
        np.testing.assert_allclose(np.mean(np.abs(ols.predict(vx)-vy)), ref["test_metric"], atol=1e-14)
        for sgd_seed in SGD_SEEDS:
            for name, kind, loss, link in sensitivity.SPECS:
                tx, ty, vx, vy, tr, te, scaler = data[kind]
                config = dict(experiment.CLASS_CONFIG if kind == "classification" else experiment.REG_CONFIG)
                torch.manual_seed(sgd_seed)
                snapshots, history = fit_checkpoints(tx, ty, loss, config, sgd_seed,
                    matched_l2=link == "logistic_l2", regression=kind == "regression")
                trajectory = f"{split_seed}/{sgd_seed}/{name}"
                histories.extend({"trajectory": trajectory, **h} for h in history)
                baseline_name = "sklearn logistic L2" if kind == "classification" else "sklearn linear"
                baseline = next(r for r in references if r["split_seed"] == split_seed and r["model"] == baseline_name)
                reference_name = {"logistic": "sklearn logistic no penalty", "logistic_l2": "sklearn logistic L2",
                                  "mse": "sklearn linear", "t5": "scipy t5 reference"}.get(link)
                reference = next((r for r in references if r["split_seed"] == split_seed and r["model"] == reference_name), None)
                for epoch, snapshot in snapshots.items():
                    w, b = snapshot["weights"], snapshot["bias"]
                    score, train_score = vx @ w + b, tx @ w + b
                    if kind == "classification":
                        prob = np.clip(stats.norm.cdf(score) if link == "probit" else special.expit(score), experiment.CLIP_EPS, 1-experiment.CLIP_EPS)
                        pred = (prob >= .5).astype(int)
                        value = float(np.mean(pred == vy))
                        train_metric = float(np.mean((train_score >= 0) == ty))
                        train_prob = np.clip(stats.norm.cdf(train_score) if link == "probit" else special.expit(train_score), experiment.CLIP_EPS, 1-experiment.CLIP_EPS)
                        objective = float(-np.mean(ty*np.log(train_prob)+(1-ty)*np.log1p(-train_prob)))
                        if link == "logistic_l2": objective += float(w@w/(2*len(ty)))
                    else:
                        pred, prob = score, [""] * len(vy)
                        value = float(np.mean(np.abs(pred-vy)))
                        train_metric = float(np.mean(np.abs(train_score-ty)))
                        objective = float(np.mean((train_score-ty)**2)) if link == "mse" else float(-stats.t.logpdf(ty-train_score, df=5, scale=1).mean())
                    np.testing.assert_allclose(objective, snapshot["training_objective"], rtol=1e-12, atol=1e-12)
                    run_id = f"{split_seed}/{epoch}/{sgd_seed}/{name}"
                    original_w = w/scaler.scale_
                    saved_parameters[run_id] = {"weights_standardized_X": w.tolist(), "bias_standardized_X": b,
                        "weights_original_X": original_w.tolist(), "bias_original_X": float(b-original_w@scaler.mean_)}
                    if epoch == 200:
                        old_id = f"{split_seed}/current/{sgd_seed}/{name}"
                        np.testing.assert_allclose(w, old_parameters[old_id]["weights_standardized_X"], rtol=0, atol=1e-13)
                        np.testing.assert_allclose(b, old_parameters[old_id]["bias_standardized_X"], rtol=0, atol=1e-13)
                        np.testing.assert_allclose(value, float(old_rows[old_id]["test_metric"]), rtol=0, atol=1e-13)
                        replicated += 1
                    row = {"run_id": run_id, "split_seed": split_seed, "sgd_seed": sgd_seed, "model": name, "dataset": kind,
                        **config, "epochs": epoch, "n_train": len(ty), "n_test": len(vy),
                        "metric": "accuracy" if kind == "classification" else "mae", "train_metric": train_metric,
                        "test_metric": value, "baseline_test_metric": baseline["test_metric"],
                        "delta_test_metric": value-baseline["test_metric"], "training_objective": objective,
                        "training_objective_gap": objective-reference["training_objective"] if reference else None,
                        "training_gradient_norm": snapshot["training_gradient_norm"]}
                    rows.append(row)
                    predictions.extend({"run_id": run_id, "dataset": kind, "data_row": int(i+1), "observed": float(yi),
                        "prediction": float(pi), "probability_y1": float(p) if kind == "classification" else ""}
                        for i, yi, pi, p in zip(te, vy, pred, prob))
            print(f"  SGD seed {sgd_seed} complete: {len(rows)}/225 checkpoints", flush=True)
        experiment.write_csv(ROOT / "runs.csv", rows)
    aggregated = summarize(rows)
    for filename, records in [("runs.csv", rows), ("aggregate.csv", aggregated), ("baseline_runs.csv", references),
                              ("test_predictions.csv", predictions), ("split.csv", splits), ("training_history.csv", histories)]:
        experiment.write_csv(ROOT / filename, records)
    (ROOT / "parameters.json").write_text(json.dumps(saved_parameters, indent=2) + "\n")
    make_figure(aggregated)
    write_report(aggregated)
    summary = {"protocol": "fixed hyperparameters; same trajectories at 200/500/1000; no CV or test-based selection",
        "split_seeds": SPLIT_SEEDS, "sgd_seeds": SGD_SEEDS, "epochs": CHECKPOINTS,
        "n_trajectories": 75, "n_checkpoints": len(rows), "replicated_original_200_epoch_runs": replicated,
        "classification_config": {k: v for k, v in experiment.CLASS_CONFIG.items() if k != 'epochs'},
        "regression_config": {k: v for k, v in experiment.REG_CONFIG.items() if k != 'epochs'},
        "versions": {"numpy": np.__version__, "scipy": experiment.scipy.__version__,
                     "sklearn": experiment.sklearn.__version__, "torch": torch.__version__},
        "loss_checks": checks, "aggregate": aggregated, "elapsed_seconds": time.perf_counter()-started,
        "breast_cancer_sha256": breast_hash, "ale_sha256": ale_hash,
        "probability_clip": [experiment.CLIP_EPS, 1-experiment.CLIP_EPS], "gradient_clip_norm": experiment.GRAD_CLIP_NORM}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (ROOT / "validation.json").write_text(json.dumps({"status": "passed", "loss_checks": "passed",
        "200_epoch_parameters_and_metrics_replicated": replicated, "fresh_baseline_metrics": "passed",
        "independent_numpy_scipy_training_objectives": "passed"}, indent=2) + "\n")
    print(f"Done in {summary['elapsed_seconds']:.1f} s: {ROOT / 'report.html'}", flush=True)
    for row in aggregated:
        print(row, flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        run()
    summary_path = ROOT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["warnings"] = sorted(set(str(w.message) for w in captured))
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
