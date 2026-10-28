#!/usr/bin/env python3
"""Homework 1, Problem 4: ALE regression and residual normality diagnostics.

Run with .venv/bin/python run_experiment.py from this folder. The script also
works from another directory: every artifact is resolved relative to this file.
Hyperparameters are selected by training-only five-fold CV, never by p-values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import time
import warnings

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

FEATURES = ["anchor_ratio", "trans_range", "node_density", "iterations"]
DATA = ROOT / "data" / "mcs_ds_edited_iter_shuffled.csv"
SOURCE = "https://archive.ics.uci.edu/dataset/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns"
ALPHA = 0.05
COLORS = ["#247ba0", "#e18b32", "#8465a8", "#388e68", "#ce5e79"]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data():
    with DATA.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FEATURES + ["ale", "sd_ale"]:
            raise ValueError(f"Unexpected dataset columns: {reader.fieldnames}")
        raw = list(reader)
    values = np.array([[float(row[col]) for col in FEATURES + ["ale", "sd_ale"]] for row in raw])
    if values.shape != (107, 6) or not np.isfinite(values).all():
        raise ValueError("Expected 107 complete, finite observations with 6 columns.")
    if np.unique(values[:, :4], axis=0).shape[0] != len(values):
        raise ValueError("Repeated feature configurations need a grouped split.")
    return values[:, :4], values[:, 4]


def scaled(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def scaled_target(model):
    # Both X and y scalers are fitted separately within every CV training fold.
    return TransformedTargetRegressor(regressor=scaled(model), transformer=StandardScaler())


def model_specs(seed):
    return [
        ("Linear regression", scaled(LinearRegression()), {}),
        ("RBF SVR", scaled_target(SVR(kernel="rbf")), {
            "regressor__model__C": [0.1, 1.0, 10.0, 100.0],
            "regressor__model__gamma": ["scale", 0.1, 1.0],
            "regressor__model__epsilon": [0.01, 0.1],
        }),
        ("Decision tree", DecisionTreeRegressor(random_state=seed), {
            "max_depth": [2, 3, 4, None], "min_samples_leaf": [2, 5, 10],
        }),
        ("Random forest", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1), {
            "max_depth": [4, None], "min_samples_leaf": [1, 3, 5],
        }),
        ("Neural network", scaled_target(MLPRegressor(
            activation="tanh", solver="lbfgs", max_iter=6000,
            max_fun=100000, tol=1e-7, random_state=seed,
        )), {
            "regressor__model__hidden_layer_sizes": [(8,), (16,)],
            "regressor__model__alpha": [0.01, 0.1, 1.0],
        }),
    ]


def diagnostics(y, prediction):
    residual = y - prediction
    if len(residual) < 20 or np.std(residual) == 0:
        raise ValueError("Normality diagnostics require at least 20 nonconstant residuals here.")
    k2, p = stats.normaltest(residual)
    w, shapiro_p = stats.shapiro(residual)
    return {
        "n": len(y), "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)), "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)),
        "skewness": float(stats.skew(residual, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(residual, bias=False)),
        "normaltest_statistic": float(k2), "normaltest_p": float(p),
        "shapiro_statistic": float(w), "shapiro_p": float(shapiro_p),
    }


def holm_adjust(pvalues):
    """Holm adjustment for the five primary, held-out normaltest p-values."""
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def plot_panels(names, predictions, y, subset, results, figure_dir):
    for kind in ["qq", "residual_vs_fitted"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
        for i, (name, ax) in enumerate(zip(names, axes.flat)):
            pred = predictions[name][subset]
            residual = y - pred
            p = next(row["normaltest_p"] for row in results if row["model"] == name and row["subset"] == subset)
            if kind == "qq":
                (normal_quantiles, ordered), (slope, intercept, _) = stats.probplot(residual, dist="norm")
                ax.scatter(normal_quantiles, ordered, s=30, color=COLORS[i], alpha=0.85)
                ax.plot(normal_quantiles, intercept + slope * normal_quantiles, color="#525866", lw=1.4)
                ax.set_xlabel("Theoretical normal quantile")
                ax.set_ylabel("Ordered residual (m)")
            else:
                ax.scatter(pred, residual, s=32, color=COLORS[i], alpha=0.85)
                ax.axhline(0, color="#525866", lw=1.2)
                ax.set_xlabel("Predicted ALE (m)")
                ax.set_ylabel("Residual = observed - predicted (m)")
            ax.set_title(f"{name}\nnormaltest p = {p:.4g}", fontsize=11)
            ax.grid(alpha=0.2)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0.04, 0.85,
            f"{subset.capitalize()} residuals | n = {len(y)}\n\n"
            + ("Points near the line support\napproximate marginal normality.\n\nThe line fits location and scale;\nit does not test zero mean."
               if kind == "qq" else "Check for curvature, unequal spread,\nand large prediction errors.\n\nA normality test alone does not\ncheck these model assumptions."),
            transform=axes.flat[-1].transAxes, va="top", fontsize=12, linespacing=1.6)
        fig.suptitle(f"ALE regression: {subset} {'normal Q-Q plots' if kind == 'qq' else 'residual diagnostics'}", fontsize=17)
        fig.savefig(figure_dir / f"{kind}_{subset}.png", dpi=170)
        plt.close(fig)

    if subset == "test":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for i, name in enumerate(names):
            pred = predictions[name][subset]
            axes[0].scatter(y, pred, s=32, alpha=0.6, color=COLORS[i], label=name)
        limits = [float(min(y.min(), min(predictions[n][subset].min() for n in names))) - 0.05,
                  float(max(y.max(), max(predictions[n][subset].max() for n in names))) + 0.05]
        axes[0].plot(limits, limits, color="#525866", ls="--", lw=1)
        axes[0].set(xlabel="Observed ALE (m)", ylabel="Predicted ALE (m)", title="Held-out predictions")
        axes[0].legend(fontsize=8)
        rows = [next(r for r in results if r["model"] == n and r["subset"] == subset) for n in names]
        axes[1].barh(names, [r["rmse"] for r in rows], color=COLORS)
        axes[1].invert_yaxis()
        axes[1].set(xlabel="Test RMSE (m)", title="Prediction error (lower is better)")
        for i, row in enumerate(rows):
            axes[1].text(row["rmse"] + 0.004, i, f"{row['rmse']:.3f}", va="center", fontsize=10)
        axes[1].set_xlim(0, max(r["rmse"] for r in rows) * 1.25)
        for ax in axes:
            ax.grid(alpha=0.18)
        fig.savefig(figure_dir / "test_performance.png", dpi=170)
        plt.close(fig)


def pformat(value):
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def write_report(summary, rows, best_params):
    test = [r for r in rows if r["subset"] == "test"]
    train = [r for r in rows if r["subset"] == "train"]
    winner = min(test, key=lambda r: r["rmse"])
    rejected = [r["model"] for r in test if r["reject_normality_holm"]]
    failed = [r["model"] for r in test if not r["reject_normality_holm"]]
    raw_rejected = [r["model"] for r in test if r["reject_normality_raw"]]
    esc = html.escape
    def table(data, primary):
        header = "<tr><th>模型</th><th>n</th><th>MAE</th><th>RMSE</th><th>R²</th><th>K²</th><th>p (原始)</th>"
        header += "<th>p (Holm)</th><th>Shapiro p</th><th>结论 (Holm)</th></tr>" if primary else "<th>Shapiro p</th></tr>"
        body = []
        for r in data:
            text = "".join(f"<td>{r[k]:.4f}</td>" for k in ["mae", "rmse", "r2", "normaltest_statistic"])
            row = f"<tr><td>{esc(r['model'])}</td><td>{r['n']}</td>{text}<td>{pformat(r['normaltest_p'])}</td>"
            if primary:
                decision = "拒绝正态性" if r["reject_normality_holm"] else "未拒绝正态性"
                row += f"<td>{pformat(r['normaltest_p_holm'])}</td><td>{pformat(r['shapiro_p'])}</td><td>{decision}</td>"
            else:
                row += f"<td>{pformat(r['shapiro_p'])}</td>"
            body.append(row + "</tr>")
        return f"<div class='scroll'><table>{header}{''.join(body)}</table></div>"
    reject_text = "、".join(rejected) if rejected else "没有模型"
    conclusion = f"在 α = 0.05 下，五个主检验做 Holm 校正后，{reject_text}拒绝残差的正态性假设。"
    if not rejected:
        conclusion = "在 α = 0.05 下，五个主检验做 Holm 校正后，所有模型均未拒绝残差的正态性假设。"
    if failed:
        conclusion += " 对于" + "、".join(failed) + "，证据不足以拒绝正态性；这不能证明残差服从正态分布。"
    params_html = "".join(f"<tr><td>{esc(n)}</td><td><code>{esc(json.dumps(p, ensure_ascii=False))}</code></td></tr>" for n, p in best_params.items())
    caution = "本次拟合和调参没有产生警告。" if not summary["warnings"] else f"本次拟合/调参产生 {len(summary['warnings'])} 条不同警告，详见 results/summary.json；报告时应保留这一限制。"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Problem 4 · ALE 回归与残差正态性实验</title><style>
body{{font-family:system-ui,-apple-system,'PingFang SC',sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#253044;line-height:1.75;background:#fcfcfd}}
h1{{font-size:30px;line-height:1.35}} h2{{font-size:21px;margin-top:34px}} a{{color:#247ba0}} code{{font-size:13px;word-break:break-word}}
.lead{{border-left:4px solid #247ba0;background:#edf5fa;padding:16px 20px}} .scroll{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}} th,td{{border-bottom:1px solid #dfe4ed;padding:10px 9px;text-align:left;white-space:nowrap}}
th{{background:#edf1f6}} img{{width:100%;height:auto;border:1px solid #e3e8ef;margin-top:10px}} .note{{color:#57677b;font-size:14px}} pre{{background:#edf1f6;padding:16px;overflow-x:auto}}
@media print{{body{{max-width:none;margin:0;font-size:11px}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}
</style></head><body>
<p class="note">Machine Learning 1 · Homework 1 · Problem 4</p>
<h1>ALE 回归模型与残差正态性实验</h1>
<p class="lead">{conclusion}<br>本次留出集上 RMSE 最小的是 <b>{esc(winner['model'])}</b>（RMSE = {winner['rmse']:.4f} m，MAE = {winner['mae']:.4f} m，R² = {winner['r2']:.4f}）。这是本次比较的描述结果，并非独立确认的模型选择结论。</p>
<h2>1. 题目与数据</h2>
<p>按 main.pdf 第 4 题，用前四个变量 <code>{', '.join(FEATURES)}</code> 预测 <code>ale</code>，忽略 <code>sd_ale</code>。来自 <a href="{SOURCE}">UCI ALE 数据集</a>的原始 CSV 有 107 行、6 列，无缺失值，ALE 单位为米；数据来自传感器定位的模拟实验。四个特征组合均唯一。</p>
<h2>2. 实验方法</h2>
<ol><li>固定 <code>random_state={summary['seed']}</code>，随机 80/20 划分：训练集 {summary['n_train']} 行，测试集 {summary['n_test']} 行。各模型共享同一划分，行号保存在 <code>results/split.csv</code>。</li>
<li>比较线性回归、RBF 支持向量回归、决策树、随机森林与单隐藏层神经网络。后四个模型只在训练集上进行 5 折交叉验证网格搜索，以平均验证 RMSE 最小为调参标准。</li>
<li>线性回归、SVR 与神经网络通过 Pipeline 标准化 X；SVR 与神经网络也标准化 y，预测后还原为米。所有变换仅在对应训练折上拟合。树模型不标准化。</li>
<li>定义残差 <code>r = y - ŷ</code>。以测试残差的 <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html">D'Agostino-Pearson normaltest</a> 为主检验，H₀ 为残差样本来自某一正态分布，H₁ 为偏离正态；α = 0.05。该检验合并偏度和峰度，并不检验均值是否为零。</li>
<li>同时比较五个模型，对五个测试集主检验的 p 值应用 Holm 校正；保留原始 p 值以便与题目直接对应。Shapiro-Wilk 检验仅作辅助诊断，表中是未经校正的 p 值。训练集检验也仅作补充。</li></ol>
<p class="note">预处理遵循 <a href="https://scikit-learn.org/stable/common_pitfalls.html">scikit-learn 关于训练/测试隔离的建议</a>。{caution}</p>
<h2>3. 测试集结果（主结果）</h2>
{table(test, True)}
<p class="note">MAE、RMSE 单位均为米，K² 是 normaltest 的统计量。原始 p &lt; 0.05 的个别判断不一定能在五模型的多重比较校正后保留。</p>
<p>如果按题目的单模型检验直接比较原始 p 与 0.05，本次拒绝正态性的模型是：{esc('、'.join(raw_rejected) or '无')}。本报告另给出 Holm 校正，以区分单模型判断与同时比较五个模型的判断。</p>
<img src="figures/qq_test.png" alt="五种模型测试残差的正态 QQ 图">
<p>QQ 图比较残差分位数与标准正态理论分位数，灰线拟合位置和尺度。点接近直线支持近似正态，明显尾部偏离或弯曲提示非正态；应结合检验结果而非仅凭 p 值判断。</p>
<img src="figures/residual_vs_fitted_test.png" alt="测试残差与预测值的关系">
<img src="figures/test_performance.png" alt="测试集预测及模型 RMSE 比较">
<h2>4. 训练集结果（补充）</h2>
{table(train, False)}
<p>训练残差受模型拟合影响，可能被灵活模型人为压缩，因此不能替代留出集诊断。训练误差明显小于测试误差可能提示过拟合。训练集 p 值未做多重比较校正，也不是主结论的依据。</p>
<img src="figures/qq_train.png" alt="训练残差的正态 QQ 图">
<h2>5. 结论与解释</h2>
<p>{conclusion}</p>
<p>仅 22 个测试残差使检验的统计功效有限。所有测试观测共享同一训练模型，而训练残差还受到拟合约束；正态性检验的常规 p 值在这里应作为近似诊断。即使边际残差接近正态，也不能据此断定条件误差处处正态、同方差、独立或均值为零。</p>
<p>拒绝正态性只能说明“高斯误差似然”这一平方损失的解释在该模型下缺少支持，不能推出平方损失不可用。不依赖正态性，条件均值 E[Y|X] 仍然是平方损失下的最优预测（在二阶矩有限时）。模型遗漏非线性关系也可能导致非正态残差。预测误差较低与残差较正态是两个不同的评价目标。</p>
<h2>6. 选定超参数与复现</h2>
<div class="scroll"><table><tr><th>模型</th><th>训练集 CV 选定参数</th></tr>{params_html}</table></div>
<p>未列出的固定参数、完整候选网格、CV 得分、版本和数据哈希保存在 <code>results/summary.json</code> 与 <code>results/cv_*.csv</code>。神经网络使用 tanh 与 L-BFGS；随机森林使用 200 棵树。</p>
<pre>cd problem4_ale
.venv/bin/python run_experiment.py</pre>
<p>可离线重跑；脚本默认读取本地原始数据，自动覆盖由脚本生成的结果、图像及本报告。环境安装步骤见 README.md。</p>
<h2>数据来源</h2>
<p class="note">Singh, A., Kotiyal, V., Sharma, S., Nagar, J., &amp; Lee, C. (2020). Average Localization Error (ALE) in sensor node localization process in WSNs. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5ZP6R">DOI: 10.24432/C5ZP6R</a>，CC BY 4.0。本地保存原始 CSV 和下载 ZIP，获取日期：2026-10-05。</p>
</body></html>"""
    (ROOT / "report.html").write_text(document, encoding="utf-8")


def run(seed):
    started = time.perf_counter()
    output = ROOT / "results"
    figures = ROOT / "figures"
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    X, y = load_data()
    train_ids, test_ids = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    assert len(set(train_ids) & set(test_ids)) == 0
    assert sorted(np.concatenate([train_ids, test_ids]).tolist()) == list(range(len(y)))
    train_set = set(train_ids)
    write_csv(output / "split.csv", [{"data_row": i + 1, "csv_line": i + 2,
        "subset": "train" if i in train_set else "test"} for i in range(len(y))])
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows, prediction_rows, predictions, params, searches, observed_warnings = [], [], {}, {}, {}, []
    names = []
    specs = model_specs(seed)
    for number, (name, estimator, grid) in enumerate(specs, 1):
        print(f"[{number}/{len(specs)}] Fitting {name} ...", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if grid:
                search = GridSearchCV(estimator, grid, scoring="neg_root_mean_squared_error",
                    cv=cv, n_jobs=1, refit=True, error_score="raise", return_train_score=True)
                search.fit(X[train_ids], y[train_ids])
                fitted = search.best_estimator_
                params[name] = search.best_params_
                cv_rmse = float(-search.best_score_)
                cv_rows = [{"params": json.dumps(p), "mean_validation_rmse": float(-v),
                    "std_validation_rmse": float(s), "mean_training_rmse": float(-t), "rank": int(rank)}
                    for p, v, s, t, rank in zip(search.cv_results_["params"],
                        search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"],
                        search.cv_results_["mean_train_score"], search.cv_results_["rank_test_score"])]
                write_csv(output / f"cv_{number:02d}.csv", cv_rows)
            else:
                scores = -cross_val_score(estimator, X[train_ids], y[train_ids], cv=cv,
                    scoring="neg_root_mean_squared_error", error_score="raise")
                fitted = estimator.fit(X[train_ids], y[train_ids])
                params[name] = {}
                cv_rmse = float(np.mean(scores))
                write_csv(output / f"cv_{number:02d}.csv", [
                    {"fold": i + 1, "validation_rmse": float(s)} for i, s in enumerate(scores)])
        observed_warnings.extend({"model": name, "category": w.category.__name__, "message": str(w.message)} for w in caught)
        names.append(name)
        predictions[name] = {}
        searches[name] = {"candidate_grid": grid, "selected_params": params[name], "cv_rmse": cv_rmse,
                         "fitted_estimator": repr(fitted)}
        for subset, ids in [("train", train_ids), ("test", test_ids)]:
            pred = fitted.predict(X[ids])
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions: {name}, {subset}")
            predictions[name][subset] = pred
            row = {"model": name, "subset": subset, **diagnostics(y[ids], pred),
                "normaltest_p_holm": "", "reject_normality_raw": bool(stats.normaltest(y[ids] - pred).pvalue < ALPHA),
                "reject_normality_holm": "", "training_cv_rmse": cv_rmse}
            rows.append(row)
            prediction_rows.extend({"model": name, "subset": subset, "data_row": int(i + 1),
                "observed_ale": float(yi), "predicted_ale": float(pi), "residual": float(yi - pi)}
                for i, yi, pi in zip(ids, y[ids], pred))
        print(f"    test RMSE={rows[-1]['rmse']:.4f}, normaltest p={rows[-1]['normaltest_p']:.6g}", flush=True)
    test = [r for r in rows if r["subset"] == "test"]
    for row, adjusted in zip(test, holm_adjust([r["normaltest_p"] for r in test])):
        row["normaltest_p_holm"] = float(adjusted)
        row["reject_normality_holm"] = bool(adjusted < ALPHA)
    write_csv(output / "metrics_normality.csv", rows)
    write_csv(output / "predictions.csv", prediction_rows)
    for subset, ids in [("train", train_ids), ("test", test_ids)]:
        plot_panels(names, predictions, y[ids], subset, rows, figures)
    # De-duplicate repetitive warnings but never silently suppress their existence.
    unique_warnings = [json.loads(s) for s in sorted({json.dumps(w) for w in observed_warnings})]
    summary = {"seed": seed, "test_fraction": 0.2, "alpha": ALPHA,
        "n_train": len(train_ids), "n_test": len(test_ids), "features": FEATURES, "target": "ale",
        "ignored_column": "sd_ale", "source": SOURCE,
        "csv_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "primary_normality_subset": "test", "multiple_testing": "Holm across five test normaltest p-values",
        "selection": "training-only 5-fold CV; minimum mean validation RMSE",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "models": searches, "metrics": rows, "warnings": unique_warnings,
        "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, rows, params)
    print(f"Done: {ROOT / 'report.html'}", flush=True)
    print(f"Warnings: {len(unique_warnings)}; elapsed: {summary['elapsed_seconds']} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Train/test, CV and estimator seed (default: 42).")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.seed)
