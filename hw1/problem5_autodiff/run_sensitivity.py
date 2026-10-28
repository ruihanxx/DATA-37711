#!/usr/bin/env python3
"""Direct-training sensitivity experiment; no CV and no winning-run selection.

Five paired train/test splits, three independent minibatch seeds, four fixed
settings, four assignment models and one logistic model with matched L2.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings

import run_experiment as experiment
import numpy as np
from scipy import optimize, special, stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from threadpoolctl import threadpool_limits

ROOT = experiment.ROOT / "sensitivity"
SPLIT_SEEDS = [0, 1, 7, 21, 42]
SGD_SEEDS = [0, 1, 42]
PRESETS = {
    "short": {"epochs": 50, "class_batch": 64, "reg_batch": 32, "class_lr": .03, "reg_lr": .01, "momentum": 0.},
    "current": {"epochs": 200, "class_batch": 64, "reg_batch": 32, "class_lr": .05, "reg_lr": .03, "momentum": 0.},
    "larger_lr": {"epochs": 200, "class_batch": 64, "reg_batch": 32, "class_lr": .1, "reg_lr": .05, "momentum": 0.},
    "momentum": {"epochs": 200, "class_batch": 32, "reg_batch": 16, "class_lr": .05, "reg_lr": .03, "momentum": .9},
}
SPECS = [
    ("Logistic BCE", "classification", experiment.logistic_bce, "logistic"),
    ("Probit BCE", "classification", experiment.probit_bce, "probit"),
    ("Logistic BCE + matched L2", "classification", experiment.logistic_bce, "logistic_l2"),
    ("Linear MSE", "regression", experiment.mse_loss, "mse"),
    ("Linear t5 NLL", "regression", experiment.student_t_nll, "t5"),
]


def fit_matched_l2(X, y, config, seed):
    """Match sklearn C=1: mean BCE + ||w||^2/(2*n), excluding bias."""
    model = nn.Linear(X.shape[1], 1, dtype=experiment.DTYPE)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    tx, ty = torch.tensor(X, dtype=experiment.DTYPE), torch.tensor(y, dtype=experiment.DTYPE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["initial_lr"], momentum=config["momentum"])
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(config["epochs"]):
        optimizer.param_groups[0]["lr"] = config["initial_lr"]/(1+epoch/config["decay_epochs"])
        for ids in torch.randperm(len(y), generator=generator).split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = experiment.logistic_bce(model(tx[ids]).squeeze(-1), ty[ids]) + model.weight.square().sum()/(2*len(y))
            if not torch.isfinite(loss): raise ValueError("Non-finite L2 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), experiment.GRAD_CLIP_NORM, error_if_nonfinite=True)
            optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    objective = experiment.logistic_bce(model(tx).squeeze(-1), ty) + model.weight.square().sum()/(2*len(y))
    objective.backward()
    norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
    return model, norm


def bce_numpy(score, y):
    p = np.clip(special.expit(score), experiment.CLIP_EPS, 1-experiment.CLIP_EPS)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def t_reference(X, y, initial):
    A = np.c_[X, np.ones(len(y))]
    def loss(beta): return float(-stats.t.logpdf(y-A@beta, df=5, scale=1).mean())
    def gradient(beta):
        r = y-A@beta
        return -A.T@(6*r/(5+r*r))/len(y)
    result = optimize.minimize(loss, initial, jac=gradient, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    if np.linalg.norm(gradient(result.x)) > 1e-6:
        raise ValueError("Student-t reference has not converged")
    return result.x, loss(result.x), float(np.linalg.norm(gradient(result.x)))


def aggregate(rows, references):
    output = []
    for preset in PRESETS:
        for name, kind, _, _ in SPECS:
            group = [r for r in rows if r["preset"]==preset and r["model"]==name]
            value = np.array([r["test_metric"] for r in group])
            delta = np.array([r["delta_test_metric"] for r in group])
            gaps = [r["training_objective_gap"] for r in group if r["training_objective_gap"] is not None]
            within = [np.std([r["test_metric"] for r in group if r["split_seed"]==seed], ddof=1) for seed in SPLIT_SEEDS]
            output.append({"preset": preset, "model": name, "dataset": kind, "n_paired_runs": len(group),
                "metric": "accuracy" if kind=="classification" else "mae",
                "mean_test_metric": float(value.mean()), "sd_test_metric": float(value.std(ddof=1)),
                "mean_baseline_metric": float(np.mean([r["baseline_test_metric"] for r in group])),
                "mean_delta": float(delta.mean()), "sd_delta": float(delta.std(ddof=1)),
                "min_delta": float(delta.min()), "max_delta": float(delta.max()),
                "mean_within_split_sgd_sd": float(np.mean(within)),
                "mean_training_objective_gap": float(np.mean(gaps)) if gaps else "",
                "max_training_objective_gap": float(np.max(gaps)) if gaps else "",
                "mean_max_test_prediction_difference": float(np.mean([r["max_test_prediction_difference"] for r in group])) if kind=="regression" else ""})
    return output


def make_figure(aggregated):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"serif", "font.size":10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    panels = [("classification", ["Logistic BCE", "Probit BCE", "Logistic BCE + matched L2"], 100., "Accuracy difference (percentage points)"),
              ("regression", ["Linear MSE", "Linear t5 NLL"], 1., "MAE difference (m)")]
    colors = ["#247ba0", "#ce5e79", "#388e68"]
    for ax, (kind, names, scale, ylabel) in zip(axes, panels):
        for i, (name, color) in enumerate(zip(names, colors)):
            selected = [next(r for r in aggregated if r["model"]==name and r["preset"]==p) for p in PRESETS]
            xpos = np.arange(len(PRESETS)) + (i-(len(names)-1)/2)*.16
            ax.errorbar(xpos, [r["mean_delta"]*scale for r in selected],
                yerr=[r["sd_delta"]*scale for r in selected], fmt="o", capsize=4, color=color, label=name)
        ax.axhline(0, color="#555", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(PRESETS)), list(PRESETS))
        ax.set(xlabel="Fixed settings", ylabel=ylabel,
               title="Classification: above zero is better" if kind=="classification" else "Regression: below zero is better")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("PyTorch minus sklearn: mean and SD over 15 paired runs", fontsize=14)
    fig.savefig(ROOT/"comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(aggregated, references):
    import html
    def fmt(value, kind, delta=False):
        return f"{100*value:+.2f} pp" if kind=="classification" and delta else f"{100*value:.2f}%" if kind=="classification" else f"{value:+.6f}" if delta else f"{value:.6f}"
    body = []
    for row in aggregated:
        kind=row["dataset"]
        body.append(f"<tr><td>{row['preset']}</td><td>{html.escape(row['model'])}</td><td>{fmt(row['mean_baseline_metric'],kind)}</td><td>{fmt(row['mean_test_metric'],kind)}</td><td>{fmt(row['mean_delta'],kind,True)}</td><td>{fmt(row['min_delta'],kind,True)} / {fmt(row['max_delta'],kind,True)}</td></tr>")
    settings = "".join(f"<tr><td>{p}</td><td>{s['epochs']}</td><td>{s['class_lr']}</td><td>{s['reg_lr']}</td><td>{s['class_batch']} / {s['reg_batch']}</td><td>{s['momentum']}</td></tr>" for p,s in PRESETS.items())
    numerical = []
    for row in aggregated:
        if row["mean_training_objective_gap"] != "":
            numerical.append(f"<tr><td>{row['preset']}</td><td>{html.escape(row['model'])}</td><td>{row['mean_training_objective_gap']:.3e}</td><td>{row['max_training_objective_gap']:.3e}</td></tr>")
    current = {r['model']:r for r in aggregated if r['preset']=='current'}
    short = {r['model']:r for r in aggregated if r['preset']=='short'}
    momentum = {r['model']:r for r in aggregated if r['preset']=='momentum'}
    l2 = current['Logistic BCE + matched L2']
    logistic, probit = current['Logistic BCE'], current['Probit BCE']
    mse, student = current['Linear MSE'], current['Linear t5 NLL']
    unpenalized_mean = np.mean([r['test_metric'] for r in references if r['model']=='sklearn logistic no penalty'])
    findings = f"""<h2>实测结论</h2>
<p>当前固定参数下，sklearn logistic 的平均准确率为 {100*logistic['mean_baseline_metric']:.2f}%，PyTorch logistic 为 {100*logistic['mean_test_metric']:.2f}%（差 {100*logistic['mean_delta']:+.2f} 个百分点），probit 为 {100*probit['mean_test_metric']:.2f}%（差 {100*probit['mean_delta']:+.2f} 个百分点）。匹配 L2 的 PyTorch 对照为 {100*l2['mean_test_metric']:.2f}%，与 baseline 的平均差为 {100*l2['mean_delta']:+.2f} 个百分点。无 L2 的 sklearn 数值拟合参考平均准确率为 {100*unpenalized_mean:.2f}%，说明更充分地降低无惩罚训练损失不保证测试准确率更好。</p>
<p>当前 MSE SGD 的平均 MAE 为 {mse['mean_test_metric']:.6f} m，baseline 为 {mse['mean_baseline_metric']:.6f} m，差 {mse['mean_delta']:+.6f} m；平均训练 MSE 距最小二乘最优值仅 {mse['mean_training_objective_gap']:.2e}。这与同一训练目标应得到接近的解相符。t 损失的平均 MAE 为 {student['mean_test_metric']:.6f} m，较 baseline 差 {student['mean_delta']:+.6f} m，但各次差值范围为 {student['min_delta']:+.6f} 至 {student['max_delta']:+.6f} m，并非每次都更好。</p>
<p>参数会改变优化误差和随机波动：50 轮配置的 MSE 平均训练目标差为 {short['Linear MSE']['mean_training_objective_gap']:.2e}，明显大于当前 200 轮配置。它的平均测试 MAE 虽然更低，但不能据此宣称算法普遍优于最小二乘。momentum 配置同时改动 momentum 和 batch，固定划分下、仅改变打乱种子时，MSE MAE 的平均标准差为 {momentum['Linear MSE']['mean_within_split_sgd_sd']:.6f} m，当前配置为 {mse['mean_within_split_sgd_sd']:.6f} m，前者的随机波动更大。</p>
<p>这些成对实验与数值核验支持：当前配置的差距基本合理，未发现 PyTorch 实现造成系统性劣势；应结合训练目标是否充分优化、正则化是否匹配和不同划分的波动解释结果。</p>"""
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Problem 5：参数与随机种子敏感性</title>
<style>body{{font-family:system-ui,-apple-system,'PingFang SC',sans-serif;max-width:1160px;margin:32px auto;padding:0 20px;color:#243047;line-height:1.75}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #dde3eb}}th{{background:#eef3f8}}img{{width:100%}}.scroll{{overflow-x:auto}}code{{background:#f0f3f6}}h2{{margin-top:32px}}</style>
<h1>Problem 5：直接训练的参数与随机种子敏感性</h1>
<p>数据划分种子为 {SPLIT_SEEDS}，SGD mini-batch 打乱种子为 {SGD_SEEDS}。每个设置对应 15 次成对实验，共 300 次 PyTorch 训练；每次均使用完整训练集，不做交叉验证，不根据测试分数筛选“最好的一次”。分类为 455/114，回归为 85/22。特征标准化仅拟合训练数据，保留概率 clip 和梯度 clip。</p>
{findings}
<h2>预先固定的超参数</h2><table><tr><th>设置</th><th>Epoch</th><th>分类初始 LR</th><th>回归初始 LR</th><th>分类 / 回归 Batch</th><th>Momentum</th></tr>{settings}</table>
<p>学习率均按 <code>lr0/(1+epoch_index/100)</code> 衰减；权重初始化为零。SGD seed 改变 mini-batch 顺序，split seed 改变训练/测试划分。</p>
<h2>与 baseline 相差多少</h2><p>分类的 baseline 是默认 L2、C=1 的 sklearn logistic，回归的 baseline 是 sklearn 最小二乘线性回归。表中差值为 PyTorch − baseline，分类以百分点（pp）表示、回归以米表示。均值和范围均来自同一划分下的成对比较。</p>
<div class="scroll"><table><tr><th>设置</th><th>PyTorch 模型</th><th>Baseline 均值</th><th>PyTorch 均值</th><th>平均差值</th><th>最小 / 最大差值</th></tr>{''.join(body)}</table></div>
<img src="comparison.png" alt="各超参数设置下成对指标差的均值和标准差">
<p>误差线为 15 次成对实验的标准差，用于描述稳定性，不是置信区间。同一划分的三次训练共享测试集，不同划分也有样本重叠，因此没有把这些结果当成 15 个独立样本做显著性检验。</p>
<h2>这些差异是否合理</h2>
<p>不能仅凭一个测试分数判断实现是否正确。分类测试集只有 114 条，每多错一条就相差约 0.88 个百分点。默认 sklearn logistic 带 L2，而作业的 PyTorch logistic 只有 BCE，两者目标不同；另外保存了无 L2 的 sklearn 对照，并增加匹配 C=1 的 PyTorch L2 诊断模型，其目标为平均 BCE + ||w||²/(2n)，不惩罚偏置。</p>
<p>无惩罚 logistic 在可分或近可分训练集上可能没有有限、稳定的极大似然参数；这里的 sklearn 无 L2 结果是数值拟合参考，不要求有限步 SGD 与其系数完全一致。Probit 改变了概率链接，也不应要求测试准确率与 logistic 完全相同。</p>
<p>MSE 线性回归与 sklearn 最小二乘的训练目标相同，应在充分优化后接近同一个解。下表直接比较训练目标值与数值参考解；较大的目标差说明训练预算或优化参数不足，不能把差异全部归因于样本波动。t 损失用独立 SciPy BFGS 求解作为优化参考，其训练目标与 MSE 不同，因此不保证测试 MAE 更小。</p>
<table><tr><th>设置</th><th>模型</th><th>平均训练目标差</th><th>最大训练目标差</th></tr>{''.join(numerical)}</table>
<h2>结果文件</h2><p><code>runs.csv</code> 保存每次训练的指标、成对差值、梯度范数和目标差；<code>baseline_runs.csv</code> 保存各划分的 baseline；<code>parameters.json</code> 保存模型参数；<code>test_predictions.csv</code> 和 <code>split.csv</code> 可用于重算指标；<code>aggregate.csv</code> 是汇总。既有单次作业实验保存在上一级目录。</p>
</html>"""
    (ROOT/"report.html").write_text(document,encoding="utf-8")


def run():
    started=time.perf_counter()
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    checks=experiment.check_losses()
    breast=load_breast_cancer()
    ale=np.genfromtxt(experiment.ROOT/"data/mcs_ds_edited_iter_shuffled.csv",delimiter=",",skip_header=1)
    datasets={"classification":(breast.data,breast.target),"regression":(ale[:,:4],ale[:,4])}
    runs,baseline_rows,predictions,split_rows,parameters=[],[],[],[],{}
    for split_seed in SPLIT_SEEDS:
        print(f"Data split seed {split_seed}",flush=True)
        data={}
        for kind,(X,y) in datasets.items():
            tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=split_seed,stratify=y if kind=="classification" else None)
            scaler=StandardScaler().fit(X[tr])
            data[kind]=(scaler.transform(X[tr]),y[tr],scaler.transform(X[te]),y[te],tr,te,scaler)
            split_rows.extend({"split_seed":split_seed,"dataset":kind,"subset":subset,"data_row":int(i+1)} for subset,ids in [("train",tr),("test",te)] for i in ids)
        tx,ty,vx,vy,tr,te,scaler=data['classification']
        l2=LogisticRegression(C=1.,penalty='l2',max_iter=10000,tol=1e-10).fit(tx,ty)
        none=LogisticRegression(penalty=None,max_iter=10000,tol=1e-8).fit(tx,ty)
        baselines={}
        for label,model in [('sklearn logistic L2',l2),('sklearn logistic no penalty',none)]:
            prob=model.predict_proba(vx)[:,1]; pred=(prob>=.5).astype(int)
            objective=bce_numpy(model.decision_function(tx),ty)+(np.sum(model.coef_**2)/(2*len(ty)) if model is l2 else 0.)
            row={"split_seed":split_seed,"dataset":"classification","model":label,"n_train":len(ty),"n_test":len(vy),"metric":"accuracy","test_metric":float(np.mean(pred==vy)),"training_objective":objective,"iterations":int(model.n_iter_[0]),"reference_gradient_norm":""}
            baseline_rows.append(row); baselines[label]=(model,row)
            predictions.extend({"run_id":f"baseline/{split_seed}/{label}","dataset":"classification","data_row":int(i+1),"observed":float(yi),"prediction":float(pi),"probability_y1":float(p)} for i,yi,pi,p in zip(te,vy,pred,prob))
        tx,ty,vx,vy,tr,te,scaler=data['regression']
        ols=LinearRegression().fit(tx,ty)
        beta_t,obj_t,grad_t=t_reference(tx,ty,np.r_[ols.coef_,ols.intercept_])
        for label,beta,obj,grad in [('sklearn linear',np.r_[ols.coef_,ols.intercept_],float(np.mean((ty-ols.predict(tx))**2)),0.),('scipy t5 reference',beta_t,obj_t,grad_t)]:
            pred=vx@beta[:-1]+beta[-1]
            row={"split_seed":split_seed,"dataset":"regression","model":label,"n_train":len(ty),"n_test":len(vy),"metric":"mae","test_metric":float(np.mean(np.abs(pred-vy))),"training_objective":obj,"iterations":"","reference_gradient_norm":grad}
            baseline_rows.append(row); baselines[label]=(beta,row)
            predictions.extend({"run_id":f"baseline/{split_seed}/{label}","dataset":"regression","data_row":int(i+1),"observed":float(yi),"prediction":float(pi),"probability_y1":""} for i,yi,pi in zip(te,vy,pred))
        for preset,settings in PRESETS.items():
            for sgd_seed in SGD_SEEDS:
                for name,kind,loss,link in SPECS:
                    tx,ty,vx,vy,tr,te,scaler=data[kind]
                    config={"epochs":settings['epochs'],"batch_size":settings['class_batch'] if kind=='classification' else settings['reg_batch'],"initial_lr":settings['class_lr'] if kind=='classification' else settings['reg_lr'],"momentum":settings['momentum'],"decay_epochs":100}
                    torch.manual_seed(sgd_seed)
                    fit_started=time.perf_counter()
                    if link=='logistic_l2':model,gradient=fit_matched_l2(tx,ty,config,sgd_seed)
                    else:model,_,gradient=experiment.fit_sgd(name,tx,ty,loss,config,initial_bias=float(ty.mean()) if kind=='regression' else 0.,seed=sgd_seed,verbose=False)
                    score=experiment.linear_score(model,vx); train_score=experiment.linear_score(model,tx)
                    w=model.weight.detach().numpy().ravel()
                    if kind=='classification':
                        prob=np.clip(stats.norm.cdf(score) if link=='probit' else special.expit(score),experiment.CLIP_EPS,1-experiment.CLIP_EPS)
                        pred=(prob>=.5).astype(int); value=float(np.mean(pred==vy)); train_metric=float(np.mean((train_score>=0)==ty))
                        objective=float(loss(torch.tensor(train_score,dtype=experiment.DTYPE),torch.tensor(ty,dtype=experiment.DTYPE)))+(np.sum(w*w)/(2*len(ty)) if link=='logistic_l2' else 0.)
                        baseline=baselines['sklearn logistic L2'][1]
                        reference=baselines['sklearn logistic no penalty' if link=='logistic' else 'sklearn logistic L2'][1] if link!='probit' else None
                        pred_difference=None
                    else:
                        pred=score;prob=['']*len(vy);value=float(np.mean(np.abs(pred-vy)));train_metric=float(np.mean(np.abs(train_score-ty)))
                        objective=float(loss(torch.tensor(train_score,dtype=experiment.DTYPE),torch.tensor(ty,dtype=experiment.DTYPE)))
                        baseline=baselines['sklearn linear'][1]
                        reference=baselines['sklearn linear' if link=='mse' else 'scipy t5 reference'][1]
                        beta=baselines['sklearn linear' if link=='mse' else 'scipy t5 reference'][0]
                        pred_difference=float(np.max(np.abs(pred-(vx@beta[:-1]+beta[-1]))))
                    run_id=f"{split_seed}/{preset}/{sgd_seed}/{name}"
                    gap=objective-reference['training_objective'] if reference else None
                    if gap is not None and gap < -2e-6:raise ValueError('Unexpected objective below reference')
                    row={"run_id":run_id,"split_seed":split_seed,"sgd_seed":sgd_seed,"preset":preset,"model":name,"dataset":kind,**config,
                        "n_train":len(ty),"n_test":len(vy),"metric":'accuracy' if kind=='classification' else 'mae',"train_metric":train_metric,"test_metric":value,
                        "baseline_model":baseline['model'],"baseline_test_metric":baseline['test_metric'],"delta_test_metric":value-baseline['test_metric'],
                        "unregularized_baseline_accuracy":baselines['sklearn logistic no penalty'][1]['test_metric'] if kind=='classification' else '',
                        "training_objective":objective,"training_objective_gap":gap,"training_gradient_norm":gradient,
                        "max_test_prediction_difference":pred_difference,"fit_seconds":round(time.perf_counter()-fit_started,4)}
                    runs.append(row);parameters[run_id]=experiment.parameters(model,scaler)
                    predictions.extend({"run_id":run_id,"dataset":kind,"data_row":int(i+1),"observed":float(yi),"prediction":float(pi),"probability_y1":float(p) if kind=='classification' else ''} for i,yi,pi,p in zip(te,vy,pred,prob))
                print(f"  {preset}, SGD seed {sgd_seed}: {len(runs)}/300 fits complete",flush=True)
    aggregated=aggregate(runs,baseline_rows)
    for filename,rows in [('runs.csv',runs),('baseline_runs.csv',baseline_rows),('test_predictions.csv',predictions),('split.csv',split_rows),('aggregate.csv',aggregated)]:experiment.write_csv(ROOT/filename,rows)
    (ROOT/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
    make_figure(aggregated);write_report(aggregated,baseline_rows)
    summary={"protocol":"predeclared paired direct-training experiments; no CV; no best-run selection","split_seeds":SPLIT_SEEDS,"sgd_seeds":SGD_SEEDS,"presets":PRESETS,
        "n_pytorch_fits":len(runs),"loss_checks":checks,"matched_l2_objective":"mean BCE + ||w||^2/(2*n_train), C=1, bias unpenalized",
        "probability_clip":{"min":experiment.CLIP_EPS,"max":1-experiment.CLIP_EPS},"gradient_clip_norm":experiment.GRAD_CLIP_NORM,
        "versions":{"numpy":experiment.np.__version__,"scipy":experiment.scipy.__version__,"sklearn":experiment.sklearn.__version__,"torch":torch.__version__},
        "aggregate":aggregated,"baseline_runs":baseline_rows,"elapsed_seconds":round(time.perf_counter()-started,3),
        "breast_cancer_sha256":hashlib.sha256(breast.data.tobytes()+breast.target.tobytes()).hexdigest(),"ale_sha256":hashlib.sha256(ale.tobytes()).hexdigest()}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f"Done: {ROOT/'report.html'}; elapsed {summary['elapsed_seconds']} s",flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');run()
    p=ROOT/'summary.json';s=json.loads(p.read_text())
    s['warnings']=[{"category":w.category.__name__,"message":str(w.message)} for w in captured]
    p.write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    print(f"Warnings: {len(captured)}",flush=True)
