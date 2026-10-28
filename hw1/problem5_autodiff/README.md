# 第 5 题：自动微分与自定义损失

## 直接运行

从作业目录执行：

```bash
cd problem5_autodiff
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_experiment.py
```

当前实现不进行交叉验证、验证集划分或超参数搜索，也不读取过去的 CV 参数缓存。每个模型用固定参数直接训练一次。

本次环境为 Python 3.10、CPU、float64。固定种子 42，80/20 划分：分类训练 455 条、测试 114 条，使用分层划分；ALE 训练 85 条、测试 22 条，沿用第 4 题的划分。特征标准化只在完整训练集上拟合。ALE 目标保留原始米单位。

## 固定参数

| 模型 | Epoch | Batch size | 初始学习率 | Momentum |
|---|---:|---:|---:|---:|
| Logistic | 200 | 64 | 0.05 | 0 |
| Probit | 200 | 64 | 0.05 | 0 |
| MSE linear | 200 | 32 | 0.03 | 0 |
| t5 linear | 200 | 32 | 0.03 | 0 |

四个模型均使用 `torch.optim.SGD`，每轮随机打乱 mini-batch。学习率按 `lr0 / (1 + epoch_index / 100)` 衰减，epoch_index 从 0 开始。梯度范数裁剪上限为 5。

Logistic 和 probit 分别使用 sigmoid 和标准正态 CDF，再用 `torch.clamp(p, 1e-6, 1-1e-6)` 裁剪概率并计算 BCE。两个模型没有正则化项；sklearn logistic 基线使用默认 L2、C=1。分类阈值为 0.5。

回归基线为 sklearn LinearRegression。两个自实现线性模型分别最小化 MSE 和 `StudentT(df=5, loc=0, scale=1)` 的平均负对数似然，尺度固定为 1。

## 输出

- `run_experiment.py`：损失、自定义模型、直接训练、基线和评估。
- `data/`：原始 ALE CSV 与 Breast Cancer 数据副本。
- `results/classification_metrics.csv`：训练/测试准确率和正确分类数量。
- `results/regression_metrics.csv`：训练/测试 MAE、MSE。
- `results/predictions.csv`：逐行预测与分类概率。
- `results/split.csv`：完整训练/测试划分及原始行号。
- `results/training_history.csv`：200 轮训练损失与学习率。
- `results/summary.json`：固定参数、实际训练样本数、数据哈希、版本、梯度检查、运行时间及警告。
- `results/validation.json`：指标重算、模型参数重建预测、clip 及直接训练协议核验。
- `results/problem5.tex`：结果片段，分别回答分类和回归的 (a)、(b)。
- `figures/`：分类和回归训练曲线。
- `archive/initial_2000_epochs/`：最初的 2000 轮版本。
- `archive/cross_validation_version/`：之前的 CV 实现、结果及不标准化对照，作为历史记录。

当前运行会更新结果、图片和 LaTeX 片段，不自动修改上一级 `hw1.tex`。

## 超参数与随机种子敏感性实验

```bash
.venv/bin/python run_sensitivity.py
```

这是直接训练的对照实验，不做交叉验证。预先固定 5 个数据划分种子（0、1、7、21、42）、3 个 mini-batch 打乱种子（0、1、42），以及 short、current、larger_lr、momentum 四组参数。每次 baseline 与 PyTorch 共享数据划分和训练集标准化。每个配置有 15 次成对比较，报告全部结果，不按测试分数挑选最好的一次。

除题目的四个模型外，额外加入一个与 sklearn C=1 匹配 L2 的 PyTorch logistic 诊断模型，总计 300 次 PyTorch 训练。另保存无正则化的 sklearn logistic 和独立 SciPy Student-t 优化参考。匹配 L2 的目标与梯度已对照安装的 sklearn 实现核验。

输出位于 `sensitivity/`：

- `report.html`：中文方法、对照表及差异解释。
- `comparison.png`：成对指标差的均值和标准差。
- `runs.csv`、`baseline_runs.csv`、`aggregate.csv`：逐次训练、baseline 及汇总指标。
- `parameters.json`、`test_predictions.csv`、`split.csv`：可核验的参数、预测和划分。
- `summary.json`、`matched_l2_check.json`、`validation.json`：协议、目标/梯度对照、指标重算核验。

15 次比较的标准差只是描述性指标：同一划分的三次训练共享测试集，不同划分也存在样本重叠，因此没有将这些运行当作独立样本做显著性检验。

## 增加训练轮数的对照

```bash
.venv/bin/python run_epoch_sensitivity.py
```

固定当前配置的学习率、batch、momentum、衰减、标准化和 clip，只比较 200、500、1000 epochs。仍使用上述 5 个数据划分种子和 3 个 SGD 种子，不做交叉验证。每条轨迹直接训练到 1000 轮并保存三个检查点，共 75 条轨迹、225 个检查点；第 200 轮参数和指标会与已有敏感性实验逐一核对。

结果另存于 `sensitivity_epochs/`，包含 `report.html`、`comparison.png`、逐次指标、参数、预测、每 10 轮的训练目标和核验结果。当前单次作业实验及 `hw1.tex` 不由这个脚本更新。

作业中的 200/500 轮表格使用固定数据划分种子 42、SGD 种子 42，沿同一训练轨迹比较。可从已核验的预测重建对应片段和补充指标：

```bash
.venv/bin/python make_epoch_comparison.py
```

该脚本生成 `results/problem5_epoch_comparison.tex`、`results/epoch_comparison_metrics.csv` 和 `results/epoch_comparison_metrics_validation.json`，不自动修改上一级 `hw1.tex`。分类额外比较 ROC-AUC、恶性类别的 F1、敏感度及特异度（原始标签 0 为正类），回归额外比较 RMSE 和 R²。各模型使用相同测试样本；指标已通过独立公式核验。作业第五题已使用该片段并去掉 loss 图；`run_experiment.py` 仍负责原有的单次 200 轮结果。

## 来源

题目：`../main.pdf` 第 5 题。

数据：[sklearn Breast Cancer Wisconsin](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html)、[UCI ALE（DOI 10.24432/C5ZP6R）](https://doi.org/10.24432/C5ZP6R)。

API：[PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)、[PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)、[sklearn LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)。
