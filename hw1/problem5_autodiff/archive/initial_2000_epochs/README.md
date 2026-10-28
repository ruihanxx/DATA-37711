# 第 5 题：自动微分与自定义损失

## 运行

从作业目录执行：

```bash
cd problem5_autodiff
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_experiment.py
```

本次环境为 Python 3.10，CPU、float64。固定种子 42，80/20 划分；分类分层划分（455/114），ALE 沿用第 4 题的划分（85/22）。只在训练数据上拟合特征标准化，不改变 ALE 目标的单位。测试集仅在训练结束后用于评价，未参与选择学习率、训练轮数或模型。

分类基线是 sklearn LogisticRegression，使用默认 L2 惩罚、C=1，增加 max_iter 以确保拟合完成。自实现 logistic 和 probit 只最小化题目要求的 BCE，没有加入惩罚。两者使用稳定的对数概率计算 BCE；推断分别使用 sigmoid 和标准正态 CDF，分类阈值 0.5。

回归基线是 sklearn LinearRegression。两个 PyTorch 模型分别最小化 MSE 与 StudentT(df=5, loc=0, scale=1) 的平均负对数似然。t 分布尺度固定为 1，残差使用原始米单位，不进行目标标准化，也不估计额外尺度。

四个模型均用 `torch.optim.SGD` 与每轮随机打乱的 mini-batch。分类：2000 轮、batch=64、初始学习率 0.05；回归：2000 轮、batch=16、初始学习率 0.03。学习率分别按 `lr0 / (1 + epoch_index / 500)` 与 `lr0 / (1 + epoch_index / 200)` 衰减，epoch_index 从 0 开始。

## 文件

- `run_experiment.py`：损失、自定义模型、mini-batch SGD、基线和评估。
- `data/`：原始 ALE CSV 与 sklearn 内置 Breast Cancer 数据副本。
- `results/classification_metrics.csv`：训练/测试 accuracy 与正确分类数量。
- `results/regression_metrics.csv`：训练/测试 MAE 与 MSE。
- `results/predictions.csv`：所有模型的逐行预测。
- `results/split.csv`：共享划分与原始数据行号（从 1 开始）。
- `results/training_history.csv`：四个模型的训练损失和学习率。
- `results/summary.json`：参数、数据哈希、版本、梯度范数和数值/梯度检查。
- `results/problem5.tex`：按 `hw1.tex` 风格生成的结果片段；差值列为相对各自 sklearn 基线的测试指标差值。
- `figures/`：分类和回归训练曲线。

运行实验会更新本目录的输出，但不自动修改上一级 `hw1.tex`。

## 来源

题目：`../main.pdf` 第 5 题。

数据：[sklearn Breast Cancer Wisconsin](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html)、[UCI ALE（DOI 10.24432/C5ZP6R）](https://doi.org/10.24432/C5ZP6R)。

API：[PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)、[PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)、[sklearn LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)。
