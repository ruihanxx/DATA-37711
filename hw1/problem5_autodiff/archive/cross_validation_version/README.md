# 第 5 题：自动微分与自定义损失

## 运行

从作业目录执行：

```bash
cd problem5_autodiff
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_experiment.py
```

本次环境为 Python 3.10，CPU、float64。固定种子 42，80/20 划分；分类分层划分（455/114），ALE 沿用第 4 题的划分（85/22）。只在训练数据上拟合特征标准化，不改变 ALE 目标的单位。保留此前的测试集；本次调参只使用训练集内的三折交叉验证，测试指标在训练完成后重新计算。

分类基线是 sklearn LogisticRegression，使用默认 L2 惩罚、C=1，增加 max_iter 以确保拟合完成。自实现 logistic 和 probit 只最小化题目要求的 BCE，没有加入惩罚。分别使用 sigmoid 和标准正态 CDF，并用 `torch.clamp(p, 1e-6, 1-1e-6)` 裁剪输出概率，再计算 `-[y log(p)+(1-y) log(1-p)]`。分类阈值 0.5。裁剪输出和极端输入处的损失/梯度均经过数值检查。

回归基线是 sklearn LinearRegression。两个 PyTorch 模型分别最小化 MSE 与 StudentT(df=5, loc=0, scale=1) 的平均负对数似然。t 分布尺度固定为 1，残差使用原始米单位，不进行目标标准化，也不估计额外尺度。

四个模型均用 `torch.optim.SGD` 与每轮随机打乱的 mini-batch，并将梯度范数裁剪至 5。学习率按 `lr0 / (1 + epoch_index / 100)` 衰减，epoch_index 从 0 开始。

三折 CV 在每个训练折内单独拟合 scaler。候选 epoch 为 50、100、200，初始学习率为 0.03、0.1、0.3，momentum 为 0、0.9；分类 batch 为 64、128，回归 batch 为 16、32。每个模型共 36 个候选组合。分类以验证 accuracy 最大为标准，平局时比较 clipped BCE；回归以验证 MAE 最小为标准，平局时比较对应验证损失。所有候选及各折得分均保存，未依据测试集结果继续调整参数。

| 模型 | Epoch | Batch | 初始学习率 | Momentum |
|---|---:|---:|---:|---:|
| Logistic | 200 | 64 | 0.3 | 0 |
| Probit | 50 | 128 | 0.3 | 0 |
| MSE linear | 100 | 32 | 0.3 | 0.9 |
| t5 linear | 100 | 16 | 0.1 | 0.9 |

默认运行复用 `results/selected_configs.json` 中的 CV 选定参数，只重新训练并评估，避免每次重复搜索。数据哈希及搜索设置不匹配时自动重新调参。需要完整复现调参时执行：

```bash
.venv/bin/python run_experiment.py --retune
```

## 文件

- `run_experiment.py`：损失、自定义模型、mini-batch SGD、基线和评估。
- `data/`：原始 ALE CSV 与 sklearn 内置 Breast Cancer 数据副本。
- `results/classification_metrics.csv`：训练/测试 accuracy 与正确分类数量。
- `results/regression_metrics.csv`：训练/测试 MAE 与 MSE。
- `results/predictions.csv`：所有模型的逐行预测。
- `results/split.csv`：共享划分与原始数据行号（从 1 开始）。
- `results/training_history.csv`：四个模型的训练损失和学习率。
- `results/cv_candidates.csv`：36 个候选/模型的合并验证得分。
- `results/cv_folds.csv`：每个候选在各折和 epoch 检查点的验证得分。
- `results/selected_configs.json`：默认重跑复用的参数及数据/设置校验信息，不含测试指标。
- `results/summary.json`：参数、数据哈希、版本、梯度范数和数值/梯度检查。
- `results/validation.json`：指标重算、参数重建预测、CV 选择及 clip 检查的结果。
- `results/problem5.tex`：按 `hw1.tex` 风格生成的结果片段；差值列为相对各自 sklearn 基线的测试指标差值。
- `figures/`：分类和回归训练曲线。
- `archive/initial_2000_epochs/`：优化前的代码、结果和图片。

运行实验会更新本目录的输出，但不自动修改上一级 `hw1.tex`。

## 不标准化的对照实验

```bash
.venv/bin/python run_no_scaling.py
```

完全使用原始 X 与 y，不中心化、不标准化，也不做其他归一化。使用与标准化实验相同的训练/测试划分、三折 CV、epoch 候选和 clipped BCE；针对原始量纲将学习率候选改为 `1e-5, 1e-4, 1e-3`。所有学习率、batch 和 momentum 仍只根据训练集内的 CV 选择。默认复用已选定参数，`--retune` 可重新搜索。

输出位于 `no_scaling/`：`comparison.csv` 是同一测试集上的比较；`summary.json` 包含选定参数、预测表现、训练设计矩阵条件数，以及固定标准化数据和 SGD 参数、只改变 BCE 实现的 clip 对照。`validation.json` 保存指标重算和原始参数重建预测的验证结果。标准化结果保留在 `results/`。

| PyTorch 模型 | 指标 | 标准化 | 原始特征 |
|---|---|---:|---:|
| Logistic SGD | Test accuracy | 95.61% | 75.44% |
| Probit SGD | Test accuracy | 97.37% | 94.74% |
| Linear MSE SGD | Test MAE (m) | 0.190610 | 0.262494 |
| Linear t5 NLL SGD | Test MAE (m) | 0.195444 | 0.259181 |

在这次不超过 200 轮的训练预算及候选网格内，移除标准化没有改善表现。设计矩阵条件数在分类任务中从约 322 增至约 1.48e6，回归任务中从约 1.61 增至约 1786，说明原始参数化的数值条件更差。此比较同时包含分别选择的学习率等设置，不能解释为所有不标准化模型必然更差。

固定标准化数据和缩短后的 SGD 设置，仅用稳定对数概率替代 clipped BCE，logistic 仍为 109/114，probit 仍为 111/114，测试标签完全一致。因此此前 logistic 从 111/114 变为 109/114 的下降，不应归因于概率 clip；此前同时改变了训练预算、学习率及其他优化设置，不能仅凭该比较确定其中某一项的单独贡献。

## 来源

题目：`../main.pdf` 第 5 题。

数据：[sklearn Breast Cancer Wisconsin](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_breast_cancer.html)、[UCI ALE（DOI 10.24432/C5ZP6R）](https://doi.org/10.24432/C5ZP6R)。

API：[PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)、[PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)、[sklearn LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)。
