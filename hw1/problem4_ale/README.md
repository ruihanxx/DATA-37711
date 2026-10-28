# 第 4 题：ALE 回归与残差正态性实验

本目录是 `../main.pdf` 第 4 题的独立实验。已下载原始数据并完成五种回归模型的运行；打开 `report.html` 阅读中文方法、结果与结论。

## 复现

在当前作业目录执行：

```bash
cd problem4_ale
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python run_experiment.py
```

已建立环境后只需最后一行。依赖版本已固定；本次使用 Python 3.10。数据保存在本地，实验运行无需联网。脚本路径与当前工作目录无关，也可以从作业目录执行 `problem4_ale/.venv/bin/python problem4_ale/run_experiment.py`。

默认随机种子是 42。`--seed 其他整数` 可改变划分，但会覆盖该目录的实验结果、图片和报告。

## 文件

- `run_experiment.py`：加载和检查数据、训练与调参、检验、绘图、生成报告。
- `requirements.txt`：固定四个直接依赖；`requirements-lock.txt` 保存本次全部依赖的版本；`.venv/` 是本地独立环境。
- `data/mcs_ds_edited_iter_shuffled.csv`：107 行的 UCI 原始 CSV，包含字段名。
- `data/ale_uci.zip`：原始下载档案。
- `report.html`：中文完整实验报告，图片引用本目录 `figures/`。
- `results/metrics_normality.csv`：训练/测试指标，normaltest、Shapiro-Wilk、Holm 校正及判断。
- `results/predictions.csv`：每个模型每一行观测的预测和残差，数据行号从 1 开始。
- `results/split.csv`：共享的训练/测试划分及原 CSV 行号。
- `results/cv_01.csv` 到 `cv_05.csv`：各模型的训练集交叉验证得分及候选参数。
- `results/summary.json`：数据哈希、环境版本、固定模型配置、候选参数、选定参数、完整结果及拟合警告。
- `figures/`：训练/测试 QQ 图、残差与预测值图、测试预测与 RMSE 比较图。

## 实验约定

只使用 `anchor_ratio`、`trans_range`、`node_density`、`iterations`，目标是 `ale`，忽略 `sd_ale`。五种模型共享 80/20 划分，训练 85 行、测试 22 行。非线性模型使用训练集内 5 折 CV 调参；标准化在对应训练折内完成。

主检验是测试残差 `y - prediction` 的 `scipy.stats.normaltest`，显著性水平 0.05，五模型的主 p 值使用 Holm 校正。原始 p 值、辅助 Shapiro-Wilk 和训练残差结果一并保留。未拒绝正态性不等于证明正态性；22 行留出样本的检验功效有限，p 值仅作为近似诊断。模型不根据正态性 p 值调参。

## 来源

数据：[UCI Average Localization Error (ALE)](https://archive.ics.uci.edu/dataset/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns)，[DOI 10.24432/C5ZP6R](https://doi.org/10.24432/C5ZP6R)。Singh, A., Kotiyal, V., Sharma, S., Nagar, J., & Lee, C. (2020)，CC BY 4.0。下载日期 2026-10-05；保留原始 CSV，未进行修改。

方法文档：[SciPy normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)、[SciPy QQ plot](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html)、[scikit-learn 防止数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)。
