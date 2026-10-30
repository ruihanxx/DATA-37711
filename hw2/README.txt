HW2

hw2.pdf contains the complete answers, experimental results, and figures.
report/main.tex and report/problem1.tex--problem3.tex are the LaTeX sources.

Code:
  problem1.py   Taxi conditional Gaussian model and probability evaluation
  problem2.py   Retail target encoding, embeddings, and combined model
  problem3.py   RKS/MLP sweeps, visualizations, and test predictions
  models.py     RKS and one-hidden-layer MLP implementations
  run_all.py    Runs all three experiments

Use Python 3.10 or 3.11. From this directory:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python run_all.py

The local .venv is already configured. To run one problem:
  .venv/bin/python problem1.py
  .venv/bin/python problem2.py
  .venv/bin/python problem3.py

Keep the original taxi_trips.parquet, retail.csv, train.csv, val.csv,
and test.csv beside these scripts. Original datasets and the local
environment are excluded from the submission ZIP.

All experiments use seed 42. Results, histories, metrics, and figures are
saved under results/problem1, results/problem2, and results/problem3.
The ZIP includes result tables and figures; the additional per-row taxi
and retail evaluation outputs remain in the local results directory.

Required Problem 3 outputs:
  rks_test_predictions.csv
  mlp_test_predictions.csv
Each has the column header prediction and 200 numeric rows, in test.csv
order. These test labels are withheld, so there is no test MSE for Problem 3.

To rebuild the report using the included figures:
  cd report
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  cp main.pdf ../hw2.pdf

Report values describe the supplied data and the saved experiments.
Small numerical differences can occur across hardware/library builds.
