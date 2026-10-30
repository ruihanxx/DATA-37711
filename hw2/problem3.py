import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader import HW2Dataset
from models import MLP, RandomKitchenSinks, mlp_sweep, sweep_rks_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'problem3'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def save_table(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def loss_plot(rows, parameter, metric, title, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[parameter] for r in rows], [r[metric] for r in rows], 'o-', color='#28639d')
    ax.set_xscale('log', base=2)
    ax.set_xticks([r[parameter] for r in rows], [str(r[parameter]) for r in rows])
    ax.set(xlabel='Random features D' if parameter == 'D' else 'Hidden-layer width', ylabel='MSE', title=title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def main():
    ds = HW2Dataset.from_csv(ROOT / 'train.csv', ROOT / 'val.csv', ROOT / 'test.csv')
    X, y, Xv, yv = ds.train.X, ds.train.y, ds.val.X, ds.val.y
    args = X, y, Xv, yv
    rks, _, best_D = sweep_rks_model([10, 20, 50, 100, 200, 400, 800, 1200], *args)
    rks_rows = rks.sweep_results_
    print('RKS sweep:', rks_rows, flush=True)
    mlp, _, best_width = mlp_sweep([8, 16, 32, 64, 128, 256], *args)
    mlp_rows = mlp.sweep_results_
    print('MLP sweep:', mlp_rows, flush=True)
    save_table(OUT / 'rks_sweep.csv', rks_rows)
    save_table(OUT / 'mlp_sweep.csv', mlp_rows)
    for name, rows, parameter in [('rks', rks_rows, 'D'), ('mlp', mlp_rows, 'hidden_width')]:
        for split in ['train', 'val']:
            loss_plot(rows, parameter, f'{split}_mse', f'{name.upper()} {split} MSE', f'{name}_{split}_mse.png')

    rks = RandomKitchenSinks(D=best_D).fit(X, y)
    mlp = MLP(2, d_hidden=best_width).fit(X, y)
    summary = {'seed': 42, 'n_train': len(X), 'n_val': len(Xv), 'n_test': len(ds.test.X),
               'rks': {'D': best_D, 'h': best_D // 2, 'sigma': 1.0, 'ridge_lambda': 1e-3,
                       'train_mse': float(np.mean((rks.predict(X) - y) ** 2)),
                       'val_mse': float(np.mean((rks.predict(Xv) - yv) ** 2))},
               'mlp': {'hidden_width': best_width, 'activation': 'tanh', 'lr': 0.01,
                       'batch_size': 600, 'epochs': 4000, 'weight_decay': 1e-5,
                       'train_mse': float(np.mean((mlp.predict(X) - y) ** 2)),
                       'val_mse': float(np.mean((mlp.predict(Xv) - yv) ** 2))}}
    for name, model in [('rks', rks), ('mlp', mlp)]:
        prediction = model.predict(ds.test.X)
        assert prediction.shape == (len(ds.test.X),) and np.isfinite(prediction).all()
        np.savetxt(ROOT / f'{name}_test_predictions.csv', prediction, delimiter=',', header='prediction', comments='', fmt='%.10f')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    axis = np.linspace(-3, 3, 160)
    x1, x2 = np.meshgrid(axis, axis)
    grid = np.column_stack([x1.ravel(), x2.ravel()])
    surfaces = [rks.predict(grid).reshape(x1.shape), mlp.predict(grid).reshape(x1.shape)]
    vmin = min(y.min(), *(z.min() for z in surfaces))
    vmax = max(y.max(), *(z.max() for z in surfaces))
    levels = np.linspace(vmin, vmax, 31)
    fig, ax = plt.subplots(figsize=(6, 5))
    dots = ax.scatter(X[:, 0], X[:, 1], c=y, s=22, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='x1', ylabel='x2', title='Training data')
    ax.set_aspect('equal')
    fig.colorbar(dots, ax=ax, label='y')
    fig.tight_layout()
    fig.savefig(OUT / 'training_data.png', dpi=200)
    plt.close(fig)
    for name, z in zip(['rks', 'mlp'], surfaces):
        fig, ax = plt.subplots(figsize=(6, 5))
        contours = ax.contourf(x1, x2, z, levels=levels, cmap='viridis')
        ax.set(xlabel='x1', ylabel='x2', title=f'{name.upper()} fitted surface')
        ax.set_aspect('equal')
        fig.colorbar(contours, ax=ax, label='Prediction')
        fig.tight_layout()
        fig.savefig(OUT / f'{name}_surface.png', dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    axes[0].scatter(X[:, 0], X[:, 1], c=y, s=12, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Training data')
    for ax, name, z in zip(axes[1:], ['RKS', 'MLP'], surfaces):
        im = ax.contourf(x1, x2, z, levels=levels, cmap='viridis')
        ax.set_title(f'{name} fitted surface')
    for ax in axes:
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='x1', ylabel='x2')
        ax.set_aspect('equal')
    fig.colorbar(im, ax=axes, label='y / prediction', shrink=0.85)
    fig.savefig(OUT / 'surface_comparison.png', dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
