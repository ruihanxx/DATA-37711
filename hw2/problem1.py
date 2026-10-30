from pathlib import Path
import copy
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "problem1"
SEED = 42


class GaussianMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )
        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data[:] = torch.tensor([0.0, 0.54])

    def forward(self, x):
        output = self.net(x)
        return output[:, 0], F.softplus(output[:, 1]) + 0.01


def gaussian_nll(y, mean, std):
    return (torch.log(std) + 0.5 * ((y - mean) / std) ** 2
            + 0.5 * math.log(2 * math.pi)).mean()


def probability_metrics(y, probability):
    event = (y < 45).astype(float)
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "brier": float(np.mean((probability - event) ** 2)),
        "log_loss": float(-np.mean(event * np.log(probability)
                                    + (1 - event) * np.log(1 - probability))),
        "accuracy": float(np.mean((probability >= 0.5) == event)),
    }


def evaluate(y, mean, std):
    probability = ndtr((45 - mean) / std)
    result = {
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "mae": float(np.mean(np.abs(y - mean))),
        "gaussian_nll": float(np.mean(np.log(std) + 0.5 * ((y - mean) / std) ** 2
                                      + 0.5 * np.log(2 * np.pi))),
        "coverage_90": float(np.mean(np.abs(y - mean) <= 1.6448536269514722 * std)),
        "mean_predicted_std": float(np.mean(std)),
    }
    result.update(probability_metrics(y, probability))
    return result, probability


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = pd.read_parquet(ROOT / "taxi_trips.parquet")
    features = [c for c in data.columns if c != "trip_duration"]
    x = data[features].to_numpy(dtype=np.float32)
    y = data["trip_duration"].to_numpy(dtype=np.float64)
    indices = rng.permutation(len(data))
    n_train, n_val = int(0.7 * len(data)), int(0.15 * len(data))
    train = indices[:n_train]
    val = indices[n_train:n_train + n_val]
    test = indices[n_train + n_val:]
    target_mean, target_std = float(y[train].mean()), float(y[train].std())
    xt = torch.from_numpy(x[train])
    yt = torch.tensor((y[train] - target_mean) / target_std, dtype=torch.float32)
    xv = torch.from_numpy(x[val])
    yv = torch.tensor((y[val] - target_mean) / target_std, dtype=torch.float32)

    model = GaussianMLP(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(xt))
        total_loss = 0.0
        for batch in order.split(1024):
            mean, std = model(xt[batch])
            loss = gaussian_nll(yt[batch], mean, std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            mean, std = model(xv)
            val_loss = gaussian_nll(yv, mean, std).item()
        history.append({"epoch": epoch,
                        "train_nll": total_loss / len(xt) + np.log(target_std),
                        "val_nll": val_loss + np.log(target_std)})
        if val_loss < best_loss - 0.0001:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: validation NLL = {val_loss + np.log(target_std):.4f}", flush=True)
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)

    design_train = np.column_stack([np.ones(len(train)), x[train]])
    linear_coef = np.linalg.lstsq(design_train, y[train], rcond=None)[0]
    linear_std = float(np.sqrt(np.mean((y[train] - design_train @ linear_coef) ** 2)))
    train_probability = float(np.mean(y[train] < 45))
    rows = []
    predictions = {}
    for split_name, split in [("validation", val), ("test", test)]:
        with torch.no_grad():
            mean, std = model(torch.from_numpy(x[split]))
        mlp_mean = mean.numpy().astype(float) * target_std + target_mean
        mlp_std = std.numpy().astype(float) * target_std
        estimates = {
            "Constant Gaussian": (np.full(len(split), target_mean), np.full(len(split), target_std)),
            "Linear Gaussian": (np.column_stack([np.ones(len(split)), x[split]]) @ linear_coef,
                                np.full(len(split), linear_std)),
            "Gaussian MLP": (mlp_mean, mlp_std),
        }
        for name, (pred_mean, pred_std) in estimates.items():
            scores, probability = evaluate(y[split], pred_mean, pred_std)
            rows.append({"split": split_name, "model": name, **scores})
            if split_name == "test":
                predictions[name] = probability
        rows.append({"split": split_name, "model": "Empirical event frequency",
                     **probability_metrics(y[split], np.full(len(split), train_probability))})
        if split_name == "test":
            pd.DataFrame({
                "row_position": split, "trip_duration": y[split],
                "predicted_mean": mlp_mean, "predicted_std": mlp_std,
                "probability_under_45": predictions["Gaussian MLP"],
            }).to_csv(OUT / "test_predictions.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    summary = {
        "seed": SEED, "n_rows": len(data), "features": features,
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "target_mean_train": target_mean, "target_std_train": target_std,
        "event_rate_train": train_probability, "event_rate_test": float(np.mean(y[test] < 45)),
        "architecture": "7-64-64-2, ReLU hidden layers, softplus standard deviation",
        "batch_size": 1024, "learning_rate": 0.001, "weight_decay": 0.0001,
        "max_epochs": 100, "patience": 12, "best_epoch": best_epoch,
        "epochs_run": len(history), "metrics": rows,
    }
    with open(OUT / "summary.json", "w") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    history_frame = pd.DataFrame(history)
    ax.plot(history_frame.epoch, history_frame.train_nll, label="Training")
    ax.plot(history_frame.epoch, history_frame.val_nll, label="Validation")
    ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="Gaussian NLL", title="Taxi Gaussian MLP training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png", dpi=180)
    plt.close(fig)

    event = (y[test] < 45).astype(float)
    calibration = []
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4))
    for name in ["Linear Gaussian", "Gaussian MLP"]:
        probability = predictions[name]
        groups = np.array_split(np.argsort(probability), 10)
        predicted = [float(probability[g].mean()) for g in groups]
        observed = [float(event[g].mean()) for g in groups]
        axes[0].plot(predicted, observed, marker="o", markersize=4, label=name)
        for j, group in enumerate(groups):
            calibration.append({"model": name, "bin": j + 1, "n": len(group),
                                "mean_probability": predicted[j], "event_rate": observed[j]})
    pd.DataFrame(calibration).to_csv(OUT / "calibration.csv", index=False)
    axes[0].plot([0.9, 1], [0.9, 1], "k--", linewidth=1)
    axes[0].set(xlim=(0.9, 1.001), ylim=(0.9, 1.001), xlabel="Mean predicted probability",
                ylabel="Observed fraction under 45 min", title="Test calibration: equal-count bins")
    axes[0].legend(fontsize=8)
    sample = rng.choice(len(test), min(4000, len(test)), replace=False)
    axes[1].scatter(mlp_mean[sample], y[test][sample], s=4, alpha=0.25)
    axes[1].plot([0, 80], [0, 80], "k--", linewidth=1)
    axes[1].set(xlim=(0, 80), ylim=(0, 80), xlabel="Predicted mean duration (min)",
                ylabel="Observed duration (min)", title="Gaussian MLP: test predictions")
    fig.tight_layout()
    fig.savefig(OUT / "test_evaluation.png", dpi=180)
    plt.close(fig)
    print(metrics.to_string(index=False), flush=True)
    print(f"Best epoch: {best_epoch}; outputs: {OUT}", flush=True)


if __name__ == "__main__":
    main()
