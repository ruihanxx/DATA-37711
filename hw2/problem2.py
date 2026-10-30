from pathlib import Path
import copy
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn


SEED = 42
FEATURES = ["StockCode", "CustomerID"]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "problem2"


def target_features(train, validation, test, strength=20):
    y = train.Quantity.to_numpy(dtype=float)
    oof = np.zeros((len(train), 2), dtype=np.float32)
    valid = np.zeros((len(validation), 2), dtype=np.float32)
    heldout = np.zeros((len(test), 2), dtype=np.float32)
    folds = KFold(5, shuffle=True, random_state=SEED)
    for fit, hold in folds.split(train):
        prior = y[fit].mean()
        for j, feature in enumerate(FEATURES):
            stats = pd.DataFrame({"category": train[feature].iloc[fit], "y": y[fit]})
            stats = stats.groupby("category").y.agg(["sum", "count"])
            mapping = (stats["sum"] + strength * prior) / (stats["count"] + strength)
            oof[hold, j] = train[feature].iloc[hold].map(mapping).fillna(prior)
    for j, feature in enumerate(FEATURES):
        stats = pd.DataFrame({"category": train[feature], "y": y})
        stats = stats.groupby("category").y.agg(["sum", "count"])
        mapping = (stats["sum"] + strength * y.mean()) / (stats["count"] + strength)
        valid[:, j] = validation[feature].map(mapping).fillna(y.mean())
        heldout[:, j] = test[feature].map(mapping).fillna(y.mean())
    transformed = [np.log1p(x) for x in (oof, valid, heldout)]
    mean, scale = transformed[0].mean(axis=0), transformed[0].std(axis=0)
    return [(x - mean) / scale for x in transformed]


def category_features(train, validation, test):
    arrays = [np.zeros((len(frame), 2), dtype=np.int64) for frame in (train, validation, test)]
    sizes = []
    for j, feature in enumerate(FEATURES):
        values = sorted(train[feature].unique())
        mapping = {value: i + 1 for i, value in enumerate(values)}
        sizes.append(len(values) + 1)
        for frame, array in zip((train, validation, test), arrays):
            array[:, j] = frame[feature].map(mapping).fillna(0).to_numpy(dtype=np.int64)
    return arrays, sizes


class RetailNet(nn.Module):
    def __init__(self, mode, category_sizes):
        super().__init__()
        self.mode = mode
        if mode != "target":
            self.embeddings = nn.ModuleList([nn.Embedding(n, 16, padding_idx=0) for n in category_sizes])
        width = {"target": 2, "embedding": 32, "hybrid": 34}[mode]
        self.layers = nn.Sequential(nn.Linear(width, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, categories, target_encoding):
        pieces = []
        if self.mode != "embedding":
            pieces.append(target_encoding)
        if self.mode != "target":
            pieces.extend(layer(categories[:, j]) for j, layer in enumerate(self.embeddings))
        return self.layers(torch.cat(pieces, dim=1)).squeeze(1)


def predict(model, categories, target_encoding):
    model.eval()
    with torch.no_grad():
        output = model(categories, target_encoding).numpy()
    return np.maximum(output, np.log(2))


def fit_model(mode, sizes, categories, encodings, targets):
    torch.manual_seed(SEED)
    model = RetailNet(mode, sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    history = []
    for epoch in range(1, 41):
        model.train()
        order = torch.randperm(len(targets[0]))
        total = 0.0
        for batch in order.split(2048):
            optimizer.zero_grad()
            fitted = model(categories[0][batch], encodings[0][batch])
            loss = ((fitted - targets[0][batch]) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        validation = predict(model, categories[1], encodings[1])
        val_loss = float(np.mean((validation - targets[1].numpy()) ** 2))
        history.append({"epoch": epoch, "training_mse": total / len(targets[0]),
                        "validation_mse": val_loss})
        print(f"{mode}: epoch {epoch:2d}, validation MSE {val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale == 6:
            break
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT / f"{mode}_history.csv", index=False)
    return model, best_epoch, history


def scores(name, split, truth, log_prediction):
    prediction = np.expm1(log_prediction)
    log_mse = float(np.mean((np.log1p(truth) - log_prediction) ** 2))
    return {"model": name, "split": split, "log_mse": log_mse,
            "rmsle": float(np.sqrt(log_mse)),
            "mae": float(np.mean(np.abs(truth - prediction))),
            "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2)))}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(HERE / "retail.csv", dtype={feature: str for feature in FEATURES})
    train, remainder = train_test_split(data, test_size=0.30, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=0.50, random_state=SEED)
    frames = [frame.reset_index(drop=True) for frame in (train, validation, test)]
    train, validation, test = frames
    target_arrays = target_features(*frames)
    category_arrays, sizes = category_features(*frames)
    encodings = [torch.tensor(array, dtype=torch.float32) for array in target_arrays]
    categories = [torch.tensor(array, dtype=torch.long) for array in category_arrays]
    targets = [torch.tensor(np.log1p(frame.Quantity.to_numpy()), dtype=torch.float32) for frame in frames]
    cardinality = []
    for j, feature in enumerate(FEATURES):
        counts = data[feature].value_counts()
        cardinality.append({"feature": feature, "unique_full": len(counts),
                            "unique_train": train[feature].nunique(),
                            "singleton_categories": int((counts == 1).sum()),
                            "median_rows_per_category": float(counts.median()),
                            "validation_unknown_rows": int((category_arrays[1][:, j] == 0).sum()),
                            "test_unknown_rows": int((category_arrays[2][:, j] == 0).sum())})
    pd.DataFrame(cardinality).to_csv(OUT / "cardinality.csv", index=False)
    rows, histories, best_epochs = [], {}, {}
    predictions = pd.DataFrame({"Quantity": test.Quantity})
    baseline = float(targets[0].mean())
    for frame, split in zip(frames[1:], ["validation", "test"]):
        rows.append(scores("constant", split, frame.Quantity.to_numpy(), np.full(len(frame), baseline)))
    predictions["constant"] = np.expm1(baseline)
    for mode in ["target", "embedding", "hybrid"]:
        model, best_epoch, history = fit_model(mode, sizes, categories, encodings, targets)
        best_epochs[mode], histories[mode] = best_epoch, history
        for i, split in [(1, "validation"), (2, "test")]:
            log_prediction = predict(model, categories[i], encodings[i])
            rows.append(scores(mode, split, frames[i].Quantity.to_numpy(), log_prediction))
            if split == "test":
                predictions[mode] = np.expm1(log_prediction)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    summary = {"seed": SEED, "rows": len(data),
               "split_sizes": dict(zip(["train", "validation", "test"], map(len, frames))),
               "cardinality": cardinality,
               "quantity_summary": {str(key): float(value) for key, value in
                                    data.Quantity.describe(percentiles=[0.5, 0.9, 0.99, 0.999]).items()},
               "dense_one_hot_float32_gib": len(data) * sum(data[f].nunique() for f in FEATURES) * 4 / 2**30,
               "target_transform": "log1p(Quantity)", "target_encoding_folds": 5,
               "target_encoding_smoothing": 20,
               "target_encoding_target": "Quantity (raw arithmetic mean)",
               "target_encoding_feature_transform": "log1p, then training-OOF standardization",
               "embedding_dimensions": [16, 16],
               "hidden_layers": [64, 32], "learning_rate": 0.003,
               "weight_decay": 0.0001, "batch_size": 2048,
               "max_epochs": 40, "early_stopping_patience": 6, "best_epochs": best_epochs,
               "metrics": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.log1p(data.Quantity), bins=60, color="steelblue")
    axes[0].set(xlabel="log(1 + Quantity)", ylabel="Transactions", title="Retail target distribution")
    for mode, history in histories.items():
        axes[1].plot([h["epoch"] for h in history], [h["validation_mse"] for h in history], label=mode)
    axes[1].set(xlabel="Epoch", ylabel="Validation log-space MSE", title="Early stopping")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "retail_results.png", dpi=160)
    plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
