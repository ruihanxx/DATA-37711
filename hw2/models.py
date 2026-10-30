import numpy as np
import torch
from torch import nn


class RandomKitchenSinks:
    def __init__(self, D=800, sigma=1.0, ridge_lambda=1e-3, seed=42):
        if D <= 0 or D % 2:
            raise ValueError('D must be a positive even number.')
        self.D = D
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.seed = seed

    def _features(self, X):
        angles = np.asarray(X) @ self.weights_ + self.phases_
        return np.stack((np.sin(angles), np.cos(angles)), axis=2).reshape(len(X), self.D) / np.sqrt(self.D / 2)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.weights_ = rng.normal(size=(X.shape[1], self.D // 2)) / self.sigma
        self.phases_ = rng.uniform(0, 2 * np.pi, self.D // 2)
        phi = self._features(X)
        self.feature_mean_ = phi.mean(axis=0)
        self.target_mean_ = np.mean(y)
        centered = phi - self.feature_mean_
        target = y - self.target_mean_
        if self.D <= len(X):
            self.coef_ = np.linalg.solve(centered.T @ centered + self.ridge_lambda * np.eye(self.D), centered.T @ target)
        else:
            self.coef_ = centered.T @ np.linalg.solve(centered @ centered.T + self.ridge_lambda * np.eye(len(X)), target)
        return self

    def predict(self, X):
        return (self._features(X) - self.feature_mean_) @ self.coef_ + self.target_mean_


def sweep_rks_model(sweep_D, X_train, y_train, X_val, y_val):
    results = []
    best_model, best_val_mse, best_D = None, np.inf, None
    for D in sweep_D:
        model = RandomKitchenSinks(D=D).fit(X_train, y_train)
        train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
        results.append({'D': D, 'train_mse': float(train_mse), 'val_mse': float(val_mse)})
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_D = model, val_mse, D
    best_model.sweep_results_ = results
    return best_model, float(best_val_mse), best_D


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_out=1, lr=0.01, batch_size=600,
                 activation='tanh', device='cpu', epochs=4000, weight_decay=1e-5, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.device, self.lr, self.batch_size = device, lr, batch_size
        self.epochs, self.weight_decay, self.seed = epochs, weight_decay, seed
        activation_layer = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation]
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), activation_layer(), nn.Linear(d_hidden, d_out))
        self.register_buffer('x_mean', torch.zeros(d_in))
        self.register_buffer('x_scale', torch.ones(d_in))
        self.register_buffer('y_mean', torch.zeros(1))
        self.register_buffer('y_scale', torch.ones(1))
        self.to(device)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        self.x_mean.copy_(X.mean(0))
        self.x_scale.copy_(X.std(0, unbiased=False).clamp_min(1e-8))
        self.y_mean.copy_(y.mean())
        self.y_scale.copy_(y.std(unbiased=False).clamp_min(1e-8))
        X, y = (X - self.x_mean) / self.x_scale, (y - self.y_mean) / self.y_scale
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(X), device=self.device) if self.batch_size < len(X) else torch.arange(len(X), device=self.device)
            for batch in order.split(self.batch_size):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.net(X[batch]), y[batch])
                loss.backward()
                optimizer.step()
        self.eval()
        return self

    def forward(self, X):
        return self.net((X - self.x_mean) / self.x_scale) * self.y_scale + self.y_mean

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self(X).cpu().numpy().ravel()


def mlp_sweep(sweep_values, X_train, y_train, X_val, y_val):
    results = []
    best_model, best_val_mse, best_width = None, np.inf, None
    for width in sweep_values:
        model = MLP(X_train.shape[1], d_hidden=width).fit(X_train, y_train)
        train_mse = np.mean((model.predict(X_train) - y_train) ** 2)
        val_mse = np.mean((model.predict(X_val) - y_val) ** 2)
        results.append({'hidden_width': width, 'train_mse': float(train_mse), 'val_mse': float(val_mse)})
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_width = model, val_mse, width
    best_model.sweep_results_ = results
    return best_model, float(best_val_mse), best_width
