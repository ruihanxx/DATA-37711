"""
Skeleton code for creating the Random Kitchen Sinks (RKS) and 1-hidden-layer MLP (MLP) models.

Note you are *welcome* to completely ignore this file and use your own code, but it's here if you want to use it as a starting point.
"""
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn

"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            D: total number of random features (= 2h in the assignment text; if you use both sin and cos features, D should be an even number)
            sigma: scale parameter for the RBF kernel, used for initializing the weight parameters
            ridge_lambda: ridge regularization strength
        """
        # TODO: initialize the model parameters
        pass

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Convert the input data X into the random kitchen sinks features. You will likely need to use the parameters initialized in the __init__ method.
        The resulting features should be of shape (n, D or D[+1]), where the plus one is for the intercept term.
        """
        # TODO: implement the feature transformation
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        # TODO: optional code before getting features
        phi = self._features(X)  # shape (n, D or D[+1])
        # TODO: fit the ridge regression model and obtain the fitted ridge coefficients
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # shape (n, D or D[+1])
        return phi @ self.coef_  # shape (n,)

def sweep_rks_model(sweep_D: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    An example sweeping function that sweeps over the number of random features D, it then returns the best model in terms of validation MSE.
    Args:
        sweep_D: list of numbers of random features to sweep over
        X_train: training input data
        y_train: training target values
        X_val: validation input data
        y_val: validation target values
    Returns:
        a tuple (best_model, best_val_mse, best_D)
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None
    for D in sweep_D:
        # TODO: fit the model and get the validation MSE
        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_D = D

    return best_model, best_val_mse, best_D

"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128, activation: str = "relu", device: str = "cpu"):
        """
        Potential hyper-parameters you can consider (feel free to add more):
            d_hidden: hidden dimension
            activation: activation function
            lr: learning rate
            batch_size: batch size
        """
        super().__init__()
        # DO NOT forget this -- GPU will make NN training much faster if you have access to one
        self.device = device
        # TODO: build up the 1-hidden-layer MLP model with modules like `nn.Linear`, `nn.Sequential`, `nn.ReLU`, etc.
        self.net = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # turn the numpy data into torch tensors and move to the device
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        self.net.train()  # make sure to set the model to training mode
        # TODO: fit the MLP model and obtain the fitted MLP coefficients
        # optimizer = xxx
        # loss_fn = yyy
        # for epoch in range(num_epochs):
        #     # set up batch data loader and fit the model within each batch
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy()

def mlp_sweep(sweep_values: List[float], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple:
    """
    See the example `sweep_rks_model` above for inspiration. Good luck!
    """
    pass
