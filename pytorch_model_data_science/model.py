"""The model used for defining PyTorch models and the following sklearn
estimator
"""

from typing import Optional

import numpy as np
import torch
from sklearn import base, metrics
from torch import nn


class PyTorchEstimator(base.BaseEstimator, base.RegressorMixin):
    """The class to define a sklearn estimator from PyTorch model"""

    def __init__(self, model: nn.Module, target: str) -> None:
        """Constructor

        Parameters
        ----------
        model: nn.Module
            the model from PyTorch
        target: str
            the task what the model does, including classification and
            regression
        """
        self.model = model
        self.target = target

        self._check_valid_target()

    def _check_valid_target(self) -> None:
        """Checks if the target is valid"""
        assert self.target in [
            "classification",
            "regression",
        ], f"Wrong target name: {self.target}"

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Simulates fit process

        Parameters
        ----------
        X: np.ndarray
            the feature matrix for training
        y: np.ndarray, default None
            the target matrix for training
        """
        pass

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Prediction for regression

        Parameters
        ----------
        X: np.ndarray
            the feature matrix
        y: np.ndarray, default None
            the target martix

        Returns
        -------
        np.ndarray
            the predicted array
        """
        return self.model(torch.Tensor(X)).detach().numpy().flatten()

    def predict_proba(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Prediction for classification

        Parameters
        ----------
        X: np.ndarray
            the feature matrix
        y: np.ndarray, default None
            the target martix

        Returns
        -------
        np.ndarray
            the predicted array
        """
        return self.model(torch.Tensor(X)).detach().numpy()

    def score(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Obtains MSE score

        Parameters
        ----------
        X: np.ndarray
            the feature matrix
        y: np.ndarray
            the real target matrix
        sample_weight: np.ndarray, default None
            the sample weight array

        Returns
        -------
        float
            the scoare
        """
        y_pred = self.predict(X)
        if self.target == "classification":
            return metrics.f1_score(y, y_pred > 0.5)
        elif self.target == "regression":
            return metrics.mean_squared_error(y, y_pred)


# linear network for regression
class DNNNetRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        l1: int = 512,
        l2: int = 128,
        l3: int = 64,
    ) -> None:
        super(DNNNetRegressor, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


# linear network for classification
class DNNNetClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        l1: int = 512,
        l2: int = 128,
        l3: int = 64,
    ) -> None:
        super(DNNNetClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, l3),
            nn.ReLU(),
            nn.Linear(l3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
