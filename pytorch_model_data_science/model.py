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
    """The class to define a 3-layer linear Pytorch regression model"""

    def __init__(
        self,
        input_size: int,
        l1: int = 512,
        l2: int = 128,
        l3: int = 64,
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the input size which is the second dim from each batch
        l1: int, default 512
            the number of output samples from the first layer
        l2: int, default 128
            the number of output samples from the second layer
        l3: int, default 64
            the number of output samples from the third layer
        """
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


# CNN network for regression
class CNNNetRegressor(nn.Module):
    """The class to define a 1-layer CNN Pytorch regression model"""

    def __init__(
        self,
        input_size: int,
        out: int = 16,
        kernel_size: int = 3,
        max_pool: int = 2,
        l1: int = 32,
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the number of features for one sample
        out: int, default 16
            the number of output channel for the CNN layer
        kernel_size: int, default 3
            the number of kernel size for the CNN layer
        max_pool: int, default 2
            the number of kernel size for the maxpool layer
        l1: int, default 32
            the number of output samples for the first linear layer
        """
        super(CNNNetRegressor, self).__init__()
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv1d(1, out, kernel_size),
            nn.MaxPool1d(max_pool),
            nn.Flatten(),
            nn.Linear(out * ((input_size - kernel_size + 1) // max_pool), l1),
            nn.ReLU(),
            nn.Linear(l1, 1),
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)


# linear network for classification
class DNNNetClassifier(nn.Module):
    """The class to define a 3-layer linear Pytorch classification model"""

    def __init__(
        self,
        input_size: int,
        l1: int = 512,
        l2: int = 128,
        l3: int = 64,
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the input size which is the second dim from each batch
        l1: int, default 512
            the number of output samples from the first layer
        l2: int, default 128
            the number of output samples from the second layer
        l3: int, default 64
            the number of output samples from the third layer
        """
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


# CNN network for classification
class CNNNetClassifier(nn.Module):
    """The class to define a 1-layer CNN Pytorch classification model"""

    def __init__(
        self,
        input_size: int,
        out: int = 16,
        kernel_size: int = 3,
        max_pool: int = 2,
        l1: int = 32,
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the number of features for one sample
        out: int, default 16
            the number of output channel for the CNN layer
        kernel_size: int, default 3
            the number of kernel size for the CNN layer
        max_pool: int, default 2
            the number of kernel size for the maxpool layer
        l1: int, default 32
            the number of output samples for the first linear layer
        """
        super(CNNNetClassifier, self).__init__()
        self.cnn_relu_stack = nn.Sequential(
            nn.Conv1d(1, out, kernel_size),
            nn.MaxPool1d(max_pool),
            nn.Flatten(),
            nn.Linear(out * ((input_size - kernel_size + 1) // max_pool), l1),
            nn.ReLU(),
            nn.Linear(l1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.cnn_relu_stack(x)
