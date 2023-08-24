"""The model used for defining PyTorch models and the following sklearn
estimator
"""

from typing import Optional

import numpy as np
import torch
from sklearn import base, metrics
from torch import nn


class EarlyStopper:
    """The class used for early stopping during training when the loss doesn't
    decrease validly after some patience steps
    """

    def __init__(self, patience: int = 1, min_delta: float = 0) -> None:
        """Constructor

        Parameters
        ----------
        patience: int, default 1
            the number of steps after which the training stops if the loss
            doesn't decrease
        min_delta: float, default 0
            the minimal delta, if the current loss is more than the sum of the
            delta and the minimal loss, the counter will be added 1 as one
            non-decreasing iteration
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def __call__(self, loss: float) -> bool:
        """Checks whether the non-valid non-decreasing loss is accumulated up to
        the limit patience

        Parameters
        ----------
        loss: float
            the current loss

        Returns
        -------
        bool
            the indicator if to stop the training
        """
        if loss < self.min_loss:
            # once there is a new minimal loss
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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
        output_size: int,
        l1: int = 512,
        l2: int = 128,
        l3: int = 64,
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the input size which is the second dim from each batch
        output_size: int
            the final output dimension
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
            nn.Linear(l3, output_size),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.flatten(x)
        return self.linear_relu_stack(x)


# CNN network for regression
class CNNNetRegressor(nn.Module):
    """The class to define a 1-layer CNN Pytorch regression model"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
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
        output_size: int
            the final output dimension
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
            nn.Linear(l1, output_size),
        )

    def forward(self, x) -> torch.Tensor:
        return self.cnn_relu_stack(x)


# linear network for classification
class DNNNetClassifier(nn.Module):
    """The class to define a 3-layer linear Pytorch classification model"""

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        l1: int = 512,
        l2: int = 128,
        l3: int = 64,
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the input size which is the second dim from each batch
        output_size: int, default 1
            the dimension of the output, the default 1 means univariate
            prediction
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
            nn.Linear(l3, output_size),
        )
        if output_size == 1:
            self.linear_relu_stack.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x) -> torch.Tensor:
        x = self.flatten(x)
        return self.linear_relu_stack(x)


# CNN network for classification
class CNNNetClassifier(nn.Module):
    """The class to define a 1-layer CNN Pytorch classification model"""

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
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
        output_size: int, default 1
            the dimension of the output, the default 1 means univariate
            prediction
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
            nn.Linear(l1, output_size),
        )
        if output_size == 1:
            self.cnn_relu_stack.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x) -> torch.Tensor:
        return self.cnn_relu_stack(x)
