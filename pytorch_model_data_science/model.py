"""The model used for defining PyTorch models and the following sklearn
estimator
"""

from typing import List, Optional, Tuple

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


# 1D DNN network
class DNN1DNet(nn.Module):
    """The class to define linear Pytorch model

    Note that the input data for this model can be arbitrary dimensions, but the
    input size is better to be set up clearly before.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        usage: str = "regression",
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_size: int
            the number of dimensions of input data
        output_size: int
            the number of dimensions of output data
        hidden_layers: list
            the list containing the in/out data size in the hidden layers
        usage: str, default "regression"
            the goal of the model, regression or classification
        """
        super(DNN1DNet, self).__init__()

        self.net = nn.Sequential()
        self.usage = usage

        # transform into 1D
        self.net.add_module("flatten", nn.Flatten())
        hidden_layers = [input_size] + hidden_layers + [output_size]

        for idx in range(len(hidden_layers) - 1):
            self.net.add_module(
                f"linear_{idx}", nn.Linear(hidden_layers[idx], hidden_layers[idx + 1])
            )
            if idx < (len(hidden_layers) - 2):
                self.net.add_module(
                    f"norm_{idx}", nn.BatchNorm1d(hidden_layers[idx + 1])
                )
                self.net.add_module(f"relu_{idx}", nn.ReLU())
        # when output size is 1, transform output to be probability
        if hidden_layers[-1] == 1 and self.usage == "classification":
            self.net.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Parameters
        ----------
        x: torch.Tensor
            the input tensor with the shape (N_samples, n0, n1, ...), which can
            be the arbitrary dimensions

        Returns
        -------
        torch.Tensor
            the output data after model processing
        """
        return self.net(x)


# 1D CNN network
class CNN1DNet(nn.Module):
    """The class to define a CNN Pytorch model

    Note that this model is strict with the input shape.
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        output_size: int,
        cnn_outputs: List[int],
        kernel_sizes: int,
        max_pools: int,
        linear_layers: List[int],
        usage: str = "regression",
    ) -> None:
        """Constructor

        Parameters
        ----------
        input_shape: tuple
            the input shape in 2D
        output_size: int
            the number of dimensions of output data
        cnn_outputs: list
            the list of CNN layers output channel numbers
        kernel_sizes: int
            the  kernel size in each CNN layer
        max_pools: int
            the pool number in the pool layers after each CNN layer
        linear_layers: list
            the list containing the in/out data size in the hidden layers,
            except the first input data size
        usage: str, default "regression"
            the goal of the model, regression or classification
        """
        super(CNN1DNet, self).__init__()

        self.net = nn.Sequential()
        self.usage = usage
        self.input_shape = input_shape
        for idx in range(len(cnn_outputs)):
            if idx == 0:
                self.net.add_module(
                    f"cnn_{idx}", nn.Conv1d(1, cnn_outputs[idx], kernel_sizes)
                )
            else:
                self.net.add_module(
                    f"cnn_{idx}",
                    nn.Conv1d(cnn_outputs[idx - 1], cnn_outputs[idx], kernel_sizes),
                )
            self.net.add_module(f"norm_{idx}", nn.BatchNorm1d(cnn_outputs[idx]))
            self.net.add_module(f"relu_{idx}", nn.ReLU())
            self.net.add_module(f"maxpool_{idx}", nn.MaxPool1d(max_pools))
            self.input_shape = (
                cnn_outputs[idx],
                (self.input_shape[-1] - kernel_sizes + 1) // max_pools,
            )

        # fully connection layer
        self.net.add_module("fully_connection", nn.Flatten())
        linear_layers = (
            [self.input_shape[0] * self.input_shape[1]] + linear_layers + [output_size]
        )
        for idx in range(len(linear_layers) - 1):
            self.net.add_module(
                f"linear_{idx}", nn.Linear(linear_layers[idx], linear_layers[idx + 1])
            )
            if idx < (len(linear_layers) - 2):
                self.net.add_module(
                    f"linear_norm_{idx}", nn.BatchNorm1d(linear_layers[idx + 1])
                )
                self.net.add_module(f"linear_relu_{idx}", nn.ReLU())

        # when output size is 1, transform output to be probability
        if linear_layers[-1] == 1 and self.usage == "classification":
            self.net.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Parameters
        ----------
        x: torch.Tensor
            the input feature data, with the shape (N_samples, channels, dimension)

        Returns
        -------
        torch.Tensor
            the output data after model processing
        """
        return self.net(x)
