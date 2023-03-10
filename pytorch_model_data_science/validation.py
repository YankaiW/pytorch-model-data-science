"""The module to get test metrics from the trained model
"""

from typing import Any, Tuple

import torch
from sklearn import metrics
from torch.utils.data import Dataset


def test_classifier_accuracy(
    network: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Any,
) -> Tuple[float, float, float, float]:
    """Tests trained model

    Parameters
    ----------
    network: torch.nn.Module
        the trained pytorch model
    dataset: Dataset
        the dataset for testing
    loss_fn:
        loss function

    Returns
    -------
    float
        loss value
    float
        precision
    float
        recall
    float
        f1 score
    """
    network.eval()
    with torch.no_grad():
        pred = network(dataset[:][0])
        loss = loss_fn(pred, dataset[:][1]).item()
    # metrics
    precision = metrics.precision_score(
        dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten() > 0.5,
    )
    recall = metrics.recall_score(
        dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten() > 0.5,
    )
    f1 = metrics.f1_score(
        dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten() > 0.5,
    )
    return loss, precision, recall, f1


def test_regressor_accuracy(
    network: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Any,
) -> Tuple[float, float, float, float]:
    """Tests trained model

    Parameters
    ----------
    network: torch.nn.Module
        the trained pytorch model
    dataloader: Dataset
        the dataset for testing
    loss_fn:
        loss function

    Returns
    -------
    float
        loss value
    float
        precision
    float
        recall
    float
        f1 score
    """
    network.eval()
    with torch.no_grad():
        pred = network(dataset[:][0])
        loss = loss_fn(pred, dataset[:][1]).item()
    # metrics
    mae = metrics.mean_absolute_error(
        dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten(),
    )
    mse = metrics.mean_squared_error(
        dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten(),
    )
    r2 = metrics.r2_score(
        dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten(),
    )
    return loss, mae, mse, r2
