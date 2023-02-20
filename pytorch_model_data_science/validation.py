"""The module to get test metrics from the trained model
"""

from typing import Any, Tuple

import torch
from sklearn import metrics
from torch.utils.data import DataLoader


def test_classifier_accuracy(
    network: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Any,
    model_type: str = "DNN",
) -> Tuple[float, float, float, float]:
    """Tests trained model

    Parameters
    ----------
    network: torch.nn.Module
        the trained pytorch model
    dataloader: DataLoader
        the dataloader for testing
    loss_fn:
        loss function
    model_type: str, default DNN
        the name of the model

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
    # transform data
    test_data = dataloader.dataset[:][0]
    if model_type in ["CNN", "LSTM"]:
        test_data = test_data[:, None, :]
    network.eval()
    with torch.no_grad():
        pred = network(test_data)
        loss = loss_fn(pred, dataloader.dataset[:][1]).item()
    # metrics
    precision = metrics.precision_score(
        dataloader.dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten() > 0.5,
    )
    recall = metrics.recall_score(
        dataloader.dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten() > 0.5,
    )
    f1 = metrics.f1_score(
        dataloader.dataset[:][1].numpy().flatten(),
        pred.detach().numpy().flatten() > 0.5,
    )
    return loss, precision, recall, f1


def test_regressor_accuracy(
    network: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Any,
    model_type: str = "DNN",
) -> Tuple[float, float, float, float]:
    """Tests trained model

    Parameters
    ----------
    network: torch.nn.Module
        the trained pytorch model
    dataloader: DataLoader
        the dataloader for testing
    loss_fn:
        loss function
    model_type: str, default DNN
        the name of the model

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
    # transform data
    test_data = dataloader.dataset[:][0]
    if model_type in ["CNN", "LSTM"]:
        test_data = test_data[:, None, :]

    network.eval()
    with torch.no_grad():
        pred = network(test_data)
        loss = loss_fn(pred, dataloader[:][1]).item()
    # metrics
    mae = metrics.mean_absolute_error(
        dataloader[:][1].numpy().flatten(),
        pred.detach().numpy().flatten(),
    )
    mse = metrics.mean_squared_error(
        dataloader[:][1].numpy().flatten(),
        pred.detach().numpy().flatten(),
    )
    r2 = metrics.r2_score(
        dataloader[:][1].numpy().flatten(),
        pred.detach().numpy().flatten(),
    )
    return loss, mae, mse, r2
