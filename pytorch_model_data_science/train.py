"""The module is used for training a PyTorch model
"""

from typing import Any

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def train_classifier(
    network: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Any,
    optimizer: Any,
    class_weight: bool = False,
    batch_size: int = 1024,
    verbose: int = 0,
) -> None:
    """Trains a clssification PyTorch model

    Parameters
    ----------
    network: torch.nn.Module
        the PyTorch classification model
    dataset: Dataset
        the training dataset
    loss_fn: Any
        the loss function
    optimizer: Any
        the optimizer
    class_weight: bool, default False
        the indicator if to consider the class weight
    batch_size: int, default 1024
        the number of the batch size
    verbose: int, default 0
    """
    size = len(dataset)
    running_loss = 0.0
    network.train()

    # class weights
    train_sampler = None
    if class_weight:
        _, counts = np.unique(dataset[:][1].numpy(), return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        train_sample_weight = [
            class_weights[int(i)] for i in dataset[:][1].numpy().flatten()
        ]
        train_sampler = WeightedRandomSampler(
            train_sample_weight,
            len(train_sample_weight),
            replacement=True,
        )

    # build dataloader
    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        shuffle=(train_sampler == None),
    )

    # training
    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = network(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 2000 == 1999:
            if verbose > 0:
                print(
                    (
                        f"loss: {(running_loss / 2000):.6f} "
                        + f"[{(batch+1) * len(X)}/{size}]"
                    )
                )
            running_loss = 0.0

    train_pred = network(dataset[:][0])
    train_loss = loss_fn(train_pred, dataset[:][1]).item()
    precision = metrics.precision_score(
        dataset[:][1].numpy().flatten(),
        train_pred.detach().numpy().flatten() > 0.5,
    )
    recall = metrics.recall_score(
        dataset[:][1].numpy().flatten(),
        train_pred.detach().numpy().flatten() > 0.5,
    )
    f1 = metrics.f1_score(
        dataset[:][1].numpy().flatten(),
        train_pred.detach().numpy().flatten() > 0.5,
    )
    print(f"{'Train':<10}{train_loss:<10}{precision:<15}{recall:<10}{f1:<10}")


def train_regressor(
    network: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Any,
    optimizer: Any,
    batch_size: int = 1024,
    verbose: int = 0,
) -> None:
    """Trains a regression PyTorch model

    Parameters
    ----------
    network: torch.nn.Module
        the PyTorch classification model
    dataset: Dataset
        the training dataset
    loss_fn: Any
        the loss function
    optimizer: Any
        the optimizer
    batch_size: int, default 1024
        the number of the batch size
    verbose: int, default 0
    """
    size = len(dataset)
    running_loss = 0.0
    network.train()

    # build dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # training
    for batch, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = network(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 2000 == 1999:
            if verbose > 0:
                print(
                    (
                        f"loss: {(running_loss / 2000):.6f} "
                        + f"[{(batch+1) * len(X)}/{size}]"
                    )
                )
            running_loss = 0.0

    train_pred = network(dataset[:][0])
    train_loss = loss_fn(train_pred, dataset[:][1]).item()
    mse = metrics.mean_squared_error(
        dataset[:][1].numpy().flatten(),
        train_pred.detach().numpy().flatten(),
    )
    mae = metrics.mean_absolute_error(
        dataset[:][1].numpy().flatten(),
        train_pred.detach().numpy().flatten(),
    )
    r2 = metrics.r2_score(
        dataset[:][1].numpy().flatten(),
        train_pred.detach().numpy().flatten(),
    )
    print(f"{'Train':<10}{train_loss:<10}{mse:<15}{mae:<10}{r2:<10}")
