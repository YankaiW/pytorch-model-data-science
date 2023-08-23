"""The module is used for training a PyTorch model
"""

from typing import Any, Optional

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def train_classifier(
    network: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Any,
    optimizer: Any,
    multiclass: bool = False,
    val_dataset: Optional[Dataset] = None,
    class_weight: bool = False,
    batch_size: int = 1024,
    epochs: int = 100,
    num_workers: int = 0,
    verbose: int = 0,
    visual_batch: int = 2000,
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
    val_dataset: Dataset, default = None
        the validation dataset
    class_weight: bool, default False
        the indicator if to consider the class weight
    batch_size: int, default 1024
        the number of the batch size
    epochs: int, default 100
        the number of training epochs
    num_workers: int, default 0
        the number of workers for data processing, default 0 means that the
        data loading is synchronous and done in the main process
    verbose: int, default 0
        0 means no logs, 1 means epoch logs, 2 means batch logs
    visual_batch: int, default 2000
        the number of batches when to show the on-going loss
    """
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
        num_workers=num_workers,
    )
    size = len(train_loader)
    # if there is not validation data, use training data instead
    if not val_dataset:
        val_dataset = dataset
    if multiclass:
        val_real = torch.argmax(val_dataset[:][1], dim=1)
        val_real = val_real.numpy()
    else:
        val_real = val_dataset[:][1].numpy().flatten()
    # define method for F1 score
    average = "weighted" if multiclass else "binary"

    # training
    for epoch in range(epochs):
        running_loss = 0.0
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % visual_batch == (visual_batch - 1):
                if verbose > 1:
                    print(
                        (
                            f"epoch {epoch + 1}  batch [{batch+1:<4}/{size}]"
                            + f"  loss: {(running_loss / visual_batch):.6f}"
                        )
                    )
                running_loss = 0.0

        with torch.no_grad():
            val_pred = network(val_dataset[:][0])
            val_loss = loss_fn(val_pred, val_dataset[:][1]).item()
        # transform for univariate or multi-class prediction
        if multiclass:
            val_pred = torch.argmax(val_pred, dim=1)
            val_pred = val_pred.numpy()
        else:
            val_pred = val_pred.detach().numpy().flatten() > 0.5
        precision = metrics.precision_score(
            val_real,
            val_pred,
            average=average,
        )
        recall = metrics.recall_score(
            val_real,
            val_pred,
            average=average,
        )
        f1 = metrics.f1_score(
            val_real,
            val_pred,
            average=average,
        )
        if verbose > 0:
            print(
                f"Epoch [{epoch+1:<3}/{epochs}] loss:{round(val_loss, 5):<8}"
                + f"precision:{round(precision, 5):<8}"
                + f"recall:{round(recall, 5):<8}F1:{round(f1, 5):<8}"
            )


def train_regressor(
    network: torch.nn.Module,
    dataset: Dataset,
    loss_fn: Any,
    optimizer: Any,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 1024,
    epochs: int = 100,
    num_workers: int = 0,
    verbose: int = 0,
    visual_batch: int = 2000,
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
    val_dataset: Dataset, default = None
        the validation dataset
    batch_size: int, default 1024
        the number of the batch size
    epochs: int, default 100
        the number of training epochs
    num_workers: int, default 0
        the number of workers for data processing, default 0 means that the
        data loading is synchronous and done in the main process
    verbose: int, default 0
        0 means no logs, 1 means epoch logs, 2 means batch logs
    visual_batch: int, default 2000
        the number of batches when to show the on-going loss
    """
    network.train()

    # build dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    size = len(train_loader)
    # if there is not validation data, use training data instead
    if not val_dataset:
        val_dataset = dataset

    # training
    for epoch in range(epochs):
        running_loss = 0.0
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % visual_batch == (visual_batch - 1):
                if verbose > 1:
                    print(
                        (
                            f"epoch {epoch + 1}  batch [{batch+1:<4}/{size}]"
                            + f"  loss: {(running_loss / visual_batch):.6f}"
                        )
                    )
                running_loss = 0.0

        with torch.no_grad():
            val_pred = network(val_dataset[:][0])
            val_loss = loss_fn(val_pred, val_dataset[:][1]).item()
        mse = metrics.mean_squared_error(
            val_dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        mae = metrics.mean_absolute_error(
            val_dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        r2 = metrics.r2_score(
            val_dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        if verbose > 0:
            print(
                f"Epoch [{epoch + 1:<3}/{epochs}] loss:{round(val_loss, 5):<8}"
                + f"MSE:{round(mse, 5):<8}MAE:{round(mae, 5):<8}"
                + f"R2:{round(r2, 5):<8}"
            )
