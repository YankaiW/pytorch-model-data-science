"""The module is used for training a PyTorch model
"""

import logging
import os
import tempfile
from typing import Any, Callable, Dict, Optional

import numpy as np
import ray
import torch
from ray import train
from sklearn import metrics
from torch.utils import data

from pytorch_model_data_science.model import *  # noqa: F403

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_classifier(
    config: Dict[str, Any],
    network_name: str,
    train_ray: ray.ObjectRef,
    loss_fn: Callable,
    val_ray: Optional[ray.ObjectRef] = None,
    val_size: Optional[float] = None,
    last_checkpoint: Optional[str] = None,
    class_weight: bool = False,
    num_workers: int = 0,
    multiclass: bool = False,
    epochs: int = 10,
    early_stopping: int = 0,
    verbose: int = 0,
    visual_batch: int = 2000,
    random_state: int = 0,
) -> None:
    """Hyperparameter tuning for a classification PyTorch model

    Parameters
    ----------
    config: dict
        the dictionary containing the hyperparameter grid
    network_name: str
        the name of the model, DNN or CNN
    train_ray: ray.ObjectRef
        the train data id represented by ray.ObjectRef
    loss_fn: Callable
        the PyTorch loss function
    val_ray: ray.ObjectRef
        the validation data id represented by ray.ObjectRef
    val_size: float, default None
        the validation data size from the train data
    last_checkpoint: str, default None
        the local checkpoint dir if want to continue from the last time
    class_weight: bool, default False
        the indicator if to use class weight when training
    num_workers: int, default 0
        the number of cpus when loading data
    multiclass: bool, default False
        the indicator if this is a multi-label classification problem
    epochs: int, default 10
        the number of epochs
    early_stopping: int, default 0
        the number of patience for early stopping, the default 0 means no early
        stopper applied
    verbose: int, default 0
        0 means no logs, 1 means epoch logs, 2 means batch logs
    visual_batch: int, default 2000
        the number of batches when to show the on-going loss
    random_state: int, default 0
        the random state
    """
    # build model
    if network_name == "DNN1DNet":
        network = DNN1DNet(
            usage="classification",
            input_size=ray.get(train_ray)[0][0].shape[-1],
            output_size=(torch.max(ray.get(train_ray)[:][1]).item() + 1),
            **config["model_parameters"],
        )
    elif network_name == "CNN1DNet":
        network = CNN1DNet(
            usage="classification",
            input_shape=(
                ray.get(train_ray)[0][0].shape[-2],
                ray.get(train_ray)[0][0].shape[-1],
            ),
            output_size=(torch.max(ray.get(train_ray)[:][1]).item() + 1),
            **config["model_parameters"],
        )
    else:
        raise NameError(f"Invalid network name: {network_name}")

    # define optimizer
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), lr=config["lr"])
    else:
        raise NameError(f"Invalid optimizer name: {config['optimizer']}")

    # load the model and optimizer from the last time
    if last_checkpoint:
        model_state, optimizer_state = torch.load(
            os.path.join(last_checkpoint, "checkpoint")
        )
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # get training and val sets
    assert val_ray != None or val_size != None, "No available validation data."
    if val_ray:
        train_subset = ray.get(train_ray)
        val_subset = ray.get(val_ray)
    else:
        val_ratio = int(len(ray.get(train_ray)) * val_size)
        train_subset, val_subset = data.random_split(
            ray.get(train_ray),
            [len(ray.get(train_ray)) - val_ratio, val_ratio],
            generator=torch.Generator().manual_seed(random_state),
        )

    # define method for metrics
    average = "weighted" if multiclass else "binary"

    # class weights
    train_sampler = None
    if class_weight:
        _, counts = np.unique(train_subset[:][1].numpy(), return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        train_sample_weight = [
            class_weights[int(i)] for i in train_subset[:][1].numpy().flatten()
        ]
        train_sampler = data.WeightedRandomSampler(
            train_sample_weight,
            len(train_sample_weight),
            replacement=True,
        )

    # build dataloaders
    train_loader = data.DataLoader(
        train_subset,
        sampler=train_sampler,
        batch_size=int(config["batch_size"]),
        shuffle=(train_sampler == None),
        num_workers=num_workers,
    )
    # check early stopping
    if early_stopping > 0:
        early_stopper = EarlyStopper(patience=early_stopping)

    # training
    size = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0.0
        network.train()
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # running loss visualization
            if batch % visual_batch == (visual_batch - 1):
                if verbose > 1:
                    print(
                        (
                            f"epoch {epoch + 1}  batch [{batch+1:<4}/{size}]"
                            + f"  loss: {(running_loss / visual_batch):.6f}"
                        )
                    )
                running_loss = 0.0

        # validation
        network.eval()
        with torch.no_grad():
            val_pred = network(val_subset[:][0])
            val_loss = loss_fn(val_pred, val_subset[:][1]).item()
        # transform for univariate or multi-class prediction
        if multiclass:
            val_pred = torch.argmax(val_pred, dim=1).numpy()
        else:
            val_pred = val_pred.detach().numpy().flatten() > 0.5
        # metrics
        accuracy = metrics.accuracy_score(val_subset[:][1].numpy(), val_pred)
        f1 = metrics.f1_score(
            val_subset[:][1].numpy(),
            val_pred,
            average=average,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(
                {"loss": val_loss, "accuracy": accuracy, "f1": f1},
                checkpoint=train.Checkpoint.from_directory(tempdir),
            )

        if early_stopping > 0 and early_stopper(loss=val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    logger.info("Finished Training")


def train_regressor(
    config: Dict[str, Any],
    network_name: str,
    train_ray: ray.ObjectRef,
    loss_fn: Callable,
    val_ray: Optional[ray.ObjectRef] = None,
    val_size: Optional[float] = None,
    last_checkpoint: Optional[str] = None,
    num_workers: int = 0,
    epochs: int = 10,
    early_stopping: int = 0,
    verbose: int = 0,
    visual_batch: int = 2000,
    random_state: int = 0,
) -> None:
    """Hyperparameter tuning for a regression PyTorch model

    Parameters
    ----------
    config: dict
        the dictionary containing the hyperparameter grid
    network_name: str
        the name of the model, DNN or CNN
    train_ray: ray.ObjectRef
        the train data id represented by ray.ObjectRef
    loss_fn: Callable
        the PyTorch loss function
    val_ray: ray.ObjectRef
        the validation data id represented by ray.ObjectRef
    val_size: float, default None
        the validation data size from the train data
    last_checkpoint: str, default None
        the local checkpoint dir if want to continue from the last time
    num_workers: int, default 0
        the number of cpus when loading data
    epochs: int, default 10
        the number of epochs
    early_stopping: int, default 0
        the number of patience for early stopping, the default 0 means no early
        stopper applied
    verbose: int, default 0
        0 means no logs, 1 means epoch logs, 2 means batch logs
    visual_batch: int, default 2000
        the number of batches when to show the on-going loss
    random_state: int, default 0
        the random state
    """
    # build model
    if network_name == "DNN1DNet":
        network = DNN1DNet(
            usage="regression",
            input_size=ray.get(train_ray)[0][0].shape[-1],
            output_size=ray.get(train_ray)[0][1].shape[-1],
            **config["model_parameters"],
        )
    elif network_name == "CNN1DNet":
        network = CNN1DNet(
            usage="regression",
            input_shape=(
                ray.get(train_ray)[0][0].shape[-2],
                ray.get(train_ray)[0][0].shape[-1],
            ),
            output_size=ray.get(train_ray)[0][1].shape[-1],
            **config["model_parameters"],
        )
    else:
        raise NameError(f"Invalid network name: {network_name}")

    # define optimizer
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=config["lr"])
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), lr=config["lr"])
    else:
        raise NameError(f"Invalid optimizer name: {config['optimizer']}")

    # load the model and optimizer from the last time
    if last_checkpoint:
        model_state, optimizer_state = torch.load(
            os.path.join(last_checkpoint, "checkpoint")
        )
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # get training and val sets
    assert val_ray != None or val_size != None, "No available validation data."
    if val_ray:
        train_subset = ray.get(train_ray)
        val_subset = ray.get(val_ray)
    else:
        train_subset, val_subset = data.random_split(
            ray.get(train_ray),
            [1 - val_size, val_size],
            generator=torch.Generator().manual_seed(random_state),
        )

    # build dataloaders
    train_loader = data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
    )
    # check early stopping
    if early_stopping > 0:
        early_stopper = EarlyStopper(patience=early_stopping)

    # training
    size = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0.0
        network.train()
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = network(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # running loss visualization
            if batch % visual_batch == (visual_batch - 1):
                if verbose > 1:
                    print(
                        (
                            f"epoch {epoch + 1}  batch [{batch+1:<4}/{size}]"
                            + f"  loss: {(running_loss / visual_batch):.6f}"
                        )
                    )
                running_loss = 0.0

        # validation
        network.eval()
        with torch.no_grad():
            val_pred = network(val_subset[:][0])
            val_loss = loss_fn(val_pred, val_subset[:][1]).item()
        # metrics
        mae = metrics.mean_absolute_error(
            val_subset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        r2 = metrics.r2_score(
            val_subset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )

        with tempfile.TemporaryDirectory() as tempdir:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(tempdir, "checkpoint.pt"),
            )
            train.report(
                {"loss": val_loss, "mae": mae, "r2": r2},
                checkpoint=train.Checkpoint.from_directory(tempdir),
            )

        if early_stopping > 0 and early_stopper(loss=val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    logger.info("Finished Training")
