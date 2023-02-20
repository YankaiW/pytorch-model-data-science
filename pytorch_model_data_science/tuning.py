"""The module used for hyperparameter tuning for PyTorch models
"""

import os
from typing import Any, Dict, Optional

import numpy as np
import ray
import torch
from ray import tune
from sklearn import metrics
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from pytorch_model_data_science import model


def tune_classifier(
    config: Dict[str, Any],
    net_structure: str,
    train_ray: ray.ObjectRef,
    loss_fn: Any,
    val_ray: Optional[ray.ObjectRef] = None,
    val_size: Optional[float] = None,
    last_checkpoint: Optional[str] = None,
    class_weight: bool = False,
    epochs: int = 10,
    verbose: int = 0,
    random_state: int = 0,
) -> None:
    """Hyperparameter tuning for a classification PyTorch model

    Parameters
    ----------
    config: dict
        the dictionary containing the hyperparameter grid
    net_structure: str
        the name of the model
    train_ray: ray.ObjectRef
        the train data id represented by ray.ObjectRef
    loss_fn: Any
        the PyTorch loss function
    val_ray: ray.ObjectRef
        the validation data id represented by ray.ObjectRef
    val_size: float, default None
        the validation data size from the train data
    last_checkpoint: str, default None
        the local checkpoint dir if want to continue from the last time
    class_weight: bool, default False
        the indicator if to use class weight when training
    epochs: int, default 10
        the number of epochs
    verbose: int, default 0
        the number of verbose indicator
    random_state: int, default 0
        the random state
    """
    # build model
    if net_structure == "DNN":
        network = model.DNNNetClassifier(
            input_size=ray.get(train_ray)[0][0].shape[0],
            **config["model_params"],
        )
    else:
        raise NameError("Wrong network name selected: " + net_structure)

    # define optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=config["lr"])

    # load the model and optimizer from the last time
    if last_checkpoint:
        model_state, optimizer_state = torch.load(
            os.path.join(last_checkpoint, "checkpoint")
        )
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # get training and val sets
    assert val_ray != None or val_size != None
    if val_ray:
        train_subset = ray.get(train_ray)
        val_subset = ray.get(val_ray)
    else:
        val_ratio = int(len(ray.get(train_ray)) * val_size)
        train_subset, val_subset = random_split(
            ray.get(train_ray),
            [len(ray.get(train_ray)) - val_ratio, val_ratio],
            generator=torch.Generator().manual_seed(random_state),
        )

    # class weights
    train_sampler = None
    if class_weight:
        _, counts = np.unique(train_subset[:][1].numpy(), return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        train_sample_weight = [
            class_weights[int(i)] for i in train_subset[:][1].numpy().flatten()
        ]
        train_sampler = WeightedRandomSampler(
            train_sample_weight,
            len(train_sample_weight),
            replacement=True,
        )

    # build dataloaders
    train_loader = DataLoader(
        train_subset,
        sampler=train_sampler,
        batch_size=int(config["batch_size"]),
        shuffle=(train_sampler == None),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    # training
    for epoch in range(epochs):
        size = len(train_subset)
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
            if batch % 2000 == 1999:
                if verbose > 0:
                    print(
                        (
                            f"loss: {(running_loss / 2000):.6f} "
                            + f"[{(batch+1)*len(X)}/{size}]"
                        )
                    )
                running_loss = 0.0

        # validation
        with torch.no_grad():
            val_pred = network(val_loader.dataset[:][0])
            val_loss = loss_fn(val_pred, val_loader.dataset[:][1]).item()
        # metrics
        precision = metrics.precision_score(
            val_loader.dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten() > 0.5,
        )
        recall = metrics.recall_score(
            val_loader.dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten() > 0.5,
        )
        f1 = metrics.f1_score(
            val_loader.dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten() > 0.5,
        )

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((network.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss, precision=precision, recall=recall, f1=f1)
    print("Finished Training")


def training_regressor(
    config: Dict[str, Any],
    net_structure: str,
    train_ray: ray.ObjectRef,
    loss_fn: Any,
    val_ray: Optional[ray.ObjectRef] = None,
    val_size: Optional[float] = None,
    last_checkpoint: Optional[str] = None,
    epochs: int = 10,
    verbose: int = 0,
    random_state: int = 0,
) -> None:
    """Hyperparameter tuning for a regression PyTorch model

    Parameters
    ----------
    config: dict
        the dictionary containing the hyperparameter grid
    net_structure: str
        the name of the model
    train_ray: ray.ObjectRef
        the train data id represented by ray.ObjectRef
    loss_fn:
        the pytorch loss function
    val_ray: ray.ObjectRef
        the validation data id represented by ray.ObjectRef
    val_size: float, default None
        the partition ratio of validation set
    last_checkpoint: str, default None
        the local checkpoint dir if want to continue from the last time
    epochs: int, default 10
        the number of epochs
    verbose: int, default 0
        the number of verbose indictor
    random_state: int, default 0
        the random state
    """
    # build model
    if net_structure == "DNN":
        network = model.DNNNetRegressor(
            input_size=ray.get(train_ray)[0][0].shape[0], **config["model_params"]
        )
    else:
        raise NameError("Wrong model name selected: " + net_structure)

    # define optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=config["lr"])

    # load the model and optimizer from the last time
    if last_checkpoint:
        model_state, optimizer_state = torch.load(
            os.path.join(last_checkpoint, "checkpoint")
        )
        network.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # split into training and val sets
    assert val_ray != None or val_size != None
    if val_ray:
        train_subset = ray.get(train_ray)
        val_subset = ray.get(val_ray)
    else:
        val_ratio = int(len(ray.get(train_ray)) * val_size)
        train_subset, val_subset = random_split(
            ray.get(train_ray),
            [len(ray.get(train_ray)) - val_ratio, val_ratio],
            generator=torch.Generator().manual_seed(random_state),
        )

    # build dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    # training
    for epoch in range(epochs):
        size = len(train_subset)
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
            if batch % 2000 == 1999:
                if verbose > 0:
                    print(
                        (
                            f"loss: {(running_loss / 2000):.6f} "
                            + f"[{(batch+1) * len(X)}/{size}]"
                        )
                    )
                running_loss = 0.0

        # validation
        with torch.no_grad():
            val_pred = network(val_loader.dataset[:][0])
            val_loss = loss_fn(val_pred, val_loader.dataset[:][1]).item()
        # metrics
        mae = metrics.mean_absolute_error(
            val_loader.dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        mse = metrics.mean_squared_error(
            val_loader.dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        r2 = metrics.r2_score(
            val_loader.dataset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((network.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss, mae=mae, mse=mse, r2=r2)
    print("Finished Training")
