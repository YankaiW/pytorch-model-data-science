"""The module used for hyperparameter tuning for PyTorch models
"""

import logging
import os
from typing import Any, Callable, Dict, Optional

import ray
import torch
from ray import tune
from sklearn import metrics
from torch.utils.data import DataLoader, random_split

from pytorch_model_data_science.model import *  # noqa: F403

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# train regression model
def train_regressor(
    config: Dict[str, Any],
    network_name: str,
    train_ray: ray.ObjectRef,
    loss_fn: Callable,
    val_ray: Optional[ray.ObjectRef] = None,
    val_size: Optional[float] = None,
    num_workers: int = 0,
    last_checkpoint: Optional[str] = None,
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
        the pytorch loss function
    val_ray: ray.ObjectRef
        the validation data id represented by ray.ObjectRef
    val_size: float, default None
        the partition ratio of validation set
    num_workers: int, default 0
        the number of cpus when loading data
    last_checkpoint: str, default None
        the local checkpoint dir if want to continue from the last time
    epochs: int, default 10
        the number of epochs
    early_stopping: int, default 0
        the number of patience for early stopping, the default 0 means no early
        stopper applied
    verbose: int, default 0
        the number of verbose indictor
    visual_batch: int, default 2000
        the number of batches when to show the on-going loss
    random_state: int, default 0
        the random state
    """
    # build model
    if network_name == "DNN":
        network = DNNNetRegressor(
            input_size=ray.get(train_ray)[0][0].shape[-1],
            output_size=ray.get(train_ray)[0][1].shape[-1],
            **config["model_params"],
        )
    elif network_name == "CNN":
        network = CNNNetRegressor(
            input_size=ray.get(train_ray)[0][0].shape[-1],
            output_size=ray.get(train_ray)[0][1].shape[-1],
            **config["model_params"],
        )
    else:
        raise NameError(f"Wrong model name selected: {network_name}")

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
        with torch.no_grad():
            val_pred = network(val_subset[:][0])
            val_loss = loss_fn(val_pred, val_subset[:][1]).item()
        # metrics
        mae = metrics.mean_absolute_error(
            val_subset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        mse = metrics.mean_squared_error(
            val_subset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )
        r2 = metrics.r2_score(
            val_subset[:][1].numpy().flatten(),
            val_pred.detach().numpy().flatten(),
        )

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((network.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss, mae=mae, mse=mse, r2=r2)
        if early_stopping > 0 and early_stopper(loss=val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    logger.info("Finished Training")
