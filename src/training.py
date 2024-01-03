from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[Any, torch.Tensor], torch.Tensor],
    device: str = "cpu",
) -> float:
    """Train the model for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Train dataloader.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_function (Callable[[Any, torch.Tensor], torch.Tensor]): Loss function.
        device (str, optional): Device to use. Defaults to "cpu".

    Returns:
        float: Mean loss value.
    """
    model.train()

    loss_values = []

    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(loss_values) / len(loss_values)


def test_loop(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_function: Callable[[Any, torch.Tensor], torch.Tensor],
    metrics: list[Callable[[Any, torch.Tensor], torch.Tensor]],
    device: str,
) -> tuple[float, list[float]]:
    """Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_loader (DataLoader): Test dataloader.
        loss_function (Callable[[Any, torch.Tensor], torch.Tensor]): Loss function.
        metrics (list[Callable[[Any, torch.Tensor], torch.Tensor]]): List of metrics.
        device (str): Device to use.

    Returns:
        tuple[float, list[float]]: Mean loss value and list of metrics values.
    """
    model.eval()

    loss_values = []

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            output = model(x)

            loss = loss_function(output, y)
            loss_values.append(loss.item())

            output = torch.nn.functional.sigmoid(output)
            for metric in metrics:
                metric.update(output, y)

    return sum(loss_values) / len(loss_values), [
        metric.compute().cpu().numpy() for metric in metrics
    ]
