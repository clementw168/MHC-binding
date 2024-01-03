from typing import Any

import torch

from src.dataset import AA_TO_INDEX, PEPTIDE_LENGTH


class LinearNet(torch.nn.Module):
    """Fully connected network."""

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dims: list[int],
        output_dimension: int,
    ):
        """Initialize the network.

        Args:
            embedding_dimension (int): Dimension of the embedding.
            hidden_dims (list[int]): List of hidden dimensions.
            output_dimension (int): Dimension of the output.
        """
        super().__init__()
        self.output_dimension = output_dimension

        self.embedding = torch.nn.Embedding(len(AA_TO_INDEX) + 1, embedding_dimension)

        hidden_dims = [PEPTIDE_LENGTH * embedding_dimension] + hidden_dims
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            )

        self.hidden_layers.append(torch.nn.Linear(hidden_dims[-1], output_dimension))

    def forward(self, x):
        """Forward pass of the fully connected network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dimension).

        Returns:
            output (torch.Tensor): Output of the network. Shape (batch_size, output_dimension).

        """
        output = x
        output = self.embedding(output)
        output = output.view(-1, PEPTIDE_LENGTH * self.embedding.embedding_dim)

        for layer_index in range(len(self.hidden_layers)):
            output = self.hidden_layers[layer_index](output)
            if layer_index != len(self.hidden_layers) - 1:
                output = torch.nn.functional.relu(output)

        return output.squeeze()


class ConvolutionalNet(torch.nn.Module):
    """Convolutional network."""

    def __init__(
        self,
        embedding_dimension: int,
        filters: list[int],
        output_dimension: int,
    ):
        """Initialize the network.

        Args:
            embedding_dimension (int): Dimension of the embedding.
            filters (list[int]): List of filters.
            output_dimension (int): Dimension of the output.
        """
        super().__init__()
        self.output_dimension = output_dimension

        self.embedding = torch.nn.Embedding(len(AA_TO_INDEX) + 1, embedding_dimension)

        filters = [embedding_dimension] + filters
        self.filters = torch.nn.ModuleList()
        for i in range(len(filters) - 1):
            self.filters.append(
                torch.nn.Conv1d(filters[i], filters[i + 1], kernel_size=3)
            )

        self.linear = torch.nn.Linear(filters[-1], output_dimension)

    def forward(self, x):
        """Forward pass of the convolutional network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dimension).

        Returns:
            output (torch.Tensor): Output of the network. Shape (batch_size, output_dimension).

        """
        output = x
        output = self.embedding(output)
        output = output.permute(0, 2, 1)

        for layer_index in range(len(self.filters)):
            output = self.filters[layer_index](output)

            if layer_index != len(self.filters) - 1:
                output = torch.nn.functional.relu(output)

        output = output.mean(dim=2)

        output = self.linear(output)

        return output.squeeze()


class MergeModels(torch.nn.Module):
    """Average the output of multiple models"""

    def __init__(
        self,
        models_list: list[Any],
        on_probabilities: bool = True,
    ):
        """Initialize the network.

        Args:
            models_list (list[torch.nn.Module]): List of models to merge.
        """
        super().__init__()
        self.models_list = torch.nn.ModuleList(models_list)
        self.on_probabilities = on_probabilities

    def forward(self, x):
        """Forward pass of the merged models.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dimension).

        Returns:
            output (torch.Tensor): Output of the network. Shape (batch_size, output_dimension).

        """
        outputs = [model(x) for model in self.models_list]

        if self.on_probabilities:
            outputs = [torch.nn.functional.sigmoid(output) for output in outputs]

        outputs = sum(outputs) / len(outputs)

        if self.on_probabilities:
            outputs = torch.special.logit(outputs)

        return outputs
