import torch
from torch import nn


class ResidualLayer(nn.Module):
    def __init__(
        self, 
        operator: nn.Module, 
        activation: nn.Module = None,
    ):
        """
        operator: Operator that is applied to the input tensor.
        activation: Activation function that is applied to the output tensor.
        """
        super().__init__()
        self.operator = operator
        self.activation = activation or nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.operator(x)) 
