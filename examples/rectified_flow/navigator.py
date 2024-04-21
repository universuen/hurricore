import torch
from torch import Tensor
from torch.nn import Module


class Navigator:
    def __init__(
        self,
        flow_model: Module,
        num_steps: int,
    ) -> None:
        self.flow_model = flow_model
        self.num_steps = num_steps
        self.step_size = 1 / num_steps
    
    @torch.no_grad()
    def navigate(self, x: Tensor) -> Tensor:
        for step in range(self.num_steps):
            t = torch.full((x.size(0), ), step * self.step_size, device=x.device)
            velocity = self.flow_model(x, t)
            x += velocity * self.step_size
        return x

    @torch.no_grad()
    def step(self, x: Tensor, step: int, reversed: bool = False) -> Tensor:
        t = torch.full((x.size(0), ), step * self.step_size, device=x.device)
        velocity = self.flow_model(x, t)
        if reversed:
            return x - velocity * self.step_size
        else:
            return x + velocity * self.step_size
