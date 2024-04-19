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
    
    @property
    def step_size(self) -> int:
        return 1 / self.num_steps
    
    @torch.no_grad()
    def navigate(self, x: Tensor) -> Tensor:
        for t in range(self.num_steps):
            t = torch.full((x.size(0), ), t * self.step_size, device=x.device)
            velocity = self.flow_model(x, t)
            x += velocity * self.step_size
        return x
