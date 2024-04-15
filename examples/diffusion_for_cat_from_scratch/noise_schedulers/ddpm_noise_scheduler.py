import torch


class DDPMNoiseScheduler:
    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_steps: int = 1000, 
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        
        self.math = dict()
        self.math['betas'] = torch.linspace(beta_start, beta_end, num_steps)
        self.math['alphas'] = 1.0 - self.math['betas']
        self.math['alphas_bar'] = self.math['alphas'].cumprod(0)
        # used in the corrupt method
        self.math['sqrt(alphas_bar)'] = self.math['alphas_bar'].sqrt()
        self.math['sqrt(1 - alphas_bar)'] = (1 - self.math['alphas_bar']).sqrt()
        # used in the recover method
        self.math['1 / sqrt(alphas)'] = 1 / self.math['alphas'].sqrt()
        self.math['(1 - alphas) / sqrt(1 - alphas_bar)'] = (1 - self.math['alphas']) / (1 - self.math['alphas_bar']).sqrt()
        self.math['sqrt(betas)'] = self.math['betas'].sqrt()
        
    
    def gather(self, math_name: str, t: torch.Tensor) -> torch.Tensor:
        assert math_name in self.math.keys(), f"Math name {math_name} not found in math dictionary"
        return self.math[math_name][t].reshape(-1, 1, 1, 1)
        
    
    def corrupt(
            self, 
            images: torch.Tensor, 
            t: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.gather('sqrt(alphas_bar)', t) * images
        std = self.gather('sqrt(1 - alphas_bar)', t)
        noise = torch.randn_like(images)
        corrupted_images = mean + std * noise
        return corrupted_images, noise
    
    
    def recover(
        self, 
        corrupted_images: torch.Tensor, 
        noise: torch.Tensor, 
        t: torch.Tensor, 
    ) -> torch.Tensor:
        mean = self.gather('1 / sqrt(alphas)', t) * (corrupted_images - self.gather('(1 - alphas) / sqrt(1 - alphas_bar)', t) * noise)
        std = self.gather('sqrt(betas)', t)
        z = torch.randn_like(corrupted_images)
        z[t == 0] = 0
        recovered_images = mean + std * torch.randn_like(corrupted_images)
        return recovered_images
