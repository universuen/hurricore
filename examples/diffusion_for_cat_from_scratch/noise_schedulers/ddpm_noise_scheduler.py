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
        self.math['sqrt(betas)'] = self.math['betas'].sqrt()
        self.math['alphas'] = 1 - self.math['betas']
        self.math['alphas_bar'] = self.math['alphas'].cumprod(0)
        self.math['sqrt(alphas_bar)'] = self.math['alphas_bar'].sqrt()
        self.math['sqrt(1 - alphas_bar)'] = (1 - self.math['alphas_bar']).sqrt()
        self.math['1 / sqrt(alphas)'] = 1 / self.math['alphas'].sqrt()
    
    
    def corrupt(
            self, 
            images: torch.Tensor, 
            t: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(images)
        coefficient_1 = self.math['sqrt(alphas_bar)'][t].view(-1, 1, 1, 1).to(images.device)
        coefficient_2 = self.math['sqrt(1 - alphas_bar)'][t].view(-1, 1, 1, 1).to(images.device)
        corrupted_images = coefficient_1 * images + coefficient_2 * noise
        return corrupted_images, noise
    
    
    def recover(
        self, 
        corrupted_images: torch.Tensor, 
        noise: torch.Tensor, 
        t: torch.Tensor, 
        with_randomness: bool = True,
    ) -> torch.Tensor:
        coefficient_1 = self.math['1 / sqrt(alphas)'][t].view(-1, 1, 1, 1).to(corrupted_images.device)
        coefficient_2 = (self.math['betas'][t] / self.math['sqrt(1 - alphas_bar)'][t]).view(-1, 1, 1, 1).to(corrupted_images.device)
        coefficient_3 = self.math['sqrt(betas)'][t].view(-1, 1, 1, 1).to(corrupted_images.device)
        mean = coefficient_1 * (corrupted_images - coefficient_2 * noise)
        std = coefficient_3
        if with_randomness:
            z = torch.randn_like(corrupted_images).to(corrupted_images.device)
            return mean + std * z
        else:
            return mean
