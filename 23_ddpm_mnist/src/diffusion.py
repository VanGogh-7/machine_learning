import torch


def extract(
    coefficients: torch.Tensor,
    timesteps: torch.Tensor,
    x_shape: tuple[int, ...],
) -> torch.Tensor:
    gathered = coefficients.to(timesteps.device).gather(0, timesteps)
    return gathered.view(timesteps.size(0), *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.num_timesteps = num_timesteps
        # The beta schedule controls how much Gaussian noise is added each step.
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        # alpha_bar is the cumulative product used to jump directly to x_t.
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), self.alpha_bars[:-1]])

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bars, timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            x_start.shape,
        )
        # Forward diffusion creates a noisy image x_t from a clean image x_0.
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alpha_bars_t = extract(
            self.sqrt_one_minus_alpha_bars,
            timesteps,
            x.shape,
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, timesteps, x.shape)
        predicted_noise = model(x, timesteps)

        # The reverse step uses the predicted noise to estimate a less noisy image.
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alpha_bars_t
        )
        posterior_variance_t = extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().view(x.size(0), 1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timesteps)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_with_intermediates(
        self,
        model: torch.nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        capture_steps: list[int],
    ) -> dict[int, torch.Tensor]:
        model.eval()
        captures: dict[int, torch.Tensor] = {}
        capture_set = set(capture_steps)
        x = torch.randn(shape, device=device)
        if self.num_timesteps in capture_set:
            captures[self.num_timesteps] = x.detach().cpu()

        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timesteps)
            if step in capture_set:
                captures[step] = x.detach().cpu()
        return captures

