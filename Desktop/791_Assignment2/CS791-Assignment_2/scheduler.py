# import torch

# class NoiseSchedulerDDPM():
#     """
#     Noise scheduler for the DDPM model

#     Args:
#         num_timesteps: int, the number of timesteps
#         type: str, the type of scheduler to use
#         **kwargs: additional arguments for the scheduler

#     This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
#     """
#     def __init__(self, num_timesteps=50, type="linear", **kwargs):

#         self.num_timesteps = num_timesteps
#         self.type = type

#         if type == "linear":
#             self.init_linear_schedule(**kwargs)
#         else:
#             raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


#     def init_linear_schedule(self, beta_start, beta_end):
#         """
#         Precompute whatever quantities are required for training and sampling
#         """

#         self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)

#         self.alphas = None

#     def __len__(self):
#         return self.num_timesteps
    
# class MaskSchedulerD3PM():
#     """
#     Mask scheduler for Discrete Diffusion (D3PM) models.

#     Args:
#         num_timesteps: int, number of timesteps in the diffusion process
#         mask_type: str, type of mask scheduling ("uniform", "linear", etc.)
#         **kwargs: additional arguments for mask scheduling

#     This object sets up the mask schedule for each timestep.
#     """

#     def __init__(self, num_timesteps=50, mask_type="uniform", **kwargs):
#         self.num_timesteps = num_timesteps
#         self.mask_type = mask_type

#         if mask_type == "linear":
#             self.init_linear_schedule(**kwargs)
#         else:
#             raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

#     def init_linear_schedule(self):
#         """
#         Initializes a linear mask schedule where the mask probability increases linearly.
#         """
#         self.mask_probs = None

#     def __len__(self):
#         return self.num_timesteps


# # import torch
# # import torch.nn.functional as F

# # class NoiseSchedulerDDPM():
# #     """
# #     Noise scheduler for the DDPM model
# #     """
# #     def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu', type="linear", **kwargs):
# #         self.num_timesteps = num_timesteps
# #         self.device = device
# #         self.type = type
# #         self.beta_start = beta_start
# #         self.beta_end = beta_end

# #         if type == "linear":
# #             self.init_linear_schedule()
# #         else:
# #             raise NotImplementedError(f"{type} scheduler is not implemented")

# #     def init_linear_schedule(self):
# #         """
# #         Precomputes schedule values for training and sampling.
# #         """
# #         self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32, device=self.device)
# #         self.alphas = 1. - self.betas
# #         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
# #         # Calculations for forward process (add_noise)
# #         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
# #         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

# #         # Calculations for reverse process (step)
# #         self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
# #         self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

# #     def add_noise(self, original_images, t):
# #         noise = torch.randn_like(original_images)
# #         shape = original_images.shape
        
# #         sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(shape[0], 1, 1, 1)
# #         sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(shape[0], 1, 1, 1)

# #         noisy_images = sqrt_alpha_t * original_images + sqrt_one_minus_alpha_t * noise
# #         return noisy_images, noise

# #     def step(self, predicted_noise, t, current_image):
# #         beta_t = self.betas[t]
# #         alpha_t = self.alphas[t]
# #         sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
# #         sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
        
# #         model_mean = sqrt_recip_alpha_t * (current_image - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
# #         if t == 0:
# #             return model_mean
# #         else:
# #             posterior_variance_t = self.posterior_variance[t]
# #             noise = torch.randn_like(current_image)
# #             return model_mean + torch.sqrt(posterior_variance_t) * noise

# #     def __len__(self):
# #         return self.num_timesteps

# # class MaskSchedulerD3PM():
# #     def __init__(self, num_timesteps=50, mask_type="uniform", **kwargs):
# #         self.num_timesteps = num_timesteps
# #         self.mask_type = mask_type

# #         if mask_type == "linear":
# #             self.init_linear_schedule(**kwargs)
# #         else:
# #             raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")

# #     def init_linear_schedule(self):
# #         self.mask_probs = None

# #     def __len__(self):
# #         return self.num_timesteps

# # import torch
# # import torch.nn.functional as F

# # # --- Helper functions to generate schedules ---

# # def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
# #     return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

# # def cosine_beta_schedule(timesteps, s=0.008):
# #     """
# #     Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
# #     """
# #     steps = timesteps + 1
# #     x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
# #     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
# #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
# #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
# #     return torch.clamp(betas, 0.0001, 0.9999)

# # # --- The main scheduler class ---

# # class NoiseSchedulerDDPM():
# #     def __init__(self, num_timesteps=1000, schedule_type="linear", device='cpu', **kwargs):
# #         self.num_timesteps = num_timesteps
# #         self.device = device

# #         if schedule_type == "linear":
# #             self.betas = linear_beta_schedule(num_timesteps, **kwargs).to(self.device)
# #         elif schedule_type == "cosine":
# #             self.betas = cosine_beta_schedule(num_timesteps, **kwargs).to(self.device)
# #         else:
# #             raise NotImplementedError(f"{schedule_type} scheduler is not implemented")
        
# #         # Pre-compute schedule values
# #         self.alphas = 1. - self.betas
# #         self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
# #         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
# #         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
# #         self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
# #         self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

# #     def add_noise(self, original_images, t):
# #         noise = torch.randn_like(original_images)
# #         shape = original_images.shape
# #         sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(shape[0], 1, 1, 1)
# #         sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(shape[0], 1, 1, 1)
# #         noisy_images = sqrt_alpha_t * original_images + sqrt_one_minus_alpha_t * noise
# #         return noisy_images, noise

# #     def step(self, predicted_noise, t, current_image):
# #         beta_t = self.betas[t]
# #         alpha_t = self.alphas[t]
# #         sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
# #         sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t)
# #         model_mean = sqrt_recip_alpha_t * (current_image - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
# #         if t == 0:
# #             return model_mean
# #         else:
# #             posterior_variance_t = self.posterior_variance[t]
# #             noise = torch.randn_like(current_image)
# #             return model_mean + torch.sqrt(posterior_variance_t) * noise

# #     def __len__(self):
# #         return self.num_timesteps

# # # (Your MaskSchedulerD3PM class can remain unchanged below this)
# # class MaskSchedulerD3PM():
# #     def __init__(self, num_timesteps=50, mask_type="uniform", **kwargs):
# #         self.num_timesteps = num_timesteps
# #         self.mask_type = mask_type
# #         if mask_type == "linear":
# #             self.init_linear_schedule(**kwargs)
# #         else:
# #             raise NotImplementedError(f"{mask_type} mask scheduler is not implemented")
# #     def init_linear_schedule(self):
# #         self.mask_probs = None
# #     def __len__(self):
# #         return self.num_timesteps

# scheduler.py

import torch
import torch.nn.functional as F
import math

class NoiseSchedulerDDPM:
    """
    Manages the noise schedule for a Denoising Diffusion Probabilistic Model (DDPM).

    The scheduler is responsible for:
    1.  Creating the noise schedule (betas) based on a linear or cosine strategy.
    2.  Pre-calculating all necessary variables (alphas, cumulative products, etc.) for
        both the forward (noising) and reverse (denoising) processes.
    3.  Providing methods to add noise to an image (forward process) and to
        compute the previous, less noisy image (reverse process).
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear', device=None):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule_type == 'linear':
            self.betas = self._linear_schedule(beta_start, beta_end)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        # Pre-calculate alphas and other required variables for the diffusion process
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Helper variables for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Helper variables for q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    def _linear_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)

    def _cosine_schedule(self, s=0.008):
        """
        Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".
        """
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _extract(self, a, t, x_shape):
        """
        Extracts values from a tensor 'a' at the specified timesteps 't',
        and reshapes it to match the image batch dimensions.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t) # <--- Corrected: t is already on the right device
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def add_noise(self, x_0, t):
        """
        Adds noise to the original images x_0 to get x_t. (Forward Process)
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        noisy_image = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_image, noise

    def sample_prev_timestep(self, x_t, predicted_noise, t):
        """
        Computes x_{t-1} from x_t and the predicted noise. (Reverse Process)
        """
        # Get pre-calculated values for the current timestep t
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        # Equation 11 in the DDPM paper
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t[0].item() == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            # Algorithm 2, step 4
            return model_mean + torch.sqrt(posterior_variance_t) * noise