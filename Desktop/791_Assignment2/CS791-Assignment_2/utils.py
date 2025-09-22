import numpy as np
import torch
import random
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

def seed_everything(seed):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)

# def compute_fid(real_images, fake_images):
#     '''
#         Args:
#             real_images (Actual images from the dataset): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
#             fake_images (Generated images by the diffusion model): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
#         Returns:
#             fid (Frechet Inception Distance): torch.Tensor
#     '''
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     fid = FrechetInceptionDistance().to(device)
#     # Move images to device
#     real_images = real_images.to(device)
#     fake_images = fake_images.to(device)
#     # Convert to uint8 and [0, 255]
#     real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
#     fake_images = (fake_images * 255).clamp(0, 255).to(torch.uint8)
#     # Convert 1 channel to 3 channels
#     real_images = real_images.repeat(1, 3, 1, 1)
#     fake_images = fake_images.repeat(1, 3, 1, 1)
#     # Resize to 299x299
#     real_images = F.interpolate(real_images.float(), size=(299, 299), mode='nearest').to(device)
#     fake_images = F.interpolate(fake_images.float(), size=(299, 299), mode='nearest').to(device)
#     # Convert back to uint8
#     real_images = real_images.clamp(0, 255).to(torch.uint8)
#     fake_images = fake_images.clamp(0, 255).to(torch.uint8)
#     # Update FID metric
#     fid.update(real_images, real=True)
#     fid.update(fake_images, real=False)
#     return fid.compute()

def compute_fid(real_images_preprocessed, fake_images_raw):
    '''
        Calculates Frechet Inception Distance.
        Args:
            real_images_preprocessed (Actual images): torch.Tensor, shape (N, 3, 299, 299), range [0, 255], type uint8
            fake_images_raw (Generated images): torch.Tensor, shape (N, 1, 28, 28), range [0, 1]
        Returns:
            fid (Frechet Inception Distance): torch.Tensor
    '''
    device = real_images_preprocessed.device
    fid = FrechetInceptionDistance().to(device)

    # --- Pre-process the FAKE images (must be done every time) ---
    # The real images are already processed, so we skip them.
    
    # Convert 1 channel to 3 channels for InceptionV3
    fake_images_processed = fake_images_raw.expand(-1, 3, -1, -1)
    
    # Resize to 299x299 using a better interpolation method
    fake_images_processed = F.interpolate(fake_images_processed, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Convert to uint8 and the [0, 255] range expected by the metric
    fake_images_processed = (fake_images_processed * 255).clamp(0, 255).to(torch.uint8)

    # Update FID metric
    fid.update(real_images_preprocessed, real=True)
    fid.update(fake_images_processed, real=False)
    
    return fid.compute()