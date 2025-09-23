# evaluate_fid.py

import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Your Project-Specific Imports ---
# Make sure these files are in the same directory
from models import DDPM
from scheduler import NoiseSchedulerDDPM
from utils import compute_fid, seed_everything

def sample(model, device, num_samples=64, num_steps=1000, image_size=28, num_channels=1):
    '''
    Generates samples from a trained DDPM.
    '''
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='cosine', device=device)
    model.eval()
    images = torch.randn((num_samples, num_channels, image_size, image_size), device=device)

    # Denoising loop
    for t in tqdm(reversed(range(num_steps)), desc="Sampling", total=num_steps):
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(images, t_tensor)
            images = scheduler.sample_prev_timestep(images, predicted_noise, t_tensor)

    # Denormalize from [-1, 1] to [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    return images

def parse_args():
    parser = argparse.ArgumentParser(description="FID Evaluation for a trained DDPM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth) checkpoint.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps the model was trained with.")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate for FID calculation.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for loading real images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. LOAD REAL IMAGES AND PRE-PROCESS FOR FID
    print("Loading and pre-processing real images...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    real_images_for_fid = next(iter(test_loader))[0].to(device)
    real_images_for_fid = (real_images_for_fid + 1) / 2
    real_images_for_fid = F.interpolate(
        real_images_for_fid.expand(-1, 3, -1, -1),
        size=(299, 299), mode='bilinear', align_corners=False
    )
    real_images_for_fid = (real_images_for_fid * 255).to(torch.uint8)
    print("Done pre-processing real images.")

    # 2. LOAD THE TRAINED MODEL
    print(f"Loading trained model from: {args.model_path}")
    model = DDPM(in_channels=1, num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 3. GENERATE SAMPLES
    print(f"Generating {args.num_samples} samples...")
    generated_samples = sample(model, device, args.num_samples, args.num_steps)

    # 4. COMPUTE AND PRINT THE FINAL FID SCORE
    print("Calculating final FID score...")
    fid_score = compute_fid(real_images_for_fid, generated_samples)
    
    print("\n" + "="*50)
    print(f"FINAL FID SCORE: {fid_score:.4f}")
    print("="*50)