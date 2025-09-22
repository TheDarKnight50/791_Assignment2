import torch
import argparse
from models import DDPM
from scheduler import NoiseSchedulerDDPM
from utils import compute_fid, seed_everything
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

def sample_batch(model, device, num_samples, num_steps, batch_size=64):
    """
    Helper function to generate a large number of samples in batches.
    """
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, beta_start=1e-4, beta_end=0.02, device=device)
    model.eval()
    
    all_samples = []
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - i)
            img = torch.randn((current_batch_size, 1, 28, 28), device=device)
            
            for t_val in reversed(range(num_steps)):
                t = torch.full((current_batch_size,), t_val, device=device, dtype=torch.long)
                predicted_noise = model(img, t)
                img = scheduler.step(predicted_noise, t[0].item(), img)
            
            all_samples.append(img.cpu())
    
    return torch.cat(all_samples, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="FID Score Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint (.pth file)")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate for evaluation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load the Trained Model ---
    print("Loading trained model...")
    model = DDPM()
    state_dict = torch.load(args.model_path, map_location=device)
    
    # --- THIS BLOCK IS NOW CORRECTED ---
    if any(k.startswith('module.') for k in state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    # ------------------------------------
        
    model.to(device)
    
    # --- 2. Generate Fake Images ---
    print(f"Generating {args.num_images} fake images...")
    fake_images = sample_batch(model, device, args.num_images, num_steps=1000, batch_size=args.batch_size)
    
    # --- 3. Load Real Images ---
    print("Loading real images from MNIST test set...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    real_images = []
    for images, _ in test_loader:
        real_images.append(images)
        if len(torch.cat(real_images, dim=0)) >= args.num_images:
            break
    real_images = torch.cat(real_images, dim=0)[:args.num_images]

    # --- 4. Un-normalize fake images and Calculate FID ---
    print("Un-normalizing fake images to [0, 1] range...")
    fake_images_unnorm = fake_images * 0.5 + 0.5

    print("Calculating FID score...")
    fid_score = compute_fid(real_images, fake_images_unnorm)
    
    print("\n" + "="*30)
    print(f"✅ Final FID Score: {fid_score.item():.4f}")
    print("="*30)