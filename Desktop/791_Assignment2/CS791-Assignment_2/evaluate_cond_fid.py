# evaluate_cond_fid.py

import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# --- Your Project-Specific Imports ---
from models import ConditionalDDPM
from scheduler import NoiseSchedulerDDPM
from utils import compute_fid, seed_everything

def sample(model, class_label, device, num_samples=64, num_steps=1000):
    """
    Generates samples from a trained Conditional DDPM for a specific class.
    """
    model.eval()
    images = torch.randn((num_samples, 1, 28, 28), device=device)
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='cosine', device=device)
    
    # Create a tensor of the desired class label for the whole batch
    labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)

    for t in tqdm(reversed(range(num_steps)), desc=f"Sampling class {class_label}", total=num_steps, leave=False):
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            # Pass the class label to the model
            predicted_noise = model(images, t_tensor, labels)
            images = scheduler.sample_prev_timestep(images, predicted_noise, t_tensor)

    # Denormalize from [-1, 1] to [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    return images

def parse_args():
    parser = argparse.ArgumentParser(description="FID Evaluation for a trained Conditional DDPM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained conditional model (.pth) checkpoint.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps the model was trained with.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate per class for FID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. LOAD THE TRAINED CONDITIONAL MODEL
    print(f"Loading trained model from: {args.model_path}")
    model = ConditionalDDPM(num_classes=10).to(device)
    # The state_dict might have a 'module.' prefix if saved from DDP
    state_dict = torch.load(args.model_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # 2. LOAD THE REAL TEST DATASET
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    all_fid_scores = []
    # 3. LOOP THROUGH EACH CLASS TO CALCULATE FID
    for class_num in range(10):
        print(f"\nEvaluating class: {class_num}")
        
        # a) Generate fake images for the current class
        fake_images = sample(model, class_num, device, args.num_samples, args.num_steps)

        # b) Load real images ONLY for the current class
        class_indices = [i for i, label in enumerate(test_dataset.targets) if label == class_num]
        real_dataset_subset = Subset(test_dataset, class_indices)
        # Ensure we have enough real samples to match the number of fake samples
        real_loader = DataLoader(real_dataset_subset, batch_size=args.num_samples, shuffle=True)
        real_images = next(iter(real_loader))[0].to(device)

        # c) Pre-process real images for the FID function
        real_images_preprocessed = (real_images + 1) / 2
        real_images_preprocessed = F.interpolate(
            real_images_preprocessed.expand(-1, 3, -1, -1),
            size=(299, 299), mode='bilinear', align_corners=False
        )
        real_images_preprocessed = (real_images_preprocessed * 255).to(torch.uint8)
        
        # d) Calculate FID score for this class
        fid_score = compute_fid(real_images_preprocessed, fake_images)
        print(f"FID Score for class {class_num}: {fid_score:.4f}")
        all_fid_scores.append(fid_score.item())

    # 4. CALCULATE AND PRINT THE FINAL AVERAGE SCORE
    average_fid = np.mean(all_fid_scores)
    print("\n" + "="*50)
    print(f"✅ AVERAGE FID SCORE ACROSS ALL CLASSES: {average_fid:.4f}")
    print("="*50)