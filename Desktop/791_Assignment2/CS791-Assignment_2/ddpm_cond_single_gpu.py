# ddpm_cond_single_gpu.py

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

from models import ConditionalDDPM
from scheduler import NoiseSchedulerDDPM
from utils import seed_everything, compute_fid

def train(model, train_loader, run_name, learning_rate, epochs, num_steps, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, num_steps, (images.shape[0],), device=device).long()
            scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='linear', device=device)
            noisy_images, noise = scheduler.add_noise(images, t)
            predicted_noise = model(noisy_images, t, labels)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
        
        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} finished with Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"{run_name}/model.pth")
    print(f"\nTraining complete. Final model saved to {run_name}/model.pth")

def sample(model, class_label, device, num_samples=64, num_steps=1000):
    model.eval()
    images = torch.randn((num_samples, 1, 28, 28), device=device)
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='cosine', device=device)
    labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)

    for t in tqdm(reversed(range(num_steps)), desc=f"Sampling class {class_label}", total=num_steps, leave=False):
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(images, t_tensor, labels)
            images = scheduler.sample_prev_timestep(images, predicted_noise, t_tensor)

    images = (images.clamp(-1, 1) + 1) / 2
    return images

def parse_args():
    parser = argparse.ArgumentParser(description="Conditional DDPM Training (Single GPU)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training. You can increase this for a single powerful GPU.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate per class for FID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using single device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ConditionalDDPM(num_classes=10).to(device)
    
    run_name = f"exps_cond_ddpm_single_gpu/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
    os.makedirs(run_name, exist_ok=True)
    print(f"Starting Single GPU training.")

    train(model, train_loader, run_name, args.learning_rate, args.epochs, args.num_steps, device)
    
    # --- Final Evaluation ---
    print("\n--- Starting Final Evaluation ---")
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    all_fid_scores = []
    for class_num in range(10):
        print(f"\nEvaluating class: {class_num}")
        
        fake_images = sample(model, class_num, device, args.num_samples, args.num_steps)
        save_image(fake_images, f"{run_name}/final_samples_class_{class_num}.png", nrow=int(args.num_samples**0.5))

        class_indices = [i for i, label in enumerate(test_dataset.targets) if label == class_num]
        real_dataset_subset = Subset(test_dataset, class_indices)
        real_loader = DataLoader(real_dataset_subset, batch_size=args.num_samples, shuffle=True)
        real_images = next(iter(real_loader))[0].to(device)

        real_images = (real_images + 1) / 2
        real_images = F.interpolate(real_images.expand(-1, 3, -1, -1), size=(299, 299), mode='bilinear', align_corners=False)
        real_images = (real_images * 255).to(torch.uint8)
        
        fid_score = compute_fid(real_images, fake_images)
        print(f"FID Score for class {class_num}: {fid_score:.4f}")
        all_fid_scores.append(fid_score.item())

    average_fid = np.mean(all_fid_scores)
    print("\n" + "="*50)
    print(f"AVERAGE FID SCORE ACROSS ALL CLASSES: {average_fid:.4f}")
    print("="*50)