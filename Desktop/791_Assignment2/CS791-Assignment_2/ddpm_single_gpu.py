# ddpm_single_gpu.py

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models import DDPM
from scheduler import NoiseSchedulerDDPM
from utils import seed_everything, compute_fid

def train(model, train_loader, run_name, learning_rate, epochs, num_steps, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0

        for images, _ in progress_bar:
            images = images.to(device)
            t = torch.randint(0, num_steps, (images.shape[0],), device=device).long()
            scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='linear', device=device)
            noisy_images, noise = scheduler.add_noise(images, t)
            predicted_noise = model(noisy_images, t)
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

def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Training (Single GPU)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using single device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = DDPM(in_channels=1).to(device)

    run_name = f"exps_ddpm_single_gpu/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
    os.makedirs(run_name, exist_ok=True)
    print(f"Starting Single GPU training.")

    train(model, train_loader, run_name, args.learning_rate, args.epochs, args.num_steps, device)
    print("\n--- Training Finished ---")