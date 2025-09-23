# # from models import DDPM
# # import torch
# # from torchvision import datasets, transforms
# # from torch.utils.data import DataLoader
# # import argparse
# # from utils import seed_everything, compute_fid
# # from scheduler import NoiseSchedulerDDPM
# # import os

# # # Add any extra imports you want here

# # def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device):
# #     raise NotImplementedError("Training loop is not implemented.")

# # def sample(model, device, num_samples=16, num_steps=1000):
# #     '''
# #     Returns:
# #         torch.Tensor, shape (num_samples, 1, 28, 28)
# #     '''

# #     raise NotImplementedError("Sampling function is not implemented.")

# # def parse_args():
# #     parser = argparse.ArgumentParser(description="DDPM Model Template")
# #     parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
# #     parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
# #     parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
# #     parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
# #     parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
# #     parser.add_argument("--seed", type=int, default=42, help="Random seed")
# #     parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Mode: train or sample")
# #     # Add any other arguments you want here
# #     return parser.parse_args()

# # if __name__ == "__main__":
# #     args = parse_args()
# #     seed_everything(args.seed)
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print("Using device:", device)

# #     ### Data Preprocessing Start ### (Do not edit this)
# #     transform = transforms.Compose([transforms.ToTensor()])
# #     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# #     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# #     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
# #     ### Data Preprocessing End ### (Do not edit this)

# #     model = DDPM(num_classes=10)
# #     model.to(device)

# #     run_name = f"exps_ddpm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr" # Change run name based on your experiments
# #     os.makedirs(run_name, exist_ok=True)

# #     if args.mode == "train":
# #         model.train()
# #         train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device)
# #     elif args.mode == "sample":
# #         model.load_state_dict(torch.load(f"{run_name}/model.pth"))
# #         model.eval()
# #         samples = sample(model, device, args.num_samples, args.num_steps)
# #         torch.save(samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt")


# from models import DDPM
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import argparse
# from utils import seed_everything, compute_fid
# from scheduler import NoiseSchedulerDDPM
# import os
# import torch.optim as optim
# # tqdm is no longer needed
# # from tqdm import tqdm 
# import torchvision.utils as vutils

# def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, num_steps=1000):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, beta_start=1e-4, beta_end=0.02, device=device)
    
#     print("🚀 Starting Training...")
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
        
#         # --- TQDM replacement logic ---
#         num_batches = len(train_loader)
#         print_interval = num_batches // 4 # Print progress 4 times per epoch

#         for batch_idx, (images, _) in enumerate(train_loader):
#             images = images.to(device)
#             t = torch.randint(0, num_steps, (images.shape[0],), device=device).long()
#             noisy_images, noise = scheduler.add_noise(images, t)
            
#             noise_pred = model(noisy_images, t)
            
#             loss = criterion(noise_pred, noise)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             # Print progress at intervals
#             if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
#                 percent_complete = (batch_idx + 1) / num_batches * 100
#                 print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{num_batches} ({percent_complete:.0f}%) - Loss: {loss.item():.4f}")

#         avg_loss = total_loss / num_batches
#         print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

#         model.eval()
#         with torch.no_grad():
#             samples = sample(model, device, num_samples=64, num_steps=num_steps)
#             vutils.save_image(samples, f"{run_name}/sample_epoch_{epoch+1}.png", normalize=True)
        
#         model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
#         torch.save(model_state, f"{run_name}/model.pth")
#         print(f"✅ Model checkpoint saved for epoch {epoch+1}")


# def sample(model, device, num_samples=64, num_steps=1000):
#     scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, beta_start=1e-4, beta_end=0.02, device=device)
#     model.eval()

#     img = torch.randn((num_samples, 1, 28, 28), device=device)
    
#     print("🎨 Generating samples...")
#     # --- TQDM replacement logic for sampling ---
#     print_interval = num_steps // 4 # Print progress 4 times
    
#     with torch.no_grad():
#         for i, t_val in enumerate(reversed(range(num_steps))):
#             t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
            
#             predicted_noise = model(img, t)
            
#             img = scheduler.step(predicted_noise, t[0].item(), img)
            
#             if (i + 1) % print_interval == 0 or (i + 1) == num_steps:
#                 percent_complete = (i + 1) / num_steps * 100
#                 print(f"Sampling... Step {i+1}/{num_steps} ({percent_complete:.0f}%)")

#     return img.cpu()

# def parse_args():
#     parser = argparse.ArgumentParser(description="DDPM Model Template")
#     parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
#     parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
#     parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
#     parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("mode", nargs='?', default="train", choices=["train", "sample"], help="Mode: train or sample")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     seed_everything(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     model = DDPM()
#     if torch.cuda.device_count() > 1:
#       print(f"Using {torch.cuda.device_count()} GPUs!")
#       model = nn.DataParallel(model)
#     model.to(device)

#     run_name = f"exps_ddpm/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr"
#     os.makedirs(run_name, exist_ok=True)

#     if args.mode == "train":
#         train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device, args.num_steps)
#     elif args.mode == "sample":
#         model_path = f"{run_name}/model.pth"
#         if not os.path.exists(model_path):
#             print(f"Error: Model checkpoint not found at {model_path}")
#             print("Please train the model first using --mode train")
#         else:
#             single_gpu_model = DDPM()
#             single_gpu_model.load_state_dict(torch.load(model_path, map_location=device))
#             single_gpu_model.to(device)
#             single_gpu_model.eval()
#             samples = sample(single_gpu_model, device, args.num_samples, args.num_steps)
#             sample_path = f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt"
#             torch.save(samples, sample_path)
#             vutils.save_image(samples, sample_path.replace('.pt', '.png'), normalize=True)
#             print(f"Saved {args.num_samples} samples to {sample_path}")

# from models import DDPM
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import argparse
# from utils import seed_everything, compute_fid
# from scheduler import NoiseSchedulerDDPM
# import os
# import torch.optim as optim
# import torchvision.utils as vutils

# def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, num_steps=1000, schedule_type="linear"):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type=schedule_type, device=device)
    
#     print(f"🚀 Starting Training with {schedule_type.capitalize()} schedule...")
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
        
#         num_batches = len(train_loader)
#         print_interval = num_batches // 4

#         for batch_idx, (images, _) in enumerate(train_loader):
#             images = images.to(device)
#             t = torch.randint(0, num_steps, (images.shape[0],), device=device).long()
#             noisy_images, noise = scheduler.add_noise(images, t)
            
#             noise_pred = model(noisy_images, t)
            
#             loss = criterion(noise_pred, noise)
            
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
#             optimizer.step()
            
#             total_loss += loss.item()
            
#             if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
#                 percent_complete = (batch_idx + 1) / num_batches * 100
#                 print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{num_batches} ({percent_complete:.0f}%) - Loss: {loss.item():.4f}")

#         avg_loss = total_loss / num_batches
#         print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

#         model.eval()
#         with torch.no_grad():
#             samples = sample(model, device, num_samples=64, num_steps=num_steps, schedule_type=schedule_type)
#             vutils.save_image(samples, f"{run_name}/sample_epoch_{epoch+1}.png", normalize=True)
        
#         model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
#         torch.save(model_state, f"{run_name}/model.pth")
#         print(f"✅ Model checkpoint saved for epoch {epoch+1}")


# def sample(model, device, num_samples=64, num_steps=1000, schedule_type="linear"):
#     scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type=schedule_type, device=device)
#     model.eval()

#     img = torch.randn((num_samples, 1, 28, 28), device=device)
    
#     print("🎨 Generating samples...")
#     print_interval = num_steps // 4
    
#     with torch.no_grad():
#         for i, t_val in enumerate(reversed(range(num_steps))):
#             t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
            
#             predicted_noise = model(img, t)
            
#             img = scheduler.step(predicted_noise, t[0].item(), img)
            
#             if (i + 1) % print_interval == 0 or (i + 1) == num_steps:
#                 percent_complete = (i + 1) / num_steps * 100
#                 print(f"Sampling... Step {i+1}/{num_steps} ({percent_complete:.0f}%)")

#     return img.cpu()

# def parse_args():
#     parser = argparse.ArgumentParser(description="DDPM Model Template")
#     parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
#     parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
#     parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
#     parser.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"], help="Noise schedule type")
#     parser.add_argument("mode", nargs='?', default="train", choices=["train", "sample"], help="Mode: train or sample")
#     # Arguments for sample mode
#     parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
#     parser.add_argument("--model_path", type=str, default=None, help="Path to model for sampling")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     seed_everything(args.seed if 'seed' in args else 42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     model = DDPM()
#     if torch.cuda.device_count() > 1:
#       print(f"Using {torch.cuda.device_count()} GPUs!")
#       model = nn.DataParallel(model)
#     model.to(device)

#     run_name = f"exps_ddpm/{args.schedule}_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr"
#     os.makedirs(run_name, exist_ok=True)

#     if args.mode == "train":
#         train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device, args.num_steps, args.schedule)
#     elif args.mode == "sample":
#         model_path = args.model_path or f"{run_name}/model.pth"
#         if not os.path.exists(model_path):
#             print(f"Error: Model checkpoint not found at {model_path}")
#         else:
#             single_gpu_model = DDPM()
#             single_gpu_model.load_state_dict(torch.load(model_path, map_location=device))
#             single_gpu_model.to(device)
#             single_gpu_model.eval()
#             samples = sample(single_gpu_model, device, args.num_samples, args.num_steps, args.schedule)
#             sample_path = f"{run_name}/{args.num_samples}samples_{args.num_steps}steps_{args.schedule}schedule.pt"
#             vutils.save_image(samples, sample_path.replace('.pt', '.png'), normalize=True)
#             print(f"Saved samples to {sample_path.replace('.pt', '.png')}")


# # ddpm.py

# # --- DDP and Standard Imports ---
# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

# # --- Your Project-Specific Imports ---
# from models import DDPM
# from scheduler import NoiseSchedulerDDPM
# from utils import seed_everything, compute_fid

# # --- Third-party Imports ---
# from torchvision import datasets, transforms
# from torchvision.utils import save_image
# from tqdm import tqdm

# def train(model, train_loader, test_loader, run_name, learning_rate, epochs, num_steps, batch_size, device, sampler, real_images_for_fid):
#     """
#     Trains the DDPM model using DistributedDataParallel.
#     """
#     local_rank = device.index
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='linear', device=device)

#     for epoch in range(epochs):
#         sampler.set_epoch(epoch)
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(local_rank != 0))
#         total_loss = 0.0

#         for images, _ in progress_bar:
#             images = images.to(device)
#             current_batch_size = images.shape[0]
#             t = torch.randint(0, num_steps, (current_batch_size,), device=device).long()
#             noisy_images, noise = scheduler.add_noise(images, t)
#             predicted_noise = model(noisy_images, t)
#             loss = F.mse_loss(predicted_noise, noise)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             if local_rank == 0:
#                 progress_bar.set_postfix({"Loss": loss.item()})

#         if local_rank == 0:
#             avg_loss = total_loss / len(train_loader)
#             print(f"Epoch {epoch+1} finished with Average Loss: {avg_loss:.4f}")

#             with torch.no_grad():
#                 sampled_images = sample(model.module, device, num_samples=64, num_steps=num_steps)
#                 save_image(sampled_images, f"{run_name}/epoch_{epoch+1}_samples.png", nrow=8)
#                 fid_score = compute_fid(real_images_for_fid, sampled_images)
#                 print(f"Epoch {epoch+1}, FID Score: {fid_score:.4f}")

#     if local_rank == 0:
#         torch.save(model.module.state_dict(), f"{run_name}/model.pth")
#         print(f"Model saved to {run_name}/model.pth")

# def sample(model, device, num_samples=64, num_steps=1000, image_size=28, num_channels=1):
#     '''
#     Generates samples from the DDPM.
#     '''
#     scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='linear', device=device)
#     model.eval()
#     images = torch.randn((num_samples, num_channels, image_size, image_size), device=device)

#     for t in tqdm(reversed(range(num_steps)), desc="Sampling", total=num_steps):
#         with torch.no_grad():
#             t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
#             predicted_noise = model(images, t_tensor)
#             images = scheduler.sample_prev_timestep(images, predicted_noise, t_tensor)

#     images = (images.clamp(-1, 1) + 1) / 2
#     return images

# def parse_args():
#     parser = argparse.ArgumentParser(description="DDP DDPM Model Template")
#     parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=128, help="Batch size PER GPU")
#     parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
#     parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
#     parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Mode: train or sample")
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()
#     seed_everything(args.seed)

#     if args.mode == "train":
#         dist.init_process_group("nccl")
#         local_rank = int(os.environ["LOCAL_RANK"])
#         torch.cuda.set_device(local_rank)
#         device = torch.device("cuda", local_rank)
        
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#         train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#         test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#         train_sampler = DistributedSampler(train_dataset)
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
#         test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

#         real_images_for_fid = None
#         if local_rank == 0:
#             print("Pre-processing a batch of real images for FID calculation...")
#             real_images_for_fid = next(iter(test_loader))[0].to(device)
#             real_images_for_fid = (real_images_for_fid + 1) / 2
#             real_images_for_fid = F.interpolate(
#                 real_images_for_fid.expand(-1, 3, -1, -1),
#                 size=(299, 299),
#                 mode='bilinear',
#                 align_corners=False
#             )
#             real_images_for_fid = (real_images_for_fid * 255).to(torch.uint8)
#             print("Done pre-processing real images.")

#         model = DDPM(in_channels=1).to(device)
#         model = DDP(model, device_ids=[local_rank])
        
#         run_name = f"exps_ddpm_ddp/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
#         if local_rank == 0:
#             os.makedirs(run_name, exist_ok=True)
#             print(f"Starting DDP training. Effective batch size: {args.batch_size * dist.get_world_size()}")

#         train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.num_steps, args.batch_size, device, train_sampler, real_images_for_fid)
        
#         dist.destroy_process_group()

#     elif args.mode == "sample":
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print("Using device:", device)
#         model = DDPM(in_channels=1)
#         run_name = f"exps_ddpm_ddp/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
#         model_path = f"{run_name}/model.pth"
        
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")
            
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.to(device)
        
#         print("Generating samples...")
#         samples = sample(model, device, args.num_samples, args.num_steps)
        
#         save_image(samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.png", nrow=int(args.num_samples**0.5))
#         torch.save(samples, f"{run_name}/{args.num_samples}samples_{args.num_steps}steps.pt")
#         print(f"Saved {args.num_samples} samples to {run_name}/")

# ddpm.py

# --- DDP and Standard Imports ---
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# --- Your Project-Specific Imports ---
from models import DDPM
from scheduler import NoiseSchedulerDDPM
from utils import seed_everything, compute_fid

# --- Third-party Imports ---
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

def train(model, train_loader, run_name, learning_rate, epochs, num_steps, device, sampler):
    """
    Trains the DDPM model for all epochs. Evaluation is handled outside this function.
    """
    local_rank = device.index
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=(local_rank != 0))
        total_loss = 0.0

        for images, _ in progress_bar:
            images = images.to(device)
            current_batch_size = images.shape[0]
            t = torch.randint(0, num_steps, (current_batch_size,), device=device).long()
            scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='linear', device=device)
            noisy_images, noise = scheduler.add_noise(images, t)
            predicted_noise = model(noisy_images, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if local_rank == 0:
                progress_bar.set_postfix({"Loss": loss.item()})

        if local_rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} finished with Average Loss: {avg_loss:.4f}")
        
        lr_scheduler.step()

    # --- SAVE FINAL MODEL (Done once at the end of all training) ---
    if local_rank == 0:
        torch.save(model.module.state_dict(), f"{run_name}/model.pth")
        print(f"\nTraining complete. Final model saved to {run_name}/model.pth")

def sample(model, device, num_samples=64, num_steps=1000, image_size=28, num_channels=1):
    '''
    Generates samples from the DDPM.
    '''
    scheduler = NoiseSchedulerDDPM(num_timesteps=num_steps, schedule_type='cosine', device=device)
    model.eval()
    images = torch.randn((num_samples, num_channels, image_size, image_size), device=device)

    for t in tqdm(reversed(range(num_steps)), desc="Sampling", total=num_steps):
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            predicted_noise = model(images, t_tensor)
            images = scheduler.sample_prev_timestep(images, predicted_noise, t_tensor)

    images = (images.clamp(-1, 1) + 1) / 2
    return images

def parse_args():
    parser = argparse.ArgumentParser(description="DDP DDPM Model Template")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size PER GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # --- DDP SETUP ---
    dist.init_process_group("gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # --- DATA PREPARATION ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- MODEL SETUP ---
    model = DDPM(in_channels=1).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    run_name = f"exps_ddpm_ddp/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
    if local_rank == 0:
        os.makedirs(run_name, exist_ok=True)
        print(f"Starting DDP training. Effective batch size: {args.batch_size * dist.get_world_size()}")

    # --- RUN TRAINING FOR ALL EPOCHS ---
    train(model, train_loader, run_name, args.learning_rate, args.epochs, args.num_steps, device, train_sampler)
    
    # --- FINAL EVALUATION (Done once on the main process after training) ---
    if local_rank == 0:
        print("\n--- Starting Final Evaluation ---")
        
        # 1. Pre-process real images for FID
        print("Pre-processing a batch of real images for FID calculation...")
        real_images_for_fid = next(iter(test_loader))[0].to(device)
        real_images_for_fid = (real_images_for_fid + 1) / 2
        real_images_for_fid = F.interpolate(
            real_images_for_fid.expand(-1, 3, -1, -1),
            size=(299, 299), mode='bilinear', align_corners=False
        )
        real_images_for_fid = (real_images_for_fid * 255).to(torch.uint8)
        print("Done pre-processing real images.")

        # 2. Load the final trained model
        eval_model = DDPM(in_channels=1).to(device)
        model_path = f"{run_name}/model.pth"
        eval_model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 3. Generate samples
        print(f"Generating {args.num_samples} final samples...")
        final_samples = sample(eval_model, device, args.num_samples, args.num_steps)
        save_image(final_samples, f"{run_name}/final_{args.num_samples}_samples.png", nrow=int(args.num_samples**0.5))
        
        # 4. Compute and print the final FID score
        print("Calculating final FID score...")
        fid_score = compute_fid(real_images_for_fid, final_samples)
        print("\n" + "="*50)
        print(f"FINAL FID SCORE: {fid_score:.4f}")
        print("="*50)

    # --- CLEANUP ---
    dist.destroy_process_group()