# generate.py

import torch
import argparse
import os
from torchvision.utils import save_image

# --- Import your models and sampling functions ---
# Make sure models.py, ddpm.py, and ddpm_cond.py are in the same directory
from models import DDPM, ConditionalDDPM
from ddpm import sample as sample_ddpm_unconditional
from ddpm_cond import sample as sample_ddpm_conditional

# NOTE: Add imports for D3PM when you have them ready
# from models import D3PM, ConditionalD3PM
# from d3pm import sample as sample_d3pm_unconditional
# from d3pm_cond import sample as sample_d3pm_conditional


def generate(args, device):
    """
    Loads a trained model and generates samples according to assignment specifications.
    """
    # --- 1. Construct the path to the trained model ---
    if args.mode in ['ddpm', 'd3pm']:
        # Path for unconditional models
        run_name = f"exps_{args.mode}_ddp/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
    elif args.mode in ['ddpm_cond', 'd3pm_cond']:
        # Path for conditional models (naming is slightly different in your script)
        model_type = args.mode.replace('_', '_') # e.g., 'ddpm_cond' -> 'cond_ddpm'
        run_name = f"exps_{model_type}/{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}steps"
    else:
        # This case should not be reached due to argparse choices
        raise ValueError(f"Invalid mode specified: {args.mode}")

    model_path = os.path.join(run_name, "model.pth")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at '{model_path}'")
        print("Please ensure the training hyperparameters match and the model has been trained.")
        return

    # --- 2. Load the appropriate model architecture ---
    print(f"Loading model for mode: '{args.mode}' from '{model_path}'...")
    if args.mode == 'ddpm':
        model = DDPM(in_channels=1)
    elif args.mode == 'ddpm_cond':
        model = ConditionalDDPM(num_classes=10)
    # --- Add D3PM model loading logic here when ready ---
    # elif args.mode == 'd3pm':
    #     model = D3PM(...)
    # elif args.mode == 'd3pm_cond':
    #     model = ConditionalD3PM(...)
    else:
        raise ValueError(f"Model loading not implemented for mode: {args.mode}")

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")

    # --- 3. Generate and save samples ---
    with torch.no_grad():
        if args.mode == 'ddpm':
            print(f"Generating {args.num_samples} unconditional samples...")
            samples = sample_ddpm_unconditional(model, device, num_samples=args.num_samples, num_steps=args.num_steps)
            save_path = "samples_ddpm.pt"
            torch.save(samples.cpu(), save_path)
            print(f"Saved samples to '{save_path}'")
            # Also save a preview image
            save_image(samples, "samples_ddpm.png", nrow=int(args.num_samples**0.5))

        elif args.mode == 'ddpm_cond':
            for class_num in range(10):
                print(f"Generating {args.num_samples} samples for class {class_num}...")
                samples = sample_ddpm_conditional(model, class_num, device, num_samples=args.num_samples, num_steps=args.num_steps)
                save_path = f"samples_ddpm_cond_{class_num}.pt"
                torch.save(samples.cpu(), save_path)
                print(f"  -> Saved samples to '{save_path}'")
                # Also save a preview image for each class
                save_image(samples, f"samples_ddpm_cond_{class_num}.png", nrow=int(args.num_samples**0.5))
        
        # --- Add D3PM sampling logic here when ready ---


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from trained DDPM and D3PM models.")
    parser.add_argument("--mode", type=str, required=True, choices=['ddpm', 'ddpm_cond'], # Add 'd3pm', 'd3pm_cond' later
                        help="Type of model to generate samples from.")
    
    # Add all arguments needed to reconstruct the model path
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs used.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size PER GPU used during training.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate used during training.")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of diffusion steps.")
    
    # As per assignment, we need 64 samples
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate per file.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.num_samples != 64:
        print(f"Warning: Assignment requires 64 samples per file, but you requested {args.num_samples}. Proceeding anyway.")

    generate(args, device)