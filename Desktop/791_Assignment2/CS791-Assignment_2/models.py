# # import torch.nn as nn

# # class D3PM(nn.Module):
# #     def __init__(self): # Add any required parameters
# #         super().__init__()
# #         # Define your model architecture here

# # class ConditionalD3PM(nn.Module):
# #     def __init__(self, num_classes): # Add any required parameters
# #         super().__init__()
# #         self.num_classes = num_classes
# #         # Define your conditional model architecture here

# # class DDPM(nn.Module):
# #     def __init__(self): # Add any required parameters
# #         super().__init__()
# #         # Define your model architecture here

# # class ConditionalDDPM(nn.Module):
# #     def __init__(self, num_classes): # Add any required parameters
# #         super().__init__()
# #         self.num_classes = num_classes
# #         # Define your conditional model architecture here

# # import torch
# # import torch.nn as nn
# # import math

# # class D3PM(nn.Module):
# #     def __init__(self): # Add any required parameters
# #         super().__init__()
# #         # NOTE: This part is for the D3PM model from Part 1.1 of the assignment.
# #         # Define your model architecture here

# # class ConditionalD3PM(nn.Module):
# #     def __init__(self, num_classes): # Add any required parameters
# #         super().__init__()
# #         self.num_classes = num_classes
# #         # NOTE: This part is for the Conditional D3PM model from Part 1.1 of the assignment.
# #         # Define your conditional model architecture here

# # # ---------------------------------------------------------------------------
# # # -- PART 1.2 IMPLEMENTATION: DDPM and ConditionalDDPM --
# # # ---------------------------------------------------------------------------

# # class SinusoidalPositionEmbeddings(nn.Module):
# #     """
# #     Module to generate sinusoidal position embeddings for the timesteps.
# #     """
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.dim = dim

# #     def forward(self, time):
# #         device = time.device
# #         half_dim = self.dim // 2
# #         embeddings = math.log(10000) / (half_dim - 1)
# #         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
# #         embeddings = time[:, None] * embeddings[None, :]
# #         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
# #         return embeddings

# # class ResTimeBlock(nn.Module):
# #     """
# #     A residual block that also takes a time embedding.
# #     """
# #     def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
# #         super().__init__()
# #         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
# #         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
# #         self.relu = nn.ReLU()
# #         self.dropout = nn.Dropout(dropout)
# #         self.norm1 = nn.GroupNorm(8, in_channels)
# #         self.norm2 = nn.GroupNorm(8, out_channels)
        
# #         # Residual connection
# #         if in_channels != out_channels:
# #             self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
# #         else:
# #             self.residual_conv = nn.Identity()

# #     def forward(self, x, t):
# #         h = self.norm1(x)
# #         h = self.relu(h)
# #         h = self.conv1(h)

# #         time_emb = self.relu(self.time_mlp(t))
# #         time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # Add spatial dimensions
# #         h = h + time_emb

# #         h = self.norm2(h)
# #         h = self.relu(h)
# #         h = self.dropout(h)
# #         h = self.conv2(h)

# #         return h + self.residual_conv(x)

# # class _UNet(nn.Module):
# #     """
# #     Core U-Net model for the DDPM. Can be used for both unconditional and conditional generation.
# #     """
# #     def __init__(self, in_channels=1, model_channels=64, out_channels=1, time_emb_dim=256, num_classes=None, dropout=0.1):
# #         super().__init__()
        
# #         # Time and class embedding
# #         self.time_mlp = nn.Sequential(
# #             SinusoidalPositionEmbeddings(time_emb_dim),
# #             nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(),
# #             nn.Linear(time_emb_dim * 4, time_emb_dim)
# #         )
# #         self.num_classes = num_classes
# #         if num_classes is not None:
# #             self.label_emb = nn.Embedding(num_classes, time_emb_dim)

# #         # U-Net architecture
# #         self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

# #         # Downsampling path
# #         self.down1 = ResTimeBlock(model_channels, model_channels, time_emb_dim, dropout)
# #         self.down2 = ResTimeBlock(model_channels, model_channels, time_emb_dim, dropout)
# #         self.pool1 = nn.Conv2d(model_channels, model_channels, 3, 2, 1) # Downsample to H/2
        
# #         self.down3 = ResTimeBlock(model_channels, model_channels*2, time_emb_dim, dropout)
# #         self.down4 = ResTimeBlock(model_channels*2, model_channels*2, time_emb_dim, dropout)
# #         self.pool2 = nn.Conv2d(model_channels*2, model_channels*2, 3, 2, 1) # Downsample to H/4

# #         # Bottleneck
# #         self.mid1 = ResTimeBlock(model_channels*2, model_channels*4, time_emb_dim, dropout)
# #         self.mid2 = ResTimeBlock(model_channels*4, model_channels*4, time_emb_dim, dropout)

# #         # Upsampling path
# #         self.unpool1 = nn.ConvTranspose2d(model_channels*4, model_channels*2, 4, 2, 1) # Upsample to H/2
# #         self.up1 = ResTimeBlock(model_channels*4, model_channels*2, time_emb_dim, dropout)
# #         self.up2 = ResTimeBlock(model_channels*2, model_channels*2, time_emb_dim, dropout)
        
# #         self.unpool2 = nn.ConvTranspose2d(model_channels*2, model_channels, 4, 2, 1) # Upsample to H
# #         self.up3 = ResTimeBlock(model_channels*2, model_channels, time_emb_dim, dropout)
# #         self.up4 = ResTimeBlock(model_channels, model_channels, time_emb_dim, dropout)
        
# #         # Output layer
# #         self.conv_out = nn.Sequential(
# #             nn.GroupNorm(8, model_channels), nn.SiLU(),
# #             nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
# #         )

# #     def forward(self, x, t, y=None):
# #         time_emb = self.time_mlp(t)
# #         if self.num_classes is not None:
# #             if y is None:
# #                 raise ValueError("Class labels 'y' must be provided for conditional model.")
# #             time_emb += self.label_emb(y)

# #         x = self.conv_in(x)
        
# #         # Downsampling
# #         s1 = self.down1(x, time_emb)
# #         s1 = self.down2(s1, time_emb)
# #         x = self.pool1(s1)

# #         s2 = self.down3(x, time_emb)
# #         s2 = self.down4(s2, time_emb)
# #         x = self.pool2(s2)

# #         # Bottleneck
# #         x = self.mid1(x, time_emb)
# #         x = self.mid2(x, time_emb)
        
# #         # Upsampling with skip connections
# #         x = self.unpool1(x)
# #         x = torch.cat([x, s2], dim=1)
# #         x = self.up1(x, time_emb)
# #         x = self.up2(x, time_emb)

# #         x = self.unpool2(x)
# #         x = torch.cat([x, s1], dim=1)
# #         x = self.up3(x, time_emb)
# #         x = self.up4(x, time_emb)

# #         return self.conv_out(x)

# # class DDPM(nn.Module):
# #     """
# #     Unconditional Denoising Diffusion Probabilistic Model.
# #     """
# #     def __init__(self): # Add any required parameters
# #         super().__init__()
# #         # This wrapper uses the core _UNet for unconditional generation.
# #         # Parameters like model_channels can be adjusted here if needed.
# #         self.model = _UNet(
# #             in_channels=1,
# #             model_channels=64,
# #             out_channels=1,
# #             time_emb_dim=256,
# #             num_classes=None # Unconditional
# #         )

# #     def forward(self, x, t):
# #         """
# #         :param x: Noisy image tensor of shape (N, 1, 28, 28)
# #         :param t: Timestep tensor of shape (N,)
# #         :return: Predicted noise tensor of shape (N, 1, 28, 28)
# #         """
# #         return self.model(x, t, y=None)

# # class ConditionalDDPM(nn.Module):
# #     """
# #     Conditional Denoising Diffusion Probabilistic Model.
# #     """
# #     def __init__(self, num_classes=10): # MNIST has 10 classes
# #         super().__init__()
# #         self.num_classes = num_classes
# #         # This wrapper uses the core _UNet for conditional generation.
# #         self.model = _UNet(
# #             in_channels=1,
# #             model_channels=64,
# #             out_channels=1,
# #             time_emb_dim=256,
# #             num_classes=self.num_classes # Conditional
# #         )

# #     def forward(self, x, t, y):
# #         """
# #         :param x: Noisy image tensor of shape (N, 1, 28, 28)
# #         :param t: Timestep tensor of shape (N,)
# #         :param y: Class label tensor of shape (N,)
# #         :return: Predicted noise tensor of shape (N, 1, 28, 28)
# #         """
# #         return self.model(x, t, y)

# # import torch
# # import torch.nn as nn
# # import math

# # class SinusoidalPositionEmbeddings(nn.Module):
# #     """
# #     Module for generating sinusoidal position embeddings for timesteps.
# #     """
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.dim = dim

# #     def forward(self, time):
# #         device = time.device
# #         half_dim = self.dim // 2
# #         embeddings = math.log(10000) / (half_dim - 1)
# #         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
# #         embeddings = time[:, None] * embeddings[None, :]
# #         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
# #         return embeddings

# # class DoubleConv(nn.Module):
# #     """
# #     A block consisting of two convolutional layers with GroupNorm and SiLU activation.
# #     It also incorporates a time embedding.
# #     """
# #     def __init__(self, in_channels, out_channels, time_emb_dim=256):
# #         super().__init__()
# #         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
# #         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
# #         self.norm1 = nn.GroupNorm(8, out_channels)
# #         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
# #         self.norm2 = nn.GroupNorm(8, out_channels)
# #         self.activation = nn.SiLU() # SiLU is a common choice in modern U-Nets

# #     def forward(self, x, t):
# #         h = self.conv1(x)
# #         h = self.norm1(h)
# #         h = self.activation(h)

# #         time_emb = self.activation(self.time_mlp(t))
# #         time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # Reshape for broadcasting
# #         h = h + time_emb

# #         h = self.conv2(h)
# #         h = self.norm2(h)
# #         h = self.activation(h)
# #         return h

# # class DDPM(nn.Module):
# #     """
# #     A robust U-Net model for the Denoising Diffusion Probabilistic Model.
# #     """
# #     def __init__(self, in_channels=1):
# #         super().__init__()
# #         time_emb_dim = 256

# #         # --- Time Embedding ---
# #         self.time_mlp = nn.Sequential(
# #             SinusoidalPositionEmbeddings(time_emb_dim),
# #             nn.Linear(time_emb_dim, time_emb_dim),
# #             nn.SiLU()
# #         )

# #         # --- Encoder (Downsampling Path) ---
# #         self.initial_conv = DoubleConv(in_channels, 64, time_emb_dim)
# #         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, time_emb_dim))
# #         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, time_emb_dim))

# #         # --- Bottleneck ---
# #         self.bot1 = DoubleConv(256, 256, time_emb_dim)

# #         # --- Decoder (Upsampling Path) ---
# #         self.up1_trans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
# #         self.up1_conv = DoubleConv(256, 128, time_emb_dim) # 256 because of skip connection

# #         self.up2_trans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
# #         self.up2_conv = DoubleConv(128, 64, time_emb_dim)  # 128 because of skip connection

# #         # --- Final Output Layer ---
# #         self.output = nn.Conv2d(64, in_channels, kernel_size=1)

# #     def forward(self, x, timestep):
# #         t = self.time_mlp(timestep)

# #         # --- Encoder ---
# #         x1 = self.initial_conv(x, t)  # (-> 64 channels)
# #         x2 = self.down1(x1)         # (-> 128 channels)
# #         x3 = self.down2(x2)         # (-> 256 channels)

# #         # --- Bottleneck ---
# #         x3 = self.bot1(x3)

# #         # --- Decoder with Skip Connections ---
# #         # Upsample, concatenate with skip connection, then convolve
# #         up1 = self.up1_trans(x3)
# #         x = self.up1_conv(torch.cat([up1, x2], dim=1), t)

# #         up2 = self.up2_trans(x)
# #         x = self.up2_conv(torch.cat([up2, x1], dim=1), t)

# #         return self.output(x)

# # # --- Placeholder classes from your original file ---
# # class D3PM(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         pass

# # class ConditionalD3PM(nn.Module):
# #     def __init__(self, num_classes):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         pass

# # class ConditionalDDPM(nn.Module):
# #     def __init__(self, num_classes):
# #         super().__init__()
# #         self.num_classes = num_classes
# #         pass


# # models.py

# import torch
# import torch.nn as nn
# import math

# class SinusoidalPositionEmbeddings(nn.Module):
#     """
#     Module for generating sinusoidal position embeddings for timesteps.
#     """
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

# class DoubleConv(nn.Module):
#     """
#     A block consisting of two convolutional layers with GroupNorm and SiLU activation.
#     It also incorporates a time embedding.
#     """
#     def __init__(self, in_channels, out_channels, time_emb_dim=256):
#         super().__init__()
#         self.time_mlp = nn.Linear(time_emb_dim, out_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.norm1 = nn.GroupNorm(8, out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.norm2 = nn.GroupNorm(8, out_channels)
#         self.activation = nn.SiLU()

#     def forward(self, x, t):
#         h = self.conv1(x)
#         h = self.norm1(h)
#         h = self.activation(h)

#         time_emb = self.activation(self.time_mlp(t))
#         time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
#         h = h + time_emb

#         h = self.conv2(h)
#         h = self.norm2(h)
#         h = self.activation(h)
#         return h

# class DDPM(nn.Module):
#     """
#     A robust U-Net model for the Denoising Diffusion Probabilistic Model.
#     """
#     def __init__(self, in_channels=1):
#         super().__init__()
#         time_emb_dim = 256

#         # --- Time Embedding ---
#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(time_emb_dim),
#             nn.Linear(time_emb_dim, time_emb_dim),
#             nn.SiLU()
#         )

#         # --- Encoder (Downsampling Path) ---
#         self.initial_conv = DoubleConv(in_channels, 64, time_emb_dim)
#         self.pool = nn.MaxPool2d(2)
#         # ** FIX: Defined layers separately instead of using nn.Sequential **
#         self.down_conv1 = DoubleConv(64, 128, time_emb_dim)
#         self.down_conv2 = DoubleConv(128, 256, time_emb_dim)

#         # --- Bottleneck ---
#         self.bot_conv1 = DoubleConv(256, 256, time_emb_dim)

#         # --- Decoder (Upsampling Path) ---
#         self.up_trans1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.up_conv1 = DoubleConv(256, 128, time_emb_dim)

#         self.up_trans2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.up_conv2 = DoubleConv(128, 64, time_emb_dim)

#         # --- Final Output Layer ---
#         self.output = nn.Conv2d(64, in_channels, kernel_size=1)

#     def forward(self, x, timestep):
#         t = self.time_mlp(timestep)

#         # --- Encoder ---
#         x1 = self.initial_conv(x, t)
        
#         # ** FIX: Explicitly call layers to pass 't' correctly **
#         p1 = self.pool(x1)
#         x2 = self.down_conv1(p1, t)
        
#         p2 = self.pool(x2)
#         x3 = self.down_conv2(p2, t)

#         # --- Bottleneck ---
#         x3 = self.bot_conv1(x3, t)

#         # --- Decoder with Skip Connections ---
#         up1 = self.up_trans1(x3)
#         x = self.up_conv1(torch.cat([up1, x2], dim=1), t)

#         up2 = self.up_trans2(x)
#         x = self.up_conv2(torch.cat([up2, x1], dim=1), t)

#         return self.output(x)


# # --- Placeholder classes from your original file ---
# class D3PM(nn.Module):
#     def __init__(self):
#         super().__init__()
#         pass

# class ConditionalD3PM(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
#         pass

# class ConditionalDDPM(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
#         pass

# models.py

import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        time_emb = self.activation(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        return h

# --- Unconditional Model ---
class DDPM(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.initial_conv = DoubleConv(in_channels, 64, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.down_conv1 = DoubleConv(64, 128, time_emb_dim)
        self.down_conv2 = DoubleConv(128, 256, time_emb_dim)
        self.bot_conv1 = DoubleConv(256, 256, time_emb_dim)
        self.up_trans1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(256, 128, time_emb_dim)
        self.up_trans2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(128, 64, time_emb_dim)
        self.output = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x1 = self.initial_conv(x, t)
        p1 = self.pool(x1)
        x2 = self.down_conv1(p1, t)
        p2 = self.pool(x2)
        x3 = self.down_conv2(p2, t)
        x3 = self.bot_conv1(x3, t)
        up1 = self.up_trans1(x3)
        x = self.up_conv1(torch.cat([up1, x2], dim=1), t)
        up2 = self.up_trans2(x)
        x = self.up_conv2(torch.cat([up2, x1], dim=1), t)
        return self.output(x)

# --- Conditional Model ---
class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        in_channels = 1
        time_emb_dim = 256
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)
        self.initial_conv = DoubleConv(in_channels, 64, time_emb_dim)
        self.pool = nn.MaxPool2d(2)
        self.down_conv1 = DoubleConv(64, 128, time_emb_dim)
        self.down_conv2 = DoubleConv(128, 256, time_emb_dim)
        self.bot_conv1 = DoubleConv(256, 256, time_emb_dim)
        self.up_trans1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(256, 128, time_emb_dim)
        self.up_trans2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(128, 64, time_emb_dim)
        self.output = nn.Conv2d(64, in_channels, kernel_size=1)

    def forward(self, x, timestep, class_label):
        t = self.time_mlp(timestep)
        c = self.label_emb(class_label)
        embedding = t + c
        
        x1 = self.initial_conv(x, embedding)
        p1 = self.pool(x1)
        x2 = self.down_conv1(p1, embedding)
        p2 = self.pool(x2)
        x3 = self.down_conv2(p2, embedding)
        x3 = self.bot_conv1(x3, embedding)
        up1 = self.up_trans1(x3)
        x = self.up_conv1(torch.cat([up1, x2], dim=1), embedding)
        up2 = self.up_trans2(x)
        x = self.up_conv2(torch.cat([up2, x1], dim=1), embedding)
        return self.output(x)