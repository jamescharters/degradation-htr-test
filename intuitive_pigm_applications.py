#!/usr/bin/env python3
"""
PIGM - INTUITIVE APPLICATIONS VERSION
This script trains an AI on familiar, single-vortex flows (like stirring a drink)
and then uses it to solve practical, easy-to-understand problems.

1.  Finds the best stirring pattern for a "latte art" effect inside a circular cup.
2.  Creates an animated GIF comparing a good vs. bad pattern for mixing cream into coffee.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio
import os

print("=" * 70)
print("PHYSICS-INFORMED GENERATIVE MODEL (PIGM) - INTUITIVE APPLICATIONS")
print("=" * 70)

# --- Boilerplate Code (Core AI and Physics) ---
# This is the essential machinery from the previous script.

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("✓ Using CPU")

# Physics Utilities
def compute_divergence(vx, vy):
    B, H, W = vx.shape
    dvx_dx = torch.zeros_like(vx)
    dvx_dx[:, :, 1:-1] = (vx[:, :, 2:] - vx[:, :, :-2]) / 2.0
    dvy_dy = torch.zeros_like(vy)
    dvy_dy[:, 1:-1, :] = (vy[:, 2:, :] - vy[:, :-2, :]) / 2.0
    return dvx_dx + dvy_dy

def compute_curl(vx, vy):
    B, H, W = vx.shape
    dvy_dx = torch.zeros_like(vy)
    dvy_dx[:, :, 1:-1] = (vy[:, :, 2:] - vy[:, :, :-2]) / 2.0
    dvx_dy = torch.zeros_like(vx)
    dvx_dy[:, 1:-1, :] = (vx[:, 2:, :] - vx[:, :-2, :]) / 2.0
    return dvy_dx - dvx_dy

def project_to_divergence_free_fft(vx, vy):
    B, H, W = vx.shape
    div = compute_divergence(vx, vy)
    div_fft = torch.fft.rfft2(div)
    kx = torch.fft.fftfreq(W, d=1.0, device=device)[:W//2+1].reshape(1, 1, -1)
    ky = torch.fft.fftfreq(H, d=1.0, device=device).reshape(1, -1, 1)
    k2 = kx**2 + ky**2
    k2[:, 0, 0] = 1.0
    phi_fft = div_fft / (-k2 * (2 * np.pi)**2)
    phi_fft[:, 0, 0] = 0
    dphi_dx_fft = 1j * (2 * np.pi) * kx * phi_fft
    dphi_dy_fft = 1j * (2 * np.pi) * ky * phi_fft
    dphi_dx = torch.fft.irfft2(dphi_dx_fft, s=(H, W))
    dphi_dy = torch.fft.irfft2(dphi_dy_fft, s=(H, W))
    vx_proj = vx - dphi_dx
    vy_proj = vy - dphi_dy
    return vx_proj, vy_proj

# U-Net Architecture
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        return x
class UNet(nn.Module):
    def __init__(self, in_channels=2, base_channels=32):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(64, base_channels * 4), nn.SiLU(), nn.Linear(base_channels * 4, base_channels * 4))
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, in_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, x, t):
        t_emb = self.get_timestep_embedding(t, 64)
        t_emb = self.time_embed(t_emb)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        b = b + t_emb.view(-1, b.size(1), 1, 1)
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        return self.out(d1)
    @staticmethod
    def get_timestep_embedding(timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# Physics-Informed Diffusion Model
class PhysicsInformedDiffusion(nn.Module):
    def __init__(self, img_size=64, timesteps=50):
        super().__init__()
        self.img_size = img_size
        self.timesteps = timesteps
        self.model = UNet()
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise, noise
    @torch.no_grad()
    def sample(self, batch_size=1):
        x = torch.randn(batch_size, 2, self.img_size, self.img_size, device=device)
        vx, vy = project_to_divergence_free_fft(x[:, 0], x[:, 1])
        x = torch.stack([vx, vy], dim=1)
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            noise_pred = self.model(x, t)
            alpha_bar = self.alpha_bars[t_val]
            x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            if t_val > 0:
                beta, alpha_bar_prev = self.betas[t_val], self.alpha_bars[t_val-1]
                posterior_var = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
                mean = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev - posterior_var) * noise_pred
                x = mean + torch.sqrt(posterior_var) * torch.randn_like(x)
            else:
                x = x0_pred
            vx_proj, vy_proj = project_to_divergence_free_fft(x[:, 0], x[:, 1])
            x = torch.stack([vx_proj, vy_proj], dim=1)
        return x

# --- NEW: INTUITIVE DATA GENERATION ---
# This function generates data that looks like stirring a drink in a cup.

def generate_single_vortex_data(n_samples=400, img_size=64):
    """
    Generates training data containing a SINGLE, central vortex.
    This teaches the AI what stirring a single drink looks like.
    """
    print(f"\nGenerating {n_samples} single-vortex training samples...")
    data = []
    for i in range(n_samples):
        # Center the vortex, with a slight random offset
        center = (np.random.uniform(0.45, 0.55), np.random.uniform(0.45, 0.55))
        strength = np.random.uniform(1.5, 2.5) * np.random.choice([-1, 1])
        
        y, x = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size), indexing='ij')
        dx, dy = x - center[0], y - center[1]
        r = np.sqrt(dx**2 + dy**2) + 1e-8
        
        vx = -dy / r * strength * np.exp(-10 * r**2)
        vy = dx / r * strength * np.exp(-10 * r**2)
        
        max_v = max(np.abs(vx).max(), np.abs(vy).max())
        if max_v > 0:
            vx /= max_v
            vy /= max_v
        
        data.append(np.stack([vx, vy], axis=0))

    print("✓ Training data generated.")
    return torch.tensor(np.array(data), dtype=torch.float32, device=device)

# --- Training Function ---
def train_model(model, data, epochs=25, batch_size=16):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    print("\nTraining AI on 'stirring a drink' data...")
    print("-" * 70)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        perm = torch.randperm(len(data))
        for i in range(0, len(data), batch_size):
            x0 = data[perm[i:i+batch_size]]
            optimizer.zero_grad()
            t = torch.randint(0, model.timesteps, (x0.shape[0],), device=device)
            x_t, noise = model.add_noise(x0, t)
            noise_pred = model.model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss / (len(data)/batch_size):.6f}")
    print("✓ Training complete!\n")

# --- NEW: INTUITIVE VISUALIZATIONS ---

def find_best_pattern_for_latte_art(model, num_candidates=32):
    """
    Finds the best and worst stirring patterns for creating stable,
    beautiful "latte art". A good pattern has high, organized rotation.
    VISUALIZED INSIDE A CIRCULAR CUP.
    """
    print("=" * 70)
    print("APPLICATION 1: The Perfect Latte Art")
    print("=" * 70)
    print("1. AI is generating candidate stirring patterns...")
    with torch.no_grad():
        candidates = model.sample(num_candidates)

    print("2. Evaluating patterns for 'artistic quality' (i.e., rotational strength)...")
    # A good proxy for "good art" is strong, consistent rotation (vorticity/curl).
    curls = compute_curl(candidates[:, 0], candidates[:, 1])
    scores = curls.abs().mean(dim=[1, 2])
    
    best_score, best_idx = torch.max(scores, dim=0)
    worst_score, worst_idx = torch.min(scores, dim=0)

    best_pattern = candidates[best_idx].cpu().numpy()
    worst_pattern = candidates[worst_idx].cpu().numpy()
    print(f"✓ Found best pattern with score {best_score:.2f} and worst with score {worst_score:.2f}.")

    # --- Plotting inside a circular "cup" ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    x, y = np.meshgrid(np.arange(model.img_size), np.arange(model.img_size))
    
    for i, (ax, pattern, title, color) in enumerate([
        (axes[0], worst_pattern, f"Chaotic Pattern\n(Score: {worst_score:.2f})", "red"),
        (axes[1], best_pattern, f"Stable 'Latte Art' Pattern\n(Score: {best_score:.2f})", "green")
    ]):
        # Create a circle patch for the cup boundary and clipping
        cup_radius = model.img_size * 0.48
        cup_center = (model.img_size / 2, model.img_size / 2)
        circle_boundary = Circle(cup_center, cup_radius, facecolor='none', edgecolor='black', linewidth=2, zorder=10)
        circle_clip = Circle(cup_center, cup_radius, transform=ax.transData)

        ax.add_patch(circle_boundary)
        ax.streamplot(x, y, pattern[1], pattern[0], color=color, density=2, linewidth=1.5, arrowsize=1.2, broken_streamlines=False)
        
        # Set background to a coffee-like color
        ax.set_facecolor('#d2b48c') # Tan color
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_clip_path(circle_clip) # Clip the streamplot to the circle

    plt.suptitle("AI Designing the Best Stirring Pattern for Latte Art", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('application_latte_art.png', dpi=150)
    plt.show()

def visualize_mixing_animation(model, num_candidates=32):
    """
    Creates an animated GIF comparing the best and worst patterns for
    mixing cream into coffee.
    """
    print("\n" + "=" * 70)
    print("APPLICATION 2: The Fastest Mix (Animation)")
    print("=" * 70)
    print("1. AI generating patterns to test for mixing speed...")
    with torch.no_grad():
        candidates = model.sample(num_candidates)

    print("2. Finding the BEST and WORST mixers...")
    curls = compute_curl(candidates[:, 0], candidates[:, 1])
    scores = curls.abs().mean(dim=[1, 2])
    best_idx = torch.argmax(scores)
    worst_idx = torch.argmin(scores)
    
    vx_best, vy_best = candidates[best_idx:best_idx+1, 0], candidates[best_idx:best_idx+1, 1]
    vx_worst, vy_worst = candidates[worst_idx:worst_idx+1, 0], candidates[worst_idx:worst_idx+1, 1]

    # --- Simulation Setup ---
    print("3. Setting up the cream & coffee simulation...")
    H, W = model.img_size, model.img_size
    
    # Initial state: half cream (1), half coffee (0)
    cream = torch.zeros(1, 1, H, W, device=device)
    cream[:, :, :, :W//2] = 1.0
    cream_best, cream_worst = cream.clone(), cream.clone()
    
    # The grid for advection
    y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
    base_grid = torch.stack([x_grid, y_grid], dim=2).unsqueeze(0)

    # Normalize velocity fields for stable animation speed
    dt = 0.3
    vx_best_norm = vx_best * dt * (2 / W)
    vy_best_norm = vy_best * dt * (2 / H)
    vx_worst_norm = vx_worst * dt * (2 / W)
    vy_worst_norm = vy_worst * dt * (2 / H)

    # --- Animation Loop ---
    print("4. Running simulation and creating GIF frames...")
    frames = []
    num_steps = 80
    for step in range(num_steps):
        # Advect the cream using the flow fields
        grid_best = base_grid - torch.stack([vx_best_norm[0], vy_best_norm[0]], dim=-1)
        cream_best = F.grid_sample(cream_best, grid_best, mode='bilinear', padding_mode='border', align_corners=False)

        grid_worst = base_grid - torch.stack([vx_worst_norm[0], vy_worst_norm[0]], dim=-1)
        cream_worst = F.grid_sample(cream_worst, grid_worst, mode='bilinear', padding_mode='border', align_corners=False)

        if step % 2 == 0:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
            
            # Create circular mask for plotting
            cup_radius = W * 0.48
            cup_center = (W / 2, H / 2)
            
            for ax, cream_data, title in [
                (axes[0], cream_worst, "Worst Mixing Pattern"),
                (axes[1], cream_best, "BEST Mixing Pattern")
            ]:
                circle_clip = Circle(cup_center, cup_radius, transform=ax.transData)
                ax.imshow(cream_data[0, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=1)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.axis('off')
                ax.set_clip_path(circle_clip)

            plt.suptitle(f"AI Test: Mixing Cream Into Coffee\nTime Step: {step+1}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)
            plt.close(fig)

    print("5. Saving animation to 'application_mixing.gif'...")
    imageio.mimsave('application_mixing.gif', frames, fps=15)
    print("✓ Done! Check your folder for the GIF.")


# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    IMG_SIZE = 64
    N_SAMPLES = 400
    EPOCHS = 25 # Training is faster on this simpler dataset

    # 1. Generate new, intuitive training data
    train_data = generate_single_vortex_data(n_samples=N_SAMPLES, img_size=IMG_SIZE)

    # 2. Create and train the model
    model = PhysicsInformedDiffusion(img_size=IMG_SIZE, timesteps=50).to(device)
    train_model(model, train_data, epochs=EPOCHS)
    
    # 3. Run the new intuitive, practical applications
    find_best_pattern_for_latte_art(model)
    visualize_mixing_animation(model)

    print("\n" + "=" * 70)
    print("All tasks complete. Check for the following output files:")
    print("  - application_latte_art.png")
    print("  - application_mixing.gif")
    print("=" * 70)
