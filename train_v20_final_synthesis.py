#!/usr/bin/env python3
"""
Quill PINN V20 (THE DEFINITIVE SYNTHESIS)
This version uses a DECOMPOSITION architecture, combining a pre-trained, frozen
AmplitudeNet (for the shape) with a WaveNet (for the process). The WaveNet is
trained on the proven, sheath-constrained physics from our toy problem. This
decouples the conflicting tasks and provides the final, robust solution.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("=" * 60)
print("QUILL PHYSICS PINN TRAINING V20 (DEFINITIVE SYNTHESIS)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# = a==========================================
# MODEL & GRADIENT UTILS
# ============================================
def positional_encoding(coords, encoding_dim):
    freqs = 2.0 ** torch.arange(encoding_dim, device=coords.device)
    args = coords.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).flatten(start_dim=1)

class AmplitudeNet(nn.Module):
    def __init__(self, encoding_dim=12):
        super().__init__()
        self.encoding_dim = encoding_dim
        input_dim = 2 * 2 * encoding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, xy):
        encoded_xy = positional_encoding(xy, self.encoding_dim)
        return self.network(encoded_xy)

class WaveNet(nn.Module):
    def __init__(self, encoding_dim=10):
        super().__init__()
        self.encoding_dim = encoding_dim
        input_dim = 3 * 2 * encoding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, y, t):
        coords = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        encoded_coords = positional_encoding(coords, self.encoding_dim)
        return torch.sigmoid(self.network(encoded_coords))

class DecomposedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude_net = AmplitudeNet()
        self.wave_net = WaveNet()
        print("✓ Decomposed model with Positional Encoding created.")
    def forward(self, x, y, t):
        xy_input = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        amplitude = self.amplitude_net(xy_input).squeeze(-1)
        wave = self.wave_net(x, y, t).squeeze(-1)
        return amplitude * wave

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET
# ============================================
class QuillDataset_V20:
    def __init__(self, prefix, metadata_path):
        self.image_final = 1.0 - (np.array(Image.open(f"{prefix}.png").convert('L')).astype(np.float32) / 255.0)
        self.height, self.width = self.image_final.shape
        with open(metadata_path, 'r') as f: self.metadata = json.load(f)
        self.total_time = self.metadata['total_time']
        print(f"✓ Loaded final frame ({self.width}x{self.height}) and metadata.")
    def get_quill_position_at_time(self, t_values):
        t_values_np = t_values.detach().cpu().numpy()
        positions = []
        for t_norm in t_values_np:
            t_sim = t_norm * self.total_time; active_stroke = None
            for stroke in self.metadata['strokes']:
                if stroke['start_time'] <= t_sim < stroke['end_time']: active_stroke = stroke; break
            if active_stroke:
                progress = (t_sim - active_stroke['start_time']) / (active_stroke['end_time'] - active_stroke['start_time'])
                points = np.array(active_stroke['points']); n_pts = len(points)
                idx_f = progress * (n_pts - 1); idx0, idx1 = int(np.floor(idx_f)), int(np.ceil(idx_f))
                if idx1 >= len(points): idx1 = idx0 = len(points) - 1
                if idx0 == idx1: current_pos = points[idx0]
                else: current_pos = points[idx0] * (1 - (idx_f - idx0)) + points[idx1] * (idx_f - idx0)
                positions.append((current_pos[0]/self.width, current_pos[1]/self.height))
            else: positions.append((np.nan, np.nan))
        return torch.tensor(positions, dtype=torch.float32, device=t_values.device)
    def get_all_points(self):
        y, x = torch.meshgrid(torch.linspace(0, 1, self.height), torch.linspace(0, 1, self.width), indexing='ij')
        xy = torch.stack([x.flatten(), y.flatten()], dim=1).to(device)
        h = torch.tensor(self.image_final.flatten(), dtype=torch.float32, device=device)
        return xy, h
    def get_on_path_points(self, n):
        stroke_start_norm = self.metadata['strokes'][0]['start_time'] / self.total_time
        stroke_end_norm = self.metadata['strokes'][-1]['end_time'] / self.total_time
        t_on_path = torch.rand(n, device=device) * (stroke_end_norm - stroke_start_norm) + stroke_start_norm
        quill_pos = self.get_quill_position_at_time(t_on_path)
        noise = torch.randn(n, 2, device=device) * (1.5 / self.width)
        x_on_path = torch.clamp(quill_pos[:, 0] + noise[:, 0], 0, 1)
        y_on_path = torch.clamp(quill_pos[:, 1] + noise[:, 1], 0, 1)
        return x_on_path, y_on_path, t_on_path
    def get_sheath_points(self, n, distance=4.0):
        x_on, y_on, t_on = self.get_on_path_points(n)
        angle = torch.rand(n, device=device) * 2 * np.pi
        offset_x = torch.cos(angle) * (distance / self.width)
        offset_y = torch.sin(angle) * (distance / self.height)
        x_sheath = torch.clamp(x_on + offset_x, 0, 1)
        y_sheath = torch.clamp(y_on + offset_y, 0, 1)
        return x_sheath, y_sheath, t_on

# ============================================
# TRAINING
# ============================================
def train_pinn_v20(model, dataset, phase1_epochs=7000, phase2_epochs=18000):
    epochs = phase1_epochs + phase2_epochs
    
    # --- PHASE 1: Train ONLY the AmplitudeNet on the final shape ---
    print("\n" + "="*60 + f"\nPHASE 1: AMPLITUDE FITTING ({phase1_epochs:,} epochs)\n" + "-"*60)
    optimizer = torch.optim.Adam(model.amplitude_net.parameters(), lr=1e-3)
    xy_all, h_all = dataset.get_all_points() # Use all points for a perfect fit
    for epoch in range(phase1_epochs):
        model.amplitude_net.train()
        optimizer.zero_grad()
        amplitude_pred = model.amplitude_net(xy_all).squeeze(-1)
        loss = ((amplitude_pred - h_all)**2).mean()
        loss.backward(); optimizer.step()
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Shape Loss: {loss.item():.6f}")

    # --- PHASE 2: Freeze AmplitudeNet, Train ONLY WaveNet on physics ---
    print("\n" + "="*60 + f"\nPHASE 2: WAVE FITTING ({phase2_epochs:,} epochs)\n" + "-"*60)
    for param in model.amplitude_net.parameters(): param.requires_grad = False
    optimizer = torch.optim.Adam(model.wave_net.parameters(), lr=5e-4)

    for epoch in range(phase2_epochs):
        model.wave_net.train()
        optimizer.zero_grad()
        
        # Physics is applied to the WaveNet, similar to the Ghost Stroke problem
        n_phys = 2048
        
        # Rule 1: Wave must be "on" (1.0) after the stroke is done
        x_post, y_post, t_post = dataset.get_on_path_points(n_phys) # Use on-path points for this
        loss_post_stroke = ((model.wave_net(x_post, y_post, torch.ones_like(t_post)) - 1.0)**2).mean()

        # Rule 2: d(Wave)/dt must be positive on the path
        x_src, y_src, t_src = dataset.get_on_path_points(n_phys); x_src.requires_grad_(); y_src.requires_grad_(); t_src.requires_grad_()
        wave_src = model.wave_net(x_src, y_src, t_src)
        dw_dt_src = compute_gradient(wave_src, t_src)
        loss_source = (torch.relu(0.5 - dw_dt_src)**2).mean() # A gentler target rate for the wave

        # Rule 3: d(Wave)/dt must be zero in the sheath
        x_sh, y_sh, t_sh = dataset.get_sheath_points(n_phys); x_sh.requires_grad_(); y_sh.requires_grad_(); t_sh.requires_grad_()
        wave_sh = model.wave_net(x_sh, y_sh, t_sh)
        dw_dt_sh = compute_gradient(wave_sh, t_sh)
        loss_sheath = (dw_dt_sh**2).mean()

        # Rule 4: Wave must be monotonic in time
        loss_monotonic = (torch.relu(-dw_dt_src)**2).mean() + (torch.relu(-dw_dt_sh)**2).mean()

        loss = 10.0 * loss_post_stroke + 1.0 * loss_source + 5.0 * loss_sheath + 1.0 * loss_monotonic
        loss.backward(); optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1+phase1_epochs:5d}/{epochs} | Wave Physics Loss: {loss.item():.6f}")

    print("\n✓ Training complete!")
    return model

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_reconstruction_v20(model, dataset, output_path='reconstruction_v20.png'):
    model.eval(); time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points), figsize=(16, 3))
    print("\nGenerating visualization...")
    with torch.no_grad():
        x_grid, y_grid = torch.linspace(0,1,dataset.width,device=device), torch.linspace(0,1,dataset.height,device=device)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
        x_flat, y_flat = X.flatten(), Y.flatten()
        for i, t_val in enumerate(time_points):
            t_tensor = torch.full_like(x_flat, t_val)
            h_pred = model(x_flat, y_flat, t_tensor).cpu().numpy().reshape(dataset.height, dataset.width)
            axes[i].imshow(1.0 - h_pred, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f't={t_val:.2f}'); axes[i].axis('off')
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.show()
    print(f"✓ Saved final visualization to {output_path}")

if __name__ == "__main__":
    prefix = "synthetic_letters/CANONE_letter_0_C"
    meta_path = f"{prefix}_metadata.json"
    print("MANDATORY: Ensure you have run the CORRECTED `quill_simulator.py`.")
    if not Path(f"{prefix}.png").exists() or not Path(meta_path).exists():
        print("✗ Data not found!"); sys.exit(1)
    dataset = QuillDataset_V20(prefix, meta_path)
    model = DecomposedPINN().to(device)
    train_pinn_v20(model, dataset)
    visualize_reconstruction_v20(model, dataset)
