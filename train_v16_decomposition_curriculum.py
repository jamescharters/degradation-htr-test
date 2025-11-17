#!/usr/bin/env python3
"""
Quill PINN V16 (DECOMPOSITION + CURRICULUM)
The definitive version. This script combines our two most powerful insights:
1. Decomposing the problem into an AmplitudeNet (the "what") and a TemporalNet (the "how").
2. Using a strict two-phase curriculum to train each network on its specific task without conflict.
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
print("QUILL PHYSICS PINN TRAINING V16 (DECOMPOSITION + CURRICULUM)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & GRADIENT UTILS
# ============================================
class AmplitudeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, xy):
        return self.network(xy)

class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t):
        return torch.sigmoid(self.network(t))

class DecomposedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude_net = AmplitudeNet()
        self.temporal_net = TemporalNet()
        print("✓ Decomposed model created.")
    def forward(self, x, y, t):
        xy_input = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        t_input = t.unsqueeze(-1)
        amplitude = self.amplitude_net(xy_input)
        temporal_profile = self.temporal_net(t_input)
        return amplitude.squeeze(-1) * temporal_profile.squeeze(-1)

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET
# ============================================
class QuillDataset_V16:
    def __init__(self, prefix, metadata_path):
        self.image_final = 1.0 - (np.array(Image.open(f"{prefix}.png").convert('L')).astype(np.float32) / 255.0)
        self.height, self.width = self.image_final.shape
        with open(metadata_path, 'r') as f: self.metadata = json.load(f)
        self.total_time = self.metadata['total_time']
        self.stroke_start_norm = self.metadata['strokes'][0]['start_time'] / self.total_time
        self.stroke_end_norm = self.metadata['strokes'][-1]['end_time'] / self.total_time
        print(f"✓ Loaded final frame ({self.width}x{self.height}) and metadata.")

    def get_final_points(self, n):
        x_idx, y_idx = np.random.randint(0,self.width,n), np.random.randint(0,self.height,n)
        x = torch.tensor(x_idx/(self.width-1), dtype=torch.float32, device=device)
        y = torch.tensor(y_idx/(self.height-1), dtype=torch.float32, device=device)
        h = torch.tensor(self.image_final[y_idx, x_idx], dtype=torch.float32, device=device)
        return x, y, h
    def get_pre_stroke_times(self, n): return torch.rand(n, device=device) * self.stroke_start_norm
    def get_post_stroke_times(self, n): return torch.rand(n, device=device) * (1.0 - self.stroke_end_norm) + self.stroke_end_norm
    def get_during_stroke_times(self, n): return torch.rand(n, device=device) * (self.stroke_end_norm - self.stroke_start_norm) + self.stroke_start_norm

# ============================================
# TRAINING
# ============================================
def train_pinn_v16(model, dataset, epochs=15000, phase1_epochs=10000):
    
    # --- PHASE 1: Train ONLY the AmplitudeNet on the final shape ---
    print("\n" + "="*60 + f"\nPHASE 1: AMPLITUDE FITTING ({phase1_epochs:,} epochs)\n" + "-"*60)
    optimizer_amp = torch.optim.Adam(model.amplitude_net.parameters(), lr=1e-3)
    for epoch in range(phase1_epochs):
        model.amplitude_net.train()
        model.temporal_net.eval()
        optimizer_amp.zero_grad()
        
        n_data = 2048
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        xy_input = torch.cat([x_fc.unsqueeze(-1), y_fc.unsqueeze(-1)], dim=-1)
        amplitude_pred = model.amplitude_net(xy_input).squeeze(-1)
        
        loss_final_shape = ((amplitude_pred - h_fc)**2).mean()
        loss_final_shape.backward()
        optimizer_amp.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Shape Loss: {loss_final_shape.item():.6f}")

    # --- PHASE 2: Freeze AmplitudeNet, Train ONLY TemporalNet on physics ---
    print("\n" + "="*60 + f"\nPHASE 2: TEMPORAL FITTING ({epochs - phase1_epochs:,} epochs)\n" + "-"*60)
    # Freeze the weights of the amplitude network
    for param in model.amplitude_net.parameters():
        param.requires_grad = False
    
    optimizer_temp = torch.optim.Adam(model.temporal_net.parameters(), lr=1e-3)
    for epoch in range(epochs - phase1_epochs):
        model.amplitude_net.eval()
        model.temporal_net.train()
        optimizer_temp.zero_grad()
        
        n_phys = 2048
        t_pre = dataset.get_pre_stroke_times(n_phys).unsqueeze(-1)
        loss_pre_stroke = (model.temporal_net(t_pre)**2).mean()

        t_post = dataset.get_post_stroke_times(n_phys).unsqueeze(-1)
        loss_post_stroke = ((model.temporal_net(t_post) - 1.0)**2).mean()

        t_during = dataset.get_during_stroke_times(n_phys).unsqueeze(-1).requires_grad_(True)
        t_profile = model.temporal_net(t_during)
        dt_dt = compute_gradient(t_profile, t_during)
        loss_monotonic = (torch.relu(-dt_dt)**2).mean()
        
        total_loss = loss_pre_stroke + loss_post_stroke + loss_monotonic
        total_loss.backward()
        optimizer_temp.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1+phase1_epochs:5d}/{epochs} | Temporal Physics Loss: {total_loss.item():.6f}")

    print("\n✓ Training complete!")
    return model

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_reconstruction_v16(model, dataset, output_path='reconstruction_v16.png'):
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points), figsize=(16, 3))
    print("\nGenerating visualization...")
    with torch.no_grad():
        x_grid = torch.linspace(0,1,dataset.width,device=device)
        y_grid = torch.linspace(0,1,dataset.height,device=device)
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
    if not Path(f"{prefix}.png").exists() or not Path(meta_path).exists():
        print("✗ Data not found! Please run quill_simulator.py first."); sys.exit(1)
    dataset = QuillDataset_V16(prefix, meta_path)
    model = DecomposedPINN().to(device)
    train_pinn_v16(model, dataset)
    visualize_reconstruction_v16(model, dataset)
