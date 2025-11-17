#!/usr/bin/env python3
"""
Quill PINN V15 (DECOMPOSITION - CORRECTED)
A completely new approach. The problem is decomposed into two networks:
1. AmplitudeNet(x, y): Learns the final shape of the letter.
2. TemporalNet(t): Learns the timing of the stroke, governed by robust physics.
The full model is h(x, y, t) = AmplitudeNet(x, y) * TemporalNet(t).
This version corrects the missing 'forward' methods and the flawed physics logic.
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
print("QUILL PHYSICS PINN TRAINING V15 (DECOMPOSITION - CORRECTED)")
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
    # --- FIX: Added forward method ---
    def forward(self, xy):
        return self.network(xy)

class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1) # No final sigmoid to allow for better gradient flow
        )
    # --- FIX: Added forward method ---
    def forward(self, t):
        # We apply sigmoid here to ensure the output is [0, 1]
        return torch.sigmoid(self.network(t))

class DecomposedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude_net = AmplitudeNet()
        self.temporal_net = TemporalNet()
        print("✓ Decomposed model created.")
    # --- FIX: Correctly call sub-networks ---
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
class QuillDataset_V15:
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
    # --- NEW: Methods for robust temporal physics ---
    def get_pre_stroke_times(self, n): return torch.rand(n, device=device) * self.stroke_start_norm
    def get_post_stroke_times(self, n): return torch.rand(n, device=device) * (1.0 - self.stroke_end_norm) + self.stroke_end_norm
    def get_during_stroke_times(self, n): return torch.rand(n, device=device) * (self.stroke_end_norm - self.stroke_start_norm) + self.stroke_start_norm

# ============================================
# TRAINING
# ============================================
def train_pinn_v15(model, dataset, epochs=15000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- LOSS 1: DATA LOSS ---
        # The final image shape must be learned by the AmplitudeNet
        n_data = 1024
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        xy_input = torch.cat([x_fc.unsqueeze(-1), y_fc.unsqueeze(-1)], dim=-1)
        amplitude_at_fc = model.amplitude_net(xy_input).squeeze(-1)
        loss_final_shape = ((amplitude_at_fc - h_fc)**2).mean()

        # --- LOSS 2: TEMPORAL PHYSICS LOSS ---
        n_phys = 1024
        # Rule A: T(t) must be 0 before the stroke
        t_pre = dataset.get_pre_stroke_times(n_phys).unsqueeze(-1)
        loss_pre_stroke = (model.temporal_net(t_pre)**2).mean()

        # Rule B: T(t) must be 1 after the stroke
        t_post = dataset.get_post_stroke_times(n_phys).unsqueeze(-1)
        loss_post_stroke = ((model.temporal_net(t_post) - 1.0)**2).mean()

        # Rule C: dT/dt must be >= 0 during the stroke
        t_during = dataset.get_during_stroke_times(n_phys).unsqueeze(-1).requires_grad_(True)
        t_profile = model.temporal_net(t_during)
        dt_dt = compute_gradient(t_profile, t_during)
        loss_monotonic = (torch.relu(-dt_dt)**2).mean()
        
        loss_physics = loss_pre_stroke + loss_post_stroke + loss_monotonic
        
        # Total Loss
        total_loss = 100.0 * loss_final_shape + 100.0 * loss_physics
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Total Loss: {total_loss.item():.6f}")
            print(f"    - Shape Loss: {loss_final_shape.item():.6f} | Temporal Physics: {loss_physics.item():.6f}")

    print("\n✓ Training complete!")
    return model

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_reconstruction_v15(model, dataset, output_path='reconstruction_v15.png'):
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
    dataset = QuillDataset_V15(prefix, meta_path)
    model = DecomposedPINN().to(device)
    train_pinn_v15(model, dataset)
    visualize_reconstruction_v15(model, dataset)
