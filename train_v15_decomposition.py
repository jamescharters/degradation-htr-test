#!/usr/bin/env python3
"""
Quill PINN V15 (DECOMPOSITION)
A completely new approach. The problem is decomposed into two networks:
1. AmplitudeNet(x, y): Learns the final shape of the letter.
2. TemporalNet(t): Learns the timing of the stroke, governed by physics.
The full model is h(x, y, t) = AmplitudeNet(x, y) * TemporalNet(t).
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
print("QUILL PHYSICS PINN TRAINING V15 (DECOMPOSITION)")
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
            nn.Linear(64, 1), nn.Sigmoid() # Sigmoid to keep amplitude in [0, 1]
        )
class TemporalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1), nn.Sigmoid() # Sigmoid to keep profile in [0, 1]
        )
class DecomposedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude_net = AmplitudeNet()
        self.temporal_net = TemporalNet()
        print("✓ Decomposed model created.")
    def forward(self, x, y, t):
        amplitude = self.amplitude_net(torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1))
        temporal_profile = self.temporal_net(t.unsqueeze(-1))
        return amplitude.squeeze(-1) * temporal_profile.squeeze(-1)

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET (from V14, unchanged)
# ============================================
class QuillDataset_V15:
    def __init__(self, prefix, metadata_path):
        self.image_final = 1.0 - (np.array(Image.open(f"{prefix}.png").convert('L')).astype(np.float32) / 255.0)
        self.height, self.width = self.image_final.shape
        with open(metadata_path, 'r') as f: self.metadata = json.load(f)
        self.total_time = self.metadata['total_time']
        print(f"✓ Loaded final frame ({self.width}x{self.height}) and metadata. Total time: {self.total_time:.2f}s")
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
                idx_f = progress*(n_pts-1); idx0, idx1 = int(np.floor(idx_f)), int(np.ceil(idx_f))
                if idx0>=n_pts: idx0=n_pts-1
                if idx1>=n_pts: idx1=n_pts-1
                if idx0==idx1: pos = points[idx0]
                else: pos = points[idx0]*(1-(idx_f-idx0)) + points[idx1]*(idx_f-idx0)
                positions.append((pos[0]/self.width, pos[1]/self.height))
            else: positions.append((np.nan, np.nan))
        return torch.tensor(positions, dtype=torch.float32, device=t_values.device)
    def get_initial_points(self, n): return torch.rand(n, 2, device=device)
    def get_final_points(self, n):
        x_idx, y_idx = np.random.randint(0,self.width,n), np.random.randint(0,self.height,n)
        x = torch.tensor(x_idx/(self.width-1), dtype=torch.float32, device=device)
        y = torch.tensor(y_idx/(self.height-1), dtype=torch.float32, device=device)
        h = torch.tensor(self.image_final[y_idx, x_idx], dtype=torch.float32, device=device)
        return x, y, h
    def get_off_path_points(self, n): return torch.rand(n, 3, device=device)
    def get_stroke_times(self, n):
        stroke_start_norm = self.metadata['strokes'][0]['start_time'] / self.total_time
        stroke_end_norm = self.metadata['strokes'][-1]['end_time'] / self.total_time
        return torch.rand(n, device=device) * (stroke_end_norm - stroke_start_norm) + stroke_start_norm

# ============================================
# TRAINING
# ============================================
def train_pinn_v15(model, dataset, epochs=15000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- DATA LOSS ---
        # The final image must match AmplitudeNet * TemporalNet(1)
        n_data = 1024
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        # We need TemporalNet(1) to be 1, so we enforce it directly.
        t_one = torch.ones(1, device=device)
        temporal_at_one = model.temporal_net(t_one.unsqueeze(-1))
        loss_t_one = ((temporal_at_one - 1.0)**2).mean()

        amplitude_at_fc = model.amplitude_net(torch.cat([x_fc.unsqueeze(-1), y_fc.unsqueeze(-1)], dim=-1)).squeeze(-1)
        loss_final_shape = ((amplitude_at_fc - h_fc)**2).mean()
        
        loss_data = loss_final_shape + loss_t_one

        # --- PHYSICS LOSS on TemporalNet ---
        # T(t) should be monotonic (dT/dt >= 0)
        n_phys = 2048
        t_phys = dataset.get_stroke_times(n_phys).unsqueeze(-1).requires_grad_(True)
        t_profile = model.temporal_net(t_phys)
        dt_dt = compute_gradient(t_profile, t_phys)
        loss_monotonic = (torch.relu(-dt_dt)**2).mean()

        # T(0) must be 0
        t_zero = torch.zeros(1, device=device)
        temporal_at_zero = model.temporal_net(t_zero.unsqueeze(-1))
        loss_t_zero = (temporal_at_zero**2).mean()

        loss_physics = loss_monotonic + loss_t_zero
        
        # Total Loss
        total_loss = 100.0 * loss_data + 1.0 * loss_physics
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Total Loss: {total_loss.item():.6f}")
            print(f"    - Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_physics.item():.6f}")

    print("\n✓ Training complete!")

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
