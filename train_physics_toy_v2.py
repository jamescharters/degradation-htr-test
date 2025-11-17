#!/usr/bin/env python3
"""
"The Ghost Stroke" V2 - A Physics-Only Toy Problem with a Sheath Constraint
This version solves the "red smudge" problem by adding a "sheath" of quiescence
points right next to the stroke path, forcing the network to learn a sharp,
localized line of change.
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
print("PHYSICS TOY V2: THE SHEATH CONSTRAINT")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & GRADIENT UTILS
# ============================================
class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        print(f"✓ SimplePINN created.")
    def forward(self, x, y, t):
        inputs = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        return self.network(inputs).squeeze(-1)

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET (with new sheath sampling)
# ============================================
class PathProvider:
    def __init__(self, prefix, metadata_path):
        with open(metadata_path, 'r') as f: self.metadata = json.load(f)
        self.total_time = self.metadata['total_time']
        self.width, self.height = 100, 100
        print(f"✓ Path Provider initialized.")
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
                if idx0>=n_pts: idx0=n_pts-1;
                if idx1>=n_pts: idx1=n_pts-1
                if idx0==idx1: pos = points[idx0]
                else: pos = points[idx0]*(1-(idx_f-idx0)) + points[idx1]*(idx_f-idx0)
                positions.append((pos[0]/self.width, pos[1]/self.height))
            else: positions.append((np.nan, np.nan))
        return torch.tensor(positions, dtype=torch.float32, device=t_values.device)
    
    def get_off_path_points(self, n): return torch.rand(n, 3, device=device)
    
    def get_on_path_points(self, n):
        stroke_start_norm = self.metadata['strokes'][0]['start_time'] / self.total_time
        stroke_end_norm = self.metadata['strokes'][-1]['end_time'] / self.total_time
        t_on_path = torch.rand(n, device=device) * (stroke_end_norm - stroke_start_norm) + stroke_start_norm
        quill_pos = self.get_quill_position_at_time(t_on_path)
        noise = torch.randn(n, 2, device=device) * (1.0 / self.width) # Tighter noise
        x_on_path = torch.clamp(quill_pos[:, 0] + noise[:, 0], 0, 1)
        y_on_path = torch.clamp(quill_pos[:, 1] + noise[:, 1], 0, 1)
        return x_on_path, y_on_path, t_on_path

    # --- NEW METHOD FOR THE SHEATH ---
    def get_sheath_points(self, n, distance=5.0):
        # First, get points on the path
        x_on, y_on, t_on = self.get_on_path_points(n)
        # Generate random offsets perpendicular to the path
        angle = torch.rand(n, device=device) * 2 * np.pi
        offset_x = torch.cos(angle) * (distance / self.width)
        offset_y = torch.sin(angle) * (distance / self.height)
        
        x_sheath = torch.clamp(x_on + offset_x, 0, 1)
        y_sheath = torch.clamp(y_on + offset_y, 0, 1)
        return x_sheath, y_sheath, t_on

# ============================================
# TRAINING
# ============================================
def train_physics_toy_v2(model, path_provider, epochs=15000, lr=3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lambda_boundary = 10.0
    lambda_source = 1.0
    lambda_sheath = 5.0 # NEW: Strong penalty for blurriness
    lambda_quiescence = 0.5 # Less important now

    print("\n--- Training Physics Toy Problem V2 ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Boundary Loss
        n_boundary = 1024
        x_b, y_b = torch.rand(n_boundary, device=device), torch.rand(n_boundary, device=device)
        h_at_start = model(x_b, y_b, torch.zeros_like(x_b))
        h_at_end = model(x_b, y_b, torch.ones_like(x_b))
        loss_boundary = (h_at_start**2).mean() + (h_at_end**2).mean()

        # Source Loss
        n_source = 2048
        x_src, y_src, t_src = path_provider.get_on_path_points(n_source)
        x_src.requires_grad_(); y_src.requires_grad_(); t_src.requires_grad_()
        h_src = model(x_src, y_src, t_src)
        h_t_src = compute_gradient(h_src, t_src)
        loss_source = (torch.relu(1.0 - h_t_src)**2).mean()

        # Sheath Loss
        n_sheath = 2048
        x_sh, y_sh, t_sh = path_provider.get_sheath_points(n_sheath, distance=5.0)
        x_sh.requires_grad_(); y_sh.requires_grad_(); t_sh.requires_grad_()
        h_sh = model(x_sh, y_sh, t_sh)
        h_t_sh = compute_gradient(h_sh, t_sh)
        loss_sheath = (h_t_sh**2).mean()

        # Quiescence Loss
        n_quiescence = 1024
        x_q, y_q, t_q = path_provider.get_off_path_points(n_quiescence).T
        x_q.requires_grad_(); y_q.requires_grad_(); t_q.requires_grad_()
        h_q = model(x_q, y_q, t_q)
        h_t_q = compute_gradient(h_q, t_q)
        loss_quiescence = (h_t_q**2).mean()

        total_loss = (lambda_boundary * loss_boundary +
                      lambda_source * loss_source +
                      lambda_sheath * loss_sheath +
                      lambda_quiescence * loss_quiescence)
        
        total_loss.backward(); optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Total Loss: {total_loss.item():.6f}")
            print(f"    - Source: {loss_source.item():.6f} | Sheath: {loss_sheath.item():.6f}")

    print("✓ Training complete!")
    return model

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_physics_toy_v2(model, output_path='physics_toy_result_v2.png'):
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(time_points), figsize=(16, 6.5))
    print("\nGenerating visualization...")
    res = 100
    x_grid, y_grid = torch.linspace(0,1,res,device=device), torch.linspace(0,1,res,device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    x_flat, y_flat = X.flatten(), Y.flatten()
    for i, t_val in enumerate(time_points):
        t_flat = torch.full_like(x_flat, t_val, requires_grad=True)
        h_pred = model(x_flat, y_flat, t_flat)
        h_t_pred = compute_gradient(h_pred, t_flat)
        h_img, h_t_img = h_pred.detach().cpu().numpy().reshape(res, res), h_t_pred.detach().cpu().numpy().reshape(res, res)
        ax_h = axes[0, i]; im_h = ax_h.imshow(h_img, cmap='viridis', origin='lower'); ax_h.set_title(f'h at t={t_val:.2f}'); ax_h.axis('off'); fig.colorbar(im_h, ax=ax_h, fraction=0.046, pad=0.04)
        ax_ht = axes[1, i]; im_ht = ax_ht.imshow(h_t_img, cmap='RdBu_r', origin='lower', vmin=-1.5, vmax=1.5); ax_ht.set_title(f'∂h/∂t at t={t_val:.2f}'); ax_ht.axis('off'); fig.colorbar(im_ht, ax=ax_ht, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.show()
    print(f"✓ Saved final visualization to {output_path}")

if __name__ == "__main__":
    prefix = "synthetic_letters/CANONE_letter_0_C"
    meta_path = f"{prefix}_metadata.json"
    if not Path(meta_path).exists():
        print("✗ Metadata not found! Please run quill_simulator.py first."); sys.exit(1)
    path_provider = PathProvider(prefix, meta_path)
    model = SimplePINN().to(device)
    trained_model = train_physics_toy_v2(model, path_provider)
    visualize_physics_toy_v2(trained_model)
