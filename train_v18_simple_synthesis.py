#!/usr/bin/env python3
"""
Quill PINN V18 (SIMPLE SYNTHESIS)
This script represents the simplest, most robust synthesis of all our insights.
It uses a single, unified network and a simple two-phase training schedule:
1. Boundary Warm-up: Give the network a head start on learning the final shape.
2. Full Physics: Train the boundary and the proven, sheath-constrained physics
   simultaneously to learn the stroke process.
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
print("QUILL PHYSICS PINN TRAINING V18 (SIMPLE SYNTHESIS)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & GRADIENT UTILS
# ============================================
# Using the positional encoding model that we proved works for learning shapes
class QuillPINN_V18(nn.Module):
    def __init__(self, encoding_dim=10):
        super().__init__()
        self.encoding_dim = encoding_dim
        # Input: 3 coords (x,y,t) * 2 (sin,cos) * encoding_dim
        input_dim = 3 * 2 * encoding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        print(f"✓ Positional Encoding PINN created (Params: {sum(p.numel() for p in self.parameters()):,})")

    def positional_encoding(self, coords):
        freqs = 2.0 ** torch.arange(self.encoding_dim, device=coords.device)
        args = coords.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).flatten(start_dim=1)

    def forward(self, x, y, t):
        coords = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        encoded_coords = self.positional_encoding(coords)
        return torch.relu(self.network(encoded_coords).squeeze(-1))

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET (with all necessary sampling methods)
# ============================================
class QuillDataset_V18:
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
    def get_on_path_points(self, n):
        stroke_start_norm = self.metadata['strokes'][0]['start_time'] / self.total_time
        stroke_end_norm = self.metadata['strokes'][-1]['end_time'] / self.total_time
        t_on_path = torch.rand(n, device=device) * (stroke_end_norm - stroke_start_norm) + stroke_start_norm
        quill_pos = self.get_quill_position_at_time(t_on_path)
        noise = torch.randn(n, 2, device=device) * (1.5 / self.width)
        x_on_path = torch.clamp(quill_pos[:, 0] + noise[:, 0], 0, 1)
        y_on_path = torch.clamp(quill_pos[:, 1] + noise[:, 1], 0, 1)
        return x_on_path, y_on_path, t_on_path
    def get_sheath_points(self, n, distance=5.0):
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
def train_pinn_v18(model, dataset, epochs=25000, warmup_epochs=5000):
    
    # --- PHASE 1: Boundary Warm-up ---
    print("\n" + "="*60 + f"\nPHASE 1: BOUNDARY WARM-UP ({warmup_epochs:,} epochs)\n" + "-"*60)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(warmup_epochs):
        model.train(); optimizer.zero_grad()
        n_data = 2048
        x_ic, y_ic = dataset.get_initial_points(n_data).T
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic)) ** 2).mean()
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        h_pred_fc = model(x_fc, y_fc, torch.ones_like(x_fc))
        loss_fc = ((h_pred_fc - h_fc) ** 2).mean()
        total_loss = 200.0 * (loss_ic + loss_fc)
        total_loss.backward(); optimizer.step()
        if (epoch + 1) % 500 == 0: print(f"  Epoch {epoch+1:5d}/{epochs} | Warm-up Loss: {total_loss.item():.6f}")

    # --- PHASE 2: Full Physics Training ---
    print("\n" + "="*60 + f"\nPHASE 2: FULL PHYSICS TRAINING ({epochs - warmup_epochs:,} epochs)\n" + "-"*60)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # New optimizer, lower LR
    for epoch in range(warmup_epochs, epochs):
        model.train(); optimizer.zero_grad()
        
        # Boundary loss (always on)
        n_data = 1024
        x_ic, y_ic = dataset.get_initial_points(n_data).T
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic)) ** 2).mean()
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        h_pred_fc = model(x_fc, y_fc, torch.ones_like(x_fc)); loss_fc = ((h_pred_fc - h_fc) ** 2).mean()
        loss_boundary = loss_ic + loss_fc

        # Full physics with sheath constraint
        n_phys = 1500
        x_src, y_src, t_src = dataset.get_on_path_points(n_phys); x_src.requires_grad_(); y_src.requires_grad_(); t_src.requires_grad_()
        h_src = model(x_src, y_src, t_src); h_t_src = compute_gradient(h_src, t_src)
        loss_source = (torch.relu(1.5 - h_t_src)**2).mean()
        
        x_sh, y_sh, t_sh = dataset.get_sheath_points(n_phys); x_sh.requires_grad_(); y_sh.requires_grad_(); t_sh.requires_grad_()
        h_sh = model(x_sh, y_sh, t_sh); h_t_sh = compute_gradient(h_sh, t_sh)
        loss_sheath = (h_t_sh**2).mean()

        x_q, y_q, t_q = dataset.get_off_path_points(n_phys).T; x_q.requires_grad_(); y_q.requires_grad_(); t_q.requires_grad_()
        h_q = model(x_q, y_q, t_q); h_t_q = compute_gradient(h_q, t_q)
        loss_quiescence = (h_t_q**2).mean()
        
        total_loss = 150.0 * loss_boundary + 50.0 * loss_source + 100.0 * loss_sheath + 25.0 * loss_quiescence
        total_loss.backward(); optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Total Loss: {total_loss.item():.4f}")
            print(f"    - Boundary: {loss_boundary.item():.6f} | Source: {loss_source.item():.6f} | Sheath: {loss_sheath.item():.6f}")

    print("\n✓ Training complete!")
    return model

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_reconstruction_v18(model, dataset, output_path='reconstruction_v18.png'):
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
    if not Path(f"{prefix}.png").exists() or not Path(meta_path).exists():
        print("✗ Data not found! Please run the CORRECTED quill_simulator.py first."); sys.exit(1)
    dataset = QuillDataset_V18(prefix, meta_path)
    model = QuillPINN_V18().to(device)
    train_pinn_v18(model, dataset)
    visualize_reconstruction_v18(model, dataset)
