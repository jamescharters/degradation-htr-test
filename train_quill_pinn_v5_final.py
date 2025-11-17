#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction V5 (FINAL)
This version implements a "curriculum learning" strategy with a warm-up phase
to overcome the local minimum problem and ensure stable convergence.
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
print("QUILL PHYSICS PINN TRAINING V5 (CURRICULUM LEARNING)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & GRADIENT UTILS
# ============================================
class QuillPINN_V5(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=8, time_encoding_dim=32):
        super().__init__()
        self.time_encoding_dim = time_encoding_dim
        input_dim = 2 + time_encoding_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1): layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        print(f"✓ Model created (Params: {sum(p.numel() for p in self.parameters()):,})")
    def encode_time(self, t):
        freqs = 2.0 ** torch.arange(self.time_encoding_dim//2, device=t.device, dtype=torch.float32)
        t_expanded = t.unsqueeze(-1) * freqs.unsqueeze(0); return torch.cat([torch.sin(2*np.pi*t_expanded), torch.cos(2*np.pi*t_expanded)], dim=-1)
    def forward(self, x, y, t):
        if x.dim()==0: x=x.unsqueeze(0)
        if y.dim()==0: y=y.unsqueeze(0)
        if t.dim()==0: t=t.unsqueeze(0)
        t_encoded = self.encode_time(t); inputs = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t_encoded], dim=-1)
        return torch.relu(self.network(inputs).squeeze(-1))

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET
# ============================================
class QuillDataset_V5:
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
    def get_initial_points(self, n): return np.random.rand(n, 2)
    def get_final_points(self, n):
        x_idx, y_idx = np.random.randint(0,self.width,n), np.random.randint(0,self.height,n)
        return x_idx/(self.width-1), y_idx/(self.height-1), self.image_final[y_idx, x_idx]
    def get_physics_points(self, n): return np.random.rand(n, 3)

# ============================================
# PHYSICS LOSS
# ============================================
def physics_loss_v5_sourceterm(model, dataset, x, y, t):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    h = model(x, y, t); h_t = compute_gradient(h, t)
    if h_t is None: return torch.tensor(0.0), torch.tensor(0.0), 0
    quill_pos = dataset.get_quill_position_at_time(t)
    quill_x, quill_y = quill_pos[:,0], quill_pos[:,1]
    dist_sq = (x-quill_x)**2 + (y-quill_y)**2
    tip_radius_sq = (5.0/dataset.width)**2
    is_under_tip = (dist_sq < tip_radius_sq) & (~torch.isnan(dist_sq))
    num_under_tip = torch.sum(is_under_tip).item()
    target_rate = 2.0
    loss_src = (torch.relu(target_rate - h_t[is_under_tip])**2).mean()
    loss_q = (h_t[~is_under_tip]**2).mean()
    if torch.isnan(loss_src): loss_src = torch.tensor(0.0, device=device)
    if torch.isnan(loss_q): loss_q = torch.tensor(0.0, device=device)
    return loss_src, loss_q, num_under_tip

# ============================================
# TRAINING (with Curriculum Learning)
# ============================================
def train_pinn_v5(model, dataset, epochs=15000, lr=1e-3, warmup_epochs=3000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Weights are now introduced gradually
    lambda_data = 150.0
    lambda_source = 20.0
    lambda_quiescence = 20.0

    print("\n" + "="*60 + f"\nTRAINING STRATEGY: CURRICULUM LEARNING\n" + "-"*60)
    print(f"  Phase 1: Data Warm-up ({warmup_epochs:,} epochs)")
    print("    - GOAL: Learn the start (t=0) and end (t=1) states only.")
    print(f"  Phase 2: Physics Fine-tuning ({epochs-warmup_epochs:,} epochs)")
    print("    - GOAL: Learn the physical process connecting the states.\n" + "="*60 + "\n")
    
    n_physics_pts = 2500
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # --- PHASE 1 & 2: Boundary losses are always active ---
        n_data = 800
        ic_pts = dataset.get_initial_points(n_data)
        x_ic, y_ic = torch.tensor(ic_pts[:,0],dtype=torch.float32,device=device), torch.tensor(ic_pts[:,1],dtype=torch.float32,device=device)
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic)) ** 2).mean()

        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        x_fc_t, y_fc_t, h_fc_t = torch.tensor(x_fc,dtype=torch.float32,device=device), torch.tensor(y_fc,dtype=torch.float32,device=device), torch.tensor(h_fc,dtype=torch.float32,device=device)
        h_pred_fc = model(x_fc_t, y_fc_t, torch.ones_like(x_fc_t))
        loss_fc = ((h_pred_fc - h_fc_t) ** 2).mean()
        loss_boundary = loss_ic + loss_fc

        # --- PHASE 2 ONLY: Activate physics losses after warm-up ---
        if epoch >= warmup_epochs:
            phys_pts = dataset.get_physics_points(n_physics_pts)
            x_p, y_p, t_p = torch.tensor(phys_pts[:,0],dtype=torch.float32,device=device), torch.tensor(phys_pts[:,1],dtype=torch.float32,device=device), torch.tensor(phys_pts[:,2],dtype=torch.float32,device=device)
            loss_src, loss_q, points_under_tip = physics_loss_v5_sourceterm(model, dataset, x_p, y_p, t_p)
            
            total_loss = (lambda_data * loss_boundary +
                          lambda_source * loss_src +
                          lambda_quiescence * loss_q)
        else: # PHASE 1
            total_loss = lambda_data * loss_boundary
            loss_src, loss_q, points_under_tip = torch.tensor(0.0), torch.tensor(0.0), 0

        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            phase = "PHYSICS" if epoch >= warmup_epochs else "WARM-UP"
            print(f"--- Epoch {epoch+1:5d}/{epochs} [PHASE: {phase}] ---")
            print(f"  TOTAL LOSS: {total_loss.item():.6f}")
            print(f"    - Boundary Loss (Weighted): {(lambda_data * loss_boundary).item():.6f}")
            if epoch >= warmup_epochs:
                print(f"    - Source Loss (Weighted):   {(lambda_source * loss_src).item():.6f}")
                print(f"    - Quiescence Loss (Weighted): {(lambda_quiescence * loss_q).item():.6f}")
                print(f"  [DIAGNOSTIC] Points under quill tip: {points_under_tip}/{n_physics_pts}")
            print("")

    print("✓ Training complete!")

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_reconstruction_v5(model, dataset, output_path='reconstruction_v5.png'):
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points), figsize=(16, 3))
    print("\nGenerating visualization...")
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            x_grid, y_grid = torch.linspace(0,1,dataset.width,device=device), torch.linspace(0,1,dataset.height,device=device)
            X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
            x_flat, y_flat, t_flat = X.flatten(), Y.flatten(), torch.full_like(X.flatten(), t_val)
            h_pred = model(x_flat, y_flat, t_flat).cpu().numpy().reshape(dataset.height, dataset.width)
            axes[i].imshow(1.0 - h_pred, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f't={t_val:.2f}'); axes[i].axis('off')
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.show()
    print(f"✓ Saved final visualization to {output_path}")

if __name__ == "__main__":
    prefix = "synthetic_letters/CANONE_letter_0_C"
    meta_path = f"{prefix}_metadata.json"
    if not Path(f"{prefix}.png").exists() or not Path(meta_path).exists():
        print("✗ Data not found! Please run quill_simulator.py first."); sys.exit(1)

    dataset = QuillDataset_V5(prefix, meta_path)
    model = QuillPINN_V5().to(device)
    train_pinn_v5(model, dataset)
    visualize_reconstruction_v5(model, dataset)
