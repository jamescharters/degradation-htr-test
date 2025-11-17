#!/usr/bin/env python3
"""
Quill PINN V19 (PRESSURE SYNTHESIS)
This script builds directly on the successful V18 baseline. It upgrades the
model to handle variable pressure data by introducing a 'RateNet' that learns
the relationship between pressure and the rate of ink deposition (∂h/∂t).
It also includes extended diagnostic visualizations.
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
print("QUILL PHYSICS PINN V19 (PRESSURE SYNTHESIS)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & GRADIENT UTILS
# ============================================

# --- NEW: RateNet to learn the physics of pressure ---
class RateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus()
        )
    def forward(self, pressure):
        return self.network(pressure)

class QuillPINN_V19(nn.Module):
    def __init__(self, encoding_dim=10):
        super().__init__()
        self.encoding_dim = encoding_dim
        # --- MODIFIED: Input dimension is now 4D (x,y,t,p) ---
        input_dim = 4 * 2 * encoding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.rate_net = RateNet() # Embed the RateNet here
        print(f"✓ Pressure-sensitive PINN created.")

    def positional_encoding(self, coords):
        freqs = 2.0 ** torch.arange(self.encoding_dim, device=coords.device)
        args = coords.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).flatten(start_dim=1)

    # --- MODIFIED: Forward pass now accepts pressure 'p' ---
    def forward(self, x, y, t, p):
        coords = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t.unsqueeze(-1), p.unsqueeze(-1)], dim=-1)
        encoded_coords = self.positional_encoding(coords)
        return torch.relu(self.network(encoded_coords).squeeze(-1))

def compute_gradient(o, i): return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True, allow_unused=True)[0]

# ============================================
# DATASET (Modified to provide pressure)
# ============================================
class QuillDataset_V19:
    def __init__(self, prefix, metadata_path):
        self.image_final = 1.0 - (np.array(Image.open(f"{prefix}.png").convert('L')).astype(np.float32) / 255.0)
        self.height, self.width = self.image_final.shape
        with open(metadata_path, 'r') as f: self.metadata = json.load(f)
        self.total_time = self.metadata['total_time']
        
        stroke = self.metadata['strokes'][0]
        self.stroke_times = np.array(stroke['times']) / self.total_time
        self.stroke_points = np.array(stroke['points']) / np.array([self.width, self.height])
        self.stroke_pressures = np.array(stroke['pressures'])
        
        print(f"✓ Loaded pressure-sensitive data.")

    def get_initial_points(self, n): return torch.rand(n, 2, device=device)
    def get_final_points(self, n):
        x_idx, y_idx = np.random.randint(0,self.width,n), np.random.randint(0,self.height,n)
        x = torch.tensor(x_idx/(self.width-1), dtype=torch.float32, device=device)
        y = torch.tensor(y_idx/(self.height-1), dtype=torch.float32, device=device)
        h = torch.tensor(self.image_final[y_idx, x_idx], dtype=torch.float32, device=device)
        p = torch.full_like(x, self.stroke_pressures.mean(), device=device)
        return x, y, p, h
    def get_off_path_points(self, n): return torch.rand(n, 4, device=device)
    
    def get_on_path_points(self, n):
        rand_indices = np.random.randint(0, len(self.stroke_times), n)
        t = torch.from_numpy(self.stroke_times[rand_indices]).float().to(device)
        xy = torch.from_numpy(self.stroke_points[rand_indices]).float().to(device)
        p = torch.from_numpy(self.stroke_pressures[rand_indices]).float().to(device)
        noise = torch.randn(n, 2, device=device) * (1.5 / self.width)
        x_on_path = torch.clamp(xy[:, 0] + noise[:, 0], 0, 1)
        y_on_path = torch.clamp(xy[:, 1] + noise[:, 1], 0, 1)
        return x_on_path, y_on_path, t, p

    def get_sheath_points(self, n, distance=5.0):
        x_on, y_on, t_on, p_on = self.get_on_path_points(n)
        angle = torch.rand(n, device=device) * 2 * np.pi
        offset_x = torch.cos(angle) * (distance / self.width)
        offset_y = torch.sin(angle) * (distance / self.height)
        x_sheath = torch.clamp(x_on + offset_x, 0, 1)
        y_sheath = torch.clamp(y_on + offset_y, 0, 1)
        return x_sheath, y_sheath, t_on, p_on

# ============================================
# TRAINING (Preserves V18's successful structure)
# ============================================
def train_pinn_v19(model, dataset, epochs=25000, warmup_epochs=5000):
    
    print("\n" + "="*60 + f"\nPHASE 1: BOUNDARY WARM-UP ({warmup_epochs:,} epochs)\n" + "-"*60)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(warmup_epochs):
        model.train(); optimizer.zero_grad()
        x_ic, y_ic = dataset.get_initial_points(2048).T
        p_ic = torch.zeros_like(x_ic)
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic), p_ic) ** 2).mean()
        x_fc, y_fc, p_fc, h_fc = dataset.get_final_points(2048)
        h_pred_fc = model(x_fc, y_fc, torch.ones_like(x_fc), p_fc)
        loss_fc = ((h_pred_fc - h_fc) ** 2).mean()
        total_loss = 200.0 * (loss_ic + loss_fc)
        total_loss.backward(); optimizer.step()
        if (epoch + 1) % 500 == 0: print(f"  Epoch {epoch+1:5d}/{epochs} | Warm-up Loss: {total_loss.item():.6f}")

    print("\n" + "="*60 + f"\nPHASE 2: FULL PHYSICS TRAINING ({epochs - warmup_epochs:,} epochs)\n" + "-"*60)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(warmup_epochs, epochs):
        model.train(); optimizer.zero_grad()
        
        x_ic, y_ic = dataset.get_initial_points(1024).T; p_ic = torch.zeros_like(x_ic)
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic), p_ic) ** 2).mean()
        x_fc, y_fc, p_fc, h_fc = dataset.get_final_points(1024)
        h_pred_fc = model(x_fc, y_fc, torch.ones_like(x_fc), p_fc); loss_fc = ((h_pred_fc - h_fc) ** 2).mean()
        loss_boundary = loss_ic + loss_fc

        n_phys = 1500
        x_src, y_src, t_src, p_src = dataset.get_on_path_points(n_phys)
        x_src.requires_grad_(); y_src.requires_grad_(); t_src.requires_grad_()
        h_src = model(x_src, y_src, t_src, p_src); h_t_src = compute_gradient(h_src, t_src)
        
        # --- MODIFIED: Target rate is now learned by RateNet ---
        target_rate = model.rate_net(p_src.unsqueeze(-1)).squeeze(-1)
        loss_source = (torch.relu(target_rate.detach() - h_t_src)**2).mean()

        x_sh, y_sh, t_sh, p_sh = dataset.get_sheath_points(n_phys)
        x_sh.requires_grad_(); y_sh.requires_grad_(); t_sh.requires_grad_()
        h_sh = model(x_sh, y_sh, t_sh, p_sh); h_t_sh = compute_gradient(h_sh, t_sh)
        loss_sheath = (h_t_sh**2).mean()

        x_q, y_q, t_q, p_q = dataset.get_off_path_points(n_phys).T
        x_q.requires_grad_(); y_q.requires_grad_(); t_q.requires_grad_()
        h_q = model(x_q, y_q, t_q, p_q); h_t_q = compute_gradient(h_q, t_q)
        loss_quiescence = (h_t_q**2).mean()
        
        # --- NEW: Loss term to train the RateNet ---
        # Supervise RateNet by correlating its output with the actual ink height (a proxy for integrated rate)
        rate_pred_for_src = model.rate_net(p_src.unsqueeze(-1)).squeeze(-1)
        loss_rate_net = ((h_src.detach() - rate_pred_for_src)**2).mean()

        total_loss = (150.0*loss_boundary + 50.0*loss_source + 100.0*loss_sheath + 25.0*loss_quiescence + 20.0*loss_rate_net)
        total_loss.backward(); optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1:5d}/{epochs} | Total Loss: {total_loss.item():.4f}")
            print(f"    - Boundary: {loss_boundary.item():.6f} | Source: {loss_source.item():.6f} | RateNet: {loss_rate_net.item():.6f}")

    print("\n✓ Training complete!")
    return model

# ============================================
# NEW DIAGNOSTIC VISUALIZATION
# ============================================
def visualize_diagnostics_v19(model, dataset, output_path='reconstruction_v19_diagnostics.png'):
    model.eval()
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.8])
    print("\nGenerating diagnostic visualization...")
    
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    vis_times = np.array(time_points)
    vis_pressures = np.interp(vis_times, dataset.stroke_times, dataset.stroke_pressures, left=0, right=0)
    x_grid, y_grid = torch.linspace(0,1,dataset.width,device=device), torch.linspace(0,1,dataset.height,device=device)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    x_flat, y_flat = X.flatten(), Y.flatten()

    for i, (t_val, p_val) in enumerate(zip(vis_times, vis_pressures)):
        # Plot Reconstruction h
        ax_h = fig.add_subplot(gs[0, i])
        t_flat = torch.full_like(x_flat, t_val)
        p_flat = torch.full_like(x_flat, p_val)
        with torch.no_grad():
            h_pred = model(x_flat, y_flat, t_flat, p_flat).cpu().numpy().reshape(dataset.height, dataset.width)
        ax_h.imshow(1.0 - h_pred, cmap='gray', vmin=0, vmax=1)
        ax_h.set_title(f'Reconstruction h\nt={t_val:.2f}, p={p_val:.2f}'); ax_h.axis('off')

        # Plot ∂h/∂t (The "Spark")
        ax_ht = fig.add_subplot(gs[1, i])
        t_flat.requires_grad_(True)
        h_pred_grad = model(x_flat, y_flat, t_flat, p_flat)
        h_t_pred = compute_gradient(h_pred_grad, t_flat)
        h_t_img = h_t_pred.detach().cpu().numpy().reshape(dataset.height, dataset.width)
        im_ht = ax_ht.imshow(h_t_img, cmap='RdBu_r', vmin=-2.0, vmax=2.0, origin='lower')
        ax_ht.set_title(f'∂h/∂t (The Spark)\nt={t_val:.2f}'); ax_ht.axis('off')

    # Plot RateNet Curve
    ax_rate = fig.add_subplot(gs[2, :])
    with torch.no_grad():
        pressures_to_test = torch.linspace(0, 1, 100, device=device).unsqueeze(-1)
        learned_rates = model.rate_net(pressures_to_test).cpu().numpy()
        ax_rate.plot(pressures_to_test.cpu().numpy(), learned_rates, label='Learned RateNet(pressure)', color='crimson', linewidth=3)
        ax_rate.set_title("The Learned Physics: Pressure vs. Ink Deposition Rate", fontsize=16)
        ax_rate.set_xlabel("Input Pressure (Normalized)", fontsize=12); ax_rate.set_ylabel("Learned Target Rate for ∂h/∂t", fontsize=12)
        ax_rate.grid(True, linestyle=':'); ax_rate.legend(); ax_rate.set_xlim(0, 1); ax_rate.set_ylim(bottom=0)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=200); plt.show()
    print(f"✓ Saved diagnostic visualization to {output_path}")

if __name__ == "__main__":
    prefix = "synthetic_letters_v2/CANONE_letter_0_C"
    meta_path = f"{prefix}_metadata.json"
    if not Path(f"{prefix}.png").exists() or not Path(meta_path).exists():
        print("✗ Data not found! Please run `quill_simulator_v2.py` first."); sys.exit(1)
        
    dataset = QuillDataset_V19(prefix, meta_path)
    model = QuillPINN_V19().to(device)
    train_pinn_v19(model, dataset)
    visualize_diagnostics_v19(model, dataset)
