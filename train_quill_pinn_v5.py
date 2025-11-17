#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction V5
"Source Term" PINN: A return to a physics-driven approach with a smarter,
localized physics loss, removing the need for intermediate anchor frames.
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
print("QUILL PHYSICS PINN TRAINING V5 (SOURCE TERM)")
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
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        print(f"✓ Model created (Params: {sum(p.numel() for p in self.parameters()):,})")

    def encode_time(self, t):
        freqs = 2.0 ** torch.arange(self.time_encoding_dim//2, device=t.device, dtype=torch.float32)
        t_expanded = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(2*np.pi*t_expanded), torch.cos(2*np.pi*t_expanded)], dim=-1)

    def forward(self, x, y, t):
        # Handle single value inputs during visualization for convenience
        if x.dim() == 0: x = x.unsqueeze(0)
        if y.dim() == 0: y = y.unsqueeze(0)
        if t.dim() == 0: t = t.unsqueeze(0)
        
        t_encoded = self.encode_time(t)
        inputs = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t_encoded], dim=-1)
        return torch.relu(self.network(inputs).squeeze(-1))

def compute_gradient(output, input_tensor):
    grad = torch.autograd.grad(
        output, input_tensor, 
        grad_outputs=torch.ones_like(output), 
        create_graph=True, retain_graph=True, allow_unused=True
    )[0]
    return grad

# ============================================
# DATASET (with corrected path interpolation)
# ============================================
class QuillDataset_V5:
    def __init__(self, prefix, metadata_path):
        self.image_final = np.array(Image.open(f"{prefix}.png").convert('L')).astype(np.float32) / 255.0
        self.image_final = 1.0 - self.image_final
        
        self.height, self.width = self.image_final.shape
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.total_time = self.metadata['total_time']
        
        print(f"✓ Loaded final frame ({self.width}x{self.height}) and metadata.")
        print(f"  Total simulation time: {self.total_time:.2f}s")

    def get_quill_position_at_time(self, t_values):
        # --- CORRECTED FUNCTION ---
        # We .detach() the tensor because this numpy-based lookup is not part of the
        # gradient computation. The original tensor remains on the computation graph.
        t_values_np = t_values.detach().cpu().numpy()
        
        positions = []
        
        for t_norm in t_values_np:
            t_sim = t_norm * self.total_time
            active_stroke = None
            for stroke in self.metadata['strokes']:
                if stroke['start_time'] <= t_sim < stroke['end_time']:
                    active_stroke = stroke
                    break
            
            if active_stroke:
                progress = (t_sim - active_stroke['start_time']) / (active_stroke['end_time'] - active_stroke['start_time'])
                points_arr = np.array(active_stroke['points'])
                idx_float = progress * (len(points_arr) - 1)
                idx0 = int(np.floor(idx_float))
                idx1 = int(np.ceil(idx_float))

                # Robustness check to prevent index out of bounds
                if idx0 >= len(points_arr): idx0 = len(points_arr) - 1
                if idx1 >= len(points_arr): idx1 = len(points_arr) - 1

                if idx0 == idx1:
                    pos = points_arr[idx0]
                else:
                    local_progress = idx_float - idx0
                    pos = points_arr[idx0] * (1 - local_progress) + points_arr[idx1] * local_progress
                
                positions.append((pos[0] / self.width, pos[1] / self.height))
            else:
                positions.append((np.nan, np.nan)) # Quill is off the page
                
        return torch.tensor(positions, dtype=torch.float32, device=t_values.device)

    def get_initial_points(self, n_points): return np.random.rand(n_points, 2)
    def get_final_points(self, n_points):
        x_idx, y_idx = np.random.randint(0, self.width, n_points), np.random.randint(0, self.height, n_points)
        return x_idx / (self.width-1), y_idx / (self.height-1), self.image_final[y_idx, x_idx]
    def get_physics_points(self, n_points): return np.random.rand(n_points, 3)

# ============================================
# THE "SOURCE TERM" PHYSICS LOSS
# ============================================
def physics_loss_v5_sourceterm(model, dataset, x, y, t):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    h = model(x, y, t)
    
    h_t = compute_gradient(h, t)
    if h_t is None: return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    quill_pos = dataset.get_quill_position_at_time(t)
    quill_x, quill_y = quill_pos[:, 0].unsqueeze(1), quill_pos[:, 1].unsqueeze(1)
    
    dist_sq = (x - quill_x)**2 + (y - quill_y)**2
    tip_radius_sq = (5.0 / dataset.width)**2
    is_under_tip = (dist_sq < tip_radius_sq) & (~torch.isnan(dist_sq))
    
    # Rule 1: Source Loss - Where the quill IS, h_t must be positive.
    target_ink_rate = 2.0
    ink_rate_under_tip = h_t[is_under_tip]
    loss_source = (torch.relu(target_ink_rate - ink_rate_under_tip)**2).mean()
    
    # Rule 2: Quiescence Loss - Where the quill IS NOT, h_t must be zero.
    ink_rate_elsewhere = h_t[~is_under_tip]
    loss_quiescence = (ink_rate_elsewhere**2).mean()
    
    if torch.isnan(loss_source): loss_source = torch.tensor(0.0, device=device)
    if torch.isnan(loss_quiescence): loss_quiescence = torch.tensor(0.0, device=device)

    return loss_source, loss_quiescence

# ============================================
# TRAINING (Driven by the new Physics Loss)
# ============================================
def train_pinn_v5(model, dataset, epochs=15000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    lambda_data = 150.0
    lambda_source = 20.0
    lambda_quiescence = 20.0

    print("\nTraining with V5 Source Term Strategy...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Boundary Condition Losses ---
        n_data = 800
        ic_pts = dataset.get_initial_points(n_data)
        x_ic, y_ic = torch.tensor(ic_pts[:,0], dtype=torch.float32, device=device), torch.tensor(ic_pts[:,1], dtype=torch.float32, device=device)
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic)) ** 2).mean()

        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        x_fc_t, y_fc_t, h_fc_t = torch.tensor(x_fc, dtype=torch.float32, device=device), torch.tensor(y_fc, dtype=torch.float32, device=device), torch.tensor(h_fc, dtype=torch.float32, device=device)
        h_pred_fc = model(x_fc_t, y_fc_t, torch.ones_like(x_fc_t))
        loss_fc = ((h_pred_fc - h_fc_t) ** 2).mean()

        # --- Physics Loss ---
        phys_pts = dataset.get_physics_points(2500)
        x_p, y_p, t_p = torch.tensor(phys_pts[:,0], dtype=torch.float32, device=device), torch.tensor(phys_pts[:,1], dtype=torch.float32, device=device), torch.tensor(phys_pts[:,2], dtype=torch.float32, device=device)
        loss_src, loss_q = physics_loss_v5_sourceterm(model, dataset, x_p, y_p, t_p)
        
        # --- Total Loss ---
        loss = (lambda_data * (loss_ic + loss_fc) +
                lambda_source * loss_src +
                lambda_quiescence * loss_q)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Source: {loss_src.item():.6f} | Quiescence: {loss_q.item():.6f}")

    print("✓ Training complete!")

# ============================================
# VISUALIZATION
# ============================================
def visualize_reconstruction_v5(model, dataset, output_path='reconstruction_v5.png'):
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points), figsize=(16, 3))
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            x_grid = torch.linspace(0, 1, dataset.width, device=device)
            y_grid = torch.linspace(0, 1, dataset.height, device=device)
            X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
            x_flat, y_flat = X.flatten(), Y.flatten()
            t_flat = torch.full_like(x_flat, t_val)
            h_pred = model(x_flat, y_flat, t_flat).cpu().numpy().reshape(dataset.height, dataset.width)
            axes[i].imshow(1.0 - h_pred, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f't={t_val:.2f}'); axes[i].axis('off')
    plt.tight_layout(); plt.savefig(output_path, dpi=200); plt.show()
    print(f"✓ Saved final visualization to {output_path}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    letter_prefix = "synthetic_letters/CANONE_letter_0_C"
    metadata_path = f"{letter_prefix}_metadata.json"
    if not Path(f"{letter_prefix}.png").exists() or not Path(metadata_path).exists():
        print(f"✗ Data not found! Please run quill_simulator.py first.")
        sys.exit(1)

    print("\n[1] Loading data and metadata for path interpolation...")
    dataset = QuillDataset_V5(letter_prefix, metadata_path)
    
    print("\n[2] Creating V5 model...")
    model = QuillPINN_V5().to(device)
    
    print("\n[3] Starting V5 training...")
    train_pinn_v5(model, dataset)
    
    print("\n[4] Generating final visualization...")
    visualize_reconstruction_v5(model, dataset)
