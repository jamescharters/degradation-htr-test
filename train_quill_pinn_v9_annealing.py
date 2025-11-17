#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction V9 (ANNEALING)
This version solves the deep local minimum problem by annealing the target rate
of the source loss. This gradually coaxes the network to produce a non-zero
temporal derivative, representing the definitive and most robust training strategy.
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
print("QUILL PHYSICS PINN TRAINING V9 (TARGET ANNEALING)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & GRADIENT UTILS
# ============================================
class QuillPINN_V9(nn.Module):
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
class QuillDataset_V9:
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
    def get_off_path_points(self, n): return np.random.rand(n, 3)
    def get_on_path_points(self, n):
        stroke_start_norm = self.metadata['strokes'][0]['start_time'] / self.total_time
        stroke_end_norm = self.metadata['strokes'][-1]['end_time'] / self.total_time
        t_on_path = np.random.uniform(stroke_start_norm, stroke_end_norm, n)
        quill_pos = self.get_quill_position_at_time(torch.from_numpy(t_on_path)).numpy()
        noise = np.random.randn(n, 2) * (2.0 / self.width)
        x_on_path = quill_pos[:, 0] + noise[:, 0]
        y_on_path = quill_pos[:, 1] + noise[:, 1]
        return np.clip(x_on_path,0,1), np.clip(y_on_path,0,1), t_on_path

# ============================================
# PHYSICS LOSS (with dynamic target rate)
# ============================================
def physics_loss_v9(model, x_src, y_src, t_src, x_q, y_q, t_q, current_target_rate):
    # Source Loss
    h_src = model(x_src, y_src, t_src)
    h_t_src = compute_gradient(h_src, t_src)
    loss_source = (torch.relu(current_target_rate - h_t_src)**2).mean()

    # Quiescence Loss
    h_q = model(x_q, y_q, t_q)
    h_t_q = compute_gradient(h_q, t_q)
    loss_quiescence = (h_t_q**2).mean()

    return loss_source, loss_quiescence

# ============================================
# TRAINING
# ============================================
def train_pinn_v9(model, dataset, epochs=15000, lr_warmup=1e-3, lr_finetune=3e-4, warmup_epochs=4000):
    
    lambda_data = 150.0; lambda_source = 50.0; lambda_quiescence = 50.0
    final_target_rate = 1.5; start_target_rate = 0.1

    print("\n" + "="*60 + f"\nTRAINING STRATEGY: V9\n" + "-"*60)
    print(f"  Phase 1: Data Warm-up ({warmup_epochs:,} epochs, LR={lr_warmup})")
    print(f"  Phase 2: Physics Fine-tuning with Annealing ({epochs-warmup_epochs:,} epochs, LR={lr_finetune})\n" + "="*60 + "\n")
    
    # --- PHASE 1: WARM-UP ---
    print("--- Starting Phase 1: WARM-UP ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_warmup)
    for epoch in range(warmup_epochs):
        model.train(); optimizer.zero_grad()
        n_data = 800
        ic_pts = dataset.get_initial_points(n_data)
        x_ic, y_ic = torch.tensor(ic_pts[:,0],dtype=torch.float32,device=device), torch.tensor(ic_pts[:,1],dtype=torch.float32,device=device)
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic)) ** 2).mean()
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        x_fc_t, y_fc_t, h_fc_t = torch.tensor(x_fc,dtype=torch.float32,device=device), torch.tensor(y_fc,dtype=torch.float32,device=device), torch.tensor(h_fc,dtype=torch.float32,device=device)
        h_pred_fc = model(x_fc_t, y_fc_t, torch.ones_like(x_fc_t))
        loss_fc = ((h_pred_fc - h_fc_t) ** 2).mean()
        total_loss = lambda_data * (loss_ic + loss_fc)
        total_loss.backward(); optimizer.step()
        if (epoch + 1) % 500 == 0: print(f"  Epoch {epoch+1:5d}/{warmup_epochs} | Warm-up Loss: {total_loss.item():.6f}")

    # --- PHASE 2: FINE-TUNING with ANNEALING ---
    print("\n--- Starting Phase 2: PHYSICS FINE-TUNING ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_finetune) # Optimizer Reset
    finetune_epochs = epochs - warmup_epochs

    for epoch in range(finetune_epochs):
        model.train(); optimizer.zero_grad()
        
        # Anneal the target rate
        anneal_progress = epoch / finetune_epochs
        current_target_rate = start_target_rate + (final_target_rate - start_target_rate) * anneal_progress
        
        # Boundary losses
        n_data = 800
        ic_pts = dataset.get_initial_points(n_data); x_ic, y_ic = torch.tensor(ic_pts[:,0],dtype=torch.float32,device=device), torch.tensor(ic_pts[:,1],dtype=torch.float32,device=device)
        loss_ic = (model(x_ic, y_ic, torch.zeros_like(x_ic)) ** 2).mean()
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data); x_fc_t, y_fc_t, h_fc_t = torch.tensor(x_fc,dtype=torch.float32,device=device), torch.tensor(y_fc,dtype=torch.float32,device=device), torch.tensor(h_fc,dtype=torch.float32,device=device)
        h_pred_fc = model(x_fc_t, y_fc_t, torch.ones_like(x_fc_t)); loss_fc = ((h_pred_fc - h_fc_t) ** 2).mean()
        loss_boundary = loss_ic + loss_fc
        
        # Physics losses with importance sampling
        n_source_pts, n_quiescence_pts = 1500, 1500
        x_src, y_src, t_src = dataset.get_on_path_points(n_source_pts); x_src_t, y_src_t, t_src_t = torch.tensor(x_src,dtype=torch.float32,device=device,requires_grad=True), torch.tensor(y_src,dtype=torch.float32,device=device,requires_grad=True), torch.tensor(t_src,dtype=torch.float32,device=device,requires_grad=True)
        off_path_pts = dataset.get_off_path_points(n_quiescence_pts); x_q, y_q, t_q = torch.tensor(off_path_pts[:,0],dtype=torch.float32,device=device,requires_grad=True), torch.tensor(off_path_pts[:,1],dtype=torch.float32,device=device,requires_grad=True), torch.tensor(off_path_pts[:,2],dtype=torch.float32,device=device,requires_grad=True)
        
        loss_source, loss_quiescence = physics_loss_v9(model, x_src_t, y_src_t, t_src_t, x_q, y_q, t_q, current_target_rate)

        total_loss = (lambda_data * loss_boundary +
                      lambda_source * loss_source +
                      lambda_quiescence * loss_quiescence)

        total_loss.backward(); optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+warmup_epochs+1:5d}/{epochs} | Fine-tune Loss: {total_loss.item():.6f}")
            print(f"    - Target Rate: {current_target_rate:.3f} | Raw Source Loss: {loss_source.item():.6f}")

    print("\n✓ Training complete!")

# ============================================
# VISUALIZATION & MAIN
# ============================================
def visualize_reconstruction_v9(model, dataset, output_path='reconstruction_v9.png'):
    model.eval(); time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
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
    dataset = QuillDataset_V9(prefix, meta_path)
    model = QuillPINN_V9().to(device)
    train_pinn_v9(model, dataset)
    visualize_reconstruction_v9(model, dataset)
