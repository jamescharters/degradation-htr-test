#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction V4
FINAL VERSION: Combines the robust anchor-driven training strategy with the
advanced V3 model and physics.
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
print("QUILL PHYSICS PINN TRAINING V4 (ANCHOR-DRIVEN)")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# MODEL & PHYSICS (Identical to V3)
# ============================================
class QuillPINN_V4(nn.Module):
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
    def encode_time(self, t): # Sinusoidal time encoding
        freqs = 2.0 ** torch.arange(self.time_encoding_dim//2, device=t.device, dtype=torch.float32)
        t_expanded = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(2*np.pi*t_expanded), torch.cos(2*np.pi*t_expanded)], dim=-1)
    def forward(self, x, y, t):
        t_encoded = self.encode_time(t)
        inputs = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), t_encoded], dim=-1)
        return torch.relu(self.network(inputs).squeeze(-1))

def compute_gradient(output, input_tensor):
    return torch.autograd.grad(output, input_tensor, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True)[0]

def physics_loss_v4(model, x, y, t): # The full, multi-component physics
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    h = model(x, y, t)
    h_t = compute_gradient(h, t)
    loss_monotonic = (torch.relu(-h_t) ** 2).mean()
    h_x = compute_gradient(h, x)
    h_y = compute_gradient(h, y)
    h_xx = compute_gradient(h_x, x)
    h_yy = compute_gradient(h_y, y)
    loss_smooth = ((h_xx + h_yy) ** 2).mean()
    return loss_monotonic, loss_smooth

# ============================================
# DATASET (UPDATED to load anchor frames)
# ============================================
class QuillDataset_V4:
    def __init__(self, prefix):
        def load_image_as_numpy(path):
            img = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
            return 1.0 - img # Invert so ink is 1, paper is 0

        self.image_final = load_image_as_numpy(f"{prefix}.png")
        self.image_t025 = load_image_as_numpy(f"{prefix}_t0.25.png")
        self.image_t050 = load_image_as_numpy(f"{prefix}_t0.50.png")
        self.image_t075 = load_image_as_numpy(f"{prefix}_t0.75.png")
        
        self.height, self.width = self.image_final.shape
        print(f"✓ Loaded final and anchor frames: {self.width}x{self.height}")

    def _sample(self, image_data, n_points):
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        x = x_idx / (self.width - 1)
        y = y_idx / (self.height - 1)
        h_true = image_data[y_idx, x_idx]
        return x, y, h_true

    def get_initial_points(self, n_points):
        return np.random.rand(n_points), np.random.rand(n_points)
    
    def get_final_points(self, n_points):
        return self._sample(self.image_final, n_points)
        
    def get_anchor_points(self, t_value, n_points):
        if t_value == 0.25: return self._sample(self.image_t025, n_points)
        if t_value == 0.50: return self._sample(self.image_t050, n_points)
        if t_value == 0.75: return self._sample(self.image_t075, n_points)
        raise ValueError("Invalid anchor time")

    def get_physics_points(self, n_points):
        return np.random.rand(n_points), np.random.rand(n_points), np.random.rand(n_points)

# ============================================
# TRAINING (UPDATED with anchor losses)
# ============================================
def train_pinn_v4(model, dataset, epochs=10000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # LOSS WEIGHTS - The key to our success
    lambda_data = 150.0      # Final and initial conditions are critical
    lambda_anchors = 150.0   # The intermediate anchors are NON-NEGOTIABLE
    lambda_monotonic = 1.0   # A guiding physics rule
    lambda_smooth = 0.01     # A gentle regularizer to prevent spikiness

    print("\nTraining with V4 Anchor-Driven Strategy...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Losses (Initial, Final, and Anchors) ---
        n_data = 500 # Points per condition
        # Initial (t=0)
        x_ic, y_ic = dataset.get_initial_points(n_data)
        x_ic_t = torch.tensor(x_ic, dtype=torch.float32, device=device)
        y_ic_t = torch.tensor(y_ic, dtype=torch.float32, device=device)
        loss_ic = (model(x_ic_t, y_ic_t, torch.zeros_like(x_ic_t)) ** 2).mean()

        # Final (t=1)
        x_fc, y_fc, h_fc = dataset.get_final_points(n_data)
        h_pred_fc = model(torch.tensor(x_fc, dtype=torch.float32, device=device),
                          torch.tensor(y_fc, dtype=torch.float32, device=device),
                          torch.ones(n_data, device=device))
        loss_fc = ((h_pred_fc - torch.tensor(h_fc, dtype=torch.float32, device=device)) ** 2).mean()

        # Anchors (t=0.25, 0.5, 0.75)
        loss_anchor_total = 0
        for t_anchor in [0.25, 0.50, 0.75]:
            x_ac, y_ac, h_ac = dataset.get_anchor_points(t_anchor, n_data)
            h_pred_ac = model(torch.tensor(x_ac, dtype=torch.float32, device=device),
                              torch.tensor(y_ac, dtype=torch.float32, device=device),
                              torch.full((n_data,), t_anchor, device=device))
            loss_anchor_total += ((h_pred_ac - torch.tensor(h_ac, dtype=torch.float32, device=device)) ** 2).mean()
            
        # --- Physics Loss ---
        x_p, y_p, t_p = dataset.get_physics_points(2000)
        x_p_t = torch.tensor(x_p, dtype=torch.float32, device=device)
        y_p_t = torch.tensor(y_p, dtype=torch.float32, device=device)
        t_p_t = torch.tensor(t_p, dtype=torch.float32, device=device)
        loss_mono, loss_sm = physics_loss_v4(model, x_p_t, y_p_t, t_p_t)
        
        # --- Total Loss ---
        loss = (lambda_data * (loss_ic + loss_fc) +
                lambda_anchors * loss_anchor_total +
                lambda_monotonic * loss_mono +
                lambda_smooth * loss_sm)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | "
                  f"Anchor Loss: {loss_anchor_total.item():.6f}")

    print("✓ Training complete!")

# ============================================
# VISUALIZATION (Identical to V3)
# ============================================
def visualize_reconstruction_v4(model, dataset, output_path='reconstruction_v4.png'):
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points), figsize=(16, 3))
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            x_grid = np.linspace(0, 1, dataset.width)
            y_grid = np.linspace(0, 1, dataset.height)
            X, Y = np.meshgrid(x_grid, y_grid)
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
            y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
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
    # Ensure you have run the simulator to generate the anchor frames first!
    letter_prefix = "synthetic_letters/CANONE_letter_0_C"
    if not Path(f"{letter_prefix}_t0.25.png").exists():
        print(f"✗ Anchor frame '{letter_prefix}_t0.25.png' not found!")
        print("  Please modify and run quill_simulator.py to generate it first.")
        sys.exit(1)

    print("\n[1] Loading data with anchor frames...")
    dataset = QuillDataset_V4(letter_prefix)
    
    print("\n[2] Creating V4 model...")
    model = QuillPINN_V4().to(device)
    
    print("\n[3] Starting V4 training...")
    train_pinn_v4(model, dataset)
    
    print("\n[4] Generating final visualization...")
    visualize_reconstruction_v4(model, dataset)
