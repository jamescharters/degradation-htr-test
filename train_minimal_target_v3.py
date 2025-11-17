#!/usr/bin/env python3
"""
A minimal, boiled-down PINN - V3: Guiding the Process.
Goal: Make the intermediate shapes more square-like by adding a smoothness penalty.

Changes from V2:
1. Re-introduced the Spatial Smoothness (Laplacian) physics loss.
2. Tuned the new `lambda_smooth` to be small, acting as a guide rather than a dictator.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ============================================
# SETUP
# ============================================
print("=" * 40)
print("MINIMAL PINN EXAMPLE - V3 (GUIDED)")
print("=" * 40)
device = torch.device("cpu")

# ============================================
# 1. THE MODEL (Identical to V2)
# ============================================
def create_model():
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1), nn.Sigmoid()
    ).to(device)

# ============================================
# 2. THE DATA (Identical to V2)
# ============================================
def get_target_image(width=100, height=100):
    image = np.zeros((height, width), dtype=np.float32)
    x_start, x_end = int(width * 0.3), int(width * 0.7)
    y_start, y_end = int(height * 0.3), int(height * 0.7)
    image[y_start:y_end, x_start:x_end] = 1.0
    return gaussian_filter(image, sigma=3)

# ============================================
# 3. THE PHYSICS LOSS (Now with Smoothness)
# ============================================
def physics_loss_guided(model, x, y, t):
    """Enforces ∂h/∂t >= 0 AND penalizes spatial roughness."""
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    inputs = torch.cat([x, y, t], dim=1)
    h = model(inputs)
    
    # Monotonicity Loss
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    loss_monotonic = (torch.relu(-h_t) ** 2).mean()
    
    # Smoothness Loss (Laplacian)
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x), create_graph=True)[0]
    h_yy = torch.autograd.grad(h_y, y, grad_outputs=torch.ones_like(h_y), create_graph=True)[0]
    laplacian = h_xx + h_yy
    loss_smooth = (laplacian ** 2).mean()
    
    return loss_monotonic, loss_smooth

# ============================================
# 4. THE TRAINING LOOP (updated with new loss term)
# ============================================
def train(model, target_image, epochs=5000, lr=1e-3):
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights - The crucial tuning step!
    lambda_data = 100.0   # Must be high to ensure final accuracy
    lambda_monotonic = 1.0   # The core physics rule
    lambda_smooth = 0.001 # NEW: A small penalty to guide, not dominate

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Loss (same as V2) ---
        n_data_pts = 500
        x_idx = np.random.randint(0, target_image.shape[1], n_data_pts)
        y_idx = np.random.randint(0, target_image.shape[0], n_data_pts)
        x_data = torch.tensor(x_idx / (target_image.shape[1] - 1), dtype=torch.float32).view(-1, 1).to(device)
        y_data = torch.tensor(y_idx / (target_image.shape[0] - 1), dtype=torch.float32).view(-1, 1).to(device)
        
        t_initial = torch.zeros_like(x_data)
        inputs_initial = torch.cat([x_data, y_data, t_initial], dim=1)
        h_pred_initial = model(inputs_initial)
        loss_initial = (h_pred_initial ** 2).mean()

        t_final = torch.ones_like(x_data)
        h_true_final = torch.tensor(target_image[y_idx, x_idx], dtype=torch.float32).view(-1, 1).to(device)
        inputs_final = torch.cat([x_data, y_data, t_final], dim=1)
        h_pred_final = model(inputs_final)
        loss_final = ((h_pred_final - h_true_final) ** 2).mean()
        loss_data = loss_initial + loss_final

        # --- Physics Loss (now has two components) ---
        n_phys_pts = 1000
        x_phys = torch.rand(n_phys_pts, 1, device=device)
        y_phys = torch.rand(n_phys_pts, 1, device=device)
        t_phys = torch.rand(n_phys_pts, 1, device=device)
        loss_mono, loss_smooth = physics_loss_guided(model, x_phys, y_phys, t_phys)
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data +
                      lambda_monotonic * loss_mono +
                      lambda_smooth * loss_smooth)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} | "
                  f"Data: {loss_data.item():.6f} | Smooth: {loss_smooth.item():.6f}")
            
    print("✓ Training complete!")
    return model

# ============================================
# 5. VISUALIZATION (Identical to V2)
# ============================================
def visualize(model, target_image):
    print("Generating visualization...")
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points) + 1, figsize=(18, 3))
    axes[0].imshow(target_image, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    res = 100
    x_grid, y_grid = torch.linspace(0, 1, res), torch.linspace(0, 1, res)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    x_flat, y_flat = X.flatten().view(-1, 1).to(device), Y.flatten().view(-1, 1).to(device)
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            t_flat = torch.full_like(x_flat, t_val)
            inputs = torch.cat([x_flat, y_flat, t_flat], dim=1)
            h_pred = model(inputs).cpu().numpy().reshape(res, res)
            ax = axes[i + 1]
            ax.imshow(h_pred, cmap='gray', origin='lower', vmin=0, vmax=1)
            ax.set_title(f't = {t_val:.2f}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig("minimal_target_result_v3.png")
    print("✓ Saved plot to minimal_target_result_v3.png")
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    model = create_model()
    target = get_target_image()
    trained_model = train(model, target)
    visualize(trained_model, target)
