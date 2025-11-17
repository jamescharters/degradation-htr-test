#!/usr/bin/env python3
"""
A minimal, boiled-down PINN - V5: Non-Negotiable Anchor.
Goal: Force the network to be accurate at the mid-point by making the penalty
      for failure non-negotiable.

Changes from V4:
1. Increased `lambda_intermediate` to be equal to `lambda_data`. This makes
   the mid-point rule as important as the final rule.
2. Increased epochs to give the model more time to solve this harder problem.
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
print("MINIMAL PINN EXAMPLE - V5 (NON-NEGOTIABLE)")
print("=" * 40)
device = torch.device("cpu")

# ============================================
# 1. THE MODEL (Identical to V2/V3/V4)
# ============================================
def create_model():
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1), nn.Sigmoid()
    ).to(device)

# ============================================
# 2. THE DATA (Identical)
# ============================================
def get_target_image(width=100, height=100):
    image = np.zeros((height, width), dtype=np.float32)
    x_start, x_end = int(width * 0.3), int(width * 0.7)
    y_start, y_end = int(height * 0.3), int(height * 0.7)
    image[y_start:y_end, x_start:x_end] = 1.0
    return gaussian_filter(image, sigma=3)

# ============================================
# 3. THE PHYSICS LOSS (Identical)
# ============================================
def physics_loss_monotonic(model, x, y, t):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    inputs = torch.cat([x, y, t], dim=1)
    h = model(inputs)
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    return (torch.relu(-h_t) ** 2).mean()

# ============================================
# 4. THE TRAINING LOOP (The CRITICAL update)
# ============================================
def train(model, target_image, epochs=8000, lr=1e-3): # CHANGED: More epochs
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights
    lambda_data = 100.0
    lambda_physics = 1.0
    lambda_intermediate = 100.0 # CHANGED: Make the mid-point NON-NEGOTIABLE

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Losses ---
        n_data_pts = 400
        x_idx = np.random.randint(0, target_image.shape[1], n_data_pts)
        y_idx = np.random.randint(0, target_image.shape[0], n_data_pts)
        x_data = torch.tensor(x_idx / (target_image.shape[1] - 1), dtype=torch.float32).view(-1, 1).to(device)
        y_data = torch.tensor(y_idx / (target_image.shape[0] - 1), dtype=torch.float32).view(-1, 1).to(device)
        h_true_final = torch.tensor(target_image[y_idx, x_idx], dtype=torch.float32).view(-1, 1).to(device)
        
        # Initial condition (t=0)
        t_initial = torch.zeros_like(x_data)
        h_pred_initial = model(torch.cat([x_data, y_data, t_initial], dim=1))
        loss_initial = (h_pred_initial ** 2).mean()

        # Final condition (t=1)
        t_final = torch.ones_like(x_data)
        h_pred_final = model(torch.cat([x_data, y_data, t_final], dim=1))
        loss_final = ((h_pred_final - h_true_final) ** 2).mean()
        
        # Intermediate anchor (t=0.5)
        t_intermediate = torch.full_like(x_data, 0.5)
        h_true_intermediate = h_true_final * 0.5
        h_pred_intermediate = model(torch.cat([x_data, y_data, t_intermediate], dim=1))
        loss_intermediate = ((h_pred_intermediate - h_true_intermediate) ** 2).mean()

        loss_data = loss_initial + loss_final
        
        # --- Physics Loss ---
        n_phys_pts = 1000
        x_phys, y_phys, t_phys = torch.rand(n_phys_pts, 1), torch.rand(n_phys_pts, 1), torch.rand(n_phys_pts, 1)
        loss_phys = physics_loss_monotonic(model, x_phys.to(device), y_phys.to(device), t_phys.to(device))
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data +
                      lambda_physics * loss_phys +
                      lambda_intermediate * loss_intermediate)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} | "
                  f"Mid-point Loss: {loss_intermediate.item():.6f}")
            
    print("✓ Training complete!")
    return model

# ============================================
# 5. VISUALIZATION (Identical)
# ============================================
def visualize(model, target_image):
    print("Generating visualization...")
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points) + 1, figsize=(18, 3))
    axes[0].imshow(target_image, cmap='gray', origin='lower', vmin=0, vmax=1); axes[0].set_title('Ground Truth'); axes[0].axis('off')
    res = 100
    x_grid, y_grid = torch.linspace(0, 1, res), torch.linspace(0, 1, res)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    x_flat, y_flat = X.flatten().view(-1, 1).to(device), Y.flatten().view(-1, 1).to(device)
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            t_flat = torch.full_like(x_flat, t_val)
            h_pred = model(torch.cat([x_flat, y_flat, t_flat], dim=1)).cpu().numpy().reshape(res, res)
            ax = axes[i + 1]
            ax.imshow(h_pred, cmap='gray', origin='lower', vmin=0, vmax=1); ax.set_title(f't = {t_val:.2f}'); ax.axis('off')
    plt.tight_layout(); plt.savefig("minimal_target_result_v5.png"); plt.show()

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    model = create_model()
    target = get_target_image()
    trained_model = train(model, target)
    visualize(trained_model, target)
