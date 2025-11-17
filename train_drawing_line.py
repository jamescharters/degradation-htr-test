#!/usr/bin/env python3
"""
PINN - V6: Drawing a Simple Stroke.
Goal: Teach the network a localized, moving process by drawing a line.

Changes from V5:
1. The target is now a horizontal line, not a square.
2. We introduce a function `get_line_image(progress)` to create partially drawn lines.
3. The anchor point loss now uses these partial lines as targets, forcing the
   network to learn the drawing process instead of a simple fade-in.
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
print("PINN EXAMPLE - V6 (DRAWING A LINE)")
print("=" * 40)
device = torch.device("cpu")

# ============================================
# 1. THE MODEL (Identical to V5)
# ============================================
def create_model():
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1), nn.Sigmoid()
    ).to(device)

# ============================================
# 2. THE DATA (NEW: A dynamic line generator)
# ============================================
def get_line_image(width=100, height=100, progress=1.0):
    """
    Returns a numpy array of a soft-edged horizontal line.
    'progress' determines how much of the line is drawn (from 0.0 to 1.0).
    """
    image = np.zeros((height, width), dtype=np.float32)
    
    # Line properties
    y_center = height // 2
    thickness = 4
    x_start, x_end = int(width * 0.2), int(width * 0.8)
    
    # Calculate the current end-point based on progress
    current_x_end = x_start + int((x_end - x_start) * progress)
    
    # Draw the line segment
    y_min, y_max = y_center - thickness // 2, y_center + thickness // 2
    if current_x_end > x_start: # Only draw if progress > 0
        image[y_min:y_max, x_start:current_x_end] = 1.0
        
    return gaussian_filter(image, sigma=1.5)

# ============================================
# 3. THE PHYSICS LOSS (Identical to V5)
# ============================================
def physics_loss_monotonic(model, x, y, t):
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    inputs = torch.cat([x, y, t], dim=1)
    h = model(inputs)
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    return (torch.relu(-h_t) ** 2).mean()

# ============================================
# 4. THE TRAINING LOOP (Using the dynamic anchor)
# ============================================
def train(model, epochs=8000, lr=1e-3):
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights - anchors are everything
    lambda_data = 100.0
    lambda_physics = 0.1 # Physics is less important now, anchors guide the process
    lambda_anchor1 = 100.0 # Anchor at t=0.33
    lambda_anchor2 = 100.0 # Anchor at t=0.66

    # Pre-generate target images
    final_image = get_line_image(progress=1.0)
    anchor1_image = get_line_image(progress=0.33)
    anchor2_image = get_line_image(progress=0.66)

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Losses ---
        n_data_pts = 300
        x_idx = np.random.randint(0, final_image.shape[1], n_data_pts)
        y_idx = np.random.randint(0, final_image.shape[0], n_data_pts)
        x_data = torch.tensor(x_idx / (final_image.shape[1] - 1), dtype=torch.float32).view(-1, 1).to(device)
        y_data = torch.tensor(y_idx / (final_image.shape[0] - 1), dtype=torch.float32).view(-1, 1).to(device)
        
        # Initial condition (t=0)
        t_initial = torch.zeros_like(x_data)
        h_pred_initial = model(torch.cat([x_data, y_data, t_initial], dim=1))
        loss_initial = (h_pred_initial ** 2).mean()

        # Final condition (t=1)
        t_final = torch.ones_like(x_data)
        h_true_final = torch.tensor(final_image[y_idx, x_idx], dtype=torch.float32).view(-1, 1).to(device)
        h_pred_final = model(torch.cat([x_data, y_data, t_final], dim=1))
        loss_final = ((h_pred_final - h_true_final) ** 2).mean()
        
        # --- DYNAMIC ANCHOR LOSSES ---
        # Anchor 1 at t=0.33
        t_anchor1 = torch.full_like(x_data, 0.33)
        h_true_anchor1 = torch.tensor(anchor1_image[y_idx, x_idx], dtype=torch.float32).view(-1, 1).to(device)
        h_pred_anchor1 = model(torch.cat([x_data, y_data, t_anchor1], dim=1))
        loss_anchor1 = ((h_pred_anchor1 - h_true_anchor1) ** 2).mean()

        # Anchor 2 at t=0.66
        t_anchor2 = torch.full_like(x_data, 0.66)
        h_true_anchor2 = torch.tensor(anchor2_image[y_idx, x_idx], dtype=torch.float32).view(-1, 1).to(device)
        h_pred_anchor2 = model(torch.cat([x_data, y_data, t_anchor2], dim=1))
        loss_anchor2 = ((h_pred_anchor2 - h_true_anchor2) ** 2).mean()

        loss_data = loss_initial + loss_final
        
        # --- Physics Loss ---
        n_phys_pts = 1000
        x_phys, y_phys, t_phys = torch.rand(n_phys_pts, 1), torch.rand(n_phys_pts, 1), torch.rand(n_phys_pts, 1)
        loss_phys = physics_loss_monotonic(model, x_phys.to(device), y_phys.to(device), t_phys.to(device))
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data +
                      lambda_physics * loss_phys +
                      lambda_anchor1 * loss_anchor1 +
                      lambda_anchor2 * loss_anchor2)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} | "
                  f"Anchor1 Loss: {loss_anchor1.item():.6f}")
            
    print("✓ Training complete!")
    return model

# ============================================
# 5. VISUALIZATION (Identical)
# ============================================
def visualize(model):
    print("Generating visualization...")
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points) + 1, figsize=(18, 3))
    
    final_image = get_line_image(progress=1.0)
    axes[0].imshow(final_image, cmap='gray', origin='lower', vmin=0, vmax=1); axes[0].set_title('Ground Truth'); axes[0].axis('off')
    
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
    plt.tight_layout(); plt.savefig("drawing_line_result.png"); plt.show()

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    model = create_model()
    trained_model = train(model)
    visualize(trained_model)
