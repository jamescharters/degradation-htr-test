#!/usr/bin/env python3
"""
A minimal, boiled-down PINN - V4: Solving the Procrastination Problem.
Goal: Force the animation to progress over time using an integral constraint.

Changes from V3:
1. Added an "Integral Constraint" loss.
2. This new rule forces the total amount of ink to increase over time,
   preventing the model from waiting until t=1 to draw anything.
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
print("MINIMAL PINN EXAMPLE - V4 (PACED)")
print("=" * 40)
device = torch.device("cpu")

# ============================================
# 1. THE MODEL (Identical to V2/V3)
# ============================================
def create_model():
    return nn.Sequential(
        nn.Linear(3, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1), nn.Sigmoid()
    ).to(device)

# ============================================
# 2. THE DATA (Identical to V2/V3)
# ============================================
def get_target_image(width=100, height=100):
    image = np.zeros((height, width), dtype=np.float32)
    x_start, x_end = int(width * 0.3), int(width * 0.7)
    y_start, y_end = int(height * 0.3), int(height * 0.7)
    image[y_start:y_end, x_start:x_end] = 1.0
    return gaussian_filter(image, sigma=3)

# ============================================
# 3. THE PHYSICS LOSS (Identical to V3)
# ============================================
def physics_loss_guided(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    inputs = torch.cat([x, y, t], dim=1)
    h = model(inputs)
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    loss_monotonic = (torch.relu(-h_t) ** 2).mean()
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x), create_graph=True)[0]
    h_yy = torch.autograd.grad(h_y, y, grad_outputs=torch.ones_like(h_y), create_graph=True)[0]
    loss_smooth = ((h_xx + h_yy) ** 2).mean()
    return loss_monotonic, loss_smooth

# ============================================
# 4. THE TRAINING LOOP (with Integral Constraint)
# ============================================
def train(model, target_image, epochs=5000, lr=1e-3):
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights
    lambda_data = 100.0
    lambda_monotonic = 1.0
    lambda_smooth = 0.001
    lambda_integral = 50.0 # NEW: A strong weight for our pacing guide

    # Pre-calculate the total ink in the final image
    total_ink_final = target_image.mean()

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Loss (same as V3) ---
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

        # --- Physics Loss (same as V3) ---
        n_phys_pts = 1000
        x_phys = torch.rand(n_phys_pts, 1, device=device)
        y_phys = torch.rand(n_phys_pts, 1, device=device)
        t_phys = torch.rand(n_phys_pts, 1, device=device)
        loss_mono, loss_smooth = physics_loss_guided(model, x_phys, y_phys, t_phys)
        
        # --- NEW: Integral Constraint Loss ---
        # We enforce this at a random intermediate time `t_mid` each epoch
        n_integral_pts = 1000 # Use many points for a good approximation of the integral
        x_int = torch.rand(n_integral_pts, 1, device=device)
        y_int = torch.rand(n_integral_pts, 1, device=device)
        t_mid = torch.rand(1, 1, device=device).expand(n_integral_pts, 1) # A single random time `t`

        inputs_mid = torch.cat([x_int, y_int, t_mid], dim=1)
        h_pred_mid = model(inputs_mid)

        # The mean of h over the canvas approximates the total ink.
        # The target total ink should scale linearly with time.
        total_ink_predicted = h_pred_mid.mean()
        total_ink_target = t_mid[0] * total_ink_final 
        
        loss_integral = (total_ink_predicted - total_ink_target)**2

        # --- Total Loss ---
        total_loss = (lambda_data * loss_data +
                      lambda_monotonic * loss_mono +
                      lambda_smooth * loss_smooth +
                      lambda_integral * loss_integral)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f} | "
                  f"Integral Loss: {loss_integral.item():.6f}")
            
    print("✓ Training complete!")
    return model

# ============================================
# 5. VISUALIZATION (Identical to V2/V3)
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
    plt.savefig("minimal_target_result_v4.png")
    print("✓ Saved plot to minimal_target_result_v4.png")
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    model = create_model()
    target = get_target_image()
    trained_model = train(model, target)
    visualize(trained_model, target)
