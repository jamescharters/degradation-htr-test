#!/usr/bin/env python3
"""
A minimal, boiled-down PINN to learn a simple spatio-temporal process.
Goal: Learn to make a square appear over time.

Core Logic:
1.  The canvas must be blank at t=0.
2.  The canvas must match the final square image at t=1.
3.  Physics: Ink can only be added, never removed (∂h/∂t >= 0).
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
print("MINIMAL PINN EXAMPLE")
print("=" * 40)
device = torch.device("cpu") # CPU is fine for this

# ============================================
# 1. THE MODEL (as a simple function)
# ============================================
def create_model():
    """Returns a simple MLP: (x, y, t) -> h"""
    return nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
        nn.Sigmoid() # Sigmoid is simple and keeps output in [0, 1]
    ).to(device)

# ============================================
# 2. THE DATA (as a simple function)
# ============================================
def get_target_image(width=100, height=100):
    """Returns a numpy array of a soft-edged square."""
    image = np.zeros((height, width), dtype=np.float32)
    x_start, x_end = int(width * 0.3), int(width * 0.7)
    y_start, y_end = int(height * 0.3), int(height * 0.7)
    image[y_start:y_end, x_start:x_end] = 1.0
    return gaussian_filter(image, sigma=3)

# ============================================
# 3. THE PHYSICS LOSS (as a simple function)
# ============================================
def physics_loss_monotonic(model, x, y, t):
    """Enforces the rule ∂h/∂t >= 0."""
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    # Concatenate inputs and get model output
    inputs = torch.cat([x, y, t], dim=1)
    h = model(inputs)
    
    # Compute derivative of h with respect to t
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    
    # Penalize any negative values of h_t
    violation = torch.relu(-h_t)
    return (violation ** 2).mean()

# ============================================
# 4. THE TRAINING LOOP (simplified)
# ============================================
def train(model, target_image, epochs=3000, lr=1e-3):
    print("Starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights
    lambda_data = 10.0
    lambda_physics = 1.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Loss ---
        # Sample random (x,y) points
        n_data_pts = 500
        x_idx = np.random.randint(0, target_image.shape[1], n_data_pts)
        y_idx = np.random.randint(0, target_image.shape[0], n_data_pts)
        
        x_data = torch.tensor(x_idx / (target_image.shape[1] - 1), dtype=torch.float32).view(-1, 1).to(device)
        y_data = torch.tensor(y_idx / (target_image.shape[0] - 1), dtype=torch.float32).view(-1, 1).to(device)
        
        # Rule 1: h(x, y, 0) = 0 (Initial Condition)
        t_initial = torch.zeros_like(x_data)
        inputs_initial = torch.cat([x_data, y_data, t_initial], dim=1)
        h_pred_initial = model(inputs_initial)
        loss_initial = (h_pred_initial ** 2).mean()

        # Rule 2: h(x, y, 1) = target_image (Final Condition)
        t_final = torch.ones_like(x_data)
        h_true_final = torch.tensor(target_image[y_idx, x_idx], dtype=torch.float32).view(-1, 1).to(device)
        inputs_final = torch.cat([x_data, y_data, t_final], dim=1)
        h_pred_final = model(inputs_final)
        loss_final = ((h_pred_final - h_true_final) ** 2).mean()
        
        loss_data = loss_initial + loss_final

        # --- Physics Loss ---
        # Rule 3: ∂h/∂t >= 0
        n_phys_pts = 1000
        x_phys = torch.rand(n_phys_pts, 1, device=device)
        y_phys = torch.rand(n_phys_pts, 1, device=device)
        t_phys = torch.rand(n_phys_pts, 1, device=device)
        loss_phys = physics_loss_monotonic(model, x_phys, y_phys, t_phys)
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data) + (lambda_physics * loss_phys)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item():.4f}")
            
    print("✓ Training complete!")
    return model

# ============================================
# 5. VISUALIZATION (simplified)
# ============================================
def visualize(model, target_image):
    print("Generating visualization...")
    model.eval()
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(1, len(time_points) + 1, figsize=(18, 3))
    
    # Plot Ground Truth
    axes[0].imshow(target_image, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Create a grid to evaluate the model on
    res = 100
    x_grid = torch.linspace(0, 1, res)
    y_grid = torch.linspace(0, 1, res)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='xy')
    x_flat = X.flatten().view(-1, 1).to(device)
    y_flat = Y.flatten().view(-1, 1).to(device)

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
    plt.savefig("minimal_target_result.png")
    print("✓ Saved plot to minimal_target_result.png")
    plt.show()

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    model = create_model()
    target = get_target_image()
    
    trained_model = train(model, target)
    
    visualize(trained_model, target)
