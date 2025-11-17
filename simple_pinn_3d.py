#!/usr/bin/env python3
"""
A 3D (2 space + 1 time) "toy" example of a Physics-Informed Neural Network (PINN).
Goal: Learn a function h(x, y, t) that models a circular ink blot expanding
      from the center of a square canvas.

The network is only told:
  1. h(x, y, 0) = 0      (Initial Condition: blank canvas)
  2. h(x, y, 1) = 1      (Final Condition: full canvas)
  3. The "Physics":
     - Monotonicity: Ink never disappears (∂h/∂t ≥ 0).
     - Smoothness: The ink surface should be smooth (penalize Laplacian).
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 40)
print("3D PINN Toy Example (2D space + time)")
print("=" * 40)

# Use CPU for this simple example
device = torch.device("cpu")

# 1. THE MODEL
# Input: (x, y, t) - three numbers
# Output: h - a single number (height)
class SimplePINN_3D(nn.Module):
    def __init__(self):
        super().__init__()
        # Increased model size for the higher dimensional problem
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        return torch.sigmoid(self.network(inputs))

# 2. THE MULTI-COMPONENT PHYSICS LOSS
def physics_loss_3d(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    
    h = model(x, y, t)
    
    # --- Monotonicity Loss (∂h/∂t ≥ 0) ---
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    violation = torch.relu(-h_t)
    loss_monotonic = (violation ** 2).mean()
    
    # --- Smoothness Loss (Laplacian) ---
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    
    h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x), create_graph=True)[0]
    h_yy = torch.autograd.grad(h_y, y, grad_outputs=torch.ones_like(h_y), create_graph=True)[0]
    
    laplacian = h_xx + h_yy
    loss_smooth = (laplacian ** 2).mean()
    
    return loss_monotonic, loss_smooth

# 3. THE TRAINING LOOP
def train_3d(model, epochs=4000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights
    lambda_data = 1.0
    lambda_monotonic = 1.0
    lambda_smooth = 0.01 # Smoothness is a "regularizer", so it has a smaller weight

    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Loss (Initial and Final Conditions) ---
        xy_data = torch.rand(200, 2, device=device) # Sample 200 (x,y) points
        x_data, y_data = xy_data[:, 0:1], xy_data[:, 1:2]
        
        # Rule 1: h(x, y, 0) = 0
        t_initial = torch.zeros_like(x_data)
        h_pred_initial = model(x_data, y_data, t_initial)
        loss_initial = (h_pred_initial ** 2).mean()
        
        # Rule 2: h(x, y, 1) = 1
        t_final = torch.ones_like(x_data)
        h_pred_final = model(x_data, y_data, t_final)
        loss_final = ((h_pred_final - 1.0) ** 2).mean()
        
        loss_data = loss_initial + loss_final
        
        # --- Physics Loss ---
        # Sample random points in the 3D space-time volume
        xyz_physics = torch.rand(1000, 3, device=device)
        x_phys, y_phys, t_phys = xyz_physics[:, 0:1], xyz_physics[:, 1:2], xyz_physics[:, 2:3]
        
        loss_mono, loss_smooth = physics_loss_3d(model, x_phys, y_phys, t_phys)
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data +
                      lambda_monotonic * loss_mono +
                      lambda_smooth * loss_smooth)
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss.item():.6f} | "
                  f"Data: {loss_data.item():.6f} | Mono: {loss_mono.item():.6f} | Smooth: {loss_smooth.item():.6f}")

    print("✓ Training complete!")
    return model

# 4. VISUALIZATION
def visualize_result_3d(model):
    print("Visualizing result as a series of snapshots...")
    model.eval()
    
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(1, len(time_points), figsize=(20, 4))
    
    # Create a 2D grid for x and y
    grid_res = 100
    x_space = torch.linspace(0, 1, grid_res)
    y_space = torch.linspace(0, 1, grid_res)
    X, Y = torch.meshgrid(x_space, y_space, indexing='xy')
    
    x_flat = X.flatten().view(-1, 1).to(device)
    y_flat = Y.flatten().view(-1, 1).to(device)
    
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            # Create a time tensor for this specific snapshot
            t_flat = torch.full_like(x_flat, t_val)
            
            # Get model prediction
            h_pred = model(x_flat, y_flat, t_flat).cpu().numpy()
            H = h_pred.reshape(grid_res, grid_res)
            
            # Plot the heatmap
            ax = axes[i]
            im = ax.imshow(H, cmap='viridis', origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=1)
            ax.set_title(f't = {t_val:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig("toy_pinn_3d_result.png")
    print("✓ Saved plot to toy_pinn_3d_result.png")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pinn_model_3d = SimplePINN_3D().to(device)
    trained_model = train_3d(pinn_model_3d)
    visualize_result_3d(trained_model)
