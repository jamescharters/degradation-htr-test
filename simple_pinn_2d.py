#!/usr/bin/env python3
"""
A 2D (1 space + 1 time) "toy" example of a Physics-Informed Neural Network (PINN).
Goal: Learn a function h(x, t) that models a line being drawn from left to right.
  1. h(x, 0) = 0      (Initial Condition: blank canvas)
  2. h(x, 1) = 1      (Final Condition: full line)
  3. The "Physics": The line "propagates" from left to right over time.
     We'll model this by penalizing (∂h/∂t + ∂h/∂x)², a simplified wave equation.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 40)
print("2D PINN Toy Example (1D space + time)")
print("=" * 40)

# Use CPU for this simple example
device = torch.device("cpu")

# 1. THE MODEL
# Input: (x, t) - two numbers
# Output: h - a single number (height)
class SimplePINN_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, t):
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        # Apply a final sigmoid to keep the output between 0 and 1, which is helpful here.
        return torch.sigmoid(self.network(inputs))

# 2. THE PHYSICS LOSS FUNCTION
def physics_loss_2d(model, x, t):
    # We need gradients with respect to both inputs 'x' and 't'
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # Get the model's output h
    h = model(x, t)
    
    # Compute derivatives using autograd
    # ∂h/∂t
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    # ∂h/∂x
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    
    # The rule is a simplified wave equation: ∂h/∂t + ∂h/∂x = 0
    # The loss is the mean squared value of the equation's residual
    wave_residual = h_t + h_x
    loss = (wave_residual ** 2).mean()
    return loss

# 3. THE TRAINING LOOP
def train_2d(model, epochs=5000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights to balance the rules
    lambda_physics = 0.1
    lambda_data = 1.0

    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Loss (Initial and Final Conditions) ---
        # We sample random points in space for these conditions
        x_data = torch.rand(100, 1, device=device)
        
        # Rule 1: h(x, 0) = 0
        t_initial = torch.zeros_like(x_data)
        h_pred_initial = model(x_data, t_initial)
        loss_initial = (h_pred_initial ** 2).mean() # Target is 0
        
        # Rule 2: h(x, 1) = 1
        t_final = torch.ones_like(x_data)
        h_pred_final = model(x_data, t_final)
        loss_final = ((h_pred_final - 1.0) ** 2).mean() # Target is 1
        
        loss_data = loss_initial + loss_final
        
        # --- Physics Loss ---
        # Rule 3: Enforce wave equation at random points in space-time
        x_physics = torch.rand(500, 1, device=device)
        t_physics = torch.rand(500, 1, device=device)
        loss_phys = physics_loss_2d(model, x_physics, t_physics)
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data) + (lambda_physics * loss_phys)
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss.item():.6f} | "
                  f"Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_phys.item():.6f}")

    print("✓ Training complete!")
    return model

# 4. VISUALIZATION
def visualize_result_2d(model):
    print("Visualizing result...")
    model.eval()
    
    # Create a grid of points in space (x) and time (t)
    x_grid = torch.linspace(0, 1, 100)
    t_grid = torch.linspace(0, 1, 100)
    X, T = torch.meshgrid(x_grid, t_grid, indexing='xy')
    
    # Flatten the grid to pass through the network
    x_flat = X.flatten().view(-1, 1).to(device)
    t_flat = T.flatten().view(-1, 1).to(device)
    
    with torch.no_grad():
        h_pred = model(x_flat, t_flat).cpu().numpy()
    
    # Reshape the output back into a 2D grid
    H = h_pred.reshape(100, 100)
    
    # Plotting
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Heatmap of h(x, t)
    ax1 = fig.add_subplot(1, 2, 1)
    c = ax1.pcolormesh(T, X, H, cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(c, ax=ax1)
    ax1.set_title("Heatmap of h(x, t)")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Space (x)")
    
    # Add a line representing the wave front x=t
    ax1.plot([0, 1], [0, 1], 'r--', label='Wavefront (x=t)')
    ax1.legend()
    
    # Plot 2: Snapshots at different times
    ax2 = fig.add_subplot(1, 2, 2)
    time_indices = [0, 24, 49, 74, 99] # Corresponds to t=0, 0.25, 0.5, 0.75, 1.0
    for i in time_indices:
        t_val = t_grid[i].item()
        ax2.plot(x_grid.numpy(), H[:, i], label=f't = {t_val:.2f}')
    
    ax2.set_title("Snapshots of h(x) at different times")
    ax2.set_xlabel("Space (x)")
    ax2.set_ylabel("Height (h)")
    ax2.grid(True, linestyle=':')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("toy_pinn_2d_result.png")
    print("✓ Saved plot to toy_pinn_2d_result.png")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pinn_model_2d = SimplePINN_2D().to(device)
    trained_model = train_2d(pinn_model_2d)
    visualize_result_2d(trained_model)
