#!/usr/bin/env python3
"""
A minimal "toy" example of a Physics-Informed Neural Network (PINN).
Goal: Learn a function y(t) that satisfies three rules:
  1. y(0) = 0      (Initial Condition)
  2. y(1) = 1      (Final Condition)
  3. dy/dt >= 0    (The "Physics": function must always increase)
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=" * 40)
print("Simplest PINN Toy Example")
print("=" * 40)

# Use CPU for this simple example
device = torch.device("cpu")

# 1. A VERY SIMPLE MODEL
# Input: t (a single number)
# Output: y (a single number)
class SimplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, t):
        return self.network(t)

# 2. THE PHYSICS LOSS FUNCTION
# This function enforces the rule dy/dt >= 0
def physics_loss(model, t):
    # We need gradients with respect to the input 't'
    t.requires_grad_(True)
    
    # Get the model's output y
    y = model(t)
    
    # Compute the derivative dy/dt using autograd
    dy_dt = torch.autograd.grad(
        outputs=y,
        inputs=t,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]
    
    # The rule is dy/dt >= 0. We penalize any violation.
    # torch.relu(-dy_dt) will be > 0 only if dy_dt is negative (a violation).
    violation = torch.relu(-dy_dt)
    
    # The loss is the mean squared violation
    loss = (violation ** 2).mean()
    return loss

# 3. THE TRAINING LOOP
def train(model, epochs=2000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Define the data points for our conditions
    t_initial = torch.tensor([[0.0]], device=device) # t=0
    y_initial = torch.tensor([[0.0]], device=device) # y should be 0

    t_final = torch.tensor([[1.0]], device=device)   # t=1
    y_final = torch.tensor([[1.0]], device=device)   # y should be 1
    
    # Loss weights to balance the rules
    lambda_physics = 1.0
    lambda_data = 10.0 # Make matching the start/end points more important

    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- Data Loss ---
        # Rule 1: y(0) = 0
        y_pred_initial = model(t_initial)
        loss_initial = ((y_pred_initial - y_initial) ** 2).mean()
        
        # Rule 2: y(1) = 1
        y_pred_final = model(t_final)
        loss_final = ((y_pred_final - y_final) ** 2).mean()
        
        loss_data = loss_initial + loss_final
        
        # --- Physics Loss ---
        # Rule 3: dy/dt >= 0
        # We check this rule at a bunch of random points in time
        t_physics = torch.rand(100, 1, device=device) # 100 random points between 0 and 1
        loss_phys = physics_loss(model, t_physics)
        
        # --- Total Loss ---
        total_loss = (lambda_data * loss_data) + (lambda_physics * loss_phys)
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss.item():.6f} | "
                  f"Data Loss: {loss_data.item():.6f} | Physics Loss: {loss_phys.item():.6f}")

    print("✓ Training complete!")
    return model

# 4. VISUALIZATION
def visualize_result(model):
    print("Visualizing result...")
    model.eval()
    with torch.no_grad():
        # Create a smooth range of time points for plotting
        t_plot = torch.linspace(0, 1, 100).view(-1, 1).to(device)
        y_pred = model(t_plot).cpu().numpy()

    # Plot the learned function
    plt.figure(figsize=(8, 6))
    plt.plot(t_plot.cpu().numpy(), y_pred, label="PINN Learned Function y(t)", color='blue', linewidth=3)
    
    # Plot the constraints
    plt.plot([0], [0], 'go', markersize=10, label="Constraint: y(0)=0")
    plt.plot([1], [1], 'ro', markersize=10, label="Constraint: y(1)=1")
    
    # Plot some true functions for comparison
    t_true = t_plot.cpu().numpy()
    plt.plot(t_true, t_true, 'k--', label="True function y=t", alpha=0.5)
    plt.plot(t_true, t_true**2, 'g--', label="True function y=t²", alpha=0.5)

    plt.title("Result of Toy PINN Example")
    plt.xlabel("Time (t)")
    plt.ylabel("Value (y)")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.savefig("toy_pinn_result.png")
    print("✓ Saved plot to toy_pinn_result.png")
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pinn_model = SimplePINN().to(device)
    trained_model = train(pinn_model)
    visualize_result(trained_model)
