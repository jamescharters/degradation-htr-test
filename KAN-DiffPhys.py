import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")

# ---------------------------------------------
# 1. Generate Target Data
# ---------------------------------------------
def generate_valid_data():
    Nx = 50
    Nt = 1000
    x = torch.linspace(-1, 1, Nx)
    t = torch.linspace(0, 1, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    I = 0.01 * torch.exp(-20 * x**2)
    S = 1.0 - I
    data_I = []
    
    for i in range(Nt):
        cur_t = t[i]
        switch = torch.sigmoid(50 * (cur_t - 0.4)) 
        beta = 10.0 * (1 - switch) + 1.0 * switch
        diff = 0.2 * (1 - switch) + 0.01 * switch
        
        I_left = torch.roll(I, 1); I_left[0] = I[0]; I_right = torch.roll(I, -1); I_right[-1] = I[-1]
        I_xx = (I_right - 2*I + I_left) / (dx**2)
        
        I = I + dt * (diff * I_xx + beta * S * I - 1.0 * I)
        S = 1.0 - I
        data_I.append(I.clone())
        
    return torch.stack(data_I), t, x, dt, dx

I_target, t_grid, x_grid, dt, dx = generate_valid_data()

# ---------------------------------------------
# 2. The KAN Layer (With Safe Initialization)
# ---------------------------------------------
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=20):
        super().__init__()
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        
        # FIX 1: Initialize weights very small to prevent initial explosion
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.01)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_expanded - grid_expanded, 2) * 10.0)
        basis = basis.view(x.size(0), -1)
        return base_output + torch.matmul(basis, self.spline_weight)

# ---------------------------------------------
# 3. The Differentiable Physics Model
# ---------------------------------------------
class DiffPhysKAN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.beta_kan = nn.Sequential(
            RBFKanLayer(1, 16, grid_size=20),
            RBFKanLayer(16, 1)
        )
        self.diff_param = nn.Parameter(torch.tensor([-3.0])) 

    def get_params(self, t):
        # FIX 2: Clamp parameters to sane physical ranges
        # Beta shouldn't be > 20. Diffusion shouldn't be > 1.0.
        raw_beta = F.softplus(self.beta_kan(t))
        beta = torch.clamp(raw_beta, min=0.0, max=20.0)
        
        raw_diff = F.softplus(self.diff_param)
        diff = torch.clamp(raw_diff, min=0.0, max=1.0)
        return beta, diff

    def forward(self, t_steps, x_grid, initial_I):
        simulated_history = []
        
        I = initial_I.clone()
        S = 1.0 - I
        
        betas, diff = self.get_params(t_steps) 
        
        dt = t_steps[1] - t_steps[0]
        dx = x_grid[1] - x_grid[0]
        
        # The Simulation Loop
        for i in range(len(t_steps)):
            beta = betas[i]
            
            # Spatial Derivatives
            I_left = torch.roll(I, 1); I_left[0] = I[0]
            I_right = torch.roll(I, -1); I_right[-1] = I[-1]
            I_xx = (I_right - 2*I + I_left) / (dx**2)
            
            # Physics Update
            dIdt = diff * I_xx + beta * S * I - 1.0 * I
            
            # Euler Step
            I = I + dt * dIdt
            
            # FIX 3: Clamp State to prevent negative population crash
            I = torch.clamp(I, min=0.0, max=10.0)
            S = 1.0 - I
            
            simulated_history.append(I.clone())
            
        return torch.stack(simulated_history)

# ---------------------------------------------
# 4. Stabilized Training Loop
# ---------------------------------------------
model = DiffPhysKAN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) 

print("\n=== Starting Stabilized DiffPhys Training ===")

t_input = t_grid.view(-1, 1)
I_init = I_target[0]

# Detach target to be safe
I_target_safe = I_target.detach()

for epoch in range(2001):
    optimizer.zero_grad()
    
    # Run Simulation
    I_simulated = model(t_input, x_grid, I_init)
    
    # Loss
    loss = torch.mean((torch.log(I_simulated + 1e-6) - torch.log(I_target_safe + 1e-6))**2)
    
    if torch.isnan(loss):
        print("WARNING: Loss is NaN. Resetting optimizer...")
        optimizer.zero_grad()
        continue
        
    loss.backward()
    
    # FIX 4: Gradient Clipping (Prevent exploding updates)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    if epoch % 200 == 0:
        with torch.no_grad():
            b_start, _ = model.get_params(torch.tensor([[0.1]]))
            b_end, _ = model.get_params(torch.tensor([[0.9]]))
        print(f"Ep {epoch} | Loss: {loss.item():.5f} | Beta(0.1): {b_start.item():.2f} | Beta(0.9): {b_end.item():.2f}")

# ---------------------------------------------
# 5. Visualization
# ---------------------------------------------
t_plot = torch.linspace(0, 1, 100).view(-1, 1)
true_switch = 1.0 / (1.0 + np.exp(-50 * (t_plot.numpy() - 0.4)))
true_beta = 10.0 * (1 - true_switch) + 1.0 * true_switch

with torch.no_grad():
    kan_beta, _ = model.get_params(t_plot)
    kan_beta = kan_beta.numpy()

plt.figure(figsize=(10, 6))
plt.plot(t_plot, true_beta, 'k--', label='True Beta', linewidth=2)
plt.plot(t_plot, kan_beta, 'r-', label='Stabilized KAN-DiffPhys', linewidth=3)
plt.title("Final Result: KAN + Differentiable Physics")
plt.xlabel("Time")
plt.ylabel("Beta")
plt.legend()
plt.grid(True)
plt.show()