import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Device Configuration
# ---------------------------------------------
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

device = get_device()
print(f"Training Lockdown Simulation on: {device}")

# ---------------------------------------------
# 2. Neural Network
# ---------------------------------------------
class SpatioTemporalSIR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3) # S, I, R
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# ---------------------------------------------
# 3. Physics with Dynamic Parameters (Lockdown)
# ---------------------------------------------
gamma = 1.0 

def get_dynamic_params(t):
    """
    Returns Beta and Diffusion as a function of time.
    Simulates a lockdown starting at t = 0.4
    """
    # Create a soft switch using Sigmoid
    # steepness = 20 ensures the transition happens quickly but smoothly
    lockdown_start = 0.4
    switch = torch.sigmoid(20 * (t - lockdown_start)) 
    
    # Beta: Starts at 10.0 (Super spreader), drops to 1.0 (Masks/Distancing)
    beta_t = 10.0 * (1 - switch) + 1.0 * switch
    
    # Diffusion: Starts at 0.2 (Travel), drops to 0.01 (Stay at Home)
    diff_t = 0.2 * (1 - switch) + 0.01 * switch
    
    return beta_t, diff_t

def compute_derivatives(u, x, t):
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t, u_xx

def pde_loss(model, x, t):
    preds = model(x, t)
    S, I, R = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
    
    S_t, S_xx = compute_derivatives(S, x, t)
    I_t, I_xx = compute_derivatives(I, x, t)
    R_t, R_xx = compute_derivatives(R, x, t)
    
    # Get the Beta and Diffusion values for THIS specific time batch
    beta_t, diff_t = get_dynamic_params(t)
    
    # PDE Residuals with Dynamic Parameters
    # Note: beta_t and diff_t are tensors of shape [batch, 1]
    res_S = S_t - (diff_t * S_xx - beta_t * S * I)
    res_I = I_t - (diff_t * I_xx + beta_t * S * I - gamma * I)
    res_R = R_t - (diff_t * R_xx + gamma * I)
    
    return torch.mean(res_S**2 + res_I**2 + res_R**2)

# ---------------------------------------------
# 4. Data Sampling
# ---------------------------------------------
def get_batch(batch_size=2000):
    # Domain
    x_domain = (torch.rand(batch_size, 1) * 2 - 1).to(device).requires_grad_(True)
    t_domain = (torch.rand(batch_size, 1)).to(device).requires_grad_(True)
    
    # Initial Condition (t=0) - Concentrated in center
    x_init = (torch.rand(batch_size // 4, 1) * 2 - 1).to(device)
    t_init = torch.zeros_like(x_init).to(device)
    
    gaussian = torch.exp(-30 * x_init**2)
    I_init_target = 0.6 * gaussian 
    S_init_target = 1.0 - I_init_target
    R_init_target = torch.zeros_like(x_init)
    
    return (x_domain, t_domain), (x_init, t_init, S_init_target, I_init_target, R_init_target)

# ---------------------------------------------
# 5. Training Loop
# ---------------------------------------------
model = SpatioTemporalSIR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 12000 # Slightly more epochs for complex dynamics

print("Starting Lockdown Simulation...")
for epoch in range(epochs):
    optimizer.zero_grad()
    (x_dom, t_dom), (x_init, t_init, S_target, I_target, R_target) = get_batch()
    
    loss_physics = pde_loss(model, x_dom, t_dom)
    
    preds_init = model(x_init, t_init)
    loss_init = torch.mean((preds_init[:, 0:1] - S_target)**2 + 
                           (preds_init[:, 1:2] - I_target)**2 + 
                           (preds_init[:, 2:3] - R_target)**2)
    
    loss = loss_physics + 5.0 * loss_init # Weight initial condition higher
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.6f}")

# ---------------------------------------------
# 6. Visualization
# ---------------------------------------------
print("Plotting...")

# Grid for Heatmap
x_np = np.linspace(-1, 1, 100)
t_np = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_np, t_np)

x_flat = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
t_flat = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

with torch.no_grad():
    preds = model(x_flat, t_flat).cpu().numpy()
    # Also get the beta curve for plotting
    beta_curve, _ = get_dynamic_params(t_flat)
    beta_curve = beta_curve.cpu().numpy()

I_grid = preds[:, 1].reshape(100, 100)

# PLOT 1: The Heatmap
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

c = ax[0].contourf(T, X, I_grid, levels=50, cmap='inferno')
ax[0].set_title("Impact of Lockdown on Spread (Heatmap)")
ax[0].set_xlabel("Time (t)")
ax[0].set_ylabel("Space (x)")
ax[0].axvline(x=0.4, color='white', linestyle='--', linewidth=2, label='Lockdown Starts')
ax[0].legend()
plt.colorbar(c, ax=ax[0], label="Infected Proportion")

# PLOT 2: Total Infected over time
# Summing I across the Space dimension for each time step
total_infected = np.sum(I_grid, axis=1)
ax[1].plot(t_np, total_infected, 'r-', linewidth=3)
ax[1].set_title("Total Active Cases (Flattening the Curve)")
ax[1].set_xlabel("Time (t)")
ax[1].set_ylabel("Total Infected (Sum over Space)")
ax[1].axvline(x=0.4, color='k', linestyle='--', label='Lockdown Starts')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()