import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# CPU is safer for these sequential loops to avoid async GPU errors
device = torch.device("cpu")

# ==========================================
# 1. Data Generation
# ==========================================
def generate_stiff_data(noise_level=0.05):
    days = 200
    t = np.linspace(0, 1, days)
    
    # Sharp Switch (Steepness 50)
    switch = 1.0 / (1.0 + np.exp(-50 * (t - 0.4)))
    beta_curve = 10.0 * (1 - switch) + 1.0 * switch # 10 -> 1
    
    I = 0.01
    S = 0.99
    gamma = 2.0 
    
    data_I = []
    dt = t[1] - t[0]
    
    for i in range(days):
        beta = beta_curve[i]
        dI = beta * S * I - gamma * I
        
        # Standard Euler Integration
        S = S - (beta * S * I * dt)
        I = I + dI * dt
        I = max(0, I)
        
        data_I.append(I)
        
    I_clean = np.array(data_I)
    noise = np.random.normal(0, noise_level * np.max(I_clean), size=len(I_clean))
    I_noisy = np.clip(I_clean + noise, 0, None)
    
    return torch.tensor(t, dtype=torch.float32).view(-1,1), \
           torch.tensor(I_noisy, dtype=torch.float32).view(-1,1), \
           torch.tensor(beta_curve, dtype=torch.float32).view(-1,1)

t_train, I_train, beta_true = generate_stiff_data(noise_level=0.05)

# ==========================================
# 2. The Competitors
# ==========================================

# CONTENDER A: Standard MLP
class MLP_DiffPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.gamma_param = nn.Parameter(torch.tensor([2.0]))

    def get_beta(self, t):
        return self.net(t)
    
    def forward(self, t, I_init):
        sim_I = []
        # Initial conditions
        I = I_init
        S = 1.0 - I 
        
        betas = self.net(t)
        gamma = F.softplus(self.gamma_param)
        dt = t[1] - t[0]
        
        for i in range(len(t)):
            beta = betas[i]
            dI = beta * S * I - gamma * I
            
            # --- FIX: Out-of-place updates ---
            S = S - (beta * S * I * dt)
            I = I + (dI * dt)
            
            # Clamp for stability
            I = torch.clamp(I, 0, 1)
            S = torch.clamp(S, 0, 1)
            
            sim_I.append(I)
            
        return torch.stack(sim_I)

# CONTENDER B: KAN-DiffPhys
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=30):
        super().__init__()
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x):
        base = F.linear(x, self.base_weight)
        x_exp = x.unsqueeze(-1)
        grid_exp = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_exp - grid_exp, 2) * 20.0)
        basis = basis.view(x.size(0), -1)
        return base + torch.matmul(basis, self.spline_weight)

class KAN_DiffPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            RBFKanLayer(1, 16, grid_size=30), 
            RBFKanLayer(16, 1)
        )
        self.gamma_param = nn.Parameter(torch.tensor([2.0]))

    def get_beta(self, t):
        return F.softplus(self.net(t))
    
    def forward(self, t, I_init):
        sim_I = []
        I = I_init
        S = 1.0 - I
        
        betas = F.softplus(self.net(t))
        gamma = F.softplus(self.gamma_param)
        dt = t[1] - t[0]
        
        for i in range(len(t)):
            beta = betas[i]
            dI = beta * S * I - gamma * I
            
            # --- FIX: Out-of-place updates ---
            S = S - (beta * S * I * dt)
            I = I + (dI * dt)
            
            I = torch.clamp(I, 0, 1)
            S = torch.clamp(S, 0, 1)
            
            sim_I.append(I)
            
        return torch.stack(sim_I)

# ==========================================
# 3. The Shootout Loop
# ==========================================
def train_model(model, name, epochs=2000):
    print(f"Training {name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (Run Simulation)
        I_pred = model(t_train, I_train[0])
        
        # Loss
        loss = torch.mean((torch.log(I_pred + 1e-6) - torch.log(I_train + 1e-6))**2)
        
        loss.backward()
        
        # Clip gradients to prevent explosion in the physics loop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Ep {epoch} | Loss: {loss.item():.5f}")
        
    return losses

print("Initializing Models...")
mlp_model = MLP_DiffPhys()
kan_model = KAN_DiffPhys()

mlp_losses = train_model(mlp_model, "MLP-DiffPhys")
kan_losses = train_model(kan_model, "KAN-DiffPhys")

# ==========================================
# 4. Generate Paper Plots
# ==========================================
with torch.no_grad():
    beta_mlp = mlp_model.get_beta(t_train).numpy()
    beta_kan = kan_model.get_beta(t_train).numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Convergence Speed
ax1.plot(mlp_losses, label='MLP (Baseline)', color='blue', alpha=0.6)
ax1.plot(kan_losses, label='KAN (Ours)', color='red', linewidth=2)
ax1.set_yscale('log')
ax1.set_title("Fig 1: Convergence Speed (Log Scale)")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("MSE Loss")
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Plot 2: Parameter Recovery
ax2.plot(t_train, beta_true, 'k--', label='True Parameter', linewidth=2)
ax2.plot(t_train, beta_mlp, 'b-', label='MLP Prediction', linewidth=2, alpha=0.6)
ax2.plot(t_train, beta_kan, 'r-', label='KAN Prediction', linewidth=3)
ax2.set_title("Fig 2: Recovering the 'Lockdown' Switch")
ax2.set_xlabel("Time")
ax2.set_ylabel("Transmission Rate (Beta)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()