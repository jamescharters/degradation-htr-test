import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

device = torch.device("cpu")

# ==========================================
# 1. IMPROVED DATA GENERATOR
# ==========================================
def create_mock_covid_csv(filename="mock_covid_data.csv"):
    days = 200
    t = np.linspace(0, 1, days)
    
    # Make the wave wider and easier to catch
    # Switch happens at 0.3
    switch = 1.0 / (1.0 + np.exp(-20 * (t - 0.3)))
    beta_curve = 4.0 * (1 - switch) + 0.5 * switch
    
    I = 0.01 # Start with 1% infected so the wave starts immediately
    S = 0.99
    gamma = 0.1 
    new_cases_list = []
    
    for i in range(days):
        beta = beta_curve[i]
        
        new_infections = beta * S * I
        dI = new_infections - gamma * I
        
        S -= new_infections
        I += dI
        
        # Clamp
        I = max(0, min(I, 1.0))
        S = max(0, min(S, 1.0))
        
        daily_count = int(new_infections * 100000)
        
        # Add noise
        noise = np.random.normal(0, 0.1 * daily_count + 1)
        final_count = max(0, int(daily_count + noise))
        new_cases_list.append(final_count)
        
    df = pd.DataFrame({'Day': np.arange(days), 'New_Cases': new_cases_list})
    df.to_csv(filename, index=False)

# ==========================================
# 2. DATASET (Normalized)
# ==========================================
class EpidemicDataset:
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        
        # Smoothing
        df['Smoothed'] = df['New_Cases'].rolling(window=7, min_periods=1).mean()
        # Active Cases Proxy
        df['Active'] = df['Smoothed'].rolling(window=14, min_periods=1).sum()
        
        data = df['Active'].values
        
        # --- CRITICAL FIX: Normalize to max ---
        # We map the highest peak to 0.5 (50% infection).
        # This ensures the physics engine has "headroom".
        scale_factor = np.max(data) * 2.0 
        data = data / scale_factor
        
        self.t_raw = torch.tensor(np.linspace(0, 1, len(data)), dtype=torch.float32)
        self.I_raw = torch.tensor(data, dtype=torch.float32)
        
    def get_data(self):
        return self.t_raw.view(-1, 1), self.I_raw.view(-1, 1)

# ==========================================
# 3. MODEL (Hot Initialization)
# ==========================================
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        
        # FIX: Initialize with POSITIVE mean (0.2) not zero mean
        # This biases the network to output non-zero values immediately
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.1 + 0.2)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1 + 0.2)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_expanded - grid_expanded, 2) * 10.0)
        basis = basis.view(x.size(0), -1)
        return base_output + torch.matmul(basis, self.spline_weight)

class KAN_DiffPhys_ODE(nn.Module):
    def __init__(self):
        super().__init__()
        # Predict Beta(t)
        # Grid Size 50: We need high resolution to catch a spike that happens in 0.05 time units
        self.beta_kan = nn.Sequential(
            RBFKanLayer(1, 32, grid_size=50),
            RBFKanLayer(32, 1)
        )
        
        # Initialize Gamma HIGH (e.g., 10.0)
        # The data dies very fast, implying people recover (or are isolated) instantly.
        self.gamma_param = nn.Parameter(torch.tensor([5.0])) 

    def get_beta(self, t):
        # REMOVED THE CLAMP. Let the physics dictate the magnitude.
        return F.softplus(self.beta_kan(t)) 
    
    def get_gamma(self):
        return F.softplus(self.gamma_param)

    def forward(self, t_steps, initial_I):
        simulated_I = []
        I = initial_I.clone()
        S = 1.0 - I
        
        betas = self.get_beta(t_steps)
        gamma = self.get_gamma()
        
        dt = t_steps[1] - t_steps[0]
        
        for i in range(len(t_steps)):
            beta = betas[i]
            
            new_infections = beta * S * I
            recoveries = gamma * I
            
            dI = new_infections - recoveries
            dS = -new_infections
            
            I = I + dt * dI
            S = S + dt * dS
            
            # Relaxed clamp to allow for the high dynamics
            I = torch.clamp(I, 0.0, 5.0) 
            S = torch.clamp(S, 0.0, 5.0)
            
            simulated_I.append(I)
            
        return torch.stack(simulated_I)

# ==========================================
# 4. EXECUTION
# ==========================================
if os.path.exists("mock_covid_data.csv"): os.remove("mock_covid_data.csv")
create_mock_covid_csv()
dataset = EpidemicDataset("mock_covid_data.csv")
t_train, I_train = dataset.get_data()

model = KAN_DiffPhys_ODE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Starting Training...")

for epoch in range(1001):
    optimizer.zero_grad()
    
    # Forward Simulation
    I_pred = model(t_train, I_train[0])
    
    # Loss: Weighted MSE
    # We multiply by I_train + 0.1 to weight the PEAK higher than the zero-tail
    weights = I_train.detach() + 0.1
    loss = torch.mean(weights * (I_pred - I_train)**2)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Ep {epoch} | Loss: {loss.item():.6f} | Gamma: {model.get_gamma().item():.3f}")

# Plot
with torch.no_grad():
    final_beta = model.get_beta(t_train).numpy()
    final_curve = model(t_train, I_train[0]).numpy()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_train, I_train, 'k.', alpha=0.3, label='Data')
plt.plot(t_train, final_curve, 'r-', linewidth=3, label='Fit')
plt.title("Calibrated Model")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_train, final_beta, 'b-', linewidth=3)
plt.title("Discovered Beta")
plt.grid(True)
plt.show()