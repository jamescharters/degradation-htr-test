import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cpu")

# ==========================================
# 1. DATA GENERATION
# ==========================================
def generate_financial_data():
    print("Simulating Market Regime Change...")
    days = 1000
    dt = 1/365 
    t = torch.linspace(0, 1, days)
    
    # True Regimes (Hidden Parameter)
    sigma_true = []
    for time in t:
        if 0.4 < time < 0.6:
            sigma_true.append(0.8) # CRISIS (High Volatility)
        elif time >= 0.6:
            sigma_true.append(0.2) # New Normal
        else:
            sigma_true.append(0.1) # Calm
            
    sigma_true = torch.tensor(sigma_true, dtype=torch.float32)
    
    # Returns (The Observable Noise)
    torch.manual_seed(42) 
    noise = torch.randn(days)
    returns = sigma_true * noise * torch.sqrt(torch.tensor(dt))
    
    # Price Path
    price = [100.0]
    for r in returns:
        price.append(price[-1] * (1 + r))
        
    return t.view(-1, 1), returns.view(-1, 1), sigma_true.view(-1, 1), price

# Generate Data
t_train, returns_train, sigma_true, price_path = generate_financial_data()
t_train.requires_grad_(True) 
dt_tensor = torch.tensor(1/365).float()

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================

# --- A. KAN Architecture ---
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=30):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_expanded - grid_expanded, 2) * 20.0)
        basis = basis.view(x.size(0), -1)
        return base_output + torch.matmul(basis, self.spline_weight)

class VolatilityKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            RBFKanLayer(1, 64, grid_size=30),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )
    def get_log_sigma(self, t):
        return self.net(t)

# --- B. MLP Architecture ---
class VolatilityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def get_log_sigma(self, t):
        return self.net(t)

# ==========================================
# 3. INITIALIZATION
# ==========================================

# 1. MLP Supervised (Baseline)
model_mlp_sup = VolatilityMLP().to(device)
opt_mlp_sup = torch.optim.Adam(model_mlp_sup.parameters(), lr=0.01)

# 2. KAN Supervised (Theoretical Limit)
model_kan_sup = VolatilityKAN().to(device)
opt_kan_sup = torch.optim.Adam(model_kan_sup.parameters(), lr=0.01)

# 3. KAN Naive (Unsupervised Data-Driven)
model_kan_naive = VolatilityKAN().to(device)
opt_kan_naive = torch.optim.Adam(model_kan_naive.parameters(), lr=0.005) # Lower LR for stability

# 4. KAN DiffPhys (Unsupervised Physics-Informed)
model_kan_diff = VolatilityKAN().to(device)
opt_kan_diff = torch.optim.Adam(model_kan_diff.parameters(), lr=0.01)

# Smart Initialization (Crucial for convergence)
global_std = torch.std(returns_train) * np.sqrt(365)
init_bias = torch.log(global_std)

with torch.no_grad():
    for m in [model_mlp_sup, model_kan_sup, model_kan_naive, model_kan_diff]:
        m.net[-1].bias.fill_(init_bias)

# ==========================================
# 4. JOINT TRAINING LOOP
# ==========================================
print("Starting Quadruple Model Training...")
criterion_mse = nn.MSELoss() 

for epoch in range(5001):
    
    # --- 1. TRAIN MLP (Supervised) ---
    opt_mlp_sup.zero_grad()
    pred_mlp_sup = torch.exp(model_mlp_sup.get_log_sigma(t_train))
    loss_mlp_sup = criterion_mse(pred_mlp_sup, sigma_true)
    loss_mlp_sup.backward()
    opt_mlp_sup.step()

    # --- 2. TRAIN KAN (Supervised) ---
    opt_kan_sup.zero_grad()
    pred_kan_sup = torch.exp(model_kan_sup.get_log_sigma(t_train))
    loss_kan_sup = criterion_mse(pred_kan_sup, sigma_true)
    loss_kan_sup.backward()
    opt_kan_sup.step()

    # --- 3. TRAIN KAN NAIVE (Unsupervised) ---
    # Goal: Fit the absolute magnitude of returns directly
    opt_kan_naive.zero_grad()
    pred_kan_naive = torch.exp(model_kan_naive.get_log_sigma(t_train))
    # Proxy target: |r| / sqrt(dt). This is a noisy estimator of Volatility.
    proxy_target = torch.abs(returns_train) / torch.sqrt(dt_tensor)
    loss_kan_naive = criterion_mse(pred_kan_naive, proxy_target)
    loss_kan_naive.backward()
    opt_kan_naive.step()

    # --- 4. TRAIN KAN DiffPhys (Unsupervised) ---
    # Goal: Maximize Likelihood of returns based on Brownian Motion
    opt_kan_diff.zero_grad()
    log_sigma_diff = model_kan_diff.get_log_sigma(t_train)
    
    # Physics NLL
    log_var = 2 * log_sigma_diff + torch.log(dt_tensor)
    var = torch.exp(log_var)
    nll_loss = torch.mean(log_var + (returns_train**2 / var))
    
    # TV Regularization
    u_grad = torch.autograd.grad(log_sigma_diff, t_train, torch.ones_like(log_sigma_diff), create_graph=True)[0]
    tv_loss = torch.mean(torch.abs(u_grad))
    
    tv_weight = 0.0 if epoch < 1000 else 0.04
    loss_kan_diff = nll_loss + tv_weight * tv_loss
    loss_kan_diff.backward()
    opt_kan_diff.step()
    
    if epoch % 1000 == 0:
        print(f"Ep {epoch} | MLP(Sup): {loss_mlp_sup:.4f} | KAN(Sup): {loss_kan_sup:.4f} | KAN(Diff): {nll_loss:.4f}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
with torch.no_grad():
    c_mlp_sup = torch.exp(model_mlp_sup.get_log_sigma(t_train)).numpy()
    c_kan_sup = torch.exp(model_kan_sup.get_log_sigma(t_train)).numpy()
    c_kan_naive = torch.exp(model_kan_naive.get_log_sigma(t_train)).numpy()
    c_kan_diff = torch.exp(model_kan_diff.get_log_sigma(t_train)).numpy()

plt.figure(figsize=(12, 12))

# Plot 1: Market Context
plt.subplot(3, 1, 1)
plt.plot(price_path, 'k-', linewidth=1.5)
plt.title("1. Market Context: Stock Price", fontsize=14)
plt.ylabel("Price ($)")
plt.grid(True, alpha=0.3)
plt.margins(x=0)

# Plot 2: Supervised Benchmarks
plt.subplot(3, 1, 2)
plt.plot(t_train.detach().numpy(), sigma_true.numpy(), 'k--', label='Ground Truth', linewidth=2)
plt.plot(t_train.detach().numpy(), c_mlp_sup, 'g-', label='MLP (Supervised)', linewidth=2, alpha=0.7)
plt.plot(t_train.detach().numpy(), c_kan_sup, 'b-', label='KAN (Supervised)', linewidth=2, alpha=0.7)
plt.title("2. Supervised Benchmarks (The Ideal)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.margins(x=0)

# Plot 3: Unsupervised Discovery
plt.subplot(3, 1, 3)
plt.plot(t_train.detach().numpy(), sigma_true.numpy(), 'k--', label='Ground Truth', linewidth=2.5)
# Naive KAN
plt.plot(t_train.detach().numpy(), c_kan_naive, 'c-', label='KAN Naive (Fit |Returns|)', linewidth=1.0, alpha=0.5)
# DiffPhys KAN
plt.plot(t_train.detach().numpy(), c_kan_diff, 'r-', label='KAN-DiffPhys (Physics NLL)', linewidth=3.0)

plt.title("3. Unsupervised Discovery: Data-Driven (Naive) vs Physics-Informed (DiffPhys)", fontsize=14)
plt.xlabel("Time (Years)")
plt.ylabel("Volatility")
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.2) # Limit y-axis because naive model might spike
plt.margins(x=0)

plt.tight_layout()
plt.show()

# ==========================================
# 6. PARAMETER COUNTS
# ==========================================
def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n==========================================")
print("       MODEL PARAMETER COMPARISON")
print("==========================================")
print(f"1. MLP (Baseline):         {count_params(model_mlp_sup)} parameters")
print(f"2. KAN (Standard):         {count_params(model_kan_sup)} parameters")
print("   (Note: KANs have fewer params but more expressivity)")
print("==========================================")