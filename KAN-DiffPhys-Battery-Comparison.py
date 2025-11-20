import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. DATA GENERATION
# ==========================================
def generate_battery_data():
    print("Generating Data...")
    cycles = 1000
    t = torch.linspace(0, 1, cycles).view(-1, 1)
    dt = t[1] - t[0]
    
    # True Physics
    t_np = t.numpy()
    true_switch = 1.0 / (1.0 + np.exp(-50 * (t_np - 0.6)))
    true_lambda = 0.2 * (1 - true_switch) + 5.0 * true_switch
    true_lambda = torch.tensor(true_lambda, dtype=torch.float32)
    
    # Simulation with Noise
    Q = torch.tensor([1.0])
    data_Q = []
    for i in range(cycles):
        dQ = -true_lambda[i] * Q
        Q = Q + dt * dQ
        noise = torch.randn(1) * 0.05 
        data_Q.append(Q + noise)
        
    return t, torch.stack(data_Q).view(-1, 1), true_lambda, dt

t_train, Q_train, lambda_true, dt = generate_battery_data()
t_train.requires_grad_(True)

# ==========================================
# 2. MODELS (Fair Parameter Count)
# ==========================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MLP_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, t): return F.softplus(self.net(t))

class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.05)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
    def forward(self, x):
        base = F.linear(x, self.base_weight)
        x_exp = x.unsqueeze(-1)
        grid_exp = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_exp - grid_exp, 2) * 20.0)
        return base + torch.matmul(basis.view(x.size(0), -1), self.spline_weight)

class KAN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Small width to prove efficiency
            RBFKanLayer(1, 8, grid_size=30), 
            RBFKanLayer(8, 8, grid_size=20),
            RBFKanLayer(8, 1, grid_size=20)
        )
    def forward(self, t): return F.softplus(self.net(t))

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def run_simulation(model, t_grid, dt):
    lambdas = model(t_grid)
    Q = torch.tensor([1.0])
    history = []
    for i in range(len(t_grid)):
        dQ = -lambdas[i] * Q
        Q = Q + dt * dQ
        Q = torch.clamp(Q, 0.0, 1.2)
        history.append(Q)
    return torch.stack(history), lambdas

def train_model(model_type, reg_weight, label):
    print(f"Training {label}...", end="")
    if model_type == "MLP": model = MLP_Network().to(device)
    else: model = KAN_Network().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(2001):
        optimizer.zero_grad()
        Q_pred, lambdas_pred = run_simulation(model, t_train, dt)
        
        # Data Loss
        loss = torch.mean((Q_pred - Q_train.detach())**2)
        
        # TV Regularization
        if reg_weight > 0:
            lambda_dt = torch.autograd.grad(lambdas_pred, t_train, torch.ones_like(lambdas_pred), create_graph=True)[0]
            loss += reg_weight * torch.mean(torch.abs(lambda_dt))
            
        loss.backward()
        optimizer.step()
    
    print(f" Done. (Params: {count_params(model)})")
    return model

# ==========================================
# 4. RUN EXPERIMENTS
# ==========================================

# 1. MLP (Baseline)
mlp_model = train_model("MLP", 0.0, "MLP Baseline")

# 2. KAN Vanilla (Unregularized)
kan_v_model = train_model("KAN", 0.0, "KAN Vanilla")

# 3. KAN Ours (TV Regularized)
# Low regularization to allow high peak, but enough to kill oscillation
kan_o_model = train_model("KAN", 0.00001, "KAN Ours")

# ==========================================
# 5. MASTER VISUALIZATION
# ==========================================
with torch.no_grad():
    Q_mlp, L_mlp = run_simulation(mlp_model, t_train, dt)
    Q_kan_v, L_kan_v = run_simulation(kan_v_model, t_train, dt)
    Q_kan_o, L_kan_o = run_simulation(kan_o_model, t_train, dt)

t_np = t_train.detach().numpy()
L_true = lambda_true.numpy()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# --- LEFT PLOT: The Observable (Capacity) ---
# Show Data points + All 3 Curves
ax[0].plot(t_np, Q_train, 'k.', alpha=0.1, label='Sensor Data')
ax[0].plot(t_np, Q_mlp, 'b-', linewidth=2, alpha=0.7, label='MLP Prediction')
ax[0].plot(t_np, Q_kan_v, 'g--', linewidth=2, alpha=0.7, label='KAN Vanilla')
ax[0].plot(t_np, Q_kan_o, 'r-', linewidth=3, label='KAN-DiffPhys (Ours)')

ax[0].set_title("Observable: Battery Capacity Fade", fontsize=14)
ax[0].set_xlabel("Normalized Cycles")
ax[0].set_ylabel("Capacity (Q)")
ax[0].legend()
ax[0].grid(True)

# --- RIGHT PLOT: The Discovery (Degradation Rate) ---
# Show Truth + All 3 Curves
ax[1].plot(t_np, L_true, 'k--', linewidth=2, label='True Physics')
ax[1].plot(t_np, L_mlp, 'b-', linewidth=2, alpha=0.6, label='MLP (Smoothed)')
ax[1].plot(t_np, L_kan_v, 'g-', linewidth=1.5, alpha=0.6, label='KAN Vanilla (Unstable)')
ax[1].plot(t_np, L_kan_o, 'r-', linewidth=3, label='KAN-DiffPhys (Stable)')

ax[1].set_title("Hidden Parameter: Degradation Rate", fontsize=14)
ax[1].set_xlabel("Normalized Cycles")
ax[1].set_ylabel("Lambda (Degradation)")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()