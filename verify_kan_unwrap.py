import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. Minimal KAN Implementation (No external lib needed)
# ==========================================
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid is a buffer (non-learnable reference points)
        h = (1 / grid_size)
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()
        self.base_activation = nn.SiLU()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * 0.1)
        nn.init.trunc_normal_(self.spline_weight, std=0.01)

    def b_splines(self, x):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_norm = (torch.tanh(x) + 1) / 2 # Normalize to 0-1 for grid
        spline_basis = self.b_splines(x_norm)
        spline_output = torch.einsum("bij,oij->bo", spline_basis, self.spline_weight)
        return base_output + spline_output

class SimpleKAN(nn.Module):
    def __init__(self, layers_hidden):
        super(SimpleKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(layers_hidden[i], layers_hidden[i+1]))  
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# 1. Generate Synthetic Data
# ==========================================
def generate_data(n_samples=1000):
    t = np.linspace(0, 6*np.pi, n_samples)
    y_true = 2.0 * t + 3.0 * np.sin(t) 
    y_wrapped = np.angle(np.exp(1j * y_true))
    return t, y_true, y_wrapped

t, y_true, y_wrapped = generate_data()

X, Y = [], []
for i in range(len(y_wrapped) - 2):
    X.append([y_wrapped[i], y_wrapped[i+1]])
    Y.append(y_true[i+1] - y_true[i]) # Target: Gradient

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
X = X / torch.pi # Simple scaling

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# ==========================================
# 2. Define Models
# ==========================================
# KAN: 2 inputs -> 4 hidden -> 1 output
model_kan = SimpleKAN([2, 4, 1])

# MLP: 2 inputs -> 32 hidden -> 1 output
# We give the MLP more neurons to try to be "fair" against the complex KAN nodes
model_mlp = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# ==========================================
# 3. Model Analysis (Parameter Count)
# ==========================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

p_kan = count_parameters(model_kan)
p_mlp = count_parameters(model_mlp)

print("-" * 40)
print(f"MODEL ANALYSIS")
print("-" * 40)
print(f"KAN Architecture: [2 -> 4 -> 1]")
print(f"KAN Parameters:   {p_kan}")
print("-" * 40)
print(f"MLP Architecture: [2 -> 32 -> 1]")
print(f"MLP Parameters:   {p_mlp}")
print("-" * 40)
print(f"Ratio (KAN/MLP):  {p_kan/p_mlp:.2f}x parameters")
print("-" * 40)

# ==========================================
# 4. Training Loop
# ==========================================
optimizer_kan = optim.AdamW(model_kan.parameters(), lr=0.01)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("\nStarting Training (600 Epochs)...")
for epoch in range(601):
    # Train KAN
    optimizer_kan.zero_grad()
    loss_kan = criterion(model_kan(X_train), Y_train)
    loss_kan.backward()
    optimizer_kan.step()
    
    # Train MLP
    optimizer_mlp.zero_grad()
    loss_mlp = criterion(model_mlp(X_train), Y_train)
    loss_mlp.backward()
    optimizer_mlp.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | KAN Loss: {loss_kan.item():.5f} | MLP Loss: {loss_mlp.item():.5f}")

# ==========================================
# 5. Results & Plotting
# ==========================================
model_kan.eval()
model_mlp.eval()

with torch.no_grad():
    pred_kan = model_kan(X_test).numpy().flatten()
    pred_mlp = model_mlp(X_test).numpy().flatten()
    ground_truth = Y_test.numpy().flatten()

# Reconstruction
recon_kan = [y_true[train_size]]
recon_mlp = [y_true[train_size]]
for i in range(len(pred_kan)):
    recon_kan.append(recon_kan[-1] + pred_kan[i])
    recon_mlp.append(recon_mlp[-1] + pred_mlp[i])

# MSE Calculation
mse_kan = np.mean((pred_kan - ground_truth)**2)
mse_mlp = np.mean((pred_mlp - ground_truth)**2)

print("\n" + "="*40)
print("FINAL RESULTS")
print("="*40)
print(f"KAN Test MSE: {mse_kan:.6f}")
print(f"MLP Test MSE: {mse_mlp:.6f}")
if mse_kan < mse_mlp:
    print(">> VIABILITY CONFIRMED: KAN outperformed MLP.")
else:
    print(">> RESULT: MLP outperformed KAN (Model tuning required).")

# Visuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title(f"Gradient Pred (Wraps) - KAN Params: {p_kan}")
# Only plotting first 50 points to see the details clearly
plt.plot(ground_truth[:50], 'k-', alpha=0.4, linewidth=3, label='True Gradient')
plt.plot(pred_kan[:50], 'r.-', label='KAN')
plt.plot(pred_mlp[:50], 'b.--', label='MLP')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Full Reconstruction (Accumulated Phase)")
# Plotting more points here to see drift
plt.plot(y_true[train_size:][:150], 'k-', alpha=0.4, linewidth=3, label='True Phase')
plt.plot(recon_kan[:150], 'r-', label='KAN Recon')
plt.plot(recon_mlp[:150], 'b--', label='MLP Recon')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()