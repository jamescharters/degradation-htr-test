import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Setup: MORE COMPLEX TASKS ---
x = torch.linspace(0, 1, 400).unsqueeze(1) # Add batch dimension for MLP

# Task A: A Cosine wave on the left side
mask_A = (x < 0.45).squeeze()
y_A_target = torch.zeros_like(x)
y_A_target[mask_A] = torch.cos(2 * torch.pi * x[mask_A]) * 0.5 + 0.5

# Task B: A Sine wave on the right side
mask_B = (x >= 0.45).squeeze()
y_B_target = torch.zeros_like(x)
y_B_target[mask_B] = torch.sin(2 * torch.pi * (x[mask_B] - 0.5)) * 0.5 + 0.5

y_full_target = y_A_target + y_B_target


# --- 2. The Models ---

# MODEL 1: Standard Model (Global Polynomial)
def global_model(x, weights):
    # Squeeze x to remove batch dim for this function
    x_flat = x.squeeze()
    basis = torch.stack([x_flat**i for i in range(len(weights))], dim=1)
    return (basis @ weights).unsqueeze(1) # Add batch dim back for consistency

# NEW --- MODEL 2: MLP Model (Global, SOTA Activations) ---
class MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(), # A smooth, modern activation function
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.layers(x)

# MODEL 3: KAN Model (Local)
def local_kan_model(x, weights, centers, log_width):
    width = torch.exp(log_width)
    # Squeeze x for broadcasting, then unsqueeze centers
    basis = torch.exp(-(x - centers.unsqueeze(0))**2 / (2 * width**2))
    return (basis @ weights).unsqueeze(1) # Add batch dim back


# --- 3. Parameters and Optimizers ---
num_params_poly = 20
num_params_kan = 24
epochs = 1000

# Poly Model
w_global = torch.zeros(num_params_poly, requires_grad=True)
optimizer_g = torch.optim.Adam([w_global], lr=0.01)

# MLP Model
mlp_model = MLP()
optimizer_m = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

# KAN Model
w_local  = torch.zeros(num_params_kan, requires_grad=True)
# centers  = torch.linspace(0, 1, num_params_kan)
c1 = torch.linspace(0, 0.4, 8)       # 8 points on the far left
c2 = torch.linspace(0.41, 0.59, 8)  # 8 points PACKED TIGHTLY around the kink
c3 = torch.linspace(0.6, 1.0, 8)       # 8 points on the far right
centers = torch.cat([c1, c2, c3])
assert len(centers) == num_params_kan

initial_width = 0.15
log_width = torch.nn.Parameter(torch.tensor(np.log(initial_width)))
optimizer_l = torch.optim.Adam([
    {'params': [w_local], 'lr': 0.01},
    {'params': [log_width], 'lr': 0.005}
])
memory_penalty_strength = 200.0

# Helper to get flattened parameters for norm calculation
def get_flat_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])


# --- 4. The Experiment ---

print("--- PHASE 1: Training on Task A (Left Side = Cosine Wave) ---")
for i in range(epochs):
    # Global Train
    optimizer_g.zero_grad(); pred_g = global_model(x[mask_A], w_global); loss_g = torch.mean((pred_g - y_A_target[mask_A])**2); loss_g.backward(); optimizer_g.step()
    # MLP Train
    optimizer_m.zero_grad(); pred_m = mlp_model(x[mask_A]); loss_m = torch.mean((pred_m - y_A_target[mask_A])**2); loss_m.backward(); optimizer_m.step()
    # Local Train
    optimizer_l.zero_grad(); pred_l = local_kan_model(x[mask_A], w_local, centers, log_width); loss_l = torch.mean((pred_l - y_A_target[mask_A])**2); loss_l.backward(); optimizer_l.step()
    
    if i % (epochs//5) == 0:
        print(f"Epoch {i:4d} | Poly Loss: {loss_g.item():.6f} | MLP Loss: {loss_m.item():.6f} | KAN Loss: {loss_l.item():.6f}")

# Snapshots after Phase 1
w_global_after_A = w_global.clone().detach()
mlp_params_after_A = get_flat_params(mlp_model).clone().detach()
w_local_after_A  = w_local.clone().detach()
print("\nPhase 1 complete. Weights have been snapshotted.")


print("\n--- PHASE 2: Training on Task B (Right Side = Sine Wave) ---")
for i in range(epochs):
    # Global Train (no penalty)
    optimizer_g.zero_grad(); pred_g = global_model(x[mask_B], w_global); loss_g = torch.mean((pred_g - y_B_target[mask_B])**2); loss_g.backward(); optimizer_g.step()
    # MLP Train (no penalty)
    optimizer_m.zero_grad(); pred_m = mlp_model(x[mask_B]); loss_m = torch.mean((pred_m - y_B_target[mask_B])**2); loss_m.backward(); optimizer_m.step()
    # KAN Train (with memory penalty)
    optimizer_l.zero_grad()
    pred_l = local_kan_model(x[mask_B], w_local, centers, log_width)
    task_loss = torch.mean((pred_l - y_B_target[mask_B])**2)
    

    left_mask_kan = centers < 0.5
    
    forgetting_penalty = torch.mean((w_local[left_mask_kan] - w_local_after_A[left_mask_kan])**2)
    total_loss = task_loss + memory_penalty_strength * forgetting_penalty
    total_loss.backward()
    optimizer_l.step()

    if i % (epochs//5) == 0:
        print(f"Epoch {i:4d} | Poly Loss: {loss_g.item():.6f} | MLP Loss: {loss_m.item():.6f} | KAN Task Loss: {task_loss.item():.6f}")


# --- 5. Final Analysis & Quantitative Report ---
print("\n--- FINAL DIAGNOSTIC REPORT ---")
with torch.no_grad():
    # --- Forgetting Metrics ---
    forget_score_g = torch.mean((global_model(x[mask_A], w_global) - y_A_target[mask_A])**2)
    forget_score_m = torch.mean((mlp_model(x[mask_A]) - y_A_target[mask_A])**2)
    forget_score_l = torch.mean((local_kan_model(x[mask_A], w_local, centers, log_width) - y_A_target[mask_A])**2)
    print("\n1. Forgetting Metric (MSE on Task A):")
    print(f"   - Polynomial Model: {forget_score_g.item():.6f}")
    print(f"   - MLP Model:        {forget_score_m.item():.6f}")
    print(f"   - KAN Model:        {forget_score_l.item():.6f}")

    # --- Weight Change Metrics ---
    delta_w_global = torch.norm(w_global - w_global_after_A)
    delta_w_mlp = torch.norm(get_flat_params(mlp_model) - mlp_params_after_A)
    left_mask_kan = centers < 0.5
    delta_w_local_left = torch.norm(w_local[left_mask_kan] - w_local_after_A[left_mask_kan])
    delta_w_local_right = torch.norm(w_local[~left_mask_kan] - w_local_after_A[~left_mask_kan])
    print("\n2. Weight Change (L2 Norm) during Phase 2:")
    print(f"   - Polynomial Model (All): {delta_w_global.item():.4f}")
    print(f"   - MLP Model (All):        {delta_w_mlp.item():.4f}")
    print(f"   - KAN Model (Left):       {delta_w_local_left.item():.4f}")
    print(f"   - KAN Model (Right):      {delta_w_local_right.item():.4f}")

    # --- 6. Visualization ---
    plt.figure(figsize=(18, 5))

    # Plot 1: Polynomial Model
    plt.subplot(1, 3, 1)
    plt.title(f"Polynomial Model (Global)\nCatastrophic Forgetting!\nTask A MSE: {forget_score_g.item():.4f}")
    plt.plot(x.numpy(), y_full_target.numpy(), 'k:', label="Full Target", alpha=0.3)
    plt.plot(x.numpy(), global_model(x, w_global).numpy(), 'r-', label="Prediction")
    plt.legend(); plt.ylim(-0.2, 1.2)

    # Plot 2: MLP Model
    plt.subplot(1, 3, 2)
    plt.title(f"MLP Model (Global)\nCatastrophic Forgetting!\nTask A MSE: {forget_score_m.item():.4f}")
    plt.plot(x.numpy(), y_full_target.numpy(), 'k:', label="Full Target", alpha=0.3)
    plt.plot(x.numpy(), mlp_model(x).numpy(), 'purple', label="Prediction")
    plt.legend(); plt.ylim(-0.2, 1.2)

    # Plot 3: KAN Model
    plt.subplot(1, 3, 3)
    final_width = torch.exp(log_width).item()
    plt.title(f"KAN Model (Local)\nMemory Preserved\nTask A MSE: {forget_score_l.item():.4f} | Width: {final_width:.3f}")
    plt.plot(x.numpy(), y_full_target.numpy(), 'k:', label="Full Target", alpha=0.3)
    plt.plot(x.numpy(), local_kan_model(x, w_local, centers, log_width).numpy(), 'g-', label="Prediction")
    plt.scatter(centers.numpy(), w_local.detach().numpy(), s=15, color='green', marker='x', label="Control Points")
    plt.legend(); plt.ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.show()