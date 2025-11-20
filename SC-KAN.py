import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Setup ---
# This section defines the two distinct tasks the models must learn sequentially.
x = torch.linspace(0, 1, 200)

# Task A: The left side of the domain should be flat at zero.
mask_A = (x < 0.5)
y_A_target = torch.zeros_like(x)

# Task B: The right side of the domain should have a sharp Gaussian bump.
mask_B = (x >= 0.5)
y_B_target = torch.zeros_like(x)
y_B_target[mask_B] = torch.exp(-100 * (x[mask_B] - 0.75)**2)


# --- 2. The Models ---
# We define two models to compare their ability to learn sequentially.

# MODEL 1: A standard model using global basis functions (polynomials).
# This mimics the behavior of a simple Multi-Layer Perceptron (MLP).
def global_model(x, weights):
    basis = torch.stack([x**i for i in range(len(weights))], dim=1)
    return basis @ weights

# MODEL 2: Our KAN-style model using local basis functions (Gaussians).
# CRITICAL: This version takes `log_width` as a parameter to learn it.
def local_kan_model(x, weights, centers, log_width):
    # Width is calculated as exp(log_width) to ensure it's always positive.
    width = torch.exp(log_width)
    
    # Gaussian basis functions (Radial Basis Functions).
    # Unsqueezing allows for correct broadcasting of dimensions.
    basis = torch.exp(-(x.unsqueeze(1) - centers.unsqueeze(0))**2 / (2 * width**2))
    return basis @ weights


# --- 3. Parameters and Optimizers ---
# Setup the learnable parameters for both models.
num_params = 10
epochs = 500

# Parameters for the global model
w_global = torch.zeros(num_params, requires_grad=True)
optimizer_g = torch.optim.Adam([w_global], lr=0.1)

# Parameters for the local KAN model
w_local  = torch.zeros(num_params, requires_grad=True)
centers  = torch.linspace(0, 1, num_params) # Fixed positions for our "neurons"

# KEY CHANGE: `log_width` is now a learnable parameter.
# We initialize it to a "bad" (wide) value to prove the model can learn to fix it.
initial_width = 0.1
log_width = torch.nn.Parameter(torch.tensor(np.log(initial_width)))

# The KAN optimizer now manages both the weights and the width parameter.
# We can assign different learning rates to different parameter groups.
optimizer_l = torch.optim.Adam([
    {'params': [w_local], 'lr': 0.1},
    {'params': [log_width], 'lr': 0.05} # A slightly smaller LR for width is often more stable
])


# --- 4. The Experiment ---

print("--- PHASE 1: Training on Task A (Left Side = 0) ---")
# Both models learn the first task.
for i in range(epochs):
    optimizer_g.zero_grad()
    pred_g = global_model(x[mask_A], w_global)
    loss_g = torch.mean((pred_g - y_A_target[mask_A])**2)
    loss_g.backward()
    optimizer_g.step()
    
    optimizer_l.zero_grad()
    pred_l = local_kan_model(x[mask_A], w_local, centers, log_width)
    loss_l = torch.mean((pred_l - y_A_target[mask_A])**2)
    loss_l.backward()
    optimizer_l.step()

# Snapshot weights after Phase 1 to measure how much they change later.
w_global_after_A = w_global.clone().detach()
w_local_after_A  = w_local.clone().detach()
print("\nPhase 1 complete. Weights have been snapshotted.")


print("\n--- PHASE 2: Training on Task B (Right Side = Spike) ---")
print(f"Initial KAN width for Phase 2: {torch.exp(log_width).item():.4f}")

# --- NEW: Define the penalty for forgetting ---
# `lambda` is the strength of our memory preservation penalty.
# A high value means "Do NOT forget, at all costs."
memory_penalty_strength = 100.0 

for i in range(epochs):
    # (Global model training remains the same)
    # ...
    optimizer_g.step()

    # --- KAN Training Loop (MODIFIED) ---
    optimizer_l.zero_grad()
    
    # 1. Calculate the primary loss on the new task (Task B)
    pred_l = local_kan_model(x[mask_B], w_local, centers, log_width)
    task_loss = torch.mean((pred_l - y_B_target[mask_B])**2)
    
    # 2. Calculate the penalty for changing the "old" weights
    left_mask = centers < 0.5
    current_left_weights = w_local[left_mask]
    original_left_weights = w_local_after_A[left_mask]
    forgetting_penalty = torch.mean((current_left_weights - original_left_weights)**2)
    
    # 3. The total loss is a combination of both
    total_loss = task_loss + memory_penalty_strength * forgetting_penalty
    
    total_loss.backward()
    optimizer_l.step()
    
    if i % (epochs//5) == 0:
        current_width = torch.exp(log_width).item()
        # Log both parts of the loss to see what's happening
        print(f"Epoch {i:3d} | Task Loss: {task_loss.item():.6f} | Penalty: {forgetting_penalty.item():.6f} | Learned Width: {current_width:.4f}")

print("\nPhase 2 complete.")


# --- 5. Final Analysis & Quantitative Report ---
print("\n--- FINAL DIAGNOSTIC REPORT ---")

with torch.no_grad():
    # Metric 1: The Forgetting Score (MSE on the old task)
    final_pred_g_A = global_model(x[mask_A], w_global)
    forget_score_g = torch.mean((final_pred_g_A - y_A_target[mask_A])**2)
    final_pred_l_A = local_kan_model(x[mask_A], w_local, centers, log_width)
    forget_score_l = torch.mean((final_pred_l_A - y_A_target[mask_A])**2)

    print("\n1. Forgetting Metric (MSE on Task A after learning Task B):")
    print(f"   - Standard Model: {forget_score_g.item():.6f}")
    print(f"   - Local KAN Model:  {forget_score_l.item():.6f}")

    # Metric 2: Weight Change Analysis
    delta_w_global = torch.norm(w_global - w_global_after_A)
    left_mask = centers < 0.5
    right_mask = centers >= 0.5
    delta_w_local_left = torch.norm(w_local[left_mask] - w_local_after_A[left_mask])
    delta_w_local_right = torch.norm(w_local[right_mask] - w_local_after_A[right_mask])

    print("\n2. Weight Change (L2 Norm) during Phase 2:")
    print(f"   - Standard Model (All Weights): {delta_w_global.item():.4f}")
    print(f"   - Local KAN Model (Left Weights):  {delta_w_local_left.item():.4f}")
    print(f"   - Local KAN Model (Right Weights): {delta_w_local_right.item():.4f}")
    
    final_learned_width = torch.exp(log_width).item()
    print(f"\n3. Final Learned Width for KAN Model: {final_learned_width:.4f}")

    # --- 6. Visualization ---
    y_pred_global = global_model(x, w_global)
    y_pred_local  = local_kan_model(x, w_local, centers, log_width)

    plt.figure(figsize=(12, 6))

    # Plot Global (Failure)
    plt.subplot(1, 2, 1)
    title_g = f"Standard Model (Global)\nCatastrophic Forgetting!\nTask A MSE: {forget_score_g.item():.4f}"
    plt.title(title_g)
    plt.plot(x.numpy(), y_pred_global.detach().numpy(), 'r-', label="Prediction")
    plt.plot(x[mask_A].numpy(), y_A_target[mask_A].numpy(), 'k--', label="Task A (Forgotten)", alpha=0.5)
    plt.plot(x[mask_B].numpy(), y_B_target[mask_B].numpy(), 'b--', label="Task B (Current)", alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.legend()
    plt.ylim(-1.5, 1.5)

    # Plot Local (Success)
    plt.subplot(1, 2, 2)
    title_l = f"KAN Model (Local)\nMemory Preserved\nTask A MSE: {forget_score_l.item():.4f} | Learned Width: {final_learned_width:.3f}"
    plt.title(title_l)
    plt.plot(x.numpy(), y_pred_local.detach().numpy(), 'g-', label="Prediction")
    plt.plot(x[mask_A].numpy(), y_A_target[mask_A].numpy(), 'k--', label="Task A (Safe)", alpha=0.5)
    plt.plot(x[mask_B].numpy(), y_B_target[mask_B].numpy(), 'b--', label="Task B (Learned)", alpha=0.5)
    plt.scatter(centers.numpy(), w_local.detach().numpy(), s=20, color='green', marker='x', label="Control Points")
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.legend()
    
    plt.tight_layout()
    plt.show()