import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Setup: MORE COMPLEX TASKS ---
x = torch.linspace(0, 1, 200)

# Task A: A Cosine wave on the left side
mask_A = (x < 0.5)
y_A_target = torch.zeros_like(x)
y_A_target[mask_A] = torch.cos(2 * torch.pi * x[mask_A]) * 0.5 + 0.5 # Cosine from 0 to 1, then scaled

# Task B: A Sine wave on the right side
mask_B = (x >= 0.5)
y_B_target = torch.zeros_like(x)
y_B_target[mask_B] = torch.sin(2 * torch.pi * (x[mask_B] - 0.5)) * 0.5 + 0.5 # Sine from 0 to 1, scaled and shifted


# --- 2. The Models (Identical to before, good!) ---
def global_model(x, weights):
    basis = torch.stack([x**i for i in range(len(weights))], dim=1)
    return basis @ weights

def local_kan_model(x, weights, centers, log_width):
    width = torch.exp(log_width)
    basis = torch.exp(-(x.unsqueeze(1) - centers.unsqueeze(0))**2 / (2 * width**2))
    return basis @ weights


# --- 3. Parameters and Optimizers (Adjusted for complexity) ---
num_params = 20 # Increased capacity for more complex functions
epochs = 1000   # More epochs for harder tasks

w_global = torch.zeros(num_params, requires_grad=True)
optimizer_g = torch.optim.Adam([w_global], lr=0.01) # Slightly reduced LR for stability

w_local  = torch.zeros(num_params, requires_grad=True)
centers  = torch.linspace(0, 1, num_params) 

# Initial width guess is still wide.
initial_width = 0.15 # Slightly wider to start, as tasks are more complex.
log_width = torch.nn.Parameter(torch.tensor(np.log(initial_width)))

# Optimizer for KAN: Adjusted LR for width to encourage more precise tuning
optimizer_l = torch.optim.Adam([
    {'params': [w_local], 'lr': 0.01},
    {'params': [log_width], 'lr': 0.005} # Smaller LR for width to prevent overshooting
])

# CRITICAL: Memory preservation penalty strength
memory_penalty_strength = 200.0 # Increased penalty to strictly enforce memory


# --- 4. The Experiment ---

print("--- PHASE 1: Training on Task A (Left Side = Cosine Wave) ---")
for i in range(epochs):
    # Global Train
    optimizer_g.zero_grad()
    pred_g = global_model(x[mask_A], w_global)
    loss_g = torch.mean((pred_g - y_A_target[mask_A])**2)
    loss_g.backward()
    optimizer_g.step()
    
    # Local Train
    optimizer_l.zero_grad()
    pred_l = local_kan_model(x[mask_A], w_local, centers, log_width)
    loss_l = torch.mean((pred_l - y_A_target[mask_A])**2)
    loss_l.backward()
    optimizer_l.step()

    if i % (epochs//5) == 0:
        print(f"Epoch {i:4d} | Global Loss: {loss_g.item():.6f} | Local KAN Loss: {loss_l.item():.6f}")


w_global_after_A = w_global.clone().detach()
w_local_after_A  = w_local.clone().detach()
print("\nPhase 1 complete. Weights have been snapshotted.")


print("\n--- PHASE 2: Training on Task B (Right Side = Sine Wave) ---")
print(f"Initial KAN width for Phase 2: {torch.exp(log_width).item():.4f}")

for i in range(epochs):
    # Global Train (no penalty)
    optimizer_g.zero_grad()
    pred_g = global_model(x[mask_B], w_global)
    loss_g = torch.mean((pred_g - y_B_target[mask_B])**2)
    loss_g.backward()
    optimizer_g.step()

    # Local Train (with memory penalty)
    optimizer_l.zero_grad()
    pred_l = local_kan_model(x[mask_B], w_local, centers, log_width)
    task_loss = torch.mean((pred_l - y_B_target[mask_B])**2)
    
    left_mask = centers < 0.5
    current_left_weights = w_local[left_mask]
    original_left_weights = w_local_after_A[left_mask]
    forgetting_penalty = torch.mean((current_left_weights - original_left_weights)**2)
    
    total_loss = task_loss + memory_penalty_strength * forgetting_penalty
    
    total_loss.backward()
    optimizer_l.step()
    
    if i % (epochs//5) == 0:
        current_width = torch.exp(log_width).item()
        print(f"Epoch {i:4d} | Task Loss: {task_loss.item():.6f} | Penalty: {forgetting_penalty.item():.6f} | Learned Width: {current_width:.4f}")

print("\nPhase 2 complete.")


# --- 5. Final Analysis & Quantitative Report ---
print("\n--- FINAL DIAGNOSTIC REPORT ---")

with torch.no_grad():
    # Metric 1: The Forgetting Score
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
    # The plot should show both task A and task B targets for context
    plt.plot(x.numpy(), y_A_target.numpy() + y_B_target.numpy(), 'k:', label="Full Target (Both Tasks)", alpha=0.3)
    plt.plot(x[mask_A].numpy(), y_A_target[mask_A].numpy(), 'k--', label="Task A Target", alpha=0.7)
    plt.plot(x[mask_B].numpy(), y_B_target[mask_B].numpy(), 'b--', label="Task B Target", alpha=0.7)
    plt.plot(x.numpy(), y_pred_global.detach().numpy(), 'r-', label="Prediction")
    
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.legend()
    plt.ylim(-0.2, 1.2) # Adjusted limits for the new target functions

    # Plot Local (Success)
    plt.subplot(1, 2, 2)
    final_width = torch.exp(log_width).item()
    title_l = f"KAN Model (Local)\nMemory Preserved\nTask A MSE: {forget_score_l.item():.4f} | Learned Width: {final_width:.3f}"
    plt.title(title_l)
    plt.plot(x.numpy(), y_A_target.numpy() + y_B_target.numpy(), 'k:', label="Full Target (Both Tasks)", alpha=0.3)
    plt.plot(x[mask_A].numpy(), y_A_target[mask_A].numpy(), 'k--', label="Task A Target", alpha=0.7)
    plt.plot(x[mask_B].numpy(), y_B_target[mask_B].numpy(), 'b--', label="Task B Target", alpha=0.7)
    plt.plot(x.numpy(), y_pred_local.detach().numpy(), 'g-', label="Prediction")
    
    plt.scatter(centers.numpy(), w_local.detach().numpy(), s=20, color='green', marker='x', label="Control Points")
    plt.axvline(0.5, color='gray', linestyle=':')
    plt.legend()
    plt.ylim(-0.2, 1.2) # Adjusted limits
    
    plt.tight_layout()
    plt.show()